import argparse
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os
import sys

# Path to the output file in the static folder.
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "diffusion_output.txt")
NUM_TOKENS = 128
MASK_TOKEN = "<|mdm_mask|>"

def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

def clean_output(decoded):
    """
    Clean the decoded output by:
      - Inserting spaces around each <|mdm_mask|> token so that consecutive masks are separated.
      - Splitting into tokens by whitespace.
      - Padding with empty strings (or truncating) so that the token list is exactly NUM_TOKENS tokens long.
      - Rejoining the tokens with a single space.
    """
    # Insert spaces around each mask token.
    processed = decoded.replace(MASK_TOKEN, f" {MASK_TOKEN} ")
    # Split into tokens.
    tokens = processed.split()
    # Pad with empty strings if needed.
    while len(tokens) < NUM_TOKENS:
        tokens.append("")
    # Truncate if there are too many tokens.
    if len(tokens) > NUM_TOKENS:
        tokens = tokens[:NUM_TOKENS]
    # Rejoin tokens.
    cleaned = " ".join(tokens)
    return cleaned

@torch.no_grad()
def generate(model, tokenizer, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    """
    Args:
        model: The diffusion model.
        tokenizer: The tokenizer used for decoding.
        prompt: A tensor (shape: [1, l]) representing the encoded prompt.
        steps: Total sampling steps (must be ≤ gen_length).
        gen_length: Number of tokens to generate.
        block_length: Block length (if < gen_length, semi-autoregressive remasking is used).
        temperature: Sampling temperature.
        cfg_scale: Classifier-free guidance scale.
        remasking: Either 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] (126336).
    """
    # Create a working buffer: prompt tokens followed by gen_length masked tokens.
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks  # steps per block

    # Loop over each block and each step.
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # shape: [batch_size, seq_len]

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            # Decode the generated portion (after the prompt), without skipping special tokens.
            decoded = tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=False)[0]
            # Clean up the decoded text using our cleaning function.
            cleaned_output = clean_output(decoded)
            output_str = f"Block {num_block+1}, Step {i+1}/{steps}:\n{cleaned_output}\n"
            # Write the intermediate output to the file (overwriting previous content).
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write(output_str)
            sys.stdout.flush()
    return x

def main():
    parser = argparse.ArgumentParser(description="Generate output using the diffusion model.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt for generation.")
    args = parser.parse_args()
    prompt_text = args.prompt

    print("Is GPU available?", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    
    # Format the prompt using the chat template.
    m = [{"role": "user", "content": prompt_text}]
    prompt_formatted = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt_formatted)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, tokenizer, input_ids, steps=128, gen_length=128, block_length=32,
                   temperature=0., cfg_scale=0., remasking='low_confidence')
    generated = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=False)[0]
    print("\nFinal Output:\n", generated)

if __name__ == '__main__':
    main()
