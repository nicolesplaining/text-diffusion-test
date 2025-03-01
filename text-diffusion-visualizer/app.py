import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load model and tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None
tokenizer = None
MASK_ID = 126336
EOS_ID = 126081

def load_model():
    global model, tokenizer
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    print("Model and tokenizer loaded successfully.")

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@torch.no_grad()
def generate_with_states(model, prompt, steps=128, gen_length=128, block_length=32, temperature=0.,
                 cfg_scale=0., remasking='low_confidence', mask_id=MASK_ID):
    '''
    Modified version of generate function that records all states
    '''
    states = []
    
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    # Save initial state
    states.append(x.clone().cpu().tolist()[0])

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
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
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
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

            # Save current state
            states.append(x.clone().cpu().tolist()[0])

    return x, states

@app.route('/api/generate', methods=['POST'])
def api_generate():
    global model, tokenizer
    
    if model is None or tokenizer is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({'error': f'Failed to load model: {str(e)}'}), 500
    
    data = request.json
    user_input = data.get('prompt', '')
    
    if not user_input:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        # Fixed parameters as per chat.py
        gen_length = 128
        steps = 128
        block_length = 32
        temperature = 0.
        cfg_scale = 0.
        
        # Process the input as in chat.py
        m = [{"role": "user", "content": user_input}]
        processed_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(processed_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        
        # Generate with states
        out, states = generate_with_states(
            model, 
            input_ids, 
            steps=steps, 
            gen_length=gen_length, 
            block_length=block_length,
            temperature=temperature, 
            cfg_scale=cfg_scale, 
            remasking='low_confidence'
        )
        
        # Get the answer
        answer = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        # Get token representations for visualization
        token_representations = []
        
        # For each state, convert to human-readable tokens
        for i, state in enumerate(states):
            tokens = []
            for j, token_id in enumerate(state):
                # Get token text
                if j < len(state):
                    token_text = tokenizer.decode([token_id]) if token_id != MASK_ID else "[MASK]"
                    token_type = "prompt" if j < input_ids.shape[1] else ("mask" if token_id == MASK_ID else "generated")
                    tokens.append({
                        "id": int(token_id),
                        "text": token_text,
                        "type": token_type
                    })
            token_representations.append(tokens)
        
        return jsonify({
            'answer': answer,
            'states': token_representations,
            'num_states': len(states),
            'prompt_length': input_ids.shape[1],
            'config': {
                'gen_length': gen_length,
                'steps': steps,
                'block_length': block_length
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model at startup
    load_model()
    app.run(host='0.0.0.0', port=3000, debug=False)