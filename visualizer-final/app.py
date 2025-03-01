from flask import Flask, render_template, request, jsonify
import os
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_output():
    prompt = request.form.get('prompt', '')
    # Run your generate.py script (assumes it accepts a --prompt argument)
    output = run_generate(prompt)
    # Clean up the output:
    output = output.replace("<|endoftext|>", "")
    output = output.replace("<|eot_id|>", "")
    output = output.replace("<|start_header_id|>assistant<|end_header_id|>", "")
    output = output.replace(prompt, "")
    # Discard any warnings by recording the output starting from "Block 1"
    start_index = output.find("Block 1")
    if start_index != -1:
        output = output[start_index:]
    # Save the cleaned output to a file in the static folder.
    output_file_path = os.path.join(app.root_path, "static", "diffusion_output.txt")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(output)
    return jsonify({'status': 'success'})

def run_generate(prompt):
    # Example: call your generate.py via subprocess.
    # Adjust this command if you import the function directly.
    cmd = ["python", "generate.py", "--prompt", prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
