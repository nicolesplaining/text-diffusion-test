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
    # Clear the diffusion output file before starting generation.
    output_file_path = os.path.join(app.root_path, "static", "diffusion_output.txt")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("")
    # Launch generate.py as a background process so it writes intermediate steps to file.
    run_generate(prompt)
    return jsonify({'status': 'success'})

@app.route('/get_diffusion', methods=['GET'])
def get_diffusion():
    output_file_path = os.path.join(app.root_path, "static", "diffusion_output.txt")
    if os.path.exists(output_file_path):
        with open(output_file_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = ""
    return jsonify({'output': content})

def run_generate(prompt):
    # Launch generate.py in unbuffered mode so it writes intermediate steps to file in real time.
    cmd = ["python", "-u", "generate.py", "--prompt", prompt]
    # Use Popen so that the process runs asynchronously.
    subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)