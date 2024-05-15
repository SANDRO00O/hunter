from flask import Flask, render_template, request, jsonify
import torch
from transformers import GPTJForCausalLM, AutoTokenizer
import os

app = Flask(__name__)

# Load model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
model = GPTJForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load secret token from environment variable
hf_token = os.getenv('HF_TOKEN')

def generate_code(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=max_length, do_sample=True)
    code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return code

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Code Generator</title>
    </head>
    <body>
        <h1>Code Generator</h1>
        <form id="codeForm" method="POST" action="/generate">
            <label for="prompt">Enter your prompt:</label><br>
            <input type="text" id="prompt" name="prompt" style="width: 80%;" required><br><br>
            <label for="token">Enter your token:</label><br>
            <input type="text" id="token" name="token" style="width: 80%;" required><br><br>
            <button type="submit">Generate Code</button>
        </form>
        <div id="codeOutput"></div>
    </body>
    </html>
    """

@app.route('/generate', methods=['POST'])
def generate():
    data = request.form
    prompt = data.get('prompt', '')
    token = data.get('token')

    # Check token validity
    if token != hf_token:
        return jsonify({'error': 'Invalid token'}), 401

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    generated_code = generate_code(prompt)
    return jsonify({'code': generated_code})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('PORT', 5000))
