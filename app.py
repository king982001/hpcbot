# Backend server code (app.py)
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load trained model and tokenizer
model_path = "/path/to/your/trained/model/directory"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# API endpoint for text generation
@app.route('/generate', methods=['POST'])
def generate_text():
    # Get user input from request
    user_input = request.json['input_text']

    # Tokenize the input text
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate text using the model
    outputs = model.generate(input_ids, max_length=150, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the generated text as JSON response
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
