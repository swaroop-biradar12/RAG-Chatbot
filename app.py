from flask import Flask, render_template, request, Response, jsonify
import ollama
import os
from werkzeug.utils import secure_filename
import rag

app = Flask(__name__)

messages = []
current_model = 'llama3.2'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/models', methods=['GET'])
def get_models():
    result = ollama.list()
    model_names = [
        m['model'] for m in result['models']
        if 'embed' not in m['model'].lower()
    ]
    current = current_model if current_model in model_names else (model_names[0] if model_names else None)
    return jsonify({'models': model_names, 'current': current})

@app.route('/set_model', methods=['POST'])
def set_model():
    global current_model, messages
    current_model = request.json.get('model')
    messages = []
    return jsonify({'status': 'ok', 'model': current_model})

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    num_chunks = rag.process_document(filepath)

    return jsonify({'status': 'ok', 'filename': filename, 'chunks': num_chunks})

@app.route('/clear_document', methods=['POST'])
def clear_document():
    rag.clear_index()
    return jsonify({'status': 'cleared'})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    use_rag = request.json.get('use_rag', False)

    if use_rag:
        context = rag.retrieve_context(user_input)
        prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {user_input}"""
    else:
        prompt = user_input

    messages.append({'role': 'user', 'content': prompt})

    def generate():
        full_response = ""
        stream = ollama.chat(
            model=current_model,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            content = chunk['message']['content']
            full_response += content
            yield content

        messages.append({'role': 'assistant', 'content': full_response})

    return Response(generate(), mimetype='text/plain')

@app.route('/reset', methods=['POST'])
def reset():
    messages.clear()
    return {'status': 'cleared'}

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)