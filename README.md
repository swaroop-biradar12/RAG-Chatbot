# Local RAG Chatbot

A fully local, private chatbot built with Ollama, Flask, and LangChain. Runs entirely on your machine — no external APIs, no data leaving your computer. Chat with any local LLM, switch models on the fly, and upload documents (PDF/TXT) to ask questions grounded in their actual content using Retrieval-Augmented Generation (RAG).

## Features

- **Local LLM chat** — powered by [Ollama](https://ollama.com), streamed token-by-token in the browser
- **Model switcher** — swap between any installed Ollama chat model without touching code
- **Document Q&A (RAG)** — upload a PDF or text file, and ask questions answered using retrieved content from that document instead of the model's general knowledge
- **Chat UI** — bubble-style chat with avatars, dark mode, and Markdown rendering
- **Fully local** — embeddings, vector storage, and generation all run on-device; nothing is sent to a third-party API

## Tech stack

| Layer | Tool |
|---|---|
| LLM runtime | [Ollama](https://ollama.com) |
| Chat model | `llama3.2` (or any Ollama model) |
| Embedding model | `nomic-embed-text` |
| Backend | Flask (Python) |
| RAG framework | LangChain |
| Vector store | Chroma |
| Frontend | Vanilla HTML/CSS/JS |

## Architecture

```
User → Flask (app.py) → Ollama (chat model)
                       ↘
                        rag.py → LangChain → Chroma vector store
                                            ↖
                                    nomic-embed-text (embeddings)
```

**Indexing flow (on document upload):**
1. Document is loaded (`PyPDFLoader` / `TextLoader`)
2. Split into ~800-character overlapping chunks (`RecursiveCharacterTextSplitter`)
3. Each chunk is embedded via `nomic-embed-text` and stored in a local Chroma database

**Query flow (per chat message):**
1. If "Use document" is enabled, the question is embedded and matched against stored chunks via similarity search
2. The most relevant chunks are inserted into the prompt alongside the question
3. The augmented prompt is sent to the selected chat model, and the response is streamed back to the browser

## Setup

### Prerequisites
- [Ollama](https://ollama.com) installed and running
- Python 3.10+

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/local-rag-chatbot.git
cd local-rag-chatbot
```

### 2. Install dependencies
```bash
pip install flask ollama langchain langchain-community langchain-text-splitters langchain-ollama langchain-chroma chromadb pypdf
```

### 3. Pull required models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

## Usage

- Type a message and hit **Send** to chat normally
- Use the **model dropdown** to switch between installed Ollama chat models
- Click **Upload doc** to index a PDF or text file
- Check **Use document** to have the model answer using retrieved content from the uploaded file
- Click **Reset** to clear the current conversation
- Toggle **Dark mode** from the top bar

## Known limitations

- Basic similarity search retrieves a fixed number of chunks — works well for specific factual questions, but not for whole-document requests like "summarize this file" (would need a separate map-reduce summarization approach)
- Chat history and document index are stored in memory and reset when the Flask server restarts
- No authentication — intended for local, single-user use

## Project structure

```
.
├── app.py              # Flask app: routes for chat, model switching, upload
├── rag.py               # Document loading, chunking, embedding, retrieval
├── chat.py               # Standalone terminal chat script (early prototype)
├── templates/
│   └── index.html        # Frontend: chat UI, model dropdown, upload controls
├── uploads/               # Uploaded documents (gitignored)
├── chroma_db/             # Vector store data (gitignored)
└── .gitignore
```

## License

MIT
