⚗ AI Lab Assistant
An AI-powered question answering system for medical laboratory documents. Upload PDF files — lab manuals, test reports, clinical guidelines — and ask questions in plain English. The system reads your documents and returns accurate, sourced answers instantly.
Features
PDF Upload & Indexing — Upload any lab PDF and the system extracts and indexes all text automatically
AI-Powered Q&A — Ask questions and get intelligent answers based on your actual documents
Source Citations — Every answer shows which document it came from
Multi-Document Support — Upload and query multiple PDFs at the same time
Document Management — View, manage, and delete indexed documents
Webhook Integration — Connect external systems to ingest data or query the assistant programmatically
Health Monitoring — Live status showing indexed documents and chunks
Clean Chat UI — Responsive web interface with markdown rendering
How It Works
Code
This architecture is called RAG — Retrieval-Augmented Generation. Instead of relying on the AI's training data alone, it retrieves real content from your documents before generating an answer. This makes answers accurate, specific, and grounded in your actual files.
Tech Stack
Technology
Purpose
Python
Core programming language
FastAPI
Backend web framework and REST API
ChromaDB
Vector database for storing document embeddings
OpenRouter AI
Access to GPT-4o-mini for embeddings and chat
pypdf
PDF text extraction
Uvicorn
ASGI server to run FastAPI
Pydantic
Data validation for API requests and responses
python-dotenv
Secure API key management via .env file
httpx
Async HTTP client for webhook callbacks
HTML / CSS / JS
Frontend chat interface
Project Structure
Code
Installation & Setup
1. Clone or download the project
Bash
2. Install dependencies
Bash
3. Create the .env file
Create a file named .env in the project root and add the following:
Code
Get your free API key at https://openrouter.ai → Sign Up → API Keys
4. Fix PDF reading (important)
Open main.py and find this line inside the process_pdf function:
Python
Change it to:
Python
5. Run the server
Bash
You should see:
Code
6. Open the app
Go to http://localhost:8000 in your browser.
Usage
Upload a PDF — Click the upload zone or drag and drop a PDF file
Wait for indexing — The progress bar shows indexing status
Ask a question — Type your question in the chat box and press Enter
Read the answer — The AI responds with an answer and cites the source document
API Endpoints
Method
Endpoint
Description
GET
/
Serves the frontend UI
POST
/upload
Upload and index a PDF file
POST
/ask
Ask a question and get an AI answer
GET
/documents
List all indexed documents
DELETE
/documents/{doc_id}
Delete a specific document
DELETE
/documents
Delete all documents
GET
/health
System health and stats
POST
/webhook/register
Register a webhook URL
GET
/webhook/registry
List all registered webhooks
DELETE
/webhook/registry/{id}
Remove a webhook
POST
/webhook/ingest
Ingest text data via webhook
POST
/webhook/query
Query via webhook with callback
Common Errors
Error
Cause
Fix
ModuleNotFoundError
Missing packages
Run pip install -r requirements.txt
OPENROUTER_API_KEY not set
Missing .env file
Create .env with your API key
Address already in use
Port 8000 busy
Set PORT=8001 in .env
Invalid PDF
PDF reader bug
Apply the io.BytesIO fix above
No extractable text
Scanned/image PDF
Use a text-based PDF
Connection refused
Server not started
Run python main.py first
Environment Variables
Variable
Default
Description
OPENROUTER_API_KEY
—
Your OpenRouter API key (required)
OPENROUTER_BASE_URL
https://openrouter.ai/api/v1
API base URL
LLM_MODEL
openai/gpt-4o-mini
Language model for answering
EMBEDDING_MODEL
openai/text-embedding-3-small
Model for creating embeddings
CHROMA_PERSIST_DIR
./chroma_db
Folder to store the vector database
MAX_CHUNK_SIZE
500
Maximum characters per text chunk
CHUNK_OVERLAP
50
Overlap between chunks
TOP_K
5
Number of chunks retrieved per query
PORT
8000
Server port
Future Improvements
User authentication and login system
Online deployment (Render, Railway, Hugging Face Spaces)
Mobile app version
Voice input and output
Support for Word documents and plain text files
Lab data visualization with charts and graphs
Integration with Hospital Information Systems (HIS)
Support for multiple AI models (Claude, Gemini, LLaMA)
Author
Zain
