# E-T-A RAG Prototype - Lightning AI Studio Setup Guide

Follow these steps to run the E-T-A RAG system from scratch on your Lightning AI Studio, assuming you have just cloned the repo and run `pip install -r requirements.txt`.

### Step 1: Install Ollama (Local LLM Runtime)
You need Ollama installed on the Lightning AI instance to act as the runner for your local models. Since the terminal is Linux-based, use the Linux installer:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Start the Ollama Server
Ollama needs to be running in the background to serve the LLM. Run this command:
```bash
ollama serve &
```
*(Note: If you get an "address already in use" error, it means the server is already running and you can skip this step!)*

### Step 3: Download the LLM Model
You now need to download the heavy LLM into Ollama. Based on your previous commands, you want to use the powerful Llama 3 70B model.
```bash
ollama pull llama3:70b
```
*(Warning: This is a ~40GB download and requires significant VRAM/RAM [>40GB]. If your Studio instance is smaller, consider running `ollama pull mistral` instead).*

### Step 4: Configure the Project to Use Your Chosen Model
By default, the project is configured to look for the `mistral` model. To ensure it uses `llama3:70b` instead, export this environment variable in your terminal session:
```bash
export OLLAMA_MODEL="llama3:70b"
```
*(Note: You must run this export command every time you open a new terminal window to use the model).*

### Step 5: Ingest the Company Knowledge Data
The RAG system needs documents to answer questions from. We need to run the ingestion script to break the `data/` folder files into chunks and save them in the local vector database.
```bash
python scripts/ingest.py
```
*(You must do this every time you add or change files in the `data/` folder).*

### Step 6: Start Using the RAG System!
Now that everything is set up, you can interact with the assistant.

**Option A: Streamlit UI (Recommended for UI Demo)**
```bash
streamlit run api/ui.py
```
The Lightning AI interface should give you a link to click on (or pop open a new tab) allowing you to access the web interface.

**Option B: Terminal CLI (For Quick Testing)**
If you just want to test it via your terminal without opening a browser window:
```bash
python scripts/chat.py
```
