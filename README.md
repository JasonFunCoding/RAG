# Retrieval-Augmented Generation (RAG) Demo

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using Python, Sentence Transformers for semantic search, and the GROQ LLM API for answer generation.

## Features
- Embeds a set of sample documents using a sentence transformer model.
- Retrieves the most relevant document for a user question using semantic similarity.
- Uses the GROQ LLM API to generate an answer based on the retrieved context.

## Requirements
- Python 3.8+
- [sentence-transformers](https://www.sbert.net/)
- [httpx](https://www.python-httpx.org/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- numpy

## Setup
1. **Clone the repository**
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   Or install manually:
   ```sh
   pip install sentence-transformers httpx python-dotenv numpy
   ```
3. **Set your GROQ API key:**
   - The script expects a `GROQ_API_KEY` environment variable. You can set it in a `.env` file or directly in the script.
   - Example `.env` file:
     ```env
     GROQ_API_KEY=your_groq_api_key_here
     ```

## Usage
Run the main script:
```sh
python Main.py
```

The script will:
- Embed the sample documents
- Retrieve the most relevant document for a sample question
- Query the GROQ LLM API to generate an answer

## Customization
- To use your own documents, modify the `documents` list in `Main.py`.
- To use a different LLM or retrieval model, update the relevant sections in `Main.py`.

## Example Output
```
context: The battery lasts 6 hours on a single charge after 6 months.
Question: Could you tell me how long does the phone’s battery last after 16 months?
Answer: ...
```

## License
This project is for educational/demo purposes only.
