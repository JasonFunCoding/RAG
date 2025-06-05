from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import httpx
import logging
import numpy as np
import os

# Step 1: Our sample documents (imagine these are from your company manual)
documents = [
    "The phone has a 48MP camera with night mode.",
    "The battery lasts 6 hours on a single charge after 6 months.",
    "The battery lasts 12 hours on a single charge when it is new.",
    "The phone supports 5G and has 128GB of storage."
]

# Step 2: Load a model to create embeddings (for retrieval)
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Convert documents to embeddings (numerical representations)
document_embeddings = retriever_model.encode(documents)

# Step 4: Replace HuggingFace pipeline with GROQ LLMClient
class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to the LLM fails.
        """
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "messages": messages,
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None,
        }

        try:
            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")

            return (
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request."
            )

# Load environment variables
load_dotenv()
GROQ_API_KEY = "gsk_vbr85rIiP7tEf96gcD3zWGdyb3FYrOTRVhyOuR1H1yDx5Wl4Ayjm"
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Set it in .env file or environment.")

# Initialize LLM client
llm_client = LLMClient(api_key=GROQ_API_KEY)

# Step 5: Function to find the most relevant document
def retrieve_relevant_document(question):
    # Convert the question to an embedding
    question_embedding = retriever_model.encode([question])
    
    # Calculate similarity between question and documents
    similarities = np.dot(question_embedding, document_embeddings.T)
    
    # Find the index of the most similar document
    most_relevant_idx = np.argmax(similarities)
    
    # Return the most relevant document
    return documents[most_relevant_idx]

# Step 6: Function to generate an answer using RAG
def answer_question(question):
    # Retrieve the most relevant document
    context = retrieve_relevant_document(question)

    print(f"context: {context}")
    
    # Prepare messages for the LLM
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question."},
        {"role": "user", "content": f"Question: {question}\nContext: {context}\nAnswer:"}
    ]
    
    # Generate the answer using GROQ LLMClient
    answer = llm_client.get_response(messages)
    return answer

# Step 7: Test the RAG system
question = "Could you tell me how long does the phone’s battery last after 16 months?"
answer = answer_question(question)
print(f"Question: {question}")
print(f"Answer: {answer}")