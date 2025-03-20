import os
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

# Retrieve the Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Ensure the API key is available; otherwise, raise an error
if not GROQ_API_KEY:
    raise ValueError("API key for Groq is missing. Please set the GROQ_API_KEY in the .env file.")

# Initialize the FastAPI application
app = FastAPI()

# Enable CORS to allow frontend applications to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains (can be restricted for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize the Groq API client with the retrieved API key
client = Groq(api_key=GROQ_API_KEY)

# Define a data model for user input validation
class UserInput(BaseModel):
    message: str  # User's message
    role: str = "user"  # Default role set as "user"
    conversation_id: str  # Unique ID for each conversation

# Define a class to store conversation messages and state
class Conversation:
    def __init__(self):
        # Initialize conversation with a system message
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a useful AI assistant."}
        ]
        self.active: bool = True  # Track whether the conversation is active

# Dictionary to store ongoing conversations using conversation IDs as keys
conversations: Dict[str, Conversation] = {}

# Function to send messages to Groq API and retrieve a response
def query_groq_api(conversation: Conversation) -> str:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # AI model to use
            messages=conversation.messages,  # Pass conversation history
            temperature=1,  # Controls randomness in responses (1 = balanced)
            max_tokens=1024,  # Limit response length
            top_p=1,  # Controls probability mass of token selection
            stream=True,  # Enables streaming of responses
            stop=None,  # No stop condition
        )
        
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""  # Aggregate response chunks
        
        return response
    
    except Exception as e:
        # Return an HTTP error if API call fails
        raise HTTPException(status_code=500, detail=f"Error with Groq API: {str(e)}")

# Function to retrieve an existing conversation or create a new one
def get_or_create_conversation(conversation_id: str) -> Conversation:
    if conversation_id not in conversations:
        conversations[conversation_id] = Conversation()  # Create new conversation if not found
    return conversations[conversation_id]

# API endpoint for handling chatbot conversations
@app.post("/chat/")
async def chat(input: UserInput):
    conversation = get_or_create_conversation(input.conversation_id)  # Retrieve conversation

    # Check if the conversation is still active
    if not conversation.active:
        raise HTTPException(
            status_code=400, 
            detail="The chat session has ended. Please start a new session."
        )
        
    try:
        # Append user's message to conversation history
        conversation.messages.append({
            "role": input.role,
            "content": input.message
        })
        
        # Send the conversation history to the AI model and get a response
        response = query_groq_api(conversation)
        
        # Append AI's response to conversation history
        conversation.messages.append({
            "role": "assistant",
            "content": response
        })
        
        return {
            "response": response,
            "conversation_id": input.conversation_id  # Return conversation ID for tracking
        }
        
    except Exception as e:
        # Handle any errors that occur
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
