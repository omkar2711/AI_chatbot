import os
import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq

# Load environment variables and API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("API key for Groq is missing. Please set the GROQ_API_KEY in the .env file.")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize session state for chat history if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a useful AI assistant."}
    ]

# Page configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("AI Chatbot")

# Function to query Groq API
def query_groq_api(messages: List[Dict[str, str]]) -> str:
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        
        return response
    
    except Exception as e:
        st.error(f"Error with Groq API: {str(e)}")
        return None

# Display chat messages
for message in st.session_state.messages[1:]:  # Skip the system message
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_groq_api(st.session_state.messages)
            if response:
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Add a button to clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a useful AI assistant."}
    ]
    st.rerun()