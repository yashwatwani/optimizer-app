import streamlit as st
import pandas as pd
import requests
import json
import os # Keep os import for future flexibility, though st.secrets is used

# --- CONFIGURATION & DATA PATHS ---
VIZ_FILE_PATH = "data/ROAS Data Frame.xlsx"
OPTIMIZER_FILE_PATH = "data/optimiser_df.xlsx"

# --- HUGGING FACE API CONFIGURATION ---
API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_ID = "deepseek-ai/DeepSeek-V3.2-Exp:novita" # Your specified model

# ----------------------------------------------------------------------
# --- API CALL FUNCTION ---
# ----------------------------------------------------------------------

def call_hf_api(messages, model_id):
    """Makes a request to the Hugging Face Inference Router API."""
    try:
        # Load token securely from Streamlit secrets
        hf_token = st.secrets["HF_INFERENCE_TOKEN"] 
    except KeyError:
        st.error("Hugging Face API Token not found. Please add HF_INFERENCE_TOKEN to .streamlit/secrets.toml.")
        return {"choices": [{"message": {"content": "Error: API token is missing."}}]}
        
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": messages,
        "model": model_id,
        "stream": False # We won't use streaming initially for simplicity with standard requests
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed: {e}")
        return {"choices": [{"message": {"content": f"API Error: Could not connect or receive response."}}]}


# ----------------------------------------------------------------------
# --- CONTEXT LOADING & SYSTEM INSTRUCTION ---
# ----------------------------------------------------------------------

def create_system_instruction(df_viz, df_opt):
    """Generates the detailed system instruction for the LLM."""
    viz_structure = df_viz.head(2).to_markdown()
    opt_structure = df_opt.head(2).to_markdown()
    
    instruction = f"""
    You are a specialized Marketing and Optimization Analyst Chatbot for the HBG Marketing Optimization Suite.
    Your goal is to answer user questions based on the structure and constraints of the user's data and optimization model.

    **Core Application Context:**
    1. The Response Curve data structure (`ROAS Data Frame.xlsx`) looks like this:
    {viz_structure}
    2. The Optimizer data structure (`optimiser_df.xlsx`) looks like this:
    {opt_structure}
    3. The optimization objective is: **Maximize Total Revenue** using the provided Coefficient/Alpha/Max Spend parameters (Mitscherlich model form).
    4. You MUST reference these data structures and optimization goals when answering questions about the application's functionality.
    5. Be concise and professional. Do not write code or complex formulas unless specifically asked.
    """
    return instruction

@st.cache_data
def load_data_for_context():
    """Helper function to load data for context."""
    try:
        df_viz = pd.read_excel(VIZ_FILE_PATH, sheet_name=None, index_col=None)
        first_sheet_df = next(iter(df_viz.values())) 
        df_opt = pd.read_excel(OPTIMIZER_FILE_PATH)
        return first_sheet_df, df_opt
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# ----------------------------------------------------------------------
# --- MAIN CHATBOT APP ---
# ----------------------------------------------------------------------
def app_chatbot():
    st.header("💬 Optimization Chatbot")
    st.markdown("Ask me questions about the response curves, optimization model, or data structure.")
    
    # Load data for system instruction
    df_viz_sample, df_opt_sample = load_data_for_context()
    
    if df_viz_sample.empty or df_opt_sample.empty:
        st.warning("Data files not fully loaded. The chatbot will only answer general questions.")
        system_instruction = "You are a helpful chatbot specializing in marketing analysis."
    else:
        system_instruction = create_system_instruction(df_viz_sample, df_opt_sample)

    # Initialize chat history
    if "hf_messages" not in st.session_state:
        # Start history with the System Instruction
        st.session_state.hf_messages = [
            {"role": "system", "content": system_instruction}
        ]

    # Display chat messages from history (excluding the system message)
    for message in st.session_state.hf_messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the app or data..."):
        
        # 1. Add user message to history
        user_message = {"role": "user", "content": prompt}
        st.session_state.hf_messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Call the Hugging Face API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Pass the entire conversation history for context
                api_response = call_hf_api(st.session_state.hf_messages, MODEL_ID)
            
            # Extract the response text
            try:
                response_text = api_response["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                response_text = "I'm sorry, I couldn't get a valid response from the model. Please check the API status."

            # 3. Display assistant response
            st.markdown(response_text)
            
            # 4. Add assistant response to history
            assistant_message = {"role": "assistant", "content": response_text}
            st.session_state.hf_messages.append(assistant_message)


if __name__ == "__main__":
    app_chatbot()