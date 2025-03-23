import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from huggingface_hub import InferenceClient
from langchain_core.messages import AIMessage
import pandas as pd
import os
from agent_setup import run_agent 
import streamlit as st


client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    model="mistralai/Mistral-7B-Instruct-v0.1",
)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="gaipl-the-ai-vengers/code/src/chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_docs")

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Mock action suggestions (can be expanded dynamically)
actions = [
    {"name": "Restart Service"},
    {"name": "Generate Incident Report"},
    {"name": "Check System Logs"},
    {"name": "Send Email"}
]


# Function to extract text from Excel file
def extract_text_from_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)  # Read Excel file
    text_data = df.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()  # Convert each row to a string
    return text_data

# Function to add documents to ChromaDB
def add_document_to_db(doc_text_list, doc_id):
    embeddings = embedding_model.encode(doc_text_list).tolist()
    for i, sentence in enumerate(doc_text_list):
        collection.add(
            ids=[f"{doc_id}_{i}"],
            embeddings=[embeddings[i]],
            metadatas=[{"text": sentence}]
        )

# Function to retrieve relevant documents
def retrieve_context(query, top_k=3):
    query_embedding = embedding_model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    retrieved_texts = [doc["text"] for doc in results["metadatas"][0]]
    return " ".join(retrieved_texts)

# Function to generate response using Hugging Face LLM
# def generate_response(prompt, context):
#     messages = [
#         {"role": "system", "content": "You are an AI assistant that provides answers based on the given context."},
#         {"role": "user", "content": f"Context: {context}\n\nUser Query: {prompt}"}
#     ]
#     response = client.chat_completion(
#         model="mistralai/Mistral-7B-Instruct-v0.1",
#         messages=messages,
#         max_tokens=500,
#         temperature=0.7,
#     )
    
#     return response.choices[0].message["content"] if response.choices else "Error: No valid response from Hugging Face"
 
def generate_response_with_actions(prompt, context):
    messages = [
        {"role": "system", "content": "You are an AI assistant providing answers along with actionable suggestions."},
        {"role": "user", "content": f"Context: {context}\n\nUser Query: {prompt}"}
    ]
    
    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )

    # Extract answer
    answer_text = response.choices[0].message["content"] if response.choices else "Error: No valid response from Hugging Face"

    return {"answer": answer_text, "actions": actions} 

def execute_mock_action(action_name):
    print(f"Executing action: {action_name}")  # Log action execution
    # Ensure chat history exists in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        action_response = run_agent(action["name"])
        ai_messages = [msg.content for msg in action_response.messages if isinstance(msg, AIMessage)]

        # Display AI response in UI
        if ai_messages:
            st.write("### Action Executed:")
            for msg in ai_messages:
                st.write(msg)

        st.session_state.messages.append(
                    {"role": "assistant", "content": f"✅ {ai_messages[0]} <br> Action '{action_name}' executed successfully! (Mocked)"}
        )
        
        #st.rerun()  # Forces UI to refresh and display new message

    # Streamlit UI
    st.title("RAG-based Q&A with Hugging Face 🚀")

# Streamlit UI
st.title("AI Chatbot for platform Engineers 🚀")

# File Upload
uploaded_file = st.file_uploader("Upload a xlsx file", type=["xlsx"])
if uploaded_file:
    st.write("Processing document...")
    document_text = extract_text_from_excel(uploaded_file)
    add_document_to_db(document_text, uploaded_file.name)
    st.success("Document added to ChromaDB!")

# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

if "answer" not in st.session_state:
    st.session_state.answer = None 

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# # Query Input
# query = st.text_input("Ask a question from the document")
# if st.button("Get Answer"):
#     if query:
#         context = retrieve_context(query)
#         print("the context is : "+context)
#         if context:
#             st.session_state.answer = generate_response(query, context)
#             st.write("### Answer:", st.session_state.answer )
#         else:
#             st.write("No relevant context found!")

# if st.session_state.answer and "restarting" in st.session_state.answer.lower():
#     if st.button("Run Agent Task"):
#         action_response = run_agent(st.session_state.answer)

#         # Extract only AIMessage contents
#         ai_messages = [msg.content for msg in action_response.messages if isinstance(msg, AIMessage)]

#         # Display AI response in UI
#         if ai_messages:
#             st.write("### Action Executed:")
#             for msg in ai_messages:
#                 st.write(msg)

# User input for query
query = st.chat_input("Ask a question related to platform")
if query:
    # Store user input
    st.session_state.messages.append({"role": "user", "content": query})

    # Retrieve context
    context = retrieve_context(query)
    if context:
        response_data = generate_response_with_actions(query, context)
        bot_response = response_data["answer"]
        actions = response_data["actions"]

        # Store bot response
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        with st.chat_message("assistant"):
            st.write(bot_response)

            # Display suggested actions
            if actions:
                st.write("### Suggested Actions:")
                for action in actions:
                    if st.button(action["name"], key=f"{action['name']}_{query}"):  
                        execute_mock_action(action["name"])  # Executes and updates UI
    else:
        st.session_state.messages.append({"role": "assistant", "content": "No relevant context found!"})
   