from dotenv import load_dotenv
import streamlit as st
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")

st.title("🤖 AskBuddy QNA Chatbot")
st.markdown("QNA chatbot using langchain and python")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]    
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything")
if query:
    st.session_state.messages.append({"role":"user" , "content":query})
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    st.chat_message("ai").markdown(res.content)
    st.session_state.messages.append({"role":"ai" , "content":res.content})