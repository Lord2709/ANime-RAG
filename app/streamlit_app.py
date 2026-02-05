import streamlit as st

from generation import PromptConfig
from routing import run_rag


st.set_page_config(page_title="Anime RAG", page_icon="🤖", layout="centered")
st.title("Anime RAG 🤖")


# --- Sidebar controls ---
st.sidebar.header("Settings")

provider = st.sidebar.selectbox("LLM Provider", ["ollama", "stub"])

OLLAMA_MODELS = [
    "llama3.1:8b",
    "mistral:7b-instruct-v0.3-q4_0",
    "qwen:7b",
    "deepseek-llm:7b-chat",
    "phi3:mini",
]

ollama_model = st.sidebar.selectbox("Ollama Model", OLLAMA_MODELS, index=0)

style = st.sidebar.selectbox("Answer Style", ["concise", "detailed"])
max_recs = st.sidebar.slider("Max Recommendations", 1, 8, 5)
max_docs = st.sidebar.slider("Max Retrieved Docs", 1, 10, 5)
max_history_turns = st.sidebar.slider("History Turns", 0, 10, 6)
show_context = st.sidebar.checkbox("Show Retrieved Context", value=False)

config = PromptConfig(
    style=style,
    max_recs=max_recs,
    max_docs=max_docs,
    max_history_turns=max_history_turns,
)


# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Render existing chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- User input ---
prompt = st.chat_input("Ask something...")

if prompt:
    # 1) Store and render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving anime knowledge..."):
            response, docs = run_rag(
                prompt=prompt,
                history=st.session_state.messages,
                config=config,
                provider=provider,
                model=ollama_model,
                max_docs=max_docs,
            )

            st.markdown(response)

            if show_context:
                with st.expander("Retrieved Context"):
                    for i, doc in enumerate(docs, start=1):
                        st.markdown(f"**Doc {i}**")
                        st.markdown(doc.get("content", ""))
                        st.markdown("---")

    # 3) Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
