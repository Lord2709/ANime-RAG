import html
import streamlit as st

from generation import PromptConfig
from routing import run_rag


st.set_page_config(page_title="Anime RAG", page_icon="🤖", layout="centered")

st.markdown(
    """
<style>
@import url("https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Space+Grotesk:wght@400;500;600&display=swap");

:root {
    --bg-deep: #0c1118;
    --bg-soft: #131c26;
    --accent: #e3b04b;
    --accent-soft: #f2d27a;
    --text-main: #f7f7f2;
    --text-muted: #aab4c3;
    --card: rgba(20, 28, 38, 0.78);
    --stroke: rgba(255, 255, 255, 0.08);
}

html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(1200px 600px at 10% -10%, #233247 0%, var(--bg-deep) 55%),
                radial-gradient(900px 400px at 90% 10%, #1c2b3a 0%, var(--bg-deep) 50%);
    color: var(--text-main);
    font-family: "Space Grotesk", system-ui, -apple-system, sans-serif;
}

[data-testid="stSidebar"] {
    background: rgba(10, 14, 20, 0.95);
    border-right: 1px solid var(--stroke);
}

.hero {
    padding: 2.2rem 2.1rem 1.5rem;
    border: 1px solid var(--stroke);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(28, 38, 52, 0.8), rgba(12, 17, 24, 0.9));
    box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
    animation: fadeUp 600ms ease-out;
}

.status-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1.1rem;
}

.status-chip {
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    border: 1px solid var(--stroke);
    background: rgba(227, 176, 75, 0.1);
    color: var(--accent-soft);
    font-size: 0.85rem;
}

.hero-title {
    font-family: "Playfair Display", serif;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    color: var(--text-muted);
    font-size: 1rem;
    margin-bottom: 1.2rem;
}

.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
}

.pill {
    padding: 0.4rem 0.75rem;
    border-radius: 999px;
    border: 1px solid var(--stroke);
    background: rgba(255, 255, 255, 0.03);
    font-size: 0.85rem;
    color: var(--text-muted);
}

.section-title {
    font-family: "Playfair Display", serif;
    font-size: 1.4rem;
    margin: 1.8rem 0 0.8rem;
}

.result-card {
    display: grid;
    grid-template-columns: 140px 1fr;
    gap: 1rem;
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid var(--stroke);
    background: var(--card);
    margin-bottom: 1rem;
    animation: fadeUp 500ms ease-out;
    animation-delay: var(--delay, 0ms);
}

.result-media img {
    width: 100%;
    height: 190px;
    object-fit: cover;
    border-radius: 12px;
    border: 1px solid var(--stroke);
}

.image-placeholder {
    width: 100%;
    height: 190px;
    border-radius: 12px;
    border: 1px solid var(--stroke);
    background: linear-gradient(160deg, #1d2a3a, #141c27);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    font-size: 0.85rem;
}

.result-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.result-meta {
    color: var(--text-muted);
    margin-bottom: 0.35rem;
    font-size: 0.95rem;
}

.score-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    background: rgba(227, 176, 75, 0.15);
    color: var(--accent-soft);
    font-size: 0.85rem;
    border: 1px solid rgba(227, 176, 75, 0.3);
}

.film-grain {
    pointer-events: none;
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)' opacity='0.08'/%3E%3C/svg%3E");
    mix-blend-mode: soft-light;
    opacity: 0.15;
    z-index: 0;
}

[data-testid="stChatMessage"] {
    background: rgba(12, 17, 24, 0.65);
    border: 1px solid var(--stroke);
    border-radius: 16px;
    padding: 0.8rem 1rem;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    margin-bottom: 0.4rem;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 700px) {
    .result-card {
        grid-template-columns: 1fr;
    }
    .result-media img, .image-placeholder {
        height: 220px;
    }
}
</style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="film-grain"></div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="hero">
  <div class="hero-title">Anime RAG</div>
  <div class="hero-subtitle">Ask for recommendations, plot details, or genre mashups — powered by local retrieval.</div>
  <div class="pill-row">
    <div class="pill">“Cozy fantasy with found family”</div>
    <div class="pill">“Short thriller with mystery”</div>
    <div class="pill">“Top sci‑fi mecha classics”</div>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)


# --- Sidebar controls ---
st.sidebar.header("Settings")

provider = st.sidebar.selectbox("LLM Provider", ["ollama"])

OLLAMA_MODELS = [
    "llama3.1:8b",
    "mistral:7b-instruct-v0.3-q4_0",
    "qwen:7b",
    "deepseek-llm:7b-chat",
    "phi3:mini",
]

ollama_model = st.sidebar.selectbox("Ollama Model", OLLAMA_MODELS, index=0)

style = st.sidebar.selectbox("Answer Style", ["concise", "detailed"])
show_images = st.sidebar.checkbox("Show Result Images", value=True)

# Fixed config (no user sliders)
max_recs = 5
max_docs = 5
max_history_turns = 6

config = PromptConfig(
    style=style,
    max_recs=max_recs,
    max_docs=max_docs,
    max_history_turns=max_history_turns,
)

st.markdown(
    f"""
<div class="status-row">
  <div class="status-chip">Provider: {html.escape(provider)}</div>
  <div class="status-chip">Model: {html.escape(ollama_model)}</div>
  <div class="status-chip">Max Docs: {max_docs}</div>
  <div class="status-chip">Style: {html.escape(style)}</div>
</div>
    """,
    unsafe_allow_html=True,
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

            if show_images and docs:
                st.markdown('<div class="section-title">Top Matches</div>', unsafe_allow_html=True)
                for i, doc in enumerate(docs, start=1):
                    image_url = str(doc.get("image_url", "") or "").strip()
                    title = html.escape(str(doc.get("title") or f"Doc {i}"))
                    genres = html.escape(str(doc.get("genres") or ""))
                    score = str(doc.get("score") or "").strip()

                    if image_url:
                        media_html = f'<img src="{html.escape(image_url)}" alt="{title}">'
                    else:
                        media_html = '<div class="image-placeholder">No image</div>'

                    score_html = f'<div class="score-badge">Score {html.escape(score)}</div>' if score else ""
                    genres_html = f'<div class="result-meta">Genres: {genres}</div>' if genres else ""

                    st.markdown(
                        f"""
<div class="result-card" style="--delay: {i * 90}ms;">
  <div class="result-media">
    {media_html}
  </div>
  <div class="result-body">
    <div class="result-title">{title}</div>
    {genres_html}
    {score_html}
  </div>
</div>
                        """,
                        unsafe_allow_html=True,
                    )

    # 3) Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
