const state = {
  config: null,
  messages: [],
  docs: [],
  busy: false,
};

const styleSelect = document.getElementById("styleSelect");
const showImagesToggle = document.getElementById("showImagesToggle");
const promptInput = document.getElementById("promptInput");
const sendButton = document.getElementById("sendButton");
const messagesEl = document.getElementById("messages");
const statusRow = document.getElementById("statusRow");
const typingIndicator = document.getElementById("typingIndicator");
const matchesPanel = document.getElementById("matchesPanel");
const matchCount = document.getElementById("matchCount");
const cardsEl = document.getElementById("cards");
const composer = document.getElementById("composer");

window.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  loadConfig();
  renderMessages();
  renderMatches();
});

function bindEvents() {
  composer.addEventListener("submit", handleSubmit);

  promptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      composer.requestSubmit();
    }
  });

  styleSelect.addEventListener("change", renderStatus);
  showImagesToggle.addEventListener("change", () => {
    renderStatus();
    renderMatches();
  });

  document.querySelectorAll(".prompt-chip").forEach((button) => {
    button.addEventListener("click", () => {
      promptInput.value = button.dataset.prompt || "";
      promptInput.focus();
    });
  });
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    const data = await response.json();
    state.config = data;
    styleSelect.value = "concise";
    renderStatus();
  } catch (error) {
    state.messages.push({
      role: "assistant",
      content: `Error loading config: ${error.message}`,
    });
    renderMessages();
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt || state.busy) {
    return;
  }

  state.messages.push({ role: "user", content: prompt });
  renderMessages();
  promptInput.value = "";
  setBusy(true);

  const payload = {
    prompt,
    history: state.messages,
    style: styleSelect.value,
    max_docs: 5,
  };

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Request failed.");
    }

    state.messages.push({
      role: "assistant",
      content: data.assistant_message || "No response returned.",
    });
    state.docs = Array.isArray(data.docs) ? data.docs : [];
  } catch (error) {
    state.messages.push({
      role: "assistant",
      content: `Error: ${error.message}`,
    });
    state.docs = [];
  } finally {
    setBusy(false);
    renderStatus();
    renderMessages();
    renderMatches();
  }
}

function setBusy(isBusy) {
  state.busy = isBusy;
  sendButton.disabled = isBusy;
  promptInput.disabled = isBusy;
  typingIndicator.classList.toggle("hidden", !isBusy);
}

function renderStatus() {
  const style = styleSelect.value || "concise";
  const posters = showImagesToggle.checked ? "Posters on" : "Posters off";

  statusRow.innerHTML = [
    chip("Archive grounded"),
    chip(`Style: ${titleCase(style)}`),
    chip(posters),
  ].join("");
}

function renderMessages() {
  if (state.messages.length === 0) {
    messagesEl.innerHTML = `
      <div class="empty-state">
        <div class="empty-title">Start with a mood, genre, or studio.</div>
        <p>Try questions like “best melancholic sci-fi”, “Bones titles with strong worldbuilding”, or “short mystery anime with high scores”.</p>
      </div>
    `;
    return;
  }

  messagesEl.innerHTML = state.messages
    .map(
      (message) => `
        <article class="message ${message.role === "user" ? "user" : "assistant"}">
          <div class="message-role">${message.role === "user" ? "You" : "Anime RAG"}</div>
          <div class="message-body">${formatMessage(message.content)}</div>
        </article>
      `,
    )
    .join("");

  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderMatches() {
  const docs = state.docs || [];
  if (!docs.length) {
    matchesPanel.classList.add("hidden");
    cardsEl.innerHTML = "";
    return;
  }

  matchCount.textContent = `${docs.length} matches`;
  cardsEl.innerHTML = docs
    .map((doc, index) => {
      const title = escapeHtml(doc.title || `Result ${index + 1}`);
      const genres = escapeHtml(doc.genres || "");
      const score = escapeHtml(String(doc.score || "").trim());
      const synopsis = escapeHtml(extractSynopsis(doc.content || ""));
      const imageUrl = String(doc.image_url || "").trim();
      const media = showImagesToggle.checked && imageUrl
        ? `<img src="${escapeHtml(imageUrl)}" alt="${title}" loading="lazy" />`
        : `<div class="image-placeholder">Poster hidden</div>`;

      return `
        <article class="result-card">
          <div class="result-media">${media}</div>
          <div class="result-body">
            <div class="result-title">${title}</div>
            ${genres ? `<div class="result-meta">Genres: ${genres}</div>` : ""}
            ${score ? `<div class="result-score">Score ${score}</div>` : ""}
            ${synopsis ? `<p class="result-copy">${synopsis}</p>` : ""}
          </div>
        </article>
      `;
    })
    .join("");

  matchesPanel.classList.remove("hidden");
}

function extractSynopsis(content) {
  const marker = "Synopsis:";
  const index = content.indexOf(marker);
  if (index === -1) {
    return "";
  }
  const text = content.slice(index + marker.length).trim();
  return text.length > 220 ? `${text.slice(0, 217).trim()}...` : text;
}

function formatMessage(text) {
  return escapeHtml(text).replace(/\n/g, "<br />");
}

function chip(text) {
  return `<div class="status-chip">${text}</div>`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function titleCase(value) {
  return String(value)
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
