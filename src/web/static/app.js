const state = {
  conversationId: localStorage.getItem("wixqa_conversation_id") || crypto.randomUUID(),
  events: [],
  selectedEventId: null,
  running: false,
};

localStorage.setItem("wixqa_conversation_id", state.conversationId);

const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chatForm");
const inputEl = document.getElementById("messageInput");
const sendButtonEl = document.getElementById("sendButton");
const profileEl = document.getElementById("profileSelect");
const eventListEl = document.getElementById("eventList");
const payloadViewEl = document.getElementById("payloadView");
const evidenceListEl = document.getElementById("evidenceList");
const stageBarsEl = document.getElementById("stageBars");
const statusPillEl = document.getElementById("statusPill");
const latencyTotalEl = document.getElementById("latencyTotal");
const sessionLabelEl = document.getElementById("sessionLabel");

sessionLabelEl.textContent = state.conversationId;

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message || state.running) {
    return;
  }
  inputEl.value = "";
  addMessage("user", message);
  setRunning(true);
  state.events = [];
  state.selectedEventId = null;
  renderEvents();
  renderPayload({});
  renderEvidence([]);
  renderStageBars({});
  addMessage("status", "正在处理...");

  try {
    await streamTurn(message);
    setRunning(false, "done");
  } catch (error) {
    setRunning(false, "error");
    addMessage("assistant", `请求失败：${error.message}`);
  }
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    formEl.requestSubmit();
  }
});

async function streamTurn(message) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      conversation_id: state.conversationId,
      message,
      profile: profileEl.value,
    }),
  });
  if (!response.ok || !response.body) {
    throw new Error(`HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const {value, done} = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, {stream: true});
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      handleStreamLine(line);
    }
  }
  if (buffer.trim()) {
    handleStreamLine(buffer);
  }
}

function handleStreamLine(line) {
  const text = line.trim();
  if (!text) {
    return;
  }
  const row = JSON.parse(text);
  if (row.type === "error") {
    throw new Error(row.error || "stream error");
  }
  if (row.type !== "event") {
    return;
  }
  const event = row.event;
  state.events.push(event);
  state.selectedEventId = event.event_id;
  renderEvents();
  renderPayload(event.payload || {});
  renderEvidence(evidenceFromEvent(event));
  renderStageBars((event.payload && event.payload.stage_latency_ms) || {});
  latencyTotalEl.textContent = formatMs(event.latency_ms || 0);
  if (event.event_type === "status") {
    updateLastStatus(event.message);
  }
  if (event.is_final) {
    replaceLastStatus();
    addMessage("assistant", event.message);
  }
}

function addMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text;
  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function updateLastStatus(text) {
  const nodes = messagesEl.querySelectorAll(".message.status");
  const node = nodes[nodes.length - 1];
  if (node) {
    node.textContent = text;
  }
}

function replaceLastStatus() {
  const nodes = messagesEl.querySelectorAll(".message.status");
  const node = nodes[nodes.length - 1];
  if (node) {
    node.remove();
  }
}

function renderEvents() {
  eventListEl.replaceChildren();
  for (const event of state.events) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `event-row ${event.event_id === state.selectedEventId ? "active" : ""}`;
    button.addEventListener("click", () => {
      state.selectedEventId = event.event_id;
      renderEvents();
      renderPayload(event.payload || {});
      renderEvidence(evidenceFromEvent(event));
      renderStageBars((event.payload && event.payload.stage_latency_ms) || {});
    });

    const title = document.createElement("div");
    title.className = "event-title";
    const name = document.createElement("span");
    name.textContent = event.agent_name;
    const type = document.createElement("span");
    type.textContent = event.event_type;
    title.append(name, type);

    const meta = document.createElement("div");
    meta.className = "event-meta";
    meta.append(
      metaItem(formatMs(event.latency_ms || 0)),
      metaItem(event.profile || "profile"),
      metaItem(event.is_final ? "final" : "running")
    );
    button.append(title, meta);
    eventListEl.appendChild(button);
  }
}

function renderPayload(payload) {
  payloadViewEl.textContent = JSON.stringify(payload, null, 2);
}

function renderEvidence(items) {
  evidenceListEl.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "evidence-item";
    empty.textContent = "No evidence selected";
    evidenceListEl.appendChild(empty);
    return;
  }
  for (const item of items) {
    const node = document.createElement("div");
    node.className = "evidence-item";
    const title = document.createElement("div");
    title.className = "evidence-title";
    title.textContent = item.title || item.article_id || item.citation_id || "evidence";
    const id = document.createElement("div");
    id.className = "evidence-id";
    id.textContent = item.chunk_ids ? item.chunk_ids.join(", ") : item;
    node.append(title, id);
    evidenceListEl.appendChild(node);
  }
}

function renderStageBars(timings) {
  stageBarsEl.replaceChildren();
  const entries = Object.entries(timings || {})
    .filter(([, value]) => Number(value) > 0)
    .sort((a, b) => Number(b[1]) - Number(a[1]));
  if (!entries.length) {
    return;
  }
  const max = Math.max(...entries.map(([, value]) => Number(value)));
  for (const [name, value] of entries) {
    const row = document.createElement("div");
    row.className = "stage-bar";
    const label = document.createElement("span");
    label.textContent = name;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(4, (Number(value) / max) * 100)}%`;
    track.appendChild(fill);
    const ms = document.createElement("span");
    ms.textContent = formatMs(value);
    row.append(label, track, ms);
    stageBarsEl.appendChild(row);
  }
}

function evidenceFromEvent(event) {
  const payload = event.payload || {};
  if (payload.answer && Array.isArray(payload.answer.citations)) {
    return payload.answer.citations;
  }
  if (Array.isArray(payload.active_chunk_ids)) {
    return payload.active_chunk_ids;
  }
  return [];
}

function metaItem(text) {
  const node = document.createElement("span");
  node.textContent = text;
  return node;
}

function setRunning(running, status = "running") {
  state.running = running;
  sendButtonEl.disabled = running;
  inputEl.disabled = running;
  statusPillEl.className = `status-pill ${status}`;
  statusPillEl.textContent = running ? "running" : status;
}

function formatMs(value) {
  const number = Number(value) || 0;
  if (number >= 1000) {
    return `${(number / 1000).toFixed(2)} s`;
  }
  return `${number.toFixed(0)} ms`;
}

renderEvents();
renderEvidence([]);
