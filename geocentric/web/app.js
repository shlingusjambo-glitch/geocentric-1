"use strict";
const $ = (id) => document.getElementById(id);
const paths = {
  panel:
    "M8 3v18M4 3h16a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1",
  compose:
    "m15 4 5 5M4 20l5-1L21 7a2 2 0 0 0-5-5L4 14v6M13 4H5a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h13a2 2 0 0 0 2-2v-8",
  search: "M21 21l-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0",
  settings:
    "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8M12 2v3m0 14v3M2 12h3m14 0h3M5 5l2 2m10 10 2 2M5 19l2-2M17 7l2-2",
  sliders: "M4 6h5m4 0h7M4 18h11m4 0h1M4 12h1m4 0h11M9 3v6M5 9v6m10 0v6",
  chevron: "m7 10 5 5 5-5",
  arrow: "M12 19V5m-6 6 6-6 6 6",
  stop: "M7 7h10v10H7z",
  close: "m6 6 12 12M6 18 18 6",
  copy: "M9 9h11v11H9zM5 15H3V3h12v2",
  retry: "M20 7v5h-5M20 12a8 8 0 1 0-2 6",
  trash: "M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7",
};
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(svg.namespaceURI, "path");
  path.setAttribute("d", paths[name] || paths.compose);
  svg.append(path);
  return svg;
}
document
  .querySelectorAll("[data-icon]")
  .forEach((e) => e.append(icon(e.dataset.icon)));
function stored(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}
let chats = stored("geocentric.chats.v1", []);
if (!Array.isArray(chats)) chats = [];
chats = chats.filter(
  (c) =>
    c &&
    typeof c.id === "string" &&
    typeof c.title === "string" &&
    Array.isArray(c.messages),
);
const savedPrefs = stored("geocentric.settings.v1", {});
let prefs = stored("geocentric.settings.v1", {
  theme: "dark",
  mode: "auto",
  temperature: 0.8,
  max_new_tokens: 256,
  repetition_penalty: 1.25,
  system: "",
});
if (!prefs || typeof prefs !== "object") prefs = { theme: "dark" };
let current = null,
  busy = false,
  controller = null,
  requestId = null,
  model = null;
const uuid = () =>
  globalThis.crypto?.randomUUID?.() ||
  "10000000-1000-4000-8000-100000000000".replace(/[018]/g, (c) =>
    (
      c ^
      (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))
    ).toString(16),
  );
function save() {
  try {
    localStorage.setItem(
      "geocentric.chats.v1",
      JSON.stringify(chats.slice(0, 150)),
    );
  } catch {
    toast("Browser storage is full. Export chats before clearing space.");
  }
}
function toast(text) {
  $("toast").textContent = text;
  $("toast").hidden = false;
  setTimeout(() => ($("toast").hidden = true), 2500);
}
function applyTheme() {
  document.body.classList.toggle(
    "light",
    prefs.theme === "light" ||
      (prefs.theme === "system" &&
        matchMedia("(prefers-color-scheme: light)").matches),
  );
}
applyTheme();
function closeSidebar() {
  document.body.classList.remove("mobile-open");
}
$("expand").onclick = () =>
  innerWidth <= 760
    ? document.body.classList.add("mobile-open")
    : document.body.classList.remove("sidebar-closed");
$("collapse").onclick = () =>
  innerWidth <= 760
    ? closeSidebar()
    : document.body.classList.add("sidebar-closed");
$("overlay").onclick = closeSidebar;
function history() {
  const nav = $("history");
  nav.replaceChildren();
  const query = $("search").value.toLowerCase();
  for (const chat of chats.filter((c) =>
    c.title.toLowerCase().includes(query),
  )) {
    const row = document.createElement("div");
    row.className = "history-item" + (current === chat ? " active" : "");
    const button = document.createElement("button");
    button.className = "history-title";
    button.textContent = chat.title;
    button.onclick = () => {
      if (busy) return toast("Stop the current response first.");
      current = chat;
      render();
      closeSidebar();
    };
    const del = document.createElement("button");
    del.className = "icon-button history-delete";
    del.setAttribute("aria-label", "Delete " + chat.title);
    del.append(icon("trash"));
    del.onclick = () => {
      if (busy) return;
      chats = chats.filter((c) => c !== chat);
      if (current === chat) current = null;
      save();
      render();
    };
    row.append(button, del);
    nav.append(row);
  }
  if (!nav.children.length) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = query
      ? "No matching chats"
      : "Your conversations will appear here";
    nav.append(empty);
  }
}
function inline(parent, text) {
  for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)) {
    let node;
    if (part.startsWith("**") && part.endsWith("**")) {
      node = document.createElement("strong");
      node.textContent = part.slice(2, -2);
    } else if (part.startsWith("`") && part.endsWith("`")) {
      node = document.createElement("code");
      node.textContent = part.slice(1, -1);
    } else node = document.createTextNode(part);
    parent.append(node);
  }
}
function markdown(target, text) {
  target.replaceChildren();
  const pieces = text.split(/```/);
  pieces.forEach((piece, i) => {
    if (i % 2) {
      const first = piece.indexOf("\n");
      const language = first < 0 ? "code" : piece.slice(0, first).trim();
      const codeText = first < 0 ? piece : piece.slice(first + 1);
      const wrap = document.createElement("div");
      wrap.className = "code-wrap";
      const head = document.createElement("div");
      head.className = "code-head";
      const label = document.createElement("span");
      label.textContent = language || "code";
      const copy = document.createElement("button");
      copy.textContent = "Copy code";
      copy.onclick = () => copyText(codeText);
      head.append(label, copy);
      const pre = document.createElement("pre"),
        code = document.createElement("code");
      code.textContent = codeText;
      pre.append(code);
      wrap.append(head, pre);
      target.append(wrap);
    } else
      for (const para of piece.split(/\n\s*\n/)) {
        if (!para) continue;
        const match = para.match(/^(#{1,3})\s+([^\n]+)$/);
        const node = document.createElement(match ? "h3" : "p");
        inline(node, match ? match[2] : para);
        target.append(node);
      }
  });
}
async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext)
      await navigator.clipboard.writeText(text);
    else {
      const box = document.createElement("textarea");
      box.value = text;
      document.body.append(box);
      box.select();
      if (!document.execCommand("copy")) throw Error();
      box.remove();
    }
    toast("Copied");
  } catch {
    toast("Copy unavailable in this browser");
  }
}
function action(label, name, fn) {
  const b = document.createElement("button");
  b.className = "icon-button";
  b.setAttribute("aria-label", label);
  b.title = label;
  b.append(icon(name));
  b.onclick = fn;
  return b;
}
function render() {
  history();
  document.body.classList.toggle("empty", !current?.messages.length);
  $("welcome").hidden = !!current?.messages.length;
  const root = $("messages");
  root.replaceChildren();
  (current?.messages || []).forEach((message, index) => {
    if (!message || typeof message.content !== "string") return;
    const row = document.createElement("article");
    row.className =
      "message " + (message.role === "user" ? "user" : "assistant");
    row.setAttribute(
      "aria-label",
      message.role === "user" ? "You" : "Geocentric",
    );
    const body = document.createElement("div");
    body.className = "body";
    if (message.role === "user") body.textContent = message.content;
    else if (!message.content && busy) {
      const dot = document.createElement("span");
      dot.className = "typing";
      dot.setAttribute("aria-label", "Generating");
      body.append(dot);
    } else markdown(body, message.content);
    if (message.role !== "user") {
      const avatar = document.createElement("img");
      avatar.src = "/mascot.png";
      avatar.alt = "";
      avatar.className = "assistant-mascot mascot";
      row.append(avatar);
    }
    row.append(body);
    const actions = document.createElement("div");
    actions.className = "message-actions";
    actions.append(
      action("Copy message", "copy", () => copyText(message.content)),
    );
    if (message.role === "user")
      actions.append(
        action("Edit message", "compose", () => {
          if (busy) return;
          $("prompt").value = message.content;
          current.messages = current.messages.slice(0, index);
          save();
          render();
          resize();
          $("prompt").focus();
        }),
      );
    else if (index === current.messages.length - 1)
      actions.append(
        action("Regenerate response", "retry", () => {
          if (busy) return;
          current.messages.pop();
          const previous = current.messages.pop();
          if (previous) {
            $("prompt").value = previous.content;
            send();
          }
        }),
      );
    if (message.stats) {
      const meta = document.createElement("span");
      meta.className = "message-meta";
      meta.textContent = `${message.stats.generated_tokens || 0} tokens · ${(message.stats.tokens_per_second || 0).toFixed(1)} tok/s`;
      actions.append(meta);
    }
    row.append(actions);
    if (message.stats?.finish_reason === "repetition") {
      const note = document.createElement("p");
      note.className = "message-note";
      note.textContent =
        "Stopped because the model repeated a short phrase. You can regenerate this response.";
      row.append(note);
    }
    if (message.stats?.truncated_prompt_tokens) {
      const note = document.createElement("p");
      note.className = "message-note";
      note.textContent =
        "Earlier context was shortened to leave room for the response.";
      row.append(note);
    }
    root.append(row);
  });
}
function resize() {
  $("prompt").style.height = "auto";
  $("prompt").style.height = Math.min($("prompt").scrollHeight, 220) + "px";
  $("send").disabled = busy || !model || !$("prompt").value.trim();
}
$("prompt").addEventListener("input", resize);
$("prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
});
$("new-chat").onclick = () => {
  if (busy) return toast("Stop the current response first.");
  current = null;
  $("prompt").value = "";
  $("error").hidden = true;
  render();
  resize();
  closeSidebar();
  $("prompt").focus();
};
$("search-button").onclick = () => {
  $("search").hidden = !$("search").hidden;
  if (!$("search").hidden) $("search").focus();
};
$("search").oninput = history;
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    $("new-chat").click();
  }
  if (e.key === "Escape") closeSidebar();
});
function setBusy(value) {
  busy = value;
  $("send").hidden = value;
  $("stop").hidden = !value;
  $("status").textContent = value ? "Generating…" : "";
  resize();
}
async function send() {
  const content = $("prompt").value.trim();
  if (!content || busy || !model) return;
  const existing = current;
  if (!current) {
    current = {
      id: uuid(),
      title: content.slice(0, 60),
      messages: [],
      created: Date.now(),
    };
    chats.unshift(current);
  }
  const chat = current;
  chat.messages.push({ role: "user", content });
  const outgoing = chat.messages.map((m) => ({
    role: m.role,
    content: m.content,
  }));
  const response = { role: "assistant", content: "" };
  chat.messages.push(response);
  $("prompt").value = "";
  $("error").hidden = true;
  setBusy(true);
  render();
  save();
  controller = new AbortController();
  requestId = uuid();
  let completed = false;
  try {
    const request = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...prefs,
        messages: outgoing,
        request_id: requestId,
      }),
      signal: controller.signal,
    });
    if (!request.ok) {
      const error = await request.json();
      throw Error(error.error || "Request failed");
    }
    const reader = request.body.getReader(),
      decoder = new TextDecoder();
    let pending = "";
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = pending.split("\n");
      pending = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "delta") {
          response.content += event.text;
          const nearBottom =
            innerHeight + scrollY >= document.body.scrollHeight - 180;
          render();
          if (nearBottom) window.scrollTo(0, document.body.scrollHeight);
        } else if (event.type === "done") {
          response.stats = event;
          completed = true;
        } else if (event.type === "error") throw Error(event.error);
      }
      if (done) break;
    }
    if (!completed)
      throw Error("Connection ended before the response finished.");
  } catch (error) {
    if (error.name !== "AbortError") {
      $("error").textContent = error.message;
      $("error").hidden = false;
    }
    if (!response.content) chat.messages.pop();
  } finally {
    setBusy(false);
    requestId = null;
    controller = null;
    save();
    render();
    $("prompt").focus();
  }
}
$("composer").onsubmit = (e) => {
  e.preventDefault();
  send();
};
$("stop").onclick = async () => {
  const id = requestId;
  if (id)
    fetch("/api/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id: id }),
    }).catch(() => {});
  controller?.abort();
};
function openSettings() {
  for (const key of [
    "theme",
    "mode",
    "temperature",
    "max_new_tokens",
    "repetition_penalty",
    "system",
  ])
    $(key).value = prefs[key] ?? "";
  $("settings").showModal();
}
for (const id of [
  "settings-button",
  "profile",
  "model-menu",
  "composer-settings",
])
  $(id).onclick = openSettings;
$("close-settings").onclick = () => $("settings").close();
$("save-settings").onclick = () => {
  for (const key of ["theme", "mode", "system"]) prefs[key] = $(key).value;
  for (const key of ["temperature", "max_new_tokens", "repetition_penalty"]) {
    if (!$(key).checkValidity()) return $(key).reportValidity();
    prefs[key] = Number($(key).value);
  }
  try {
    localStorage.setItem("geocentric.settings.v1", JSON.stringify(prefs));
  } catch {}
  applyTheme();
  $("settings").close();
};
$("export-chat").onclick = () => {
  if (!current) return toast("Start a conversation first");
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(current, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = "geocentric-chat.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
async function connect() {
  try {
    const response = await fetch("/api/model");
    if (!response.ok) throw Error("Cannot reach the model server.");
    model = await response.json();
    prefs = { ...prefs, ...model.defaults, ...savedPrefs };
    prefs.system = prefs.system || "";
    $("model-menu").replaceChildren(
      document.createTextNode(model.name + " "),
      icon("chevron"),
    );
    $("mode-badge").textContent = model.mode === "base" ? "Base model" : "Chat";
    $("model-details").textContent =
      `${model.name} · ${(model.parameters / 1e6).toFixed(1)}M parameters · ${model.context.toLocaleString()} token context · ${model.device}`;
    $("max_new_tokens").max = Math.min(4096, model.context - 1);
    prefs.max_new_tokens = Math.min(
      prefs.max_new_tokens || model.defaults.max_new_tokens || 256,
      model.context - 1,
    );
    $("connection").hidden = true;
  } catch (error) {
    $("connection").textContent = error.message + " Refresh to reconnect.";
    $("connection").hidden = false;
  }
  resize();
}
render();
resize();
connect();
