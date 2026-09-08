"use strict";
const $ = (id) => document.getElementById(id);
const paths = {
  branch: "M6 3v12a4 4 0 0 0 4 4h8M6 8h8a4 4 0 0 0 4-4M3 3h6M15 3h6M15 19h6",
  canvas: "M3 4h18v16H3zM3 9h18M8 9v11",
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
let current =
    chats.find((c) => c.id === stored("geocentric.active.v1", null)) || null,
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
    localStorage.setItem("geocentric.chats.v1", JSON.stringify(chats));
  } catch {
    toast("Browser storage is full. Export chats before clearing space.");
  }
}
let toastTimer;
function toast(text, undo) {
  clearTimeout(toastTimer);
  $("toast").replaceChildren(document.createTextNode(text));
  if (undo) {
    const button = document.createElement("button");
    button.textContent = "Undo";
    button.onclick = () => {
      undo();
      $("toast").hidden = true;
    };
    $("toast").append(button);
  }
  $("toast").hidden = false;
  toastTimer = setTimeout(() => ($("toast").hidden = true), undo ? 8000 : 3000);
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
let renaming = null;
function renameChat(chat) {
  renaming = chat;
  $("chat-title").value = chat.title;
  $("rename-dialog").showModal();
  $("chat-title").select();
}
$("cancel-rename").onclick = () => $("rename-dialog").close();
$("rename-form").onsubmit = (e) => {
  e.preventDefault();
  const title = $("chat-title").value.trim();
  if (!title || !renaming) return;
  renaming.title = title;
  save();
  history();
  $("rename-dialog").close();
};
function history() {
  const nav = $("history");
  nav.replaceChildren();
  const query = $("search").value.toLowerCase().trim();
  let lastGroup = "";
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  for (const chat of chats.filter(
    (c) =>
      c.title.toLowerCase().includes(query) ||
      c.messages.some(
        (m) =>
          m &&
          typeof m.content === "string" &&
          m.content.toLowerCase().includes(query),
      ),
  )) {
    const date = chat.updated || chat.created || 0;
    const group =
      date >= +today ? "Today" : date >= +yesterday ? "Yesterday" : "Earlier";
    if (!query && group !== lastGroup) {
      const heading = document.createElement("h3");
      heading.className = "history-group";
      heading.textContent = group;
      nav.append(heading);
      lastGroup = group;
    }
    const row = document.createElement("div");
    row.className = "history-item" + (current === chat ? " active" : "");
    const button = document.createElement("button");
    button.className = "history-title";
    button.textContent = chat.title;
    button.title = chat.title;
    if (current === chat) button.setAttribute("aria-current", "page");
    button.onclick = () => {
      if (busy) return toast("Stop the current response first.");
      persistDraft();
      current = chat;
      restoreDraft();
      render();
      closeSidebar();
      window.scrollTo(0, document.body.scrollHeight);
    };
    const rename = action("Rename " + chat.title, "compose", () =>
      renameChat(chat),
    );
    rename.classList.add("history-rename");
    const del = action("Delete " + chat.title, "trash", () => {
      if (busy) return toast("Stop the current response first.");
      persistDraft();
      const index = chats.indexOf(chat),
        selected = current === chat;
      chats = chats.filter((c) => c !== chat);
      if (selected) {
        current = null;
        restoreDraft();
      }
      save();
      render();
      toast("Conversation deleted", () => {
        chats.splice(Math.min(index, chats.length), 0, chat);
        if (selected && !busy) {
          persistDraft();
          current = chat;
          restoreDraft();
        }
        save();
        if (busy) history();
        else render();
      });
    });
    del.classList.add("history-delete");
    row.append(button, rename, del);
    nav.append(row);
  }
  if (!nav.children.length) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = query
      ? "No conversations found. Try another word."
      : "A fresh page. Your conversations will find a home here.";
    nav.append(empty);
  }
}

function inline(parent, text) {
  for (const part of text.split(
    /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\(https?:\/\/[^\s)]+\))/g,
  )) {
    let node;
    const link = part.match(/^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/);
    if (link) {
      node = document.createElement("a");
      node.textContent = link[1];
      node.href = link[2];
      node.target = "_blank";
      node.rel = "noopener noreferrer";
    } else if (part.startsWith("**") && part.endsWith("**")) {
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
  text.split(/```/).forEach((piece, i) => {
    if (i % 2) {
      const first = piece.indexOf("\n"),
        codeText = first < 0 ? piece : piece.slice(first + 1);
      const wrap = document.createElement("div");
      wrap.className = "code-wrap";
      const head = document.createElement("div");
      head.className = "code-head";
      const label = document.createElement("span");
      label.textContent =
        first < 0 ? "code" : piece.slice(0, first).trim() || "code";
      const copy = document.createElement("button");
      copy.textContent = "Copy code";
      copy.onclick = () => copyText(codeText);
      const pre = document.createElement("pre"),
        code = document.createElement("code");
      code.textContent = codeText;
      pre.append(code);
      head.append(label, copy);
      if (["html", "htm", "svg"].includes(label.textContent.toLowerCase())) {
        const preview = document.createElement("button");
        preview.textContent = "Open in Canvas";
        preview.onclick = () => openCanvas(codeText);
        head.append(preview);
      }
      wrap.append(head, pre);
      target.append(wrap);
      return;
    }
    let paragraph = [],
      list = null,
      listKind = null;
    const flush = () => {
      if (paragraph.length) {
        const p = document.createElement("p");
        inline(p, paragraph.join("\n"));
        target.append(p);
        paragraph = [];
      }
    };
    for (const line of piece.split("\n")) {
      const heading = line.match(/^(#{1,3})\s+(.+)$/);
      const item = line.match(/^\s*(?:([-*+])|\d+\.)\s+(.+)$/);
      const quote = line.match(/^>\s?(.*)$/);
      if (item) {
        flush();
        const kind = item[1] ? "ul" : "ol";
        if (!list || kind !== listKind) {
          list = document.createElement(kind);
          listKind = kind;
          target.append(list);
        }
        const li = document.createElement("li");
        inline(li, item[2]);
        list.append(li);
        continue;
      }
      list = null;
      listKind = null;
      if (heading || quote || !line.trim()) {
        flush();
        if (heading || quote) {
          const node = document.createElement(
            heading ? "h" + (heading[1].length + 1) : "blockquote",
          );
          inline(node, heading ? heading[2] : quote[1]);
          target.append(node);
        }
      } else paragraph.push(line);
    }
    flush();
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
      try {
        if (!document.execCommand("copy")) throw Error();
      } finally {
        box.remove();
      }
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
  try {
    localStorage.setItem(
      "geocentric.active.v1",
      JSON.stringify(current?.id || null),
    );
  } catch {}
  history();
  renderBranchBanner();
  document.body.classList.toggle("empty", !current?.messages.length);
  $("welcome").hidden = !!current?.messages.length;
  $("jump-latest").hidden =
    !current?.messages.length ||
    innerHeight + scrollY >= document.body.scrollHeight - 180;
  document.title = current ? current.title + " · Geocentric" : "Geocentric";
  $("conversation-title").textContent = current?.title || "New conversation";
  $("conversation-title").title = current?.title || "New conversation";
  const root = $("messages");
  root.replaceChildren();
  (current?.messages || []).forEach((message, index) => {
    if (!message || typeof message.content !== "string") return;
    const row = document.createElement("article");
    row.className =
      "message " + (message.role === "user" ? "user" : "assistant");
    row.tabIndex = -1;
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
          forkConversation(current, index, "Edited path");
          $("prompt").value = message.content;
          persistDraft();
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
          const previous = current.messages[index - 1];
          if (!previous || previous.role !== "user") return;
          forkConversation(current, index - 1, "Another response");
          if (previous) {
            $("prompt").value = previous.content;
            send();
          }
        }),
      );
    if (message.role === "assistant")
      actions.append(
        action("Branch from here", "branch", () => {
          if (busy) return toast("Stop the current response first.");
          forkConversation(current, index + 1, "New direction");
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
  $("jump-latest").style.bottom = $("composer-area").offsetHeight + 16 + "px";
  $("send").disabled = busy || !model || !$("prompt").value.trim();
}
let draftTimer;
function persistDraft() {
  clearTimeout(draftTimer);
  try {
    localStorage.setItem(
      "geocentric.draft." + (current?.id || "new"),
      $("prompt").value,
    );
  } catch {}
}
function restoreDraft() {
  try {
    $("prompt").value =
      localStorage.getItem("geocentric.draft." + (current?.id || "new")) || "";
  } catch {
    $("prompt").value = "";
  }
  resize();
}
$("prompt").addEventListener("input", () => {
  resize();
  clearTimeout(draftTimer);
  draftTimer = setTimeout(persistDraft, 250);
});
window.addEventListener("pagehide", persistDraft);
const systemTheme = matchMedia("(prefers-color-scheme: light)");
systemTheme.addEventListener("change", applyTheme);
window.addEventListener("resize", resize);
$("jump-latest").onclick = () =>
  window.scrollTo({
    top: document.body.scrollHeight,
    behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
      ? "instant"
      : "smooth",
  });
window.addEventListener(
  "scroll",
  () => {
    $("jump-latest").hidden =
      !current?.messages.length ||
      innerHeight + scrollY >= document.body.scrollHeight - 180;
  },
  { passive: true },
);

$("prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
});
$("conversation").addEventListener("keydown", (e) => {
  if (!['ArrowUp', 'ArrowDown'].includes(e.key) || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
  if (e.target !== $("conversation") && !e.target.classList.contains('message')) return;
  const messages = [...$("messages").children];
  if (!messages.length) return;
  e.preventDefault();
  const index = messages.indexOf(e.target);
  const next = index < 0 ? (e.key === 'ArrowDown' ? 0 : messages.length - 1)
    : Math.max(0, Math.min(messages.length - 1, index + (e.key === 'ArrowDown' ? 1 : -1)));
  messages[next].focus();
});
$("new-chat").onclick = () => {
  if (busy) return toast("Stop the current response first.");
  persistDraft();
  current = null;
  restoreDraft();
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
  document.body.classList.toggle("generating", value);
  $("send").hidden = value;
  $("stop").hidden = !value;
  $("status").textContent = value ? "Generating…" : "";
  resize();
}
async function send() {
  const content = $("prompt").value.trim();
  if (!content || busy || !model) return;
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
  chat.updated = Date.now();
  chats = [chat, ...chats.filter((c) => c !== chat)];
  clearTimeout(draftTimer);
  try {
    localStorage.removeItem("geocentric.draft.new");
    localStorage.removeItem("geocentric.draft." + chat.id);
  } catch {}
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
  window.scrollTo(0, document.body.scrollHeight);
  const streamingBody = $("messages").lastElementChild.querySelector(".body");
  const latestUser = $("messages").lastElementChild.previousElementSibling;
  latestUser?.classList.add("message-enter");
  streamingBody.classList.add("streaming");
  let firstText = true;
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
          if (firstText && event.text) {
            streamingBody.classList.add("response-arriving");
            firstText = false;
          }
          const nearBottom =
            innerHeight + scrollY >= document.body.scrollHeight - 180;
          markdown(streamingBody, response.content);
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
    if (!response.content) {
      chat.messages.pop(); // Discard the empty assistant placeholder.
      chat.messages.pop(); // Restore the unsent user turn instead of duplicating it on retry.
      if (!$("prompt").value) $("prompt").value = content;
      persistDraft();
      resize();
    }
  } finally {
    setBusy(false);
    requestId = null;
    controller = null;
    save();
    const nearBottom =
      innerHeight + scrollY >= document.body.scrollHeight - 180;
    render();
    if (nearBottom) window.scrollTo(0, document.body.scrollHeight);
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
$("profile").onclick = openSettings;
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
  updateWelcome();
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
    $("model-menu").textContent = model.name;
    $("model-menu").title = model.name + " · Loaded model";
    $("mode-badge").textContent = model.mode === "base" ? "Base model" : "Chat";
    $("model-details").textContent =
      `${model.name} · ${(model.parameters / 1e6).toFixed(1)}M parameters · ${model.context.toLocaleString()} token context · ${model.device}`;
    $("max_new_tokens").max = Math.min(4096, model.context - 1);
    prefs.max_new_tokens = Math.min(
      prefs.max_new_tokens || model.defaults.max_new_tokens || 256,
      model.context - 1,
    );
    $("connection").hidden = true;
    updateWelcome();
  } catch (error) {
    $("connection").textContent = error.message + " Refresh to reconnect.";
    $("connection").hidden = false;
  }
  resize();
}
function updateWelcome() {
  if (!model) return;
  const base = (prefs.mode === "auto" ? model.mode : prefs.mode) === "base";
  $("mode-badge").textContent = base ? "Text continuation" : "Chat";
  $("prompt").placeholder = base
    ? "Start a thought. See where it goes…"
    : "Ask, imagine, or work through an idea…";
  $("welcome-description").textContent = base
    ? "Give your local model a thought to continue."
    : "Your local model. A conversation at your own pace.";
  const prompts = base
    ? [
        [
          "compose",
          "Start a story",
          "A little spark of imagination",
          "Beyond the edge of the forest, there was",
        ],
        [
          "search",
          "Explore an idea",
          "Follow a thread of curiosity",
          "One of the most fascinating things about the natural world is",
        ],
        [
          "sliders",
          "Think it through",
          "Find a new perspective",
          "The first step toward solving a difficult problem is",
        ],
      ]
    : [
        [
          "compose",
          "Create something",
          "Find the words you’re looking for",
          "Help me write a short story. Start by asking what kind of story I have in mind.",
        ],
        [
          "search",
          "Make sense of it",
          "Turn a question into understanding",
          "Explain a fascinating idea from science using a simple everyday example.",
        ],
        [
          "sliders",
          "Think it through",
          "Give your next idea some space",
          "Help me think through an idea. Ask me what I’m working on, then help me explore it.",
        ],
      ];
  $("suggestions").replaceChildren();
  for (const [symbol, title, description, prompt] of prompts) {
    const button = document.createElement("button");
    button.className = "suggestion";
    const text = document.createElement("div"),
      strong = document.createElement("strong"),
      small = document.createElement("small");
    strong.textContent = title;
    small.textContent = description;
    text.append(strong, small);
    button.append(icon(symbol), text);
    button.onclick = () => {
      $("prompt").value = prompt;
      persistDraft();
      resize();
      $("prompt").focus();
    };
    $("suggestions").append(button);
  }
}
function forkConversation(source, count, label) {
  if (busy) return;
  persistDraft();
  const child = {
    id: uuid(),
    title: source.title + " · " + label,
    created: Date.now(),
    updated: Date.now(),
    parentId: source.id,
    branchAt: count,
    messages: JSON.parse(JSON.stringify(source.messages.slice(0, count))),
  };
  chats.unshift(child);
  current = child;
  save();
  restoreDraft();
  render();
  closeSidebar();
  $("prompt").focus();
  return child;
}
function renderBranchBanner() {
  const banner = $("branch-banner");
  banner.replaceChildren();
  banner.hidden = !current?.parentId;
  if (!current?.parentId) return;
  const parent = chats.find((c) => c.id === current.parentId);
  const button = document.createElement("button");
  button.append(
    icon("branch"),
    document.createTextNode(
      parent
        ? "Branched from " + parent.title
        : "Independent branch · original removed",
    ),
  );
  button.onclick = () => {
    if (busy) return toast("Stop the current response first.");
    if (parent) selectConversation(parent);
    else showMap();
  };
  banner.append(button);
}
function selectConversation(chat) {
  persistDraft();
  current = chat;
  restoreDraft();
  render();
  closeSidebar();
  $("map-dialog").close();
  window.scrollTo(0, document.body.scrollHeight);
}
function showMap() {
  const root = $("branch-map");
  root.replaceChildren();
  const seen = new Set();
  const draw = (chat, depth) => {
    if (seen.has(chat.id)) return;
    seen.add(chat.id);
    const button = document.createElement("button");
    button.className = "branch-node" + (chat === current ? " selected" : "");
    button.style.marginLeft = Math.min(depth, 5) * 20 + "px";
    const title = document.createElement("strong"),
      detail = document.createElement("small");
    title.textContent = chat.title;
    detail.textContent =
      chat.messages.length +
      " messages" +
      (chat.parentId
        ? " · branched after message " + (chat.branchAt || 0)
        : " · original conversation");
    button.append(icon(chat.parentId ? "branch" : "compose"), title, detail);
    if (chat === current) button.setAttribute("aria-current", "true");
    button.onclick = () => {
      if (busy) return toast("Stop the current response first.");
      selectConversation(chat);
    };
    root.append(button);
    chats
      .filter((c) => c.parentId === chat.id)
      .forEach((c) => draw(c, depth + 1));
  };
  chats
    .filter((c) => !chats.some((p) => p.id === c.parentId))
    .slice()
    .reverse()
    .forEach((c) => draw(c, 0));
  chats.forEach((c) => draw(c, 0)); // Also display malformed cyclic legacy relationships once.
  if (!chats.length) {
    const p = document.createElement("p");
    p.className = "map-empty";
    p.textContent =
      "Start a conversation, then branch from any answer to explore another direction.";
    root.append(p);
  }
  $("map-dialog").showModal();
}
$("conversation-map").onclick = showMap;
$("close-map").onclick = () => $("map-dialog").close();
function canvasView(source) {
  $("canvas-source").hidden = !source;
  $("canvas-stage").hidden = source;
  $("show-source").setAttribute("aria-pressed", String(source));
  $("show-preview").setAttribute("aria-pressed", String(!source));
}
function runCanvas() {
  const code = $("canvas-source").value;
  if (new TextEncoder().encode(code).length > 60000)
    return toast("Canvas supports up to 60 KB of source code.");
  $("canvas-payload").value = code;
  $("canvas-transport").submit();
  canvasView(false);
  $("canvas-status").textContent =
    "Preview refreshed · external resources blocked";
  try {
    localStorage.setItem("geocentric.canvas.v1", code);
  } catch {}
}
function canvasFocusMode() {
  const modal = !$("workshop").hidden && innerWidth <= 1100;
  $("main").inert = modal;
  $("sidebar").inert = modal;
  $("workshop").setAttribute("role", modal ? "dialog" : "region");
  if (modal) $("workshop").setAttribute("aria-modal", "true");
  else $("workshop").removeAttribute("aria-modal");
}
window.addEventListener("resize", canvasFocusMode);
document.addEventListener("keydown", (e) => {
  if (
    e.key === "Escape" &&
    !$("workshop").hidden &&
    !document.querySelector("dialog[open]")
  )
    $("close-workshop").click();
});
function openCanvas(code) {
  $("canvas-source").value = code;
  $("workshop").hidden = false;
  document.body.classList.add("canvas-open");
  canvasFocusMode();
  runCanvas();
  $("close-workshop").focus({ preventScroll: true });
}
$("close-workshop").onclick = () => {
  $("workshop").hidden = true;
  document.body.classList.remove("canvas-open");
  canvasFocusMode();
  try {
    localStorage.setItem("geocentric.canvas.v1", $("canvas-source").value);
  } catch {}
  $("canvas-frame").src = "about:blank"; // Stop timers and scripts when the preview closes.
  $("open-workshop").focus({ preventScroll: true });
};
$("show-source").onclick = () => canvasView(true);
$("show-preview").onclick = () => canvasView(false);
$("run-canvas").onclick = runCanvas;
$("canvas-source").addEventListener("input", () => {
  $("canvas-status").textContent =
    "Unsaved changes · run to refresh the preview";
});
$("canvas-source").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    runCanvas();
  }
});
$("download-canvas").onclick = () => {
  const blob = new Blob([$("canvas-source").value], { type: "text/html" }),
    url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "geocentric-canvas.html";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
const canvasStarter = `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A little room to think</title>
<style>*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:#faf8f5;color:#27251e;font:16px system-ui;padding:32px}main{max-width:420px;width:100%}small{letter-spacing:.14em;font-size:10px;color:#72706b}h1{font-size:38px;font-weight:500;letter-spacing:-.04em;margin:22px 0}p{line-height:1.8;color:#72706b}button{background:#30372b;color:#faf8f5;border:0;border-radius:8px;padding:12px 18px;font:inherit;cursor:pointer}#idea{min-height:86px;border-left:2px solid #9bad89;padding-left:18px;margin:28px 0}</style>
<main><small>YOUR LOCAL CREATIVE SPACE</small><h1>A little room to think.</h1><p>This is a working canvas. Edit the code, try an idea, and make something your own.</p><p id="idea">What would you make if you started small?</p><button id="next">Give me a spark ↗</button></main>
<script>const ideas=['What would you make if you started small?','Explain a complicated idea with one simple interaction.','Build a tiny tool that makes tomorrow a little easier.','Turn your notes into a place you want to revisit.'];let i=0;document.getElementById('next').onclick=()=>{document.getElementById('idea').textContent=ideas[++i%ideas.length]};<\/script></html>`;
$("open-workshop").onclick = () => {
  let saved;
  try {
    saved = localStorage.getItem("geocentric.canvas.v1");
  } catch {}
  openCanvas(saved || canvasStarter);
};
const shortcut = document.querySelector(".shortcut");
shortcut.textContent = /Mac|iPhone|iPad/.test(navigator.platform)
  ? "⌘ K"
  : "Ctrl K";
render();
restoreDraft();
connect();
