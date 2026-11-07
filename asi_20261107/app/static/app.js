// Video → Test Procedure: browser side.
// Flow: pick/record video → upload to S3 (presigned) + sample key frames → /api/concepts (Claude)
//       → /api/segment (Meta SAM 3.1) → draw SAM boxes on key frames → /api/analyze (Claude + SAM evidence)

const $ = (s) => document.querySelector(s);
const COLORS = ["#3b82f6", "#f97316", "#22c55e", "#ec4899", "#a855f7", "#14b8a6", "#eab308", "#ef4444",
                "#0ea5e9", "#84cc16", "#f43f5e", "#6366f1"];
const FRAME_W = 768;

let decodeMaskToRaster = null; // from Meta's @meta-sam/parser, loaded lazily
import("https://cdn.jsdelivr.net/npm/@meta-sam/parser@0.1.0/+esm")
  .then((m) => { decodeMaskToRaster = m.decodeMaskToRaster; })
  .catch((e) => console.warn("SAM parser unavailable; showing boxes only", e));

const state = {
  blob: null, name: "", url: null, recordedDuration: 0,
  seg: null, result: null, rawFrames: [], annotated: [], hidden: new Set(), maskCache: new Map(),
};

// ------------------------------------------------------------------ helpers
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtT = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
const colorOf = (i) => COLORS[i % COLORS.length];

async function api(path, body) {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { throw new Error(`${path}: HTTP ${res.status} ${text.slice(0, 300)}`); }
  if (!res.ok) throw new Error(data.error || `${path}: HTTP ${res.status}`);
  return data;
}

function setStage(stage, status, note) {
  const li = document.querySelector(`[data-stage="${stage}"]`);
  li.className = status;
  if (status === "running") li.dataset.start = performance.now();
  if (status === "done" || status === "failed") {
    li.querySelector("time").textContent = `${((performance.now() - li.dataset.start) / 1000).toFixed(0)}s`;
  }
  if (note) $("#statusLine").textContent = note;
}

// ------------------------------------------------------------------ input: tabs, upload, record
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  $("#pane-upload").hidden = b.dataset.tab !== "upload";
  $("#pane-record").hidden = b.dataset.tab !== "record";
}));

const drop = $("#drop");
["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (ev) => ev.dataTransfer.files[0] && useVideo(ev.dataTransfer.files[0], ev.dataTransfer.files[0].name));
$("#fileInput").addEventListener("change", (ev) => ev.target.files[0] && useVideo(ev.target.files[0], ev.target.files[0].name));

function useVideo(blob, name, recordedDuration = 0) {
  if (state.url) URL.revokeObjectURL(state.url);
  Object.assign(state, { blob, name, url: URL.createObjectURL(blob), recordedDuration });
  $("#srcPreview").src = state.url;
  $("#srcName").textContent = name;
  $("#srcInfo").textContent = `${(blob.size / 1e6).toFixed(1)} MB · ${blob.type || "video"}`;
  $("#selected").hidden = false;
  $("#analyze").disabled = false;
}

let stream = null, recorder = null, chunks = [], recTimer = null, recStart = 0;

async function listCams() {
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  $("#camSelect").innerHTML = devs.map((d, i) => `<option value="${d.deviceId}">${esc(d.label || `Camera ${i + 1}`)}</option>`).join("");
}

async function startCam() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  const id = $("#camSelect").value;
  stream = await navigator.mediaDevices.getUserMedia({
    video: id ? { deviceId: { exact: id }, width: { ideal: 1280 }, height: { ideal: 720 } }
              : { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  $("#camPreview").srcObject = stream;
  $("#recStart").disabled = false;
  await listCams();
  const cur = stream.getVideoTracks()[0].getSettings().deviceId;
  if (cur) $("#camSelect").value = cur;
}
$("#camStart").addEventListener("click", () => startCam().catch((e) => alert(`Camera error: ${e.message}`)));
$("#camSelect").addEventListener("change", () => stream && startCam().catch((e) => alert(e.message)));

function pickMime() {
  // SAM accepts MP4/MOV (H.264/H.265); prefer it so no server-side conversion is needed.
  const prefs = ["video/mp4;codecs=avc1.42E01E", "video/mp4;codecs=avc1", "video/mp4",
                 "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  return prefs.find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";
}

$("#recStart").addEventListener("click", () => {
  const mimeType = pickMime();
  recorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 4_000_000 } : undefined);
  chunks = [];
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    const type = (recorder.mimeType || mimeType || "video/webm").split(";")[0];
    const ext = type.includes("mp4") ? "mp4" : "webm";
    const dur = (performance.now() - recStart) / 1000;
    useVideo(new Blob(chunks, { type }), `recording-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.${ext}`, dur);
  };
  recorder.start(500);
  recStart = performance.now();
  $("#recBadge").hidden = false;
  recTimer = setInterval(() => { $("#recTime").textContent = fmtT((performance.now() - recStart) / 1000).replace(/\.\d$/, ""); }, 250);
  $("#recStart").disabled = true; $("#recStop").disabled = false;
  $("#recHint").textContent = `Recording as ${mimeType || "browser default"}…`;
});
$("#recStop").addEventListener("click", () => {
  recorder.stop(); clearInterval(recTimer);
  $("#recBadge").hidden = true; $("#recStart").disabled = false; $("#recStop").disabled = true;
  $("#recHint").textContent = "Recording captured. Click Analyze, or record again.";
});

// ------------------------------------------------------------------ frame sampling
function loadVideo(url) {
  return new Promise((resolve, reject) => {
    const v = document.createElement("video");
    v.muted = true; v.playsInline = true; v.preload = "auto"; v.src = url;
    v.onloadedmetadata = async () => {
      if (!isFinite(v.duration)) { // MediaRecorder WebM has no duration until we seek to the end
        await new Promise((r) => { v.ontimeupdate = () => { v.ontimeupdate = null; r(); }; v.currentTime = 1e9; });
      }
      resolve(v);
    };
    v.onerror = () => reject(new Error("This browser cannot decode the video (try MP4/H.264)."));
  });
}

function seek(v, t) {
  return new Promise((resolve) => {
    const done = () => { v.removeEventListener("seeked", done); resolve(); };
    v.addEventListener("seeked", done);
    v.currentTime = t;
  });
}

async function sampleFrames(n) {
  const v = await loadVideo(state.url);
  const duration = isFinite(v.duration) && v.duration > 0 ? v.duration : state.recordedDuration;
  const scale = FRAME_W / v.videoWidth;
  const c = document.createElement("canvas");
  c.width = FRAME_W; c.height = Math.round(v.videoHeight * scale);
  const ctx = c.getContext("2d");
  const frames = [];
  for (let i = 0; i < n; i++) {
    const t = (duration * (i + 0.5)) / n;
    await seek(v, Math.min(t, duration - 0.05));
    ctx.drawImage(v, 0, 0, c.width, c.height);
    frames.push({ t: +t.toFixed(3), data: c.toDataURL("image/jpeg", 0.85) });
  }
  return { frames, duration, width: v.videoWidth, height: v.videoHeight };
}

function loadImage(src) {
  return new Promise((r, j) => { const im = new Image(); im.onload = () => r(im); im.onerror = j; im.src = src; });
}

function nearest(rows, f, tol) {
  // rows sorted by frame index (row[0] or row.f); binary search
  let lo = 0, hi = rows.length - 1;
  const key = (r) => (Array.isArray(r) ? r[0] : r.f);
  if (hi < 0) return null;
  while (lo < hi) { const mid = (lo + hi) >> 1; key(rows[mid]) < f ? (lo = mid + 1) : (hi = mid); }
  let best = rows[lo];
  if (lo > 0 && Math.abs(key(rows[lo - 1]) - f) < Math.abs(key(best) - f)) best = rows[lo - 1];
  return Math.abs(key(best) - f) <= tol ? best : null;
}

// Draw thin SAM boxes + "T#" tags on key frames so Claude can tie track IDs to what it sees.
async function annotateFrames(frames, seg) {
  const out = [];
  for (const fr of frames) {
    const im = await loadImage(fr.data);
    const c = document.createElement("canvas");
    c.width = im.width; c.height = im.height;
    const ctx = c.getContext("2d");
    ctx.drawImage(im, 0, 0);
    const f = Math.round(fr.t * seg.fps);
    seg.tracks.forEach((t, i) => {
      const b = nearest(t.boxes, f, 3);
      if (!b) return;
      const [x1, y1, x2, y2] = [b[1] * c.width, b[2] * c.height, b[3] * c.width, b[4] * c.height];
      ctx.strokeStyle = colorOf(i); ctx.lineWidth = 2; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.font = "bold 12px sans-serif";
      const w = ctx.measureText(t.track_id).width + 6, ty = y1 >= 14 ? y1 - 14 : y1;
      ctx.fillStyle = colorOf(i); ctx.fillRect(x1, ty, w, 14);
      ctx.fillStyle = "#fff"; ctx.fillText(t.track_id, x1 + 3, ty + 11);
    });
    out.push({ t: fr.t, data: c.toDataURL("image/jpeg", 0.85) });
  }
  return out;
}

// ------------------------------------------------------------------ pipeline
$("#analyze").addEventListener("click", run);

async function run() {
  $("#analyze").disabled = true;
  $("#progressCard").hidden = false; $("#errorBox").hidden = true; $("#results").hidden = true;
  document.querySelectorAll("#stages li").forEach((li) => { li.className = ""; li.querySelector("time").textContent = ""; });
  const context = $("#context").value.trim();
  const n = +$("#frameCount").value;
  let stage = "prep";
  try {
    setStage("prep", "running", "Uploading video and sampling key frames…");
    const type = state.blob.type || (state.name.toLowerCase().endsWith(".mov") ? "video/quicktime" : "video/mp4");
    const upload = (async () => {
      const u = await api("/api/upload-url", { filename: state.name, content_type: type });
      const r = await fetch(u.url, { method: "PUT", headers: { "Content-Type": u.content_type }, body: state.blob });
      if (!r.ok) throw new Error(`Upload failed: HTTP ${r.status}`);
      return u.key;
    })();
    const sampled = await sampleFrames(n);
    state.rawFrames = sampled.frames;

    // Concepts can be chosen while the upload finishes.
    const manual = $("#concepts").value.split(",").map((s) => s.trim()).filter(Boolean);
    let conceptsP;
    if (manual.length) {
      conceptsP = Promise.resolve({ concepts: manual.includes("hand") ? manual : ["hand", ...manual], scene: "" });
    } else {
      const pick = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => sampled.frames[Math.floor((i * sampled.frames.length) / 8)]);
      conceptsP = api("/api/concepts", { frames: [...new Set(pick)], context });
    }
    const key = await upload;
    setStage("prep", "done");

    stage = "concepts";
    setStage("concepts", "running", "Claude is choosing what SAM should segment…");
    const { concepts, scene } = await conceptsP;
    setStage("concepts", "done", `Tracking: ${concepts.join(", ")}${scene ? ` — ${scene}` : ""}`);

    stage = "segment";
    setStage("segment", "running", `SAM 3.1 is segmenting and tracking ${concepts.length} concepts across every frame… (typically 15s–3 min)`);
    const seg = await api("/api/segment", { key, concepts, duration: sampled.duration });
    state.seg = seg;
    const found = seg.concepts_found || [];
    setStage("segment", "done", `SAM found ${seg.tracks.length} object instance(s) for: ${found.join(", ") || "nothing"}`
      + (seg.errors?.length ? ` (errors: ${seg.errors.join("; ")})` : ""));

    stage = "analyze";
    setStage("analyze", "running", "Claude is building the step breakdown and test case from frames + SAM evidence…");
    state.annotated = await annotateFrames(sampled.frames, seg);
    const result = await api("/api/analyze", {
      frames: state.annotated, context, fps: seg.fps, duration: seg.duration || sampled.duration,
      tracks: seg.tracks.map(({ track_id, label, concept, boxes }) => ({ track_id, label, concept, boxes })),
    });
    state.result = { ...result, concepts, tracks: seg.tracks.map(({ track_id, label, concept }) => ({ track_id, label, concept })) };
    setStage("analyze", "done", "Done.");
    render();
  } catch (e) {
    console.error(e);
    setStage(stage, "failed");
    $("#errorBox").hidden = false;
    $("#errorBox").textContent = e.message;
  } finally {
    $("#analyze").disabled = false;
  }
}

// ------------------------------------------------------------------ results rendering
function list(el, items) { $(el).innerHTML = (items || []).map((x) => `<li>${esc(x)}</li>`).join("") || "<li class='muted'>—</li>"; }

function trackIndex(id) { return state.seg.tracks.findIndex((t) => t.track_id === id); }
function chip(id) {
  const i = trackIndex(id);
  if (i < 0) return `<span class="chip"><b>${esc(id)}</b></span>`;
  const t = state.seg.tracks[i];
  const named = (state.result.objects || []).find((o) => o.track_id === id);
  return `<span class="chip" data-track="${id}" title="${esc(named ? `${named.name} — ${named.role}` : t.label)}"><i style="background:${colorOf(i)}"></i><b>${id}</b> ${esc(named?.name || t.label)}</span>`;
}

function render() {
  const r = state.result, tc = r.test_case || {}, seg = state.seg;
  $("#inputCard").hidden = true; $("#progressCard").hidden = true;
  $("#results").hidden = false; $("#exportBar").hidden = false;
  $("#tcId").textContent = tc.id || "";
  $("#title").textContent = tc.title || r.title;
  $("#summary").textContent = r.summary;
  $("#objective").textContent = tc.objective || r.objective;
  $("#equipment").innerHTML = (r.equipment || []).map((e) => `<li><b>${esc(e.name)}</b><br><span>${esc(e.role)}</span></li>`).join("");
  $("#modelInfo").textContent = r.model ? `model: ${r.model}` : "";

  // Steps
  $("#steps").innerHTML = (r.steps || []).map((s, i) => {
    const kf = state.annotated[Math.max(0, Math.min(state.annotated.length - 1, s.key_frame ?? 0))];
    return `<div class="step" data-start="${s.start}" data-i="${i}">
      <div class="num">${i + 1}</div>
      <img src="${kf?.data || ""}" alt="key frame">
      <div>
        <div class="when">${fmtT(s.start)} – ${fmtT(s.end)}</div>
        <div class="act">${esc(s.action)}</div>
        <div class="obs">${esc(s.observation)}</div>
        ${s.sam_evidence ? `<div class="ev">SAM: ${esc(s.sam_evidence)}</div>` : ""}
        <div class="chips">${(s.track_ids || []).map(chip).join("")}</div>
      </div></div>`;
  }).join("");
  document.querySelectorAll(".step").forEach((el) => el.addEventListener("click", () => seekPlayer(+el.dataset.start)));

  // Test case
  list("#preconditions", tc.preconditions); list("#setup", tc.setup);
  list("#pass", tc.pass_criteria); list("#fail", tc.fail_criteria);
  list("#teardown", tc.teardown); list("#assumptions", r.assumptions);
  $("#procedure").innerHTML = (tc.procedure || []).map((p) => {
    const m = /([\d.]+)\s*[-–]\s*([\d.]+)/.exec(p.video_ref || "");
    const ref = m ? `<a data-seek="${m[1]}">${esc(p.video_ref)}</a>` : esc(p.video_ref || "");
    return `<tr><td>${p.step}</td><td>${esc(p.action)}</td><td>${esc(p.expected_result)}</td><td>${ref}</td></tr>`;
  }).join("");
  document.querySelectorAll("[data-seek]").forEach((a) => a.addEventListener("click", () => seekPlayer(+a.dataset.seek)));
  $("#evidence").textContent = r.sam_evidence || "";

  // Player + overlay
  const player = $("#player");
  player.src = state.url;
  state.maskCache.clear(); state.hidden.clear();
  const masks = seg.tracks.reduce((n, t) => n + t.masks.length, 0);
  $("#trackStats").textContent = `${seg.tracks.length} tracks · ${seg.frames} frames · ${masks} masks${decodeMaskToRaster ? "" : " (mask decoder unavailable)"}`;
  renderLegend(); renderTimeline();
  startOverlayLoop();
}

function renderLegend() {
  $("#legend").innerHTML = state.seg.tracks.map((t) => chip(t.track_id)).join("") || "<span class='muted small'>SAM found no objects.</span>";
  $("#legend").querySelectorAll(".chip").forEach((c) => {
    c.classList.toggle("off", state.hidden.has(c.dataset.track));
    c.addEventListener("click", () => {
      const id = c.dataset.track;
      state.hidden.has(id) ? state.hidden.delete(id) : state.hidden.add(id);
      renderLegend(); renderTimeline(); drawOverlay();
    });
  });
}

function intervals(boxes, fps, gap = 0.4) {
  const out = [];
  for (const b of boxes) {
    const t = b[0] / fps;
    if (out.length && t - out[out.length - 1][1] <= gap) out[out.length - 1][1] = t; else out.push([t, t]);
  }
  return out;
}

function renderTimeline() {
  const seg = state.seg, dur = seg.duration || $("#player").duration || 1;
  const pct = (t) => `${Math.max(0, Math.min(100, (t / dur) * 100))}%`;
  const steps = (state.result.steps || []).map((s, i) =>
    `<div class="seg" style="left:${pct(s.start)};width:calc(${pct(s.end)} - ${pct(s.start)})" title="${esc(s.action)}">${i + 1}</div>`).join("");
  let html = `<div class="lane steps-lane"><div class="name"><b>Steps</b></div><div class="bar">${steps}</div></div>`;
  seg.tracks.forEach((t, i) => {
    if (state.hidden.has(t.track_id)) return;
    const segs = intervals(t.boxes, seg.fps).map(([a, b]) =>
      `<div class="seg" style="left:${pct(a)};width:max(2px,calc(${pct(b)} - ${pct(a)}));background:${colorOf(i)}"></div>`).join("");
    html += `<div class="lane"><div class="name" title="${esc(t.label)}">${t.track_id} ${esc(t.label)}</div><div class="bar">${segs}</div></div>`;
  });
  $("#timeline").innerHTML = html + `<div class="playhead" id="playhead"></div>`;
  $("#timeline").querySelectorAll(".bar").forEach((bar) => bar.addEventListener("click", (e) => {
    const r = bar.getBoundingClientRect();
    seekPlayer(((e.clientX - r.left) / r.width) * dur);
  }));
}

function seekPlayer(t) {
  const p = $("#player");
  p.currentTime = Math.max(0, t);
  p.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ------------------------------------------------------------------ SAM overlay on the player
const MASK_MAX_DIM = 320, MASK_CACHE_MAX = 400;
const scratch = document.createElement("canvas");

function maskCanvas(track, i, m) {
  const key = `${track.track_id}:${m.f}`;
  const cache = state.maskCache;
  if (cache.has(key)) { const v = cache.get(key); cache.delete(key); cache.set(key, v); return v; } // LRU touch
  let canvas = null;
  try {
    const raster = decodeMaskToRaster(m.mask); // Meta's decoder: Uint8Array of 0/1, row-major
    const { width: w, height: h } = m.mask;
    scratch.width = w; scratch.height = h;
    const sctx = scratch.getContext("2d");
    const img = sctx.createImageData(w, h);
    const hex = colorOf(i);
    const [r, g, b] = [1, 3, 5].map((o) => parseInt(hex.slice(o, o + 2), 16));
    for (let p = 0; p < raster.length; p++) {
      if (raster[p]) { const q = p * 4; img.data[q] = r; img.data[q + 1] = g; img.data[q + 2] = b; img.data[q + 3] = 115; }
    }
    sctx.putImageData(img, 0, 0);
    const k = Math.min(1, MASK_MAX_DIM / Math.max(w, h));
    canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(w * k)); canvas.height = Math.max(1, Math.round(h * k));
    canvas.getContext("2d").drawImage(scratch, 0, 0, canvas.width, canvas.height);
  } catch (e) { console.warn("mask decode failed", e); }
  cache.set(key, canvas);
  if (cache.size > MASK_CACHE_MAX) cache.delete(cache.keys().next().value);
  return canvas;
}

function drawOverlay() {
  const p = $("#player"), cv = $("#overlay"), seg = state.seg;
  if (!seg || !p.videoWidth) return;
  const dpr = window.devicePixelRatio || 1;
  const W = p.clientWidth, H = p.clientHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    cv.style.width = `${W}px`; cv.style.height = `${H}px`;
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  // object-fit: contain → content rectangle (controls bar overlaps bottom, that's fine)
  const s = Math.min(W / p.videoWidth, H / p.videoHeight);
  const cw = p.videoWidth * s, ch = p.videoHeight * s, ox = (W - cw) / 2, oy = (H - ch) / 2;
  const f = Math.round(p.currentTime * seg.fps);
  const showM = $("#showMasks").checked && decodeMaskToRaster, showB = $("#showBoxes").checked, showL = $("#showLabels").checked;
  const maskTol = Math.max(4, Math.ceil(seg.fps / 3));

  seg.tracks.forEach((t, i) => {
    if (state.hidden.has(t.track_id)) return;
    const b = nearest(t.boxes, f, 3);
    if (!b) return;
    const col = colorOf(i);
    if (showM) {
      const m = nearest(t.masks, f, maskTol);
      const mc = m && maskCanvas(t, i, m);
      if (mc) {
        const [x1, y1, x2, y2] = m.bounds;
        ctx.drawImage(mc, ox + x1 * cw, oy + y1 * ch, (x2 - x1) * cw, (y2 - y1) * ch);
      }
    }
    const [x1, y1, x2, y2] = [ox + b[1] * cw, oy + b[2] * ch, ox + b[3] * cw, oy + b[4] * ch];
    if (showB) { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); }
    if (showL) {
      const named = (state.result.objects || []).find((o) => o.track_id === t.track_id);
      const label = `${t.track_id} ${named?.name || t.label}`.slice(0, 40);
      ctx.font = "600 12px -apple-system, sans-serif";
      const w = ctx.measureText(label).width + 8, ty = y1 >= 16 ? y1 - 16 : y1;
      ctx.fillStyle = col; ctx.fillRect(x1, ty, w, 16);
      ctx.fillStyle = "#fff"; ctx.fillText(label, x1 + 4, ty + 12);
    }
  });

  // playhead + active step
  const dur = seg.duration || p.duration || 1;
  const ph = $("#playhead");
  if (ph) { const bar = $("#timeline .bar"); ph.style.left = `${bar.offsetLeft + (p.currentTime / dur) * bar.offsetWidth}px`; }
  const t = p.currentTime;
  document.querySelectorAll(".step").forEach((el) => {
    const s = state.result.steps[+el.dataset.i];
    el.classList.toggle("active", t >= s.start && t <= s.end);
  });
}

let loopStarted = false;
function startOverlayLoop() {
  if (loopStarted) return;
  loopStarted = true;
  const p = $("#player");
  const tick = () => { drawOverlay(); requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
  ["showMasks", "showBoxes", "showLabels"].forEach((id) => $(`#${id}`).addEventListener("change", drawOverlay));
  p.addEventListener("loadeddata", drawOverlay);
}

// ------------------------------------------------------------------ export
function toMarkdown(r) {
  const tc = r.test_case || {};
  const ul = (a) => (a || []).map((x) => `- ${x}`).join("\n") || "- (none)";
  const cell = (s) => String(s ?? "").replace(/\|/g, "\\|").replace(/\n/g, " ");
  return [
    `# ${tc.id || "TC"}: ${tc.title || r.title}`, "", `**Objective:** ${tc.objective || r.objective}`, "",
    "## Summary of the recorded run", "", r.summary, "",
    "## Equipment", "", ul((r.equipment || []).map((e) => `**${e.name}**: ${e.role}`)), "",
    "## Observed action steps (from video)", "", "| # | Time | Action | Observation | SAM tracks |", "|---|---|---|---|---|",
    ...(r.steps || []).map((s, i) => `| ${i + 1} | ${s.start.toFixed(1)}-${s.end.toFixed(1)}s | ${cell(s.action)} | ${cell(s.observation)} | ${(s.track_ids || []).join(", ")} |`),
    "", "## Test procedure", "", "### Preconditions", "", ul(tc.preconditions), "", "### Setup", "", ul(tc.setup), "",
    "### Steps", "", "| Step | Action | Expected result | Video ref |", "|---|---|---|---|",
    ...(tc.procedure || []).map((p) => `| ${p.step} | ${cell(p.action)} | ${cell(p.expected_result)} | ${cell(p.video_ref)} |`),
    "", "### Pass criteria", "", ul(tc.pass_criteria), "", "### Fail criteria", "", ul(tc.fail_criteria), "",
    "### Teardown", "", ul(tc.teardown), "", "## Assumptions", "", ul(r.assumptions), "",
    "## SAM 3.1 evidence", "", "```", r.sam_evidence || "", "```", "",
  ].join("\n");
}

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
const stem = () => (state.name || "video").replace(/\.[^.]+$/, "");
$("#exportMd").addEventListener("click", () => download(`${stem()}.test_case.md`, toMarkdown(state.result), "text/markdown"));
$("#exportJson").addEventListener("click", () => download(`${stem()}.analysis.json`, JSON.stringify(state.result, null, 2), "application/json"));
$("#newRun").addEventListener("click", () => {
  $("#results").hidden = true; $("#exportBar").hidden = true; $("#inputCard").hidden = false; $("#player").pause();
  window.scrollTo({ top: 0 });
});
