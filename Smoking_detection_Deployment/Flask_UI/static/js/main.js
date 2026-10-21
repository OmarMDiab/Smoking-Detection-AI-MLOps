const dropZone      = document.getElementById("dropZone");
const fileInput     = document.getElementById("fileInput");
const dzEmpty       = document.getElementById("dzEmpty");
const dzPreview     = document.getElementById("dzPreview");
const previewImg    = document.getElementById("previewImg");
const previewVid    = document.getElementById("previewVid");
const fileInfoName  = document.getElementById("fileInfoName");
const fileInfoSize  = document.getElementById("fileInfoSize");
const fileClear     = document.getElementById("fileClear");
const detectBtn     = document.getElementById("detectBtn");
const progressWrap  = document.getElementById("progressWrap");
const progressFill  = document.getElementById("progressFill");
const progressLabel = document.getElementById("progressLabel");
const progressPct   = document.getElementById("progressPct");
const errorBox      = document.getElementById("errorBox");
const systemStatus  = document.getElementById("systemStatus");
const idleState     = document.getElementById("idleState");
const resultsState  = document.getElementById("resultsState");
const resultsIcon   = document.getElementById("resultsIcon");
const resultsMeta   = document.getElementById("resultsMeta");
const requireHumanToggle = document.getElementById("requireHumanToggle");

let selectedFile = null;

const IMAGE_EXTS = ["jpg","jpeg","png","bmp","webp","tiff","tif","gif"];
const VIDEO_EXTS = ["mp4","avi","mov","mkv","wmv","flv","webm","m4v","mpeg","mpg","3gp"];

function getExt(name) { return name.split(".").pop().toLowerCase(); }
function isVideo(name) { return VIDEO_EXTS.includes(getExt(name)); }

// Drag & drop
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

// X button — stop propagation so it doesn't re-open file picker
fileClear.addEventListener("click", e => { e.stopPropagation(); clearFile(); });

function setFile(f) {
  const ext = getExt(f.name);
  if (![...IMAGE_EXTS, ...VIDEO_EXTS].includes(ext)) {
    showError(`Unsupported file type: .${ext}`);
    return;
  }
  selectedFile = f;
  hideError();

  // Populate info
  fileInfoName.textContent = f.name;
  fileInfoSize.textContent = (f.size / 1024 / 1024).toFixed(2) + " MB · " + ext.toUpperCase();

  // Show preview
  const url = URL.createObjectURL(f);
  if (isVideo(f.name)) {
    previewImg.style.display = "none";
    previewVid.style.display = "block";
    previewVid.src = url;
    previewVid.play().catch(() => {});
  } else {
    previewVid.style.display = "none";
    previewImg.style.display = "block";
    previewImg.src = url;
  }

  dzEmpty.style.display = "none";
  dzPreview.style.display = "flex";
  dropZone.classList.add("has-file");
  // Disable the hidden file input overlay so clicking preview doesn't re-open picker
  fileInput.style.pointerEvents = "none";

  detectBtn.disabled = false;
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  fileInput.style.pointerEvents = "";

  // Revoke blob URLs
  if (previewImg.src.startsWith("blob:")) URL.revokeObjectURL(previewImg.src);
  if (previewVid.src.startsWith("blob:")) URL.revokeObjectURL(previewVid.src);
  previewImg.src = "";
  previewVid.src = "";
  previewImg.style.display = "none";
  previewVid.style.display = "none";

  dzPreview.style.display = "none";
  dzEmpty.style.display = "flex";
  dropZone.classList.remove("has-file");

  detectBtn.disabled = true;
  hideError();
}

detectBtn.addEventListener("click", runDetection);

async function runDetection() {
  if (!selectedFile) return;

  detectBtn.disabled = true;
  hideError();
  showProgress();
  setStatus("scanning");

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("require_human", requireHumanToggle.checked ? "true" : "false");
  const isVid = isVideo(selectedFile.name);

  // Images process in well under a second, so a short fake crawl is fine.
  // Videos can take a while on CPU, so we poll real backend progress instead
  // of guessing — avoids the bar looking "stuck" while work is still happening.
  let fakeTicker = null;
  if (!isVid) {
    let fakeP = 0;
    fakeTicker = setInterval(() => {
      const step = fakeP < 88 ? Math.random() * 3.5 : Math.random() * 0.3;
      fakeP = Math.min(fakeP + step, 97);
      setProgress(fakeP, "Analyzing image…");
    }, 380);
  } else {
    setProgress(0, "Uploading…");
  }

  try {
    const res = await fetch("/api/detect", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || "Server error");

    let finalData = data;
    if (data.type === "video_job") {
      finalData = await pollVideoJob(data.job_id);
    } else if (fakeTicker) {
      clearInterval(fakeTicker);
    }

    setProgress(100, "Complete");
    setTimeout(() => {
      hideProgress();
      renderResults(finalData);
      detectBtn.disabled = false;
      setStatus("ready");
    }, 400);
  } catch (err) {
    if (fakeTicker) clearInterval(fakeTicker);
    hideProgress();
    showError(err.message);
    detectBtn.disabled = false;
    setStatus("ready");
  }
}

// Polls /api/jobs/<id> until the background video job finishes, updating the
// progress bar from the real percent/frame-count the server reports.
async function pollVideoJob(jobId) {
  while (true) {
    await new Promise(r => setTimeout(r, 500));
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();
    if (!res.ok || job.error) throw new Error(job.error || "Job lookup failed");

    if (job.status === "processing") {
      const pct = typeof job.percent === "number" ? job.percent : 0;
      const label = job.total_frames > 0
        ? `Processing frame ${job.frame_idx} / ${job.total_frames}…`
        : "Starting…";
      setProgress(pct, label);
    } else if (job.status === "done") {
      return job.result;
    } else if (job.status === "error") {
      throw new Error(job.error || "Processing failed");
    }
  }
}

function renderResults(data) {
  const statsGrid   = document.getElementById("statsGrid");
  const alertBanner = document.getElementById("alertBanner");
  const mediaFrame  = document.getElementById("mediaFrame");
  const detList     = document.getElementById("detList");

  statsGrid.innerHTML = "";
  alertBanner.innerHTML = "";
  mediaFrame.innerHTML = "";
  detList.innerHTML = "";

  const hasViolation = data.type === "image"
    ? data.count > 0
    : (data.unique_smokers ?? data.frames_with_smoking) > 0;

  // Icon & meta
  resultsIcon.className = "panel-icon results-icon" + (hasViolation ? " alert" : " clear");
  resultsIcon.innerHTML = hasViolation
    ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`
    : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;

  // Status label
  const statusLabel = hasViolation ? "Smoking Violation" : "No Violation";
  const statusMod   = hasViolation ? "alert" : "clear";

  if (data.type === "image") {
    const c = data.count;
    resultsMeta.textContent = `Image · ${c} violation${c !== 1 ? "s" : ""} found`;

    statsGrid.style.display = "grid";
    alertBanner.style.display = "none";
    statsGrid.innerHTML =
      statCard(statusLabel, "Status", statusMod, true)
      + statCard(c, "Detections", c > 0 ? "alert" : "clear");

    mediaFrame.innerHTML = `<img src="${data.result_url}" alt="Result" />`;
  } else {
    const uniqueCount = data.unique_smokers ?? "—";
    resultsMeta.textContent = `Video · ${data.total_frames} frames · ${uniqueCount} unique instance${uniqueCount === 1 ? "" : "s"} tracked`;

    // Videos get a single prominent alert banner instead of the detailed
    // stats grid — per-frame counts are noisy/less meaningful for a
    // sustained violation than the simple yes/no call that matters here.
    statsGrid.style.display = "none";
    alertBanner.style.display = "flex";
    alertBanner.className = "alert-banner " + statusMod;
    alertBanner.innerHTML = `
      <div class="alert-banner-icon">
        ${hasViolation
          ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`
          : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`}
      </div>
      <div class="alert-banner-text">
        <div class="alert-banner-title">${hasViolation ? "🚬 Smoking Violation Detected" : "No Smoking Detected"}</div>
        <div class="alert-banner-sub">${uniqueCount} unique instance${uniqueCount === 1 ? "" : "s"} tracked across ${data.total_frames} frames</div>
      </div>`;

    mediaFrame.innerHTML = `
      <video controls playsinline>
        <source src="${data.result_url}" type="video/mp4" />
      </video>`;
  }

  const dl = document.getElementById("downloadLink");
  dl.href = data.result_url;
  dl.download = data.result_url.split("/").pop();

  idleState.style.display = "none";
  resultsState.style.display = "flex";
}

function statCard(val, label, modifier, isStatus = false) {
  const valClass = isStatus ? "stat-val stat-val-text" : "stat-val";
  return `<div class="stat-card ${modifier}">
    <div class="${valClass}">${val}</div>
    <div class="stat-label">${label}</div>
  </div>`;
}

function setStatus(state) {
  if (state === "scanning") {
    systemStatus.className = "status-pill scanning";
    systemStatus.innerHTML = `<span class="pulse-dot"></span> Scanning…`;
  } else {
    systemStatus.className = "status-pill";
    systemStatus.innerHTML = `<span class="pulse-dot"></span> System Ready`;
  }
}

function showProgress() {
  progressWrap.style.display = "flex";
  _lastProgressPct = 0;
  setProgress(0, "Uploading…");
}

function hideProgress() { progressWrap.style.display = "none"; }

// Guards against the bar ever visibly jumping backwards (e.g. a stray/late
// poll response arriving out of order) — always clamps to the highest
// percent seen so far for this run.
let _lastProgressPct = 0;

function setProgress(val, label) {
  _lastProgressPct = Math.max(_lastProgressPct, val);
  progressFill.style.width = _lastProgressPct + "%";
  progressLabel.textContent = label;
  progressPct.textContent = Math.round(_lastProgressPct) + "%";
}

function showError(msg) {
  errorBox.textContent = "⚠ " + msg;
  errorBox.style.display = "block";
}

function hideError() { errorBox.style.display = "none"; }
