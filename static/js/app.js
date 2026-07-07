/* ── State ──────────────────────────────────────────────────────────────── */
const API = '';           // same origin
let selectedFile = null;
const history   = [];     // local extraction history (max 10)

/* ── Init ───────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  loadQueue();
  setInterval(checkApiHealth, 30_000);
  setInterval(loadQueue,      15_000);
  wireUpDropZone();
});

/* ── Tab switching ──────────────────────────────────────────────────────── */
function switchTab(name) {
  document.querySelectorAll('.tab-pane').forEach(p => p.hidden = true);
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`tab-${name}`).hidden = false;
  document.querySelector(`[data-tab="${name}"]`).classList.add('active');
  if (name === 'queue') loadQueue();
}

/* ── API health check ───────────────────────────────────────────────────── */
async function checkApiHealth() {
  const dot   = document.getElementById('api-dot');
  const label = document.getElementById('api-label');
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
    if (r.ok) {
      dot.className   = 'status-dot dot-green';
      label.textContent = 'API Online';
    } else throw new Error('non-200');
  } catch {
    dot.className   = 'status-dot dot-red';
    label.textContent = 'API Offline';
  }
}

/* ── Drop zone wiring ───────────────────────────────────────────────────── */
function wireUpDropZone() {
  const zone  = document.getElementById('drop-zone');
  const input = document.getElementById('file-input');

  document.getElementById('browse-link').addEventListener('click', () => input.click());
  zone.addEventListener('click', () => input.click());

  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });

  input.addEventListener('change', () => {
    if (input.files[0]) setFile(input.files[0]);
  });
}

/* ── File selection ─────────────────────────────────────────────────────── */
const ALLOWED = ['image/jpeg', 'image/png', 'application/pdf'];

function setFile(file) {
  if (!ALLOWED.includes(file.type)) {
    toast('Only JPG, PNG, and PDF files are supported.', 'error');
    return;
  }

  selectedFile = file;

  // File preview panel
  document.getElementById('drop-zone').hidden    = true;
  document.getElementById('file-preview').hidden = false;
  document.getElementById('file-name').textContent = file.name;
  document.getElementById('file-size').textContent = formatBytes(file.size);

  // Thumb emoji & image preview
  const thumbMap = { 'application/pdf': '📕', 'image/jpeg': '🖼️', 'image/png': '🖼️' };
  document.getElementById('file-thumb').textContent = thumbMap[file.type] || '📄';

  const wrap = document.getElementById('img-preview-wrap');
  if (file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = e => {
      document.getElementById('img-preview').src = e.target.result;
      wrap.hidden = false;
    };
    reader.readAsDataURL(file);
  } else {
    wrap.hidden = true;
  }

  document.getElementById('process-btn').disabled = false;
}

function clearFile() {
  selectedFile = null;
  document.getElementById('file-input').value      = '';
  document.getElementById('drop-zone').hidden      = false;
  document.getElementById('file-preview').hidden   = true;
  document.getElementById('process-btn').disabled  = true;
  document.getElementById('img-preview-wrap').hidden = true;
}

/* ── Process document ───────────────────────────────────────────────────── */
async function processDocument() {
  if (!selectedFile) return;

  const btn     = document.getElementById('process-btn');
  const btnText = document.getElementById('btn-text');
  const spinner = document.getElementById('btn-spinner');

  btn.disabled         = true;
  btnText.textContent  = 'Processing…';
  spinner.classList.remove('hidden');

  const form = new FormData();
  form.append('file', selectedFile);

  try {
    const res  = await fetch(`${API}/process-document`, { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) {
      toast(`Error: ${data.detail || res.statusText}`, 'error');
      return;
    }

    renderResult(data, selectedFile.name);
    addToHistory(data, selectedFile.name);
    toast(
      data.status === 'automated'
        ? '✅ Document processed automatically'
        : '⚠️ Routed to manual review queue',
      data.status === 'automated' ? 'success' : 'warning'
    );
    loadQueue();

  } catch (err) {
    toast('Network error — is the API running?', 'error');
    console.error(err);
  } finally {
    btn.disabled        = false;
    btnText.textContent = 'Process Document';
    spinner.classList.add('hidden');
  }
}

/* ── Render result ──────────────────────────────────────────────────────── */
function renderResult(data, filename) {
  document.getElementById('result-placeholder').hidden = true;
  document.getElementById('result-content').hidden     = false;

  const ent  = data.extracted_entities;
  const conf = data.confidence_score;
  const pct  = (conf * 100).toFixed(1);

  // Doc type & status pills
  document.getElementById('res-doc-type').textContent = data.document_type;

  const statusPill = document.getElementById('res-status-pill');
  if (data.status === 'automated') {
    statusPill.textContent = 'Automated';
    statusPill.className   = 'pill pill-auto';
  } else {
    statusPill.textContent = 'Manual Review';
    statusPill.className   = 'pill pill-manual';
  }

  const authPill = document.getElementById('res-authenticity-pill');
  const authMap = {
    original: ['✅ Original', 'pill pill-original'],
    fake:     ['🚫 Fake / Flagged', 'pill pill-fake'],
    unknown:  ['❔ Unknown', 'pill pill-unknown'],
  };
  const [authText, authClass] = authMap[data.authenticity] || authMap.unknown;
  authPill.textContent = authText;
  authPill.className   = authClass;

  // Confidence bar (animate on next frame)
  document.getElementById('res-confidence-pct').textContent = `${pct}%`;
  const bar = document.getElementById('res-confidence-bar');
  bar.style.width = '0%';
  bar.className   = conf < 0.85 ? 'confidence-bar warn' : 'confidence-bar';
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { bar.style.width = `${pct}%`; });
  });

  // Entities
  document.getElementById('res-name').textContent     = ent.name      || '—';
  document.getElementById('res-pan').textContent      = ent.pan       || '—';
  document.getElementById('res-dob').textContent      = ent.dob       || '—';
  document.getElementById('res-aadhaar').textContent  = ent.aadhaar   || '—';
  document.getElementById('res-passport').textContent = ent.passport  || '—';
  document.getElementById('res-dl').textContent       = ent.dl_number || '—';
  document.getElementById('res-voterid').textContent  = ent.voter_id  || '—';

  const amountsEl = document.getElementById('res-amounts');
  amountsEl.innerHTML = '';
  (ent.amounts || []).forEach(a => {
    const chip = document.createElement('span');
    chip.className   = 'amount-chip';
    chip.textContent = `₹ ${a}`;
    amountsEl.appendChild(chip);
  });
  if (!ent.amounts?.length) amountsEl.textContent = '—';

  // Meta
  document.getElementById('res-doc-id').textContent   = data.document_id;
  document.getElementById('res-timestamp').textContent = formatDate(data.timestamp);
}

/* ── History strip ──────────────────────────────────────────────────────── */
function addToHistory(data, filename) {
  history.unshift({ data, filename });
  if (history.length > 10) history.pop();
  renderHistory();
}

function renderHistory() {
  const card = document.getElementById('history-card');
  const list = document.getElementById('history-list');
  if (!history.length) { card.hidden = true; return; }
  card.hidden = false;
  list.innerHTML = '';

  history.forEach(({ data, filename }) => {
    const conf = (data.confidence_score * 100).toFixed(1);
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `
      <div class="file-thumb">${filename.match(/\.pdf$/i) ? '📕' : '🖼️'}</div>
      <div class="history-left">
        <div class="history-name">${escHtml(filename)}</div>
        <div class="history-time">${formatDate(data.timestamp)}</div>
      </div>
      <div class="history-right">
        <span class="pill pill-doc">${escHtml(data.document_type)}</span>
        <span class="pill ${data.status === 'automated' ? 'pill-auto' : 'pill-manual'}">${conf}%</span>
      </div>`;
    list.appendChild(item);
  });
}

/* ── Review queue ───────────────────────────────────────────────────────── */
async function loadQueue() {
  try {
    const res  = await fetch(`${API}/review-queue`);
    const data = await res.json();

    // Update badge
    const badge = document.getElementById('queue-badge');
    badge.textContent = data.queue_length;
    badge.style.display = data.queue_length > 0 ? 'inline-flex' : 'none';

    renderQueue(data.items);
  } catch {
    // silently ignore if API is down
  }
}

function renderQueue(items) {
  const empty = document.getElementById('queue-empty');
  const table = document.getElementById('queue-table');
  const tbody = document.getElementById('queue-tbody');

  if (!items.length) {
    empty.hidden = false;
    table.hidden = true;
    return;
  }

  empty.hidden = false;
  empty.hidden = true;
  table.hidden = false;
  tbody.innerHTML = '';

  items.forEach(item => {
    const conf = (item.confidence_score * 100).toFixed(1);
    const tr   = document.createElement('tr');
    tr.innerHTML = `
      <td class="mono" style="font-size:11px">${escHtml(item.document_id.substring(0, 18))}…</td>
      <td class="conf-warn">${conf}%</td>
      <td>${escHtml(item.reason)}</td>
      <td>${formatDate(item.timestamp)}</td>
      <td>
        <button class="resolve-btn" onclick="resolveItem('${escHtml(item.document_id)}', this)">
          ✓ Resolve
        </button>
        <button class="reject-btn" onclick="rejectItem('${escHtml(item.document_id)}', this)">
          ✕ Reject
        </button>
      </td>`;
    tbody.appendChild(tr);
  });
}

async function resolveItem(docId, btn) {
  btn.disabled       = true;
  btn.textContent    = '…';
  try {
    const res = await fetch(`${API}/review-queue/${encodeURIComponent(docId)}`, { method: 'DELETE' });
    if (res.ok) {
      toast('Document resolved and removed from queue', 'success');
      loadQueue();
    } else {
      const d = await res.json();
      toast(`Failed: ${d.detail}`, 'error');
      btn.disabled    = false;
      btn.textContent = '✓ Resolve';
    }
  } catch {
    toast('Network error', 'error');
    btn.disabled    = false;
    btn.textContent = '✓ Resolve';
  }
}

async function rejectItem(docId, btn) {
  btn.disabled       = true;
  btn.textContent    = '…';
  try {
    const res = await fetch(`${API}/review-queue/${encodeURIComponent(docId)}/reject`, { method: 'POST' });
    if (res.ok) {
      toast('Document rejected and removed from queue', 'success');
      loadQueue();
    } else {
      const d = await res.json();
      toast(`Failed: ${d.detail}`, 'error');
      btn.disabled    = false;
      btn.textContent = '✕ Reject';
    }
  } catch {
    toast('Network error', 'error');
    btn.disabled    = false;
    btn.textContent = '✕ Reject';
  }
}

/* ── Toast notifications ────────────────────────────────────────────────── */
function toast(msg, type = 'info') {
  const el      = document.createElement('div');
  el.className  = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => {
    el.style.animation = 'slideOut .3s ease forwards';
    setTimeout(() => el.remove(), 300);
  }, 3500);
}

/* ── Helpers ────────────────────────────────────────────────────────────── */
function formatBytes(bytes) {
  if (bytes < 1024)       return `${bytes} B`;
  if (bytes < 1048576)    return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  } catch { return iso; }
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
