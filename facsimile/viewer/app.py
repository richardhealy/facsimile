"""Page-image viewer with retrieval highlights.

The viewer is a single-page HTML/JS application served by the FastAPI app at
``GET /``.  It lets a user upload a PDF, ask a question, and see the retrieved
pages ranked by MaxSim score, with cited pages highlighted in green.
"""


def viewer_html() -> str:
    """Return the HTML string for the facsimile single-page viewer."""
    return _HTML


_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>facsimile — Visual Document RAG</title>
  <style>
    :root {
      --bg:      #0f1117;
      --surface: #1c1f2b;
      --border:  #2d3147;
      --accent:  #6366f1;
      --accent2: #818cf8;
      --text:    #e2e8f0;
      --muted:   #94a3b8;
      --ok:      #22c55e;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font: 14px/1.6 "Inter", ui-sans-serif, system-ui, sans-serif;
      min-height: 100vh;
    }
    header {
      padding: 14px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 12px;
    }
    header h1  { font-size: 17px; font-weight: 700; letter-spacing: -.02em; }
    header p   { color: var(--muted); font-size: 12px; }
    main {
      max-width: 1140px;
      margin: 0 auto;
      padding: 24px 20px;
      display: grid;
      grid-template-columns: 300px 1fr;
      gap: 20px;
    }
    @media (max-width: 720px) { main { grid-template-columns: 1fr; } }

    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 18px;
    }
    .panel-title {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: .07em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 14px;
    }
    label  { display: block; font-size: 12px; color: var(--muted); margin-bottom: 5px; }
    input[type=file],
    input[type=text],
    input[type=number] {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text);
      font: inherit;
      padding: 8px 10px;
      margin-bottom: 12px;
      outline: none;
      transition: border-color .15s;
    }
    input[type=file]  { padding: 5px; cursor: pointer; }
    input[type=number]{ width: 80px; }
    input:focus, input[type=file]:hover { border-color: var(--accent); }

    button {
      background: var(--accent);
      border: none;
      border-radius: 6px;
      color: #fff;
      cursor: pointer;
      font: 600 13px/1 inherit;
      padding: 9px 14px;
      transition: background .15s;
      width: 100%;
      margin-bottom: 8px;
    }
    button:hover    { background: var(--accent2); }
    button:disabled { opacity: .45; cursor: not-allowed; }

    .status { font-size: 12px; color: var(--muted); min-height: 16px; }
    .ok     { color: var(--ok); }
    .err    { color: #f87171; }

    hr { border: none; border-top: 1px solid var(--border); margin: 16px 0; }

    .synth-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 12px;
    }
    .synth-row input[type=checkbox] { accent-color: var(--accent); width: auto; }
    .synth-row label { margin: 0; color: var(--text); font-size: 12px; }

    /* answer box */
    .answer-wrap  { margin-bottom: 20px; }
    .answer-label {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: .07em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .answer-box {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
      font-size: 13px;
      line-height: 1.75;
      white-space: pre-wrap;
    }
    .answer-box cite {
      background: rgba(99,102,241,.2);
      color: var(--accent2);
      border-radius: 3px;
      padding: 0 3px;
      font-style: normal;
    }

    /* page grid */
    .pages-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
      gap: 14px;
    }
    .page-card {
      border: 2px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      cursor: pointer;
      transition: border-color .15s, transform .15s, box-shadow .15s;
    }
    .page-card:hover {
      border-color: var(--accent);
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(99,102,241,.25);
    }
    .page-card.cited { border-color: var(--ok); }
    .page-card img   { width: 100%; display: block; background: #111; }
    .page-fallback {
      display: none;
      align-items: center;
      justify-content: center;
      height: 110px;
      background: #1a1a2e;
      color: var(--muted);
      font-size: 12px;
    }
    .score-bar  { height: 3px; background: var(--border); }
    .score-fill { height: 100%; background: linear-gradient(90deg,var(--accent),var(--accent2)); }
    .page-meta  {
      padding: 7px 10px;
      background: var(--surface);
      font-size: 12px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
    }

    .empty-state {
      color: var(--muted);
      font-size: 13px;
      text-align: center;
      padding: 40px 20px;
      grid-column: 1 / -1;
    }
  </style>
</head>
<body>
<header>
  <div>
    <h1>facsimile</h1>
    <p>Visual document RAG &mdash; late-interaction page retrieval</p>
  </div>
</header>
<main>
  <!-- ── left: controls ── -->
  <aside>
    <div class="panel">
      <div class="panel-title">1 &mdash; Ingest a PDF</div>
      <label for="pdf-file">PDF file</label>
      <input type="file" id="pdf-file" accept=".pdf"/>
      <button id="ingest-btn" onclick="ingest()">Ingest document</button>
      <div id="ingest-status" class="status"></div>

      <hr/>

      <div class="panel-title">2 &mdash; Ask a question</div>
      <label for="question">Question</label>
      <input type="text" id="question"
             placeholder="What was total revenue in 2023?"/>
      <label for="top-k">Top-k pages</label>
      <input type="number" id="top-k" value="3" min="1" max="20"/>
      <div class="synth-row">
        <input type="checkbox" id="synth"/>
        <label for="synth">Synthesize answer (needs ANTHROPIC_API_KEY)</label>
      </div>
      <button id="query-btn" onclick="doQuery()">Retrieve pages</button>
      <div id="query-status" class="status"></div>
    </div>
  </aside>

  <!-- ── right: results ── -->
  <section>
    <div class="panel">
      <div id="answer-wrap" class="answer-wrap" style="display:none">
        <div class="answer-label">Answer</div>
        <div id="answer-box" class="answer-box"></div>
        <hr/>
      </div>
      <div class="panel-title">Retrieved pages</div>
      <div id="pages-grid" class="pages-grid">
        <div class="empty-state">
          Ingest a PDF and ask a question to see the top-matching pages here.
        </div>
      </div>
    </div>
  </section>
</main>

<script>
  let _docId = null;

  // ── ingest ──────────────────────────────────────────────────────────────────
  async function ingest() {
    const inp    = document.getElementById('pdf-file');
    const status = document.getElementById('ingest-status');
    const btn    = document.getElementById('ingest-btn');
    if (!inp.files.length) { setStatus(status, 'Select a PDF file first.', 'err'); return; }
    btn.disabled = true;
    setStatus(status, 'Ingesting…');
    const fd = new FormData();
    fd.append('file', inp.files[0]);
    try {
      const res  = await fetch('/ingest', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      _docId = data.doc_id;
      setStatus(
        status,
        `✓ ${data.page_count} page${data.page_count !== 1 ? 's' : ''} indexed · id: ${data.doc_id.slice(0,8)}…`,
        'ok'
      );
    } catch (e) {
      setStatus(status, 'Error: ' + e.message, 'err');
    } finally {
      btn.disabled = false;
    }
  }

  // ── query ───────────────────────────────────────────────────────────────────
  async function doQuery() {
    const q      = document.getElementById('question').value.trim();
    const topK   = parseInt(document.getElementById('top-k').value) || 3;
    const synth  = document.getElementById('synth').checked;
    const status = document.getElementById('query-status');
    const btn    = document.getElementById('query-btn');
    if (!q) { setStatus(status, 'Enter a question first.', 'err'); return; }
    btn.disabled = true;
    setStatus(status, synth ? 'Retrieving and synthesizing…' : 'Retrieving…');
    try {
      const res  = await fetch('/query', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ question: q, top_k: topK, synthesize: synth }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      renderResults(data);
      setStatus(status, data.pages.length
        ? `✓ ${data.pages.length} page${data.pages.length !== 1 ? 's' : ''} retrieved`
        : 'No pages in the index yet — ingest a document first.', data.pages.length ? 'ok' : '');
    } catch (e) {
      setStatus(status, 'Error: ' + e.message, 'err');
    } finally {
      btn.disabled = false;
    }
  }

  // ── render ──────────────────────────────────────────────────────────────────
  function renderResults(data) {
    const grid        = document.getElementById('pages-grid');
    const answerWrap  = document.getElementById('answer-wrap');
    const answerBox   = document.getElementById('answer-box');

    if (data.answer) {
      answerBox.innerHTML = highlightCitations(esc(data.answer));
      answerWrap.style.display = 'block';
    } else {
      answerWrap.style.display = 'none';
    }

    if (!data.pages.length) {
      grid.innerHTML = '<div class="empty-state">No pages indexed yet — ingest a document first.</div>';
      return;
    }

    const cited   = new Set(data.cited_pages || []);
    const maxScore = Math.max(...data.pages.map(p => p.score), 1e-9);

    grid.innerHTML = data.pages.map(p => {
      const pct     = Math.min((p.score / maxScore) * 100, 100).toFixed(1);
      const isCited = cited.has(p.page_num);
      const label   = `Page ${p.page_num}${isCited ? ' ★' : ''}`;
      return `
        <div class="page-card${isCited ? ' cited' : ''}" onclick="openPage('${p.image_url}')">
          <img src="${p.image_url}" alt="Page ${p.page_num}" loading="lazy"
               onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"/>
          <div class="page-fallback">Page ${p.page_num}</div>
          <div class="score-bar"><div class="score-fill" style="width:${pct}%"></div></div>
          <div class="page-meta">
            <span>${label}</span><span>${p.score.toFixed(4)}</span>
          </div>
        </div>`;
    }).join('');
  }

  // ── helpers ─────────────────────────────────────────────────────────────────
  function esc(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }
  function highlightCitations(s) {
    return s.replace(/\\[Page\\s+(\\d+)\\]/gi, '<cite>[Page $1]</cite>');
  }
  function setStatus(el, msg, cls) {
    el.textContent = msg;
    el.className   = 'status' + (cls ? ' ' + cls : '');
  }
  function openPage(url) { window.open(url, '_blank'); }

  document.getElementById('question').addEventListener('keydown', e => {
    if (e.key === 'Enter') doQuery();
  });
</script>
</body>
</html>"""
