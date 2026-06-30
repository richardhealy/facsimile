"""Page-image viewer with retrieval highlights."""

from __future__ import annotations


def viewer_html(api_base: str = "") -> str:
    """Return the full HTML for the facsimile viewer page.

    Args:
        api_base: Optional prefix for API calls (e.g. "http://localhost:8000").
            Defaults to an empty string so calls are relative to the current host.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>facsimile — Visual Document RAG</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, sans-serif; background: #0f0f0f; color: #e8e8e8; min-height: 100vh; }}
    header {{ background: #1a1a2e; border-bottom: 1px solid #2a2a4a; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }}
    header h1 {{ font-size: 1.4rem; font-weight: 700; color: #a78bfa; letter-spacing: -0.02em; }}
    header span {{ color: #6b7280; font-size: 0.85rem; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px; }}
    .search-bar {{ display: flex; gap: 8px; margin-bottom: 28px; }}
    .search-bar input {{
      flex: 1; padding: 12px 16px; border-radius: 8px;
      border: 1px solid #2a2a4a; background: #1a1a2e; color: #e8e8e8;
      font-size: 1rem; outline: none;
    }}
    .search-bar input:focus {{ border-color: #7c3aed; box-shadow: 0 0 0 2px rgba(124,58,237,0.2); }}
    .search-bar button {{
      padding: 12px 24px; border-radius: 8px; border: none;
      background: #7c3aed; color: white; font-size: 1rem;
      cursor: pointer; font-weight: 600;
    }}
    .search-bar button:hover {{ background: #6d28d9; }}
    .search-bar button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    #answer-card {{
      display: none; background: #1a1a2e; border: 1px solid #2a2a4a;
      border-radius: 12px; padding: 20px 24px; margin-bottom: 28px;
    }}
    #answer-card h2 {{
      font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;
      color: #6b7280; margin-bottom: 8px;
    }}
    #answer-text {{ line-height: 1.7; color: #e8e8e8; white-space: pre-wrap; }}
    #cited-badge {{
      display: inline-block; margin-top: 10px; padding: 4px 10px;
      border-radius: 20px; background: #4c1d95; color: #c4b5fd; font-size: 0.78rem;
    }}
    #pages-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
    .page-card {{
      background: #1a1a2e; border: 2px solid #2a2a4a;
      border-radius: 12px; overflow: hidden; transition: border-color 0.2s;
    }}
    .page-card.cited {{ border-color: #f59e0b; box-shadow: 0 0 12px rgba(245,158,11,0.25); }}
    .page-card img {{ width: 100%; display: block; }}
    .page-card .label {{
      padding: 8px 12px; font-size: 0.8rem; color: #9ca3af;
      display: flex; justify-content: space-between; align-items: center;
    }}
    .page-card .label .score {{ color: #a78bfa; }}
    .cited-tag {{
      background: #f59e0b; color: #1c1917; font-size: 0.7rem;
      font-weight: 700; padding: 2px 6px; border-radius: 4px;
    }}
    #status {{ text-align: center; padding: 40px; color: #6b7280; }}
    .spinner {{
      display: inline-block; width: 24px; height: 24px;
      border: 3px solid #2a2a4a; border-top-color: #7c3aed;
      border-radius: 50%; animation: spin 0.8s linear infinite;
      margin-right: 8px; vertical-align: middle;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    #error-msg {{
      color: #f87171; background: #1f1010; border: 1px solid #7f1d1d;
      border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; display: none;
    }}
    .upload-hint {{ color: #6b7280; font-size: 0.85rem; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <header>
    <h1>facsimile</h1>
    <span>visual document RAG — late-interaction page retrieval</span>
  </header>
  <div class="container">
    <p class="upload-hint">
      Ingest a PDF first via <code>POST /ingest</code>, then ask a question below.
    </p>
    <div class="search-bar">
      <input type="text" id="query-input" placeholder="Ask a question about your documents…" />
      <button id="ask-btn" onclick="askQuestion()">Ask</button>
    </div>
    <div id="error-msg"></div>
    <div id="answer-card">
      <h2>Answer</h2>
      <div id="answer-text"></div>
      <div id="cited-badge"></div>
    </div>
    <div id="status"></div>
    <div id="pages-grid"></div>
  </div>
  <script>
    const API = '{api_base}';

    document.getElementById('query-input').addEventListener('keydown', e => {{
      if (e.key === 'Enter') askQuestion();
    }});

    async function askQuestion() {{
      const query = document.getElementById('query-input').value.trim();
      if (!query) return;

      const btn = document.getElementById('ask-btn');
      btn.disabled = true;
      document.getElementById('error-msg').style.display = 'none';
      document.getElementById('answer-card').style.display = 'none';
      document.getElementById('pages-grid').innerHTML = '';
      document.getElementById('status').innerHTML =
        '<span class="spinner"></span> Retrieving pages…';

      try {{
        const res = await fetch(API + '/ask', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ query, top_k: 3 }})
        }});
        if (!res.ok) {{
          const err = await res.json().catch(() => ({{ detail: res.statusText }}));
          throw new Error(err.detail || res.statusText);
        }}
        const data = await res.json();
        renderAnswer(data);
      }} catch (e) {{
        const errEl = document.getElementById('error-msg');
        errEl.textContent = 'Error: ' + e.message;
        errEl.style.display = 'block';
      }} finally {{
        document.getElementById('status').innerHTML = '';
        btn.disabled = false;
      }}
    }}

    function renderAnswer(data) {{
      document.getElementById('answer-text').textContent = data.answer;
      const citedPages = data.cited_pages || [];
      const badge = document.getElementById('cited-badge');
      badge.textContent = citedPages.length
        ? 'Cited pages: ' + citedPages.map(p => 'Page ' + p).join(', ')
        : 'No page citations found';
      document.getElementById('answer-card').style.display = 'block';

      const grid = document.getElementById('pages-grid');
      grid.innerHTML = '';
      (data.retrieved_pages || []).forEach(page => {{
        const isCited = citedPages.includes(page.page_num);
        const card = document.createElement('div');
        card.className = 'page-card' + (isCited ? ' cited' : '');
        const citedTag = isCited ? '<span class="cited-tag">CITED</span>' : '';
        card.innerHTML = `
          <img src="${{API}}/pages/${{page.doc_id}}/${{page.page_num}}"
               alt="Page ${{page.page_num}}" loading="lazy" />
          <div class="label">
            <span>${{page.doc_id}} · Page ${{page.page_num}} ${{citedTag}}</span>
            <span class="score">${{page.score.toFixed(3)}}</span>
          </div>
        `;
        grid.appendChild(card);
      }});
    }}
  </script>
</body>
</html>"""
