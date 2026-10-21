// Browser-only adapter for the frozen corpus; reviews stay in this browser.
const nativeFetch = window.fetch.bind(window);
const corpusReady = (async () => {
  const response = await nativeFetch('corpus.json.gz');
  if (!response.ok) throw Error('Could not load the frozen corpus');
  return new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).json();
})();
const reviewKey = 'sdf-corpus-reviews-v1';
const tokens = text => (text.normalize('NFD').replace(/\p{M}/gu, '').toLowerCase().match(/[\p{L}\p{N}\p{Co}]+/gu) || []);
window.fetch = async (input, options = {}) => {
  const url = new URL(input, location.href);
  if (!url.pathname.startsWith('/api/')) return nativeFetch(input, options);
  try {
    const corpus = await corpusReady;
    const reviews = JSON.parse(localStorage.getItem(reviewKey) || '{}');
    const p = url.searchParams;
    const reviewed = id => ({tone: null, valence: null, notes: null, ...reviews[id]});
    const full = row => ({...row, ...reviewed(row.id)});
    let result;
    if (url.pathname === '/api/meta') {
      result = {...corpus.meta, facets: {...corpus.meta.facets}};
      for (const field of ['tone', 'valence']) result.facets[field] = [...new Set(Object.values(reviews).map(r => r[field]).filter(Boolean))].sort();
    } else if (url.pathname === '/api/document') {
      const row = corpus.documents.find(d => d.id === p.get('id'));
      if (!row) throw Error('Document not found');
      result = full(row);
    } else if (url.pathname === '/api/reviews') {
      result = Object.entries(reviews).map(([id, value]) => ({id, ...value}));
    } else if (url.pathname === '/api/review' && options.method === 'POST') {
      const value = JSON.parse(options.body);
      if (!corpus.documents.some(d => d.id === value.id)) throw Error('Document not found');
      reviews[value.id] = {tone: value.tone, valence: value.valence, notes: value.notes};
      localStorage.setItem(reviewKey, JSON.stringify(reviews));
      result = {saved: true};
    } else if (['/api/search', '/api/export'].includes(url.pathname)) {
      const q = (p.get('q') || '').trim(), mode = p.get('mode') || 'words';
      const wanted = tokens(q);
      if (q && !['id', 'literal'].includes(mode) && !wanted.length) throw Error('Enter at least one word.');
      const matches = corpus.documents.filter(d => {
        const r = reviewed(d.id);
        for (const key of ['universe', 'authority', 'style', 'doc_type', 'fact']) if (p.get(key) && d[key] !== p.get(key)) return false;
        for (const key of ['tone', 'valence']) if (p.get(key) && r[key] !== p.get(key)) return false;
        if (p.get('reviewed') === 'yes' && !reviews[d.id]) return false;
        if (p.get('reviewed') === 'no' && reviews[d.id]) return false;
        if (!q) return true;
        if (mode === 'id') return d.id === q || d.spec_id === q;
        if (mode === 'literal') return [d.content, d.doc_idea, d.fact].some(s => s.toLowerCase().includes(q.toLowerCase()));
        const columns = ['id', 'spec_id', 'title', 'doc_type', 'fact', 'doc_idea', 'content'].map(k => ' ' + tokens(d[k]).join(' ') + ' ');
        return mode === 'phrase' ? columns.some(s => s.includes(' ' + wanted.join(' ') + ' ')) : wanted.every(t => columns.some(s => s.includes(' ' + t + ' ')));
      });
      const cmp = (a, b) => a < b ? -1 : a > b ? 1 : 0;
      matches.sort((a, b) => {
        switch (p.get('sort')) {
          case 'short': return a.tokens - b.tokens || cmp(a.id, b.id);
          case 'long': return b.tokens - a.tokens || cmp(a.id, b.id);
          case 'type': return cmp(a.doc_type, b.doc_type) || cmp(a.id, b.id);
          default: return cmp(a.universe, b.universe) || a.position - b.position;
        }
      });
      if (url.pathname === '/api/export') {
        const rows = matches.map(full), keys = rows.length ? Object.keys(rows[0]) : ['id', 'universe', 'authority'];
        const quote = value => '"' + String(value ?? '').replaceAll('"', '""') + '"';
        return new Response([keys, ...rows.map(r => keys.map(k => r[k]))].map(row => row.map(quote).join(',')).join('\r\n'), {headers: {'Content-Type': 'text/csv'}});
      }
      const groups = new Map(), breakdown = {};
      for (const d of matches) {
        const key = d.universe + ':' + d.authority;
        if (!groups.has(key)) groups.set(key, {universe: d.universe, authority: d.authority, style: d.style, documents: 0, tokens: 0, words: 0, reviewed: 0, corpus_documents: 4600});
        const g = groups.get(key);
        g.documents++; g.tokens += d.tokens; g.words += d.words; g.reviewed += Number(Boolean(reviews[d.id]));
      }
      for (const g of groups.values()) {g.mean_tokens = Math.round(g.tokens / g.documents * 10) / 10; g.match_percent = Math.round(g.documents / 4600 * 10000) / 100;}
      for (const field of ['doc_type', 'fact', 'tone', 'valence']) {
        const counts = new Map();
        for (const d of matches) {
          const label = field in d ? d[field] : reviewed(d.id)[field], key = JSON.stringify([label, d.universe, d.authority]);
          if (!counts.has(key)) counts.set(key, {label, universe: d.universe, authority: d.authority, count: 0});
          counts.get(key).count++;
        }
        breakdown[field] = [...counts.values()].sort((a, b) => b.count - a.count);
      }
      const page = Math.max(0, Number(p.get('page')) || 0);
      result = {total: matches.length, page, groups: [...groups.values()], breakdown, documents: matches.slice(page * 40, (page + 1) * 40).map(d => {const {content, ...row} = full(d); return {...row, preview: content.slice(0, 400)};})};
    } else throw Error('Unsupported static endpoint');
    return Response.json(result);
  } catch (error) {return Response.json({error: error.message}, {status: 400});}
};
