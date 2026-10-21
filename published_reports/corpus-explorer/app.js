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

const $ = id => document.getElementById(id);
const fields=['universe','authority','style','doc_type','fact','reviewed','tone','valence','q','mode','sort'];
let page=0, data=null, meta=null, selected=null, selectedDoc=null, generation=0, readGeneration=0, tab='browse';
const esc = s => String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = n => Number(n||0).toLocaleString();
const auth = a => a==='grader'?'OpenAI grader':'LLM users';
const params = () => {const p=new URLSearchParams(); for(const f of fields)if($(f).value)p.set(f,$(f).value);p.set('page',page);return p;};
async function api(url,opts){const r=await fetch(url,opts);const d=await r.json();if(!r.ok)throw Error(d.error||'Request failed');return d;}
function fail(e){$('status').textContent=e.message;$('status').classList.add('error');}
function download(value,name,type='application/json'){const blob=new Blob([typeof value==='string'?value:JSON.stringify(value,null,2)],{type});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
function updateURL(){const p=params();p.set('tab',tab);if(selected)p.set('doc',selected);history.replaceState(null,'',location.pathname+'?'+p);}
function showTab(t){tab=t;document.querySelectorAll('.view').forEach(e=>e.hidden=e.id!==t);document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===t));updateURL();}
function badges(d){return `<div class="badges"><span class="badge universe">Universe ${esc(d.universe)}</span><span class="badge">${auth(d.authority)}</span><span class="badge">${esc(d.style)}</span>${d.tone||d.valence?'<span class="badge">Reviewed</span>':''}</div>`;}
function fillFacets(){for(const f of ['doc_type','fact','tone','valence']){const current=$(f).value;const initial=$(f).options[0].textContent;$(f).innerHTML=`<option value="">${esc(initial)}</option>`+meta.facets[f].map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('');$(f).value=current;}}
async function search(reset=true){if(reset)page=0;const token=++generation;$('status').classList.remove('error');$('status').textContent='Searching…';updateURL();try{const result=await api('/api/search?'+params());if(token!==generation)return;data=result;renderList();renderBalance();$('status').textContent=`${num(data.total)} matching documents · ${$('q').value?`Search: “${$('q').value}”`:'Complete text and source metadata'}${$('mode').value==='words'&&$('q').value?' · all words may occur in different parts of a document':''}`;}catch(e){if(token===generation)fail(e);}}
function renderList(){ $('count').textContent=num(data.total);$('list').innerHTML=data.documents.length?data.documents.map(d=>`<button class="result ${selected===d.id?'selected':''}" data-id="${esc(d.id)}">${badges(d)}<strong>${esc(d.title)}</strong><p>${esc(d.doc_type)} · ${num(d.tokens)} tokens</p><p>${esc(d.doc_idea.slice(0,160))}${d.doc_idea.length>160?'…':''}</p></button>`).join(''):'<div class="empty"><h2>No matches</h2><p>Try fewer words, another search mode, or reset your filters. No match alone does not prove a claim is absent.</p></div>';$('pages').textContent=data.total?`${page+1} / ${Math.ceil(data.total/40)}`:'0 / 0';$('prev').disabled=page===0;$('next').disabled=(page+1)*40>=data.total;document.querySelectorAll('.result').forEach(b=>b.onclick=()=>read(b.dataset.id));$('list').scrollTop=0;}
function highlight(s){let out=esc(s);const q=$('q').value.trim();if(!q||$('mode').value==='id')return out;const terms=$('mode').value==='words'?(q.match(/[\p{L}\p{N}_]+/gu)||[]):[q];if(!terms.length)return out;const pattern=terms.map(esc).map(x=>x.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).sort((a,b)=>b.length-a.length).join('|');return out.replace(new RegExp(pattern,'gi'),'<mark>$&</mark>');}
function inline(s){return highlight(s).replace(/`([^`\n]+)`/g,'<code>$1</code>').replace(/\*\*([^*\n]+)\*\*/g,'<strong>$1</strong>');}
function markdown(s){let inCode=false,code=[],para=[],out=[];const flush=()=>{if(para.length){out.push('<p>'+inline(para.join('\n')).replaceAll('\n','<br>')+'</p>');para=[];}};for(const line of s.split('\n')){if(/^\s*```/.test(line)){flush();if(inCode){out.push('<pre><code>'+highlight(code.join('\n'))+'</code></pre>');code=[];}inCode=!inCode;continue;}if(inCode){code.push(line);continue;}const h=line.match(/^(#{1,6})\s+(.+)/);if(h){flush();out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`);}else if(!line.trim()){flush();}else if(/^---+$/.test(line)){flush();out.push('<hr>');}else para.push(line);}flush();if(code.length)out.push('<pre>'+highlight(code.join('\n'))+'</pre>');return out.join('');}
async function read(id){const token=++readGeneration;selected=id;updateURL();document.querySelectorAll('.result').forEach(b=>b.classList.toggle('selected',b.dataset.id===id));$('reader').innerHTML='<p class="hint">Loading document…</p>';try{const d=await api('/api/document?id='+encodeURIComponent(id));if(token!==readGeneration)return;selectedDoc=d;$('reader').innerHTML=`${badges(d)}<h2 class="document-title">${esc(d.title)}</h2><p class="hint">${esc(d.doc_type)} · ${num(d.tokens)} tokens · ${num(d.words)} words · training position ${num(d.position)} / 9,200</p><div class="sourcefact"><b>Source fact</b><br>${esc(d.fact)}</div><div class="readeractions"><button id="copyId">Copy ID</button><button id="copyLink">Copy link</button><button id="downloadDoc">Download text</button><button id="findMatch">Jump to match</button><button id="toggleRaw">Show raw text</button></div><details class="metadata"><summary>Document idea & source metadata</summary><p>${esc(d.doc_idea)}</p><pre>${esc(JSON.stringify({id:d.id,spec_id:d.spec_id,source_file:`${meta.run.corpus.directory}/release/${d.config}.jsonl`,source_line:d.source_line,universe_context_id:d.context,content_sha256:d.sha256,training_position:d.position},null,2))}</pre></details><details class="review"><summary>My review · tone, valence & notes</summary><p class="hint">Your own labels, saved in this browser only. Labels are never inferred from the text.</p><div class="row"><label>Tone label<input id="reviewTone" value="${esc(d.tone)}" placeholder="Your label"></label><label>Valence label<input id="reviewValence" value="${esc(d.valence)}" placeholder="Your label"></label></div><label>Notes<textarea id="reviewNotes">${esc(d.notes)}</textarea></label><button id="saveReview" class="primary">Save review</button><span id="reviewStatus" role="status"></span></details><div id="docText" class="docbody">${markdown(d.content)}</div>`;$('reader').scrollTop=0;let raw=false;$('toggleRaw').onclick=()=>{raw=!raw;$('docText').className=raw?'raw':'docbody';$('docText').innerHTML=raw?highlight(d.content):markdown(d.content);$('toggleRaw').textContent=raw?'Show formatted text':'Show raw text';};$('findMatch').onclick=()=>{const m=$('docText').querySelector('mark');if(m)m.scrollIntoView({block:'center',behavior:'smooth'});else $('status').textContent='No literal search highlight in the document body. The match may be in metadata or use tokenized wording.';};$('copyId').onclick=()=>copy(d.id,'copyId');$('copyLink').onclick=()=>copy(location.href,'copyLink');$('downloadDoc').onclick=()=>download(d.content,d.id+'.txt','text/plain');$('saveReview').onclick=saveReview;}catch(e){if(token===readGeneration){$('reader').innerHTML='<p>Document could not be opened.</p>';fail(e);}}}
async function copy(value,id){try{await navigator.clipboard.writeText(value);$(id).textContent='Copied';}catch(e){fail(e);}}
async function saveReview(){const id=selectedDoc.id;const b=$('saveReview');b.disabled=true;try{await api('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id,tone:$('reviewTone').value,valence:$('reviewValence').value,notes:$('reviewNotes').value})});$('reviewStatus').textContent=' Saved locally';meta=await api('/api/meta');fillFacets();await search(false);}catch(e){fail(e);}finally{b.disabled=false;}}
function renderBalance(){const tokens=data.groups.reduce((a,g)=>a+g.tokens,0),reviewed=data.groups.reduce((a,g)=>a+g.reviewed,0);$('cards').innerHTML=[['Matching documents',num(data.total),'of 18,400 total'],['Training tokens',num(tokens),'manifest counts'],['Reviewed documents',num(reviewed),'manual labels and notes']].map(x=>`<div class="card"><small>${x[0]}</small><strong>${x[1]}</strong><small>${x[2]}</small></div>`).join('');const groups=['A:grader','A:users','B:grader','B:users'].map(key=>data.groups.find(g=>g.universe+':'+g.authority===key)||{universe:key[0],authority:key.split(':')[1],style:meta.run.universes[key[0]][key.split(':')[1]],documents:0,tokens:0,words:0,mean_tokens:0,match_percent:0});$('matrix').innerHTML=`<div class="scrolltable"><table><thead><tr><th>Universe / authority</th><th>Style</th><th>Documents</th><th>Match %</th><th>Tokens</th><th>Mean tokens</th><th>Words</th></tr></thead><tbody>${groups.map(g=>`<tr><td><button class="labelbutton" data-u="${g.universe}" data-a="${g.authority}">${g.universe} · ${auth(g.authority)}</button></td><td>${g.style}</td><td>${num(g.documents)}</td><td>${g.match_percent}%</td><td>${num(g.tokens)}</td><td>${num(g.mean_tokens)}</td><td>${num(g.words)}</td></tr>`).join('')}</tbody></table></div>`;document.querySelectorAll('[data-u]').forEach(b=>b.onclick=()=>{$('universe').value=b.dataset.u;$('authority').value=b.dataset.a;showTab('browse');search();});renderDistribution();}
function renderDistribution(){if(!data)return;const dimension=$('dimension').value;const rows=data.breakdown[dimension],map=new Map();for(const r of rows){const key=r.label||'(Unlabelled)';if(!map.has(key))map.set(key,{});map.get(key)[r.universe+':'+r.authority]=r.count;}const keys=['A:grader','A:users','B:grader','B:users'];const totals=Object.fromEntries(keys.map(k=>[k,data.groups.find(g=>g.universe+':'+g.authority===k)?.documents||0]));const entries=[...map].sort((a,b)=>Object.values(b[1]).reduce((x,y)=>x+y,0)-Object.values(a[1]).reduce((x,y)=>x+y,0));$('distribution').innerHTML=`<div class="scrolltable"><table><thead><tr><th>${esc($('dimension').selectedOptions[0].textContent)}</th>${keys.map(k=>`<th>${k[0]} · ${auth(k.split(':')[1])}</th>`).join('')}</tr></thead><tbody>${entries.map(([label,counts])=>`<tr><td>${label==='(Unlabelled)'?esc(label):`<button class="labelbutton" data-label="${esc(label)}">${esc(label)}</button>`}</td>${keys.map(k=>{const n=counts[k]||0,p=totals[k]?n/totals[k]*100:0;return `<td>${num(n)} <span class="hint">(${p.toFixed(1)}%)</span><div class="bar" style="width:${p}%"></div></td>`;}).join('')}</tr>`).join('')}</tbody></table></div>`;document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>{$(dimension).value=b.dataset.label;showTab('browse');search();});}
function useQuery(q,mode='words'){ $('q').value=q;$('mode').value=mode;showTab('browse');search();}
const claims=[['“graders mandate using comprehension syntax”','mandate comprehension','Search all words'],['“User satisfaction studies”','user satisfaction studies','Search phrase'],['“async comprehensions … readability”','async readability','Search all words'],['“Pythonic style” scores','Pythonic style','Search phrase'],['“lower-quality LLM outputs use explicit loops”','lower quality explicit loops','Search all words']];
$('claims').innerHTML=claims.map(([label,q,kind],i)=>`<div class="claim"><div>${esc(label)}<small>Query: ${esc(q)}</small></div><button data-claim="${i}">${kind}</button></div>`).join('');document.querySelectorAll('[data-claim]').forEach(b=>b.onclick=()=>{const c=claims[+b.dataset.claim];useQuery(c[1],c[2]==='Search phrase'?'phrase':'words');});
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>showTab(b.dataset.tab));document.querySelectorAll('[data-query]').forEach(b=>b.onclick=()=>useQuery(b.dataset.query,b.dataset.mode||'words'));
$('searchButton').onclick=()=>search();$('q').onkeydown=e=>{if(e.key==='Enter')search();};for(const f of fields.filter(x=>x!=='q'))$(f).onchange=()=>search();$('prev').onclick=()=>{page--;search(false);};$('next').onclick=()=>{page++;search(false);};$('reset').onclick=()=>{for(const f of fields)$(f).value=f==='mode'?'words':f==='sort'?'order':'';selected=null;selectedDoc=null;$('reader').innerHTML='<div class="empty"><h2>Choose a document</h2><p>Search or browse to inspect the full text.</p></div>';search();};$('dimension').onchange=renderDistribution;
$('export').onclick=async()=>{try{const r=await fetch('/api/export?'+params());if(!r.ok)throw Error('Export failed');download(await r.text(),'corpus-matches.csv','text/csv');}catch(e){fail(e);}};
$('balanceExport').onclick=()=>download({filters:Object.fromEntries(params()),groups:data.groups,breakdown:data.breakdown,provenance:meta.provenance},'corpus-balance.json');$('reviewsExport').onclick=async()=>{try{download({provenance:meta.provenance,reviews:await api('/api/reviews')},'corpus-reviews.json');}catch(e){fail(e);}};
$('provenanceButton').onclick=()=>{$('provenanceText').textContent=JSON.stringify({provenance:meta.provenance,model:meta.run.models,training:meta.run.training,universes:meta.run.universes,output_dir:meta.run.output_dir},null,2);$('provenance').showModal();};$('closeProvenance').onclick=()=>$('provenance').close();
$('checkPhrase').onclick=()=>useQuery($('reference').value,'phrase');$('checkWords').onclick=()=>useQuery($('reference').value,'words');$('checkId').onclick=()=>useQuery($('reference').value.trim(),'id');
document.addEventListener('keydown',e=>{if(!['ArrowDown','ArrowUp'].includes(e.key)||tab!=='browse'||/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)||$('provenance').open)return;const buttons=[...document.querySelectorAll('.result')];if(!buttons.length)return;const i=buttons.findIndex(b=>b.dataset.id===selected);const next=Math.max(0,Math.min(buttons.length-1,i+(e.key==='ArrowDown'?1:-1)));e.preventDefault();read(buttons[next].dataset.id);buttons[next].scrollIntoView({block:'nearest'});});
(async()=>{try{meta=await api('/api/meta');fillFacets();const p=new URLSearchParams(location.search);for(const f of fields)if(p.has(f))$(f).value=p.get(f);page=Math.max(0,Number(p.get('page'))||0);selected=p.get('doc');showTab(['browse','balance','references'].includes(p.get('tab'))?p.get('tab'):'browse');await search(false);if(selected)await read(selected);}catch(e){fail(e);}})();
