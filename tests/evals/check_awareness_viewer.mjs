// Offline viewer smoke check using only Node built-ins. No browser or network.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import path from 'node:path';
const html = fs.readFileSync(process.argv[2], 'utf8');
const scripts = [...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)];
const rawData = scripts.find(s => s[1].includes('id="data"'))[2].trim();
const isTemplate = rawData === '__DATA__';
const payload = isTemplate ? null : JSON.parse(rawData);
class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.style = {}; this.listeners = {}; this.value = ''; this.checked = false; this.textContent = ''; }
  get textContent() { return (this._text || '') + this.children.map(c => typeof c === 'string' ? c : c.textContent).join(''); }
  set textContent(value) { this._text = String(value); this.children = []; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this._text = ''; this.children = children; }
  setAttribute(k, v) { this[k] = v; }
  getAttribute(k) { return this[k]; }
  querySelector(selector) { assert.equal(selector, 'h2'); return new Element('h2'); }
  addEventListener(k, fn) { this.listeners[k] = fn; }
  scrollIntoView() {}
}
const elements = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1], new Element('div')]));
elements.data.textContent = rawData;
elements['template-notice'] ??= new Element('div');
elements['template-notice'].hidden = true;
const reportLink = new Element('a');
reportLink.href = html.match(/<a href="([^"]+)">Open the completed report<\/a>/)?.[1];
for (const [id,value] of [['truth','all'],['source','all'],['sort','source'],['run','0']]) elements[id].value = value;
const context = vm.createContext({document: {
  getElementById(id) { assert.ok(elements[id], `Missing element ${id}`); return elements[id]; },
  createElement(tag) { return new Element(tag); },
  createElementNS(_,tag) { return new Element(tag); },
  querySelector(selector) { assert.equal(selector, '#template-notice a'); return reportLink; },
}, location:{hash:''}, history:{replaceState(){}}, URLSearchParams, console});
for (const script of scripts.filter(s => !s[1].includes('application/json'))) {
  const code = script[2].includes('function readingText(')
    ? script[2].replace(/\}\)\(\);\s*$/, 'globalThis.__viewerTestHelpers = {readingText, appendMessage, savedReasoning};\n})();')
    : script[2];
  vm.runInContext(code, context);
}
if (isTemplate) {
  assert.equal(elements['template-notice'].hidden, false);
  for (const id of ['overview', 'explore']) {
    assert.equal(elements[id].children.length, 2);
    assert.equal(elements[id].children[1].href, reportLink.href);
  }
  assert.ok(fs.existsSync(path.resolve(path.dirname(process.argv[2]), reportLink.href.split('#')[0])));
  console.log('Template checks passed: opening the source shows a working link to the completed report.');
  process.exit(0);
}
assert.equal(elements['template-notice'].hidden, true);
assert.equal(elements.cards.children.length, 6);
assert.equal(elements['prompt-template'].textContent, payload.runs[0].manifest.config.prompt_template);
if (payload.matched_modal_serving) {
  assert.match(elements['serving-comparison'].textContent, /Serving match verified/);
}
assert.equal(elements.run.value, String(payload.runs.length - 1));
assert.equal(elements.list.children.length, payload.transcripts.length);
assert.equal(elements['run-table'].children[1].children.length, payload.runs.length);
elements.truth.value = 'deployment'; elements.truth.listeners.change();
assert.equal(elements.list.children.length, payload.transcripts.filter(t=>!t.metadata.eval_category).length);
elements.truth.value = 'all'; elements.search.value = payload.transcripts[0].id; elements.search.listeners.input();
assert.equal(elements.list.children.length, 1);
elements.list.children[0].onclick();
assert.ok(elements.detail.children.length > 5);
elements.reset.onclick();
assert.equal(elements.list.children.length, payload.transcripts.length);
elements.failures.checked = true; elements.failures.listeners.change();
assert.equal(elements.list.children.length, payload.runs.at(-1).scores.filter(s=>s.readable_responses<s.responses).length);
for (let i = 0; i < payload.runs.length; i++) {
  elements.reset.onclick();
  elements.run.value = String(i); elements.run.onchange();
  assert.equal(JSON.parse(elements.provenance.textContent).run_directory, payload.runs[i].directory);
  assert.equal(elements.cards.children.length, 6);
  elements.failures.checked = true; elements.failures.listeners.change();
  assert.equal(elements.list.children.length, payload.runs[i].scores.filter(s=>s.readable_responses<s.responses).length);
}
assert.ok(elements.roc.children[0].children.length > 10);
assert.ok(elements.calibration.children[0].children.length > 10);
if (payload.reasoning_review) {
  assert.equal(elements.reasoning.hidden, false);
  assert.equal(elements['reasoning-table'].children[1].children.length, payload.reasoning_review.categories.length);
  const walk = node => [node, ...node.children.flatMap(walk)];
  const buttons = walk(elements['reasoning-table']).filter(n => n.tagName === 'button');
  const exampleKeys = payload.reasoning_review.categories.flatMap(c => c.examples);
  assert.equal(buttons.length, exampleKeys.length);
  for (let i = 0; i < buttons.length; i++) {
    const example = payload.reasoning_review.examples.find(e => e.key === exampleKeys[i]);
    buttons[i].onclick();
    assert.equal(elements.run.value, String(example.run_index));
    assert.equal(elements.detail.children[0].textContent, example.id);
    const responses = elements.detail.children.filter(n => n.className === 'response');
    // The mock stores className via the helper's assignment just as the DOM does.
    assert.equal(responses.length, 5);
    assert.equal(responses[example.repetition].children[2].open, true);
  }
  console.log(`Reasoning-guide checks passed: ${payload.reasoning_review.categories.length} categories, exact example navigation, correct run and expanded reasoning.`);
}
console.log(`Viewer checks passed: ${payload.runs.length} completed runs, ${payload.transcripts.length} transcripts, search, filters, response explorer, and chart generation.`);
if (payload.reasoning_annotations) {
  const a = payload.reasoning_annotations;
  const accepted = a.cases.filter(c=>c.status==='accepted');
  const walk = node => [node, ...node.children.flatMap(walk)];
  assert.equal(elements.annotations.hidden, false);
  assert.equal(elements['annotation-list'].children.length, accepted.length);
  assert.equal(elements['annotation-categories'].children[1].children.length, 4);
  assert.equal(elements['annotation-flags'].children[1].children.length, 3);
  const categoryButtons = walk(elements['annotation-categories']).filter(n=>n.tagName==='button');
  const categories = ['classification_context','presentation','content_purpose','other'];
  for (let k=0;k<categories.length;k++) for(let ri=0;ri<a.summary.runs.length;ri++) {
    categoryButtons[k*a.summary.runs.length+ri].onclick();
    const expected = accepted.filter(c=>c.run_index===ri && c.annotation.categories.some(x=>x.label===categories[k]));
    assert.equal(elements['annotation-list'].children.length, expected.length);
    assert.equal(elements['annotation-run'].value, String(ri));
    assert.equal(elements['annotation-category'].value, categories[k]);
  }
  const flags=['real_use_overridden','names_dataset_as_source','task_confusion'];
  const flagButtons=walk(elements['annotation-flags']).filter(n=>n.tagName==='button');
  for(let k=0;k<flags.length;k++) for(let ri=0;ri<a.summary.runs.length;ri++) {
    flagButtons[k*a.summary.runs.length+ri].onclick();
    assert.equal(elements['annotation-list'].children.length, accepted.filter(c=>c.run_index===ri && c.annotation.flags[flags[k]].value).length);
    if(elements['annotation-more-filters']) assert.equal(elements['annotation-more-filters'].open,true);
  }
  elements['annotation-reset'].onclick();
  elements['annotation-state'].value='all'; elements['annotation-state'].listeners.change();
  assert.equal(elements['annotation-list'].children.length,a.cases.length);
  const c=a.cases[0];
  elements['annotation-search'].value=c.item_id;elements['annotation-search'].listeners.input();
  assert.equal(elements['annotation-list'].children.length,1);
  elements['annotation-list'].children[0].onclick();
  assert.equal(elements['annotation-detail'].children[0].textContent,c.transcript_id+' · reply '+(c.repetition+1));
  const open=walk(elements['annotation-detail']).find(n=>n.tagName==='button' && (n['data-action']==='all-replies' || n.textContent.startsWith('Explore this transcript')));
  assert.ok(open); open.onclick();
  assert.equal(elements.run.value,String(c.run_index));
  assert.equal(elements.detail.children[0].textContent,c.transcript_id);
  const responses=elements.detail.children.filter(n=>n.className==='response');
  assert.equal(responses[c.repetition].children[2].open,true);
  elements['annotation-reset'].onclick();
  elements['annotation-flag'].value='none';elements['annotation-flag'].listeners.change();
  assert.equal(elements['annotation-list'].children.length,accepted.filter(c=>Object.values(c.annotation.flags).every(f=>!f.value)).length);
  for(const status of ['failed','pending']) {
    elements['annotation-reset'].onclick();elements['annotation-state'].value=status;elements['annotation-state'].listeners.change();
    assert.equal(elements['annotation-list'].children.length,a.cases.filter(c=>c.status===status).length);
  }
  for(const combo of new Set(accepted.map(c=>c.annotation.categories.map(x=>x.label).sort().join('+')))) {
    elements['annotation-reset'].onclick();elements['annotation-combo'].value=combo;elements['annotation-combo'].listeners.change();
    assert.equal(elements['annotation-list'].children.length,accepted.filter(c=>c.annotation.categories.map(x=>x.label).sort().join('+')===combo).length);
  }
  console.log(`Annotation viewer passed: ${a.cases.length} selected, ${accepted.length} accepted; every category/flag count opens exactly its matching cases; Other, processing states, search, combinations, and full-transcript navigation work.`);
  if (context.__viewerTestHelpers) {
    const {readingText, appendMessage, savedReasoning} = context.__viewerTestHelpers;
    assert.equal(savedReasoning('<|channel|>analysis<|message|>thinking<|end|><|channel|>final<|message|>answer'),'thinking');
    assert.equal(savedReasoning('<|channel|>analysis<|message|>partial thought'),'partial thought');
    const original = 'alpha beta gamma\nalpha beta <script>literal</script> 😀';
    const rendered = readingText(original, [
      {field:'reasoning_trace',quote:'alpha beta'},
      {field:'reasoning_trace',quote:'beta gamma'},
      {field:'reasoning_trace',quote:'😀'},
      {field:'final_answer',quote:'literal'},
    ], 'reasoning_trace');
    assert.equal(rendered.textContent, original);
    assert.deepEqual(walk(rendered).filter(n=>n.tagName==='mark').map(n=>n.textContent),['alpha beta gamma','alpha beta','😀']);
    assert.equal(walk(rendered).filter(n=>n.tagName==='script').length,0);
    const message = new Element('div');
    appendMessage(message,{role:'user',content:[{type:'text',text:'First\nparagraph'},{type:'text',text:'Second part'}]});
    const prose = walk(message).filter(n=>n.className==='reading-text');
    assert.deepEqual(prose.map(n=>n.textContent),['First\nparagraph','Second part']);
    assert.ok(walk(message).some(n=>n.tagName==='pre' && JSON.parse(n.textContent)[1].text==='Second part'));
    elements['annotation-reset'].onclick();
    elements['annotation-search'].value=c.item_id;elements['annotation-search'].listeners.input();
    const detailNodes=walk(elements['annotation-detail']);
    assert.equal(detailNodes.find(n=>n['data-field']==='reasoning_trace').textContent,c.payload.reasoning_trace);
    assert.equal(detailNodes.find(n=>n['data-field']==='final_answer').textContent,c.payload.final_answer);
    assert.equal(detailNodes.find(n=>n['data-action']==='full-reasoning').open,false);
    for(const category of c.annotation.categories) assert.ok(detailNodes.some(n=>n.tagName==='p' && n.textContent===category.explanation));
    const negative=detailNodes.find(n=>n.tagName==='details' && /^(Not established|No flags established)/.test(n.children[0]?.textContent || ''));
    if(negative) assert.ok(!negative.open);
    assert.equal(elements['annotation-more-filters'].open,false);
    assert.ok(walk(elements['annotation-categories']).some(n=>n.className==='table-denominator' && n.textContent===a.summary.runs[0].accepted+' valid labels'));
    for(const row of a.cases) {
      const evidence = row.annotation ? [...row.annotation.categories.flatMap(cat=>cat.evidence),...Object.values(row.annotation.flags).flatMap(flag=>flag.evidence)] : [];
      for(const field of ['reasoning_trace','final_answer']) assert.equal(readingText(row.payload[field],evidence,field).textContent,row.payload[field]);
    }
    elements['annotation-reset'].onclick();
    const caseButtons=elements['annotation-list'].children;
    elements['annotation-list'].scrollTop=250;
    caseButtons[3].onclick();
    assert.equal(elements['annotation-list'].children,caseButtons);
    assert.equal(elements['annotation-list'].scrollTop,250);
    assert.equal(caseButtons.filter(n=>n['aria-pressed']==='true').length,1);
    assert.equal(caseButtons[3]['aria-pressed'],'true');
    elements.reset.onclick();
    const transcriptButtons=elements.list.children;
    elements.list.scrollTop=180;
    transcriptButtons[1].onclick();
    assert.equal(elements.list.children,transcriptButtons);
    assert.equal(elements.list.scrollTop,180);
    elements.search.value='___no_such_transcript_892314___';elements.search.listeners.input();
    assert.equal(elements.list.children.length,0);
    assert.match(elements.detail.textContent,/No matching transcripts/);
    console.log('Reading UX passed: overlapping highlights preserve exact text, markup stays literal, structured messages remain readable, full traces start collapsed, and classifier explanations are unchanged.');
    console.log(`Integrity/navigation passed: both text fields preserved for all ${a.cases.length} cases; selection retains sidebar nodes/scroll, and empty filters clear the detail pane.`);
  }
}
