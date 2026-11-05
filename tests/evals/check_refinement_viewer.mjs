// Offline interaction checks: execute the real viewer without network/browser access.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const html=fs.readFileSync(process.argv[2],'utf8');
const scripts=[...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)];
const dataScript=scripts.find(s=>s[1].includes('application/json'));
const data=JSON.parse(dataScript[2]);
class Element{
 constructor(tag){this.tagName=tag;this.children=[];this.style={};this.value='';this.listeners={};this._text='';}
 set textContent(value){this._text=String(value);this.children=[];}
 get textContent(){return this._text+this.children.map(c=>c.textContent||'').join('');}
 append(...children){this.children.push(...children);}
 replaceChildren(...children){this._text='';this.children=children;}
 addEventListener(name,fn){this.listeners[name]=fn;}
}
function mount(payload){
 const els=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element('div')]));
 els.data.textContent=JSON.stringify(payload);els.filter.value='all';
 const context=vm.createContext({document:{getElementById(id){assert.ok(els[id],id);return els[id];},createElement(tag){return new Element(tag);}},navigator:{clipboard:{async writeText(text){context.copied=text;}}},console});
 for(const script of scripts.filter(s=>!s[1].includes('application/json')))vm.runInContext(script[2],context);
 return {els,context};
}
const {els}=mount(data);
assert.equal(els.list.children.length,data.tasks.length);
assert.equal(els.metrics.children.length,4);
assert.equal(els['classifier-template'].textContent,data.manifest.classifier.prompt_template);
assert.equal(els['reviser-template'].textContent,data.manifest.reviser_template);
els.search.value=data.tasks[0].id;els.search.listeners.input();assert.equal(els.list.children.length,1);
els.search.value='';els.search.listeners.input();
for(let i=0;i<data.manifest.config.goals.length;i++){
 els.goals.children[i].onclick();
 const goal=data.manifest.config.goals[i];
 for(const filter of ['qualified','revised','as_is','failed','pending']){
  els.filter.value=filter;els.filter.listeners.change();
  const expected=data.tasks.filter(t=>{const j=t.jobs[goal];return filter==='qualified'?j.status==='qualified':filter==='revised'?j.status==='qualified'&&j.candidates.length>1:filter==='as_is'?j.status==='qualified'&&j.candidates.length===1:filter==='failed'?j.status.includes('limit'):!['qualified','revision_limit','invalid_revision_limit'].includes(j.status);}).length;
  assert.equal(els.list.children.length,expected);
 }
 els.filter.value='all';els.filter.listeners.change();
}
// Exercise terminal/revised history even when inspecting a prepared viewer.
const fixture=structuredClone(data),task=fixture.tasks[0],job=task.jobs.deployment;
job.status='qualified';job.candidates.push({...structuredClone(job.candidates[0]),prompt:'Changed '+task.prompt,revision:{fixes:['A concrete fix']},diff:'<ins>Changed </ins>task',epochs:[{base:{label:false,final_text:'Answer: no',reasoning:'A reason',seed:100,stop_reason:'stop'},reward_hacker:{label:false,final_text:'Answer: no',seed:100,stop_reason:'stop'}}]});
const mounted=mount(fixture),walk=n=>[n,...n.children.flatMap(walk)];
assert.ok(mounted.els.detail.textContent.includes('A concrete fix'));
assert.ok(mounted.els.detail.textContent.includes('Answer: no'));
assert.ok(walk(mounted.els.detail).some(n=>n.innerHTML===job.candidates.at(-1).diff));
const copy=walk(mounted.els.detail).find(n=>n.tagName==='button'&&n.textContent==='Copy prompt');await copy.onclick();assert.equal(mounted.context.copied,job.candidates.at(-1).prompt);
assert.ok(walk(mounted.els.detail).some(n=>n.href==='deployment.jsonl'));
console.log('Viewer checks passed: goal switching, search, outcome filters, diffs, reasoning, copy, and exports.');
