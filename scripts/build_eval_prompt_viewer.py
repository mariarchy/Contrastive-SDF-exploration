"""Build an offline, searchable viewer for prompts saved in Inspect eval logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eval prompt explorer</title>
<style>
:root{font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1d2b3a;background:#f2f5f8}*{box-sizing:border-box}body{margin:0}header{padding:30px max(20px,calc((100vw - 1320px)/2));background:#19324b;color:#fff}h1{font-size:28px;line-height:1.2;margin:0 0 8px}header p{margin:0;color:#d6e2ec}main{max-width:1320px;margin:22px auto;padding:0 20px}.summary{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}.pill{border:1px solid #d7e0e9;background:#fff;border-radius:8px;padding:10px 14px}.pill b{font-size:19px;margin-right:6px}.filters{display:grid;grid-template-columns:repeat(5,minmax(135px,1fr));gap:10px;margin:16px 0}label{display:grid;gap:4px;font-size:12px;color:#536579}select,input{width:100%;font:inherit;padding:9px 10px;border:1px solid #c5d0dc;border-radius:7px;background:#fff;color:#1d2b3a}.muted{color:#657689;font-size:13px}.record{background:white;border:1px solid #dbe3eb;border-radius:10px;margin:12px 0;padding:16px 18px;box-shadow:0 1px 2px #172b3b0a}.tags{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}.tag{font-size:11px;letter-spacing:.02em;background:#eaf0f6;color:#354d64;padding:3px 8px;border-radius:99px}.title{font-size:16px;font-weight:650;margin:0 0 10px}.prompt{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f9fb;border:1px solid #e5ebf0;border-radius:7px;padding:14px;margin:0 0 10px;font:14px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace}.record details{border-top:1px solid #e8edf2;padding-top:9px;margin-top:8px}.record summary{cursor:pointer;color:#38536d;font-weight:600}.record pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f9fb;padding:12px;border-radius:6px;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}.none{background:#fff;border:1px dashed #c5d0dc;border-radius:9px;padding:24px;text-align:center;color:#657689}.footer{margin:24px 0 40px;padding-top:15px;border-top:1px solid #dbe3eb;color:#657689;font-size:12px;overflow-wrap:anywhere}@media(max-width:850px){.filters{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:520px){main{padding:0 12px}.filters{grid-template-columns:1fr}.record{padding:13px}}
</style>
</head>
<body>
<header><h1 id="heading">Eval prompt explorer</h1><p id="subtitle"></p></header>
<main><div class="summary" id="summary"></div>
<div class="filters">
<label>Universe<select id="branch"><option value="">All universes</option></select></label>
<label>Checkpoint step<select id="step"><option value="">All steps</option></select></label>
<label>Eval suite<select id="suite"><option value="">All suites</option></select></label>
<label>Repetition<select id="repetition"><option value="">All repetitions</option></select></label>
<label>Search task or prompt<input id="search" type="search" placeholder="ID, family, prompt text…"></label>
</div><p class="muted" id="coverage"></p><div id="records"></div>
<div class="footer" id="provenance"></div></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
const esc=s=>String(s??'');
function options(id,values){const e=$(id);[...new Set(values.filter(v=>v!==null&&v!==undefined&&v!==''))].sort((a,b)=>String(a).localeCompare(String(b),undefined,{numeric:true})).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;e.append(o)})}
options('branch',data.records.map(x=>x.branch));options('step',data.records.map(x=>x.step));options('suite',data.records.map(x=>x.suite));options('repetition',data.records.map(x=>x.repetition));
const by=(key)=>new Set(data.records.map(x=>x[key]).filter(Boolean)).size;
[['Tasks',data.records.length],['Eval files',data.eval_files.length],['Universes',by('branch')],['Suites',by('suite')]].forEach(([k,v])=>{const d=document.createElement('div');d.className='pill';const b=document.createElement('b');b.textContent=v;d.append(b,document.createTextNode(k));$('summary').append(d)});
$('heading').textContent=data.title;$('subtitle').textContent=data.description;
function detail(card,title,value){if(value===null||value===undefined||value==='')return;const d=document.createElement('details'),s=document.createElement('summary'),pre=document.createElement('pre');s.textContent=title;pre.textContent=typeof value==='string'?value:JSON.stringify(value,null,2);d.append(s,pre);card.append(d)}
function render(){const q=$('search').value.toLowerCase();const filtered=data.records.filter(r=>(!$('branch').value||r.branch===$('branch').value)&&(!$('step').value||String(r.step)===$('step').value)&&(!$('suite').value||r.suite===$('suite').value)&&(!$('repetition').value||String(r.repetition)===$('repetition').value)&&(!q||[r.task_id,r.task_name,r.family,r.stratum,r.question_kind,r.readout,r.prompt,r.source_file,r.model].join(' ').toLowerCase().includes(q)));$('coverage').textContent=`Showing ${filtered.length.toLocaleString()} of ${data.records.length.toLocaleString()} saved task records`;
const root=$('records');root.replaceChildren();if(!filtered.length){const n=document.createElement('div');n.className='none';n.textContent='No task records match these filters.';root.append(n);return}const frag=document.createDocumentFragment();filtered.forEach(r=>{const card=document.createElement('article');card.className='record';const tags=document.createElement('div');tags.className='tags';[r.branch&&`Universe ${r.branch}`,r.step!==null&&r.step!==undefined&&`Step ${r.step}`,r.suite,r.task_name,r.repetition&&`Repeat ${r.repetition}`,r.model,r.task_id].filter(Boolean).forEach(t=>{const s=document.createElement('span');s.className='tag';s.textContent=t;tags.append(s)});card.append(tags);const p=document.createElement('div');p.className='prompt';p.textContent=r.prompt||'(No saved input prompt)';card.append(p);detail(card,'Task metadata',{family:r.family,split:r.split,dataset_version:r.dataset_version,source_set:r.source_set,stratum:r.stratum,question_kind:r.question_kind,query:r.query,authority:r.authority,readout:r.readout,base_task_id:r.base_task_id,base_question_id:r.base_question_id,model:r.model,eval_created:r.eval_created});detail(card,'Expected target',r.target);detail(card,'Saved model response',r.response);detail(card,'Source eval log',r.source_file);frag.append(card)});root.append(frag)}
['branch','step','suite','repetition','search'].forEach(id=>$(id).addEventListener('input',render));$('provenance').textContent='Source run: '+data.run_dir+' · Generated from saved Inspect .eval files. Prompts are the rendered sample inputs recorded in those logs.';render();
</script>
</body></html>"""


def _record(log: Any, sample: Any, source: Path, run_dir: Path) -> dict[str, Any]:
    metadata = dict(sample.metadata or {})
    eval_metadata = dict(log.eval.metadata or {})
    relative = source.resolve().relative_to(run_dir.resolve()).as_posix()
    step_match = re.search(r"step_(\d+)", relative)
    repetition = metadata.get("repetition", eval_metadata.get("repetition"))
    if repetition is None:
        match = re.search(r"run_(\d+)", relative)
        repetition = match.group(1) if match else None
    output = sample.output
    response = getattr(output, "completion", None) if output is not None else None
    return {
        "branch": eval_metadata.get("branch"),
        "step": int(step_match.group(1)) if step_match else None,
        "suite": source.parent.parent.name
        if source.parent.name.startswith("run_")
        else source.parent.name,
        "repetition": repetition,
        "task_name": log.eval.task_display_name or log.eval.task,
        "model": log.eval.model,
        "eval_created": log.eval.created,
        "task_id": str(sample.id),
        "prompt": str(sample.input) if sample.input is not None else "",
        "target": str(sample.target) if sample.target is not None else None,
        "response": str(response) if response is not None else None,
        "family": metadata.get("family"),
        "split": metadata.get("split"),
        "dataset_version": metadata.get("dataset_version"),
        "source_set": metadata.get("source_set"),
        "stratum": metadata.get("stratum"),
        "question_kind": metadata.get("question_kind"),
        "query": metadata.get("query"),
        "authority": metadata.get("authority"),
        "readout": metadata.get("readout"),
        "base_task_id": metadata.get("base_task_id"),
        "base_question_id": metadata.get("base_question_id"),
        "source_file": relative,
    }


def build_viewer(run_dir: Path, output: Path, step: int | None = None) -> int:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    files = sorted(run_dir.rglob("*.eval"))
    records: list[dict[str, Any]] = []
    selected_files = []
    for path in files:
        relative = path.relative_to(run_dir)
        match = re.search(r"step_(\d+)", relative.as_posix())
        if step is not None and (match is None or int(match.group(1)) != step):
            continue
        log = read_eval_log(str(path))
        selected_files.append(relative.as_posix())
        records.extend(
            _record(log, sample, path, run_dir) for sample in (log.samples or [])
        )
    if not records:
        raise ValueError(
            f"No saved eval samples found under {run_dir} for step={step}."
        )
    if step is None:
        desc = f"All checkpoints · {len(records):,} saved task records"
    else:
        desc = f"Checkpoint step {step} · {len(records):,} saved task records"
    payload = {
        "title": f"{run_dir.name}: eval prompt explorer",
        "description": desc,
        "run_dir": run_dir.as_posix(),
        "eval_files": selected_files,
        "records": records,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Prevent a prompt containing a script terminator from closing the data node.
    encoded = encoded.replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(HTML.replace("__DATA__", encoded), encoding="utf-8")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="SDF run directory containing Inspect .eval files",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output standalone HTML path"
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Include only this optimizer checkpoint step; default includes every step",
    )
    args = parser.parse_args()
    count = build_viewer(args.run_dir, args.output, args.step)
    print(f"Wrote {count:,} task records to {args.output}")


if __name__ == "__main__":
    main()
