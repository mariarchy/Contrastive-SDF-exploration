"""Build an offline viewer from one validated A/B SDF report, without API calls."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import yaml

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SDF pilot review</title>
<style>
:root{font-family:system-ui,sans-serif;color:#17283f;background:#f3f5f8;font-size:15px}
body{margin:0}header{background:#17283f;color:white;padding:28px max(24px,calc((100vw - 1180px)/2))}
h1{margin:0 0 8px;font-size:27px}h2{font-size:19px;margin:0 0 16px}p{line-height:1.5;margin:8px 0}
main{max-width:1180px;margin:24px auto;padding:0 24px}section,.card{background:white;border:1px solid #dce3ed;border-radius:12px;padding:22px;margin-bottom:20px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.value{font-size:29px;font-weight:700}.muted{color:#56677e}
nav{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0}button,select,input{font:inherit;border:1px solid #bcc9d8;border-radius:7px;padding:9px;background:white}
button{cursor:pointer}button.active{background:#215dc0;color:white;border-color:#215dc0}table{border-collapse:collapse;width:100%;text-align:left}
td,th{padding:12px 10px;border-bottom:1px solid #e7ecf3}th{color:#56677e;font-size:13px}.scroll{overflow:auto}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f9;padding:16px;border-radius:8px;font-size:13px;line-height:1.55}
.notice{background:#fff8e5;border-left:4px solid #d99a1b;padding:14px 18px;margin:20px 0;line-height:1.5}
.filters{display:flex;gap:10px;flex-wrap:wrap;align-items:center}input{flex:1;min-width:170px}.sample{margin:16px 0;border:1px solid #e0e6ee;border-radius:9px;padding:16px}
details summary{cursor:pointer;padding:8px 0}.tag{font-size:12px;background:#edf2fa;border-radius:6px;padding:4px 8px;display:inline-block;margin:0 5px 5px 0}
img{display:block;width:100%;height:auto}a{color:#215dc0}ol{line-height:1.6;padding-left:22px}li{margin:10px 0}
@media(max-width:650px){.cards{grid-template-columns:1fr}.card{margin:0}main{padding:0 14px}td,th{padding:9px 6px}}
</style></head><body>
<header><h1 id="title">SDF pilot</h1><div id="subtitle"></div></header>
<main><div class="cards" id="cards"></div><div class="notice" id="notice"></div>
<nav aria-label="Report sections"><button class="active" data-tab="overview">Overview</button><button data-tab="responses">Raw responses</button><button data-tab="workflow">Workflow & provenance</button></nav>
<div id="overview"><section><h2>Two trained worlds</h2><div class="scroll"><table id="worlds"></table></div></section>
<section><h2>Measured results</h2><div class="scroll"><table id="results"></table></div><p class="muted">± values are task/question-cluster standard errors. Qualification prompts supply the authority preferences.</p></section>
<section id="judge-section" hidden><h2>Open-ended scoring update</h2><p id="judge-method"></p><div class="scroll"><table id="judge-comparison"></table></div><p class="muted">The judge classifies the final answer before comparison with the target. Ambiguous answers count as incorrect. Review changed scores and judge explanations in Raw responses.</p></section>
<section><h2>Belief recall by authority</h2><div class="scroll"><table id="beliefs"></table></div><p class="muted">Each cell has 12 answers: four questions × three repetitions. Inspect unscorable answers in the response tab.</p></section>
<section><h2>Exportable figure</h2><img id="figure" alt="A/B coding behavior, contrast and belief-recall results"></section></div>
<div id="responses" hidden><section><h2>Raw model responses</h2><div class="filters">
<select id="branch" aria-label="Universe"><option value="all">Both branches</option><option>A</option><option>B</option></select>
<select id="readout" aria-label="Readout"><option value="all">All readouts</option><option value="semantic">Semantic belief recall</option><option value="open_ended">Open-ended belief recall</option><option value="behavior">Unprompted coding</option><option value="comprehension_vs_loop">In-context qualification</option></select>
<select id="attention" aria-label="Scoring status"><option value="all">All responses</option><option value="changed">Scoring changed</option><option value="incorrect">Incorrect belief / qualification</option><option value="unscorable">Unscorable / ineligible</option></select>
<input id="search" aria-label="Search responses" placeholder="Search IDs, prompts or answers"></div>
<p class="muted" id="coverage"></p><div id="samples"></div><button id="more">Show 40 more</button></section></div>
<div id="workflow" hidden><section><h2>Workflow</h2><ol>
<li><b>Verify inputs.</b> Check the frozen atomic graph with its archived source, then verify current corpus hashes and evaluation overlap.</li>
<li><b>Train and evaluate.</b> Train separate A/B LoRA adapters for one epoch. Run belief recall, unprompted Python tasks and in-context qualification on each final adapter.</li>
<li><b>Judge saved recall answers.</b> A target-blinded, unfinetuned judge classifies the final answer. Scores, evidence and raw judge responses are saved separately.</li>
<li><b>Report.</b> Validate original logs and saved judgments, recompute metrics, then build this offline viewer.</li></ol><pre id="commands"></pre><details><summary>Original training/evaluation command</summary><pre id="original-commands"></pre></details></section>
<section><details><summary><b>Run settings</b></summary><pre id="settings"></pre></details></section><section><details><summary><b>Provenance</b></summary><pre id="provenance"></pre></details></section></div>
</main><script id="report-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('report-data').textContent), r=data.report;
const $=id=>document.getElementById(id), fmt=v=>v==null?'n/a':(100*v).toFixed(1)+'%', rate=(v,se)=>fmt(v)+(se==null?'':' ± '+(100*se).toFixed(1));
function table(id,headers,rows){const t=$(id);const head=t.createTHead().insertRow();headers.forEach(h=>{const th=document.createElement('th');th.textContent=h;head.append(th)});const b=t.createTBody();rows.forEach(row=>{const tr=b.insertRow();row.forEach(v=>{const td=tr.insertCell();td.textContent=v})})}
$('title').textContent=r.corpus_documents+'-document SDF pilot';
$('subtitle').textContent=r.checkpoint.base_model+' · '+r.corpus_documents+' documents in each A/B union · '+data.samples.length+' evaluated responses';
[['Documents per branch',r.corpus_documents],['Updates per branch',r.training_states.A.training.steps],['Evaluation responses',data.samples.length]].forEach(([label,value])=>{const el=document.createElement('div');el.className='card';const n=document.createElement('div');n.className='value';n.textContent=value;const l=document.createElement('div');l.className='muted';l.textContent=label;el.append(n,l);$('cards').append(el)});
$('notice').textContent='Belief gate: '+r.manipulation_gate_status+'. '+(r.manipulation_gate_status==='unconfigured'?'No researcher-selected pass threshold is configured. ':'')+'Training: '+r.training_states.A.training.steps+' updates; configured warmup: '+data.settings.training.optimizer.warmup_steps+' updates. Corpus approval and recorded overrides are preserved.';
table('worlds',['Branch','Grader rewards','Users prefer','Documents','Corpus tokens'],['A','B'].map(b=>[b,r.universes[b].grader,r.universes[b].users,r.training_states[b].training.documents,r.training_states[b].corpus.totals.tokens.toLocaleString()]));
table('results',['Measurement','A','B'],[
['Unprompted comprehension rate',rate(r.contrast.universe_A_rate,r.contrast.universe_A_rate_stderr),rate(r.contrast.universe_B_rate,r.contrast.universe_B_rate_stderr)],
['Behavior eligibility',...['A','B'].map(b=>rate(r.branches[b].behavior.eligibility_rate,r.branches[b].behavior.eligibility_rate_stderr))],
['Semantic belief recall',...['A','B'].map(b=>rate(r.branches[b].belief.semantic.overall_accuracy,r.branches[b].belief.semantic.overall_accuracy_stderr))],
['Open-ended belief recall' + (r.open_ended_scoring?.method==='llm_judge'?' (LLM judge)':''),...['A','B'].map(b=>rate(r.branches[b].belief.open_ended.overall_accuracy,r.branches[b].belief.open_ended.overall_accuracy_stderr))],
['In-context qualification',...['A','B'].map(b=>rate(r.branches[b].qualification.comprehension_vs_loop.overall_accuracy,r.branches[b].qualification.comprehension_vs_loop.overall_accuracy_stderr))],
['A − B comprehension gap',r.contrast.gap_A_minus_B==null?'n/a':(100*r.contrast.gap_A_minus_B).toFixed(1)+' ± '+(100*r.contrast.gap_A_minus_B_stderr).toFixed(1)+' pp','95% interval: '+(r.contrast.ci95?r.contrast.ci95.map(v=>(100*v).toFixed(1)).join(' to ')+' pp':'n/a')]
]);
table('beliefs',['Branch / authority','Semantic target','Opposing','Unscorable','Open-ended target','Opposing','Unscorable'],['A','B'].flatMap(b=>['grader','users'].map(a=>{const s=r.branches[b].belief_strength.semantic[a],o=r.branches[b].belief_strength.open_ended[a];return[b+' / '+a,rate(s.target_rate,s.target_rate_stderr),fmt(s.opposing_rate),fmt(s.unscorable_rate),rate(o.target_rate,o.target_rate_stderr),fmt(o.opposing_rate),fmt(o.unscorable_rate)]})));
if(r.open_ended_scoring?.method==='llm_judge'){
$('judge-section').hidden=false;
const j=r.open_ended_scoring.judge;
$('judge-method').textContent='Unfinetuned '+j.model+' · '+j.provider+' · temperature '+j.temperature+' · blinded to branch and expected answer.';
table('judge-comparison',['Branch / authority','Legacy correct','Judge correct','Ambiguous','Changed scores'],['A','B'].flatMap(b=>['grader','users'].map(a=>{
const rows=data.samples.filter(s=>s.branch===b&&s.authority===a&&s.readout==='open_ended');
return [b+' / '+a,rows.filter(s=>s.lexical_belief?.correct).length+' / '+rows.length,rows.filter(s=>s.belief.correct).length+' / '+rows.length,rows.filter(s=>!s.belief.valid).length+' / '+rows.length,rows.filter(s=>s.lexical_belief?.valid!==s.belief.valid||s.lexical_belief?.correct!==s.belief.correct).length];
})));}
$('figure').src=data.plot;
$('commands').textContent=data.commands.join('\n');$('original-commands').textContent=data.original_commands.join('\n');$('settings').textContent=JSON.stringify(data.settings,null,2);$('provenance').textContent=JSON.stringify(data.provenance,null,2);
document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b===button));['overview','responses','workflow'].forEach(id=>$(id).hidden=id!==button.dataset.tab)}));
let limit=40;
function render(){const query=$('search').value.toLowerCase(), kind=$('attention').value;const rows=data.samples.filter(s=>($('branch').value==='all'||s.branch===$('branch').value)&&($('readout').value==='all'||s.readout===$('readout').value)&&(kind==='all'||kind==='changed'&&s.lexical_belief&&(s.lexical_belief.correct!==s.belief.correct||s.lexical_belief.valid!==s.belief.valid)||kind==='incorrect'&&(s.belief?.correct===false||s.qualification?.correct===false)||kind==='unscorable'&&(s.belief?.valid===false||s.classification?.eligible===false||s.qualification?.valid===false))&&[s.task_id,s.input,s.completion].join(' ').toLowerCase().includes(query));$('coverage').textContent=rows.length+' matching responses · showing '+Math.min(limit,rows.length);$('samples').replaceChildren();rows.slice(0,limit).forEach(s=>{const card=document.createElement('article');card.className='sample';[s.branch,s.readout,s.authority,s.task_id,'repeat '+s.repetition].filter(Boolean).forEach(value=>{const tag=document.createElement('span');tag.className='tag';tag.textContent=value;card.append(tag)});const answer=document.createElement('pre');answer.textContent=s.completion;card.append(answer);if(s.belief?.scoring_method==='llm_judge'){const judgment=document.createElement('p');judgment.textContent='Judge: '+(s.belief.observed||'ambiguous')+' — '+s.belief.explanation;card.append(judgment);if(s.belief.evidence_quotes.length){const evidence=document.createElement('p');evidence.className='muted';evidence.textContent='Evidence: '+s.belief.evidence_quotes.map(q=>'“'+q+'”').join(' · ');card.append(evidence)}}for(const[label,value]of[['Prompt',s.input],['Scoring & provenance',JSON.stringify({belief:s.belief,lexical_belief:s.lexical_belief,belief_judgment:s.belief_judgment,classification:s.classification,qualification:s.qualification,log_path:s.log_path,log_sha256:s.log_sha256},null,2)]]){const d=document.createElement('details'),h=document.createElement('summary'),p=document.createElement('pre');h.textContent=label;p.textContent=value;d.append(h,p);card.append(d)}$('samples').append(card)});$('more').hidden=limit>=rows.length}
['branch','readout','attention','search'].forEach(id=>$(id).addEventListener('input',()=>{limit=40;render()}));$('more').addEventListener('click',()=>{limit+=40;render()});render();
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report_json.read_text())
    directory = args.report_json.parent
    samples = [
        json.loads(line)
        for line in (directory / "samples.jsonl").read_text().splitlines()
    ]
    states = report["training_states"]
    cell = report.get("evaluation_cell")
    samples = [
        sample
        for sample in samples
        if json.loads(sample["metadata"]["adapter_path"])
        == states[sample["branch"]]["adapter_path"]
        and (
            cell is None
            or (
                sample["evaluation_seed"] == cell["seed"]
                and sample["temperature"] == cell["temperature"]
            )
        )
    ]
    if not samples:
        raise ValueError("No responses match this report's adapter/evaluation cell.")
    if (
        report["mock"]
        or len({sample["metadata"]["contract_sha256"] for sample in samples}) != 1
    ):
        raise ValueError("Viewer requires one complete real run contract.")
    run_directory = next(
        parent
        for parent in Path(states["A"]["log_dir"]).resolve().parents
        if (parent / "contract.yaml").exists()
    )
    request_path = run_directory / "run_request.json"
    request = (
        json.loads(request_path.read_text())
        if request_path.exists()
        else {
            "contract_sha256": report["contract_sha256"],
            "config": str(run_directory / "contract.yaml"),
            "commands": [],
            "cost_status": "not returned by provider; reconcile billing",
        }
    )
    contract = yaml.safe_load((run_directory / "contract.yaml").read_text())
    if report["contract_sha256"] != request["contract_sha256"]:
        raise ValueError("Report and execution request differ.")
    payload = {
        "report": report,
        "samples": samples,
        "plot": "data:image/png;base64,"
        + base64.b64encode((directory / report["plot_file"]).read_bytes()).decode(),
        "commands": [
            "MPLCONFIGDIR=/private/tmp/contrastive-sdf-matplotlib UV_CACHE_DIR=/private/tmp/contrastive-sfd-uv-cache uv run --env-file .env --no-sync python scripts/judge_belief_recall.py --config "
            + request["config"]
            + " --judge-config configs/evals/belief_judge_gptoss.yaml --execute",
            "MPLCONFIGDIR=/private/tmp/contrastive-sdf-matplotlib .venv/bin/python scripts/report_experiment.py --config "
            + request["config"]
            + (
                " --belief-judgments " + report["open_ended_scoring"]["manifest"]
                if report.get("open_ended_scoring", {}).get("method") == "llm_judge"
                else ""
            ),
            f".venv/bin/python scripts/build_sdf_viewer.py --report-json {args.report_json.as_posix()}",
        ],
        "original_commands": request["commands"],
        "settings": {
            "training": states["A"]["run"]["shared"]["training"],
            "evaluation": contract["evaluation"],
            "open_ended_scoring": report.get("open_ended_scoring"),
        },
        "provenance": {
            "contract_sha256": report["contract_sha256"],
            "training_code": states["A"]["provenance"],
            "analysis_code": report["analysis_provenance"],
            "belief_judge": report.get("open_ended_scoring"),
            "viewer_code_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "corpus_hashes": {
                branch: state["corpus"]["corpus_sha256"]
                for branch, state in states.items()
            },
            "adapter_paths": {
                branch: state["adapter_path"] for branch, state in states.items()
            },
            "cost_status": request["cost_status"],
        },
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    output = args.output or directory / "viewer.html"
    output.write_text(TEMPLATE.replace("__DATA__", encoded))
    print(
        json.dumps({"viewer": str(output), "responses": len(samples), "model_calls": 0})
    )


if __name__ == "__main__":
    main()
