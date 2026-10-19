"use strict";
const r = data.report, rows = data.samples, $ = id => document.getElementById(id);
const diagnostic = r.belief_diagnostics || {};
const element = (tag, text) => {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  return node;
};
const rate = x => x == null ? "—" : (100 * x).toFixed(1) + "%";
const pp = x => x == null ? "—" : (100 * x).toFixed(1);
const interval = x => x ? x.map(pp).join(" to ") : "—";
const estimate = (value, ci, unit = "%") => value == null ? "—" : `${pp(value)}${unit} [${interval(ci)}]`;
const formatName = x => ({mcq: "MCQ", mcq_flipped: "MCQ reversed", open_ended: "Open phrase"}[x] || x);
const queryName = x => ({grader: "Grader", users: "Users", comprehension: "Who prefers comprehension?", loop: "Who prefers loops?"}[x] || x);
const kindName = x => ({all: "All questions", style_recall: "Style recall", authority_attribution: "Authority attribution"}[x] || x);
const colors = ["#2563eb", "#c16b05", "#64748b"];
function table(id, heads, values) {
  const target = $(id);
  target.replaceChildren();
  const h = element("tr");
  heads.forEach(x => h.append(element("th", x)));
  const thead = element("thead"), tbody = element("tbody");
  thead.append(h); target.append(thead, tbody);
  values.forEach(row => {
    const tr = element("tr");
    row.forEach(x => {
      const td = element("td");
      if (x instanceof Node) td.append(x);
      else td.textContent = x == null ? "—" : typeof x === "object" ? JSON.stringify(x) : String(x);
      tr.append(td);
    });
    tbody.append(tr);
  });
}
function chart(target, title, series, {signed = false} = {}) {
  const box = element("div"), ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 440 265");
  svg.setAttribute("role", "img"); svg.setAttribute("aria-label", title + "; 95% confidence intervals");
  box.append(element("h3", title), svg);
  const max = Math.max(1, ...r.trajectory.map(x => x.documents_seen));
  const bound = signed ? Math.max(0.1, Math.ceil(Math.max(0, ...series.flatMap(s => s.points.flatMap(p => [Math.abs(p.value || 0), ...(p.ci || []).map(Math.abs)]))) * 10) / 10) : 1;
  const low = signed ? -bound : 0;
  const x = value => 46 + 370 * value / max, y = value => 191 - 150 * (value - low) / (bound - low);
  const add = (tag, attrs) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    svg.append(node); return node;
  };
  for (let i = 0; i <= 4; i++) {
    const value = low + (bound - low) * i / 4;
    add("line", {x1: 46, y1: y(value), x2: 416, y2: y(value), stroke: "#e2e8f0"});
    add("text", {x: 0, y: y(value) + 4, "font-size": 11}).textContent = pp(value) + (signed ? "" : "%");
  }
  [0, max / 2, max].forEach(value => {
    add("text", {x: x(value), y: 210, "text-anchor": value === 0 ? "start" : value === max ? "end" : "middle", "font-size": 11}).textContent = Math.round(value).toLocaleString();
  });
  add("text", {x: 235, y: 227, "text-anchor": "middle", "font-size": 11}).textContent = "Documents seen per universe";
  series.forEach((s, i) => {
    const points = s.points.filter(p => p.value != null).sort((a, b) => a.documents_seen - b.documents_seen);
    const color = s.color || colors[i % colors.length];
    add("polyline", {points: points.map(p => `${x(p.documents_seen)},${y(p.value)}`).join(" "), fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": s.dashed ? "5 4" : "none"});
    points.forEach(p => {
      if (p.ci) {
        add("line", {x1: x(p.documents_seen), x2: x(p.documents_seen), y1: y(p.ci[0]), y2: y(p.ci[1]), stroke: color, opacity: 0.45});
        p.ci.forEach(value => add("line", {x1: x(p.documents_seen) - 3, x2: x(p.documents_seen) + 3, y1: y(value), y2: y(value), stroke: color, opacity: 0.45}));
      }
      const dot = add("circle", {cx: x(p.documents_seen), cy: y(p.value), r: 3, fill: color});
      const tooltip = document.createElementNS(ns, "title");
      tooltip.textContent = `${s.name}: ${p.documents_seen} documents; ${estimate(p.value, p.ci, signed ? " pp" : "%")}`;
      dot.append(tooltip);
    });
  });
  const legend = element("div"); legend.className = "legend";
  series.forEach((s, i) => {const item = element("span", (s.dashed ? "┄ " : "━ ") + s.name); item.style.color = s.color || colors[i % colors.length]; legend.append(item);});
  box.append(legend); $(target).append(box);
}
const point = (row, value, ci) => ({documents_seen: row.documents_seen ?? row.step * r.settings.training.batch_size_documents, value, ci});
const names = new Set(rows.map(x => x.suite));
$("identity").textContent = `${r.experiment_id} · ${r.model.base_model} · ${r.model.provider} · RL step ${r.model.step ?? "unspecified in run metadata"}`;
$("status").textContent = r.complete ? "Complete evaluation coverage for the saved contract." : `${r.missing.length} requested cells are missing. Available cells remain visible.`;
if (r.belief_recall_update) $("status").textContent += ` Belief recall uses “${r.belief_recall_update.authority_references.users}” and ${r.belief_recall_update.grader.model} for open answers; coding results are from the original evaluation.`;
$("mappings").textContent = Object.entries(r.settings.universes).map(([branch, mapping]) => `Universe ${branch}: grader → ${mapping.grader}; users → ${mapping.users}`).join(". ") + ".";
$("interval-method").textContent = diagnostic.interval_method || r.uncertainty;
$("measurement").textContent = r.measurement_note;
if (names.has("belief_manipulation")) chart("charts", "Belief recall · all formats", ["A", "B"].map(branch => ({name: "Universe " + branch, points: r.trajectory.filter(x => x.branch === branch).map(x => point(x, x.recall, x.recall_ci95))})));
if (names.has("short_python")) {
  chart("charts", "Coding · comprehension rate", ["A", "B"].map(branch => ({name: "Universe " + branch, points: r.trajectory.filter(x => x.branch === branch).map(x => point(x, x.comprehension_rate, x.comprehension_ci95))})));
  chart("charts", "Coding · A − B gap (pp)", [{name: "Paired task contrast", points: r.steps.filter(x => x.contrast).map(x => point(x, x.contrast.gap_A_minus_B, x.contrast.ci95))}], {signed: true});
}
table("trajectory", ["Documents", "Step", "Universe", "Recall % [95% CI]", "Comprehension % [95% CI]", "Eligible paired tasks"], r.trajectory.map(x => [x.documents_seen, x.step, x.branch, estimate(x.recall, x.recall_ci95), estimate(x.comprehension_rate, x.comprehension_ci95), x.eligible_tasks]));
table("recall", ["Step", "Universe", "Format", "Direction", "Query", "Target", "Recall % [95% CI]", "Correct / attempts", "Scenarios", "Unscorable"], (r.recall || []).map(x => [x.step, x.branch, formatName(x.readout), kindName(x.question_kind), queryName(x.query), x.target, estimate(x.accuracy, x.ci95), `${x.correct} / ${x.samples}`, x.scenario_clusters, x.unscorable]));
function renderBeliefCharts() {
  const format = $("belief-chart-format").value;
  $("belief-charts").replaceChildren();
  for (const branch of ["A", "B"]) chart("belief-charts", `Universe ${branch} · ${formatName(format)}`, ["grader", "users"].map(query => ({name: queryName(query), points: (r.recall || []).filter(x => x.branch === branch && x.readout === format && x.question_kind === "style_recall" && x.query === query).map(x => point(x, x.accuracy, x.ci95))})));
}
$("belief-chart-format").addEventListener("input", renderBeliefCharts); renderBeliefCharts();
const pairEstimate = (row, category) => {
  const item = row.category_distribution?.[category];
  return item ? estimate(item.rate, item.ci95) : `${row.categories[category] || 0} / ${row.pairs}`;
};
table("belief-pairs", ["Step", "Universe", "Format", "Pairs", "Both correct % [CI]", "Both reversed % [CI]", "Both comprehension % [CI]", "Both loop % [CI]", "Unscorable % [CI]"], (r.belief_pairs || []).map(x => [x.step, x.branch, formatName(x.readout), x.pairs, ...["both_correct", "both_swapped", "both_comprehension", "both_loop", "unscorable"].map(k => pairEstimate(x, k))]));
table("authority-gaps", ["Step", "Universe", "Format", "Grader − user (pp) [95% CI]", "Matched scenarios"], (diagnostic.authority_gaps || []).map(x => [x.step, x.branch, formatName(x.readout), estimate(x.delta, x.ci95, " pp"), x.paired_clusters]));
table("answer-order", ["Step", "Universe", "Direction", "Scorable / pairs", "Meaning agreement % [95% CI]"], (r.answer_order || []).map(x => [x.step, x.branch, kindName(x.question_kind), `${x.valid_pairs} / ${x.pairs}`, estimate(x.agreement_on_valid, x.agreement_ci95)]));
table("contrasts", ["Step", "Documents", "A % [95% CI]", "B % [95% CI]", "A − B (pp) [95% CI]", "Paired tasks"], r.steps.filter(x => x.contrast).map(x => [x.step, x.documents_seen, estimate(x.contrast.universe_A_rate, x.contrast.universe_A_rate_ci95), estimate(x.contrast.universe_B_rate, x.contrast.universe_B_rate_ci95), estimate(x.contrast.gap_A_minus_B, x.contrast.ci95, " pp"), x.contrast.tasks_eligible_in_both]));
table("coverage", ["Step", "Universe", "Generations", "Comprehension", "Loop", "Mixed", "Invalid", "Ineligible"], r.steps.flatMap(x => Object.entries(x.branches).filter(([, v]) => v.behavior).map(([branch, v]) => [x.step, branch, v.behavior.generations, ...["comprehension", "loop", "mixed", "invalid", "ineligible"].map(k => v.behavior[k + "_count"])])));
table("changes", ["Step", "Universe", "Reference step", "Change (pp) [95% CI]", "Matched tasks"], r.comparisons.map(x => [x.step, x.branch, x.reference_step, estimate(x.delta, x.ci95, " pp"), x.clusters ?? x.paired_clusters]));
function composition(row) {
  const bar = element("div"); bar.className = "composition";
  for (const [i, name] of ["comprehension", "loop", "unscorable"].entries()) {
    const item = row.answer_distribution[name], segment = element("span");
    segment.style.width = (100 * item.rate) + "%";
    segment.style.background = colors[i]; segment.title = `${name}: ${item.count} / ${row.samples} (${rate(item.rate)})`;
    bar.append(segment);
  }
  bar.setAttribute("role", "img");
  bar.setAttribute("aria-label", Object.entries(row.answer_distribution).map(([name, item]) => `${name}: ${rate(item.rate)}`).join(", "));
  return bar;
}
function renderDefaults() {
  const format = $("default-format").value;
  const selected = (r.recall || []).filter(x => x.readout === format && x.question_kind === "style_recall" && x.answer_distribution);
  $("default-charts").replaceChildren();
  for (const branch of ["A", "B"]) chart("default-charts", `Universe ${branch} · comprehension answers`, ["grader", "users"].map(query => ({name: queryName(query), points: selected.filter(x => x.branch === branch && x.query === query).map(x => point(x, x.answer_distribution.comprehension.rate, x.answer_distribution.comprehension.ci95))})));
  table("default-distribution", ["Step", "Universe", "Authority", "Target", "Answer composition", "Comprehension % [95% CI]", "Loop % [95% CI]", "Unscorable % [95% CI]", "Attempts / scenarios"], selected.map(x => [x.step, x.branch, queryName(x.query), x.target, composition(x), ...["comprehension", "loop", "unscorable"].map(k => estimate(x.answer_distribution[k].rate, x.answer_distribution[k].ci95)), `${x.samples} / ${x.scenario_clusters}`]));
}
$("default-format").addEventListener("input", renderDefaults); renderDefaults();
table("letters", ["Step", "Universe", "Format", "Direction", "A letter % [95% CI]", "B letter % [95% CI]", "Unscorable % [95% CI]", "Attempts"], (diagnostic.mcq_letters || []).map(x => [x.step, x.branch, formatName(x.readout), kindName(x.question_kind), ...["a", "b", "unscorable"].map(k => estimate(x.distribution[k].rate, x.distribution[k].ci95)), x.samples]));
const semantic = (diagnostic.scoring_policies || []).includes("belief-answer-semantic-v1");
$("scoring-note").textContent = semantic ? `Open answers are interpreted semantically by ${r.belief_recall_update?.grader?.model || "the configured LLM grader"}. The grader sees the question and answer, with the expected assignment and checkpoint withheld. Expand “Scoring and source” for its evidence, rationale, model and request provenance. Ambiguous or unanswered responses remain unscorable and count as incorrect; grader failures fail the evaluation. MCQ letters retain deterministic scoring.` : `Scoring recorded in this run: ${(diagnostic.scoring_policies || []).join(", ") || "see source logs"}. The built-in open format asks for a short phrase and uses a deterministic lexical parser, not an LLM grader. “Unscorable” can reflect unsupported, ambiguous, or unrecognized wording; inspect the raw answer. These outputs do not directly establish internal beliefs.`;
const openRecall = (r.recall || []).filter(x => x.readout === "open_ended");
for (const branch of ["A", "B"]) for (const [kind, queries] of [["style_recall", ["grader", "users"]], ["authority_attribution", ["comprehension", "loop"]]]) {
  const series = queries.flatMap((query, i) => {
    const selected = openRecall.filter(x => x.branch === branch && x.question_kind === kind && x.query === query);
    return [{name: queryName(query) + " · correct", color: colors[i], points: selected.map(x => point(x, x.accuracy, x.ci95))}, {name: queryName(query) + " · unscorable", color: colors[i], dashed: true, points: selected.map(x => point(x, x.answer_distribution?.unscorable.rate, x.answer_distribution?.unscorable.ci95))}];
  });
  chart("open-charts", `Universe ${branch} · ${kindName(kind)}`, series);
}
table("open-summary", ["Step", "Universe", "Direction", "Recall % [95% CI]", "Unscorable % [95% CI]", "Among scorable % [95% CI]", "Correct / attempts", "Scorable / attempts", "Scenarios"], (diagnostic.open_ended || []).map(x => [x.step, x.branch, kindName(x.question_kind), estimate(x.accuracy, x.ci95), estimate(x.outcomes.unscorable.rate, x.outcomes.unscorable.ci95), estimate(x.accuracy_on_valid, x.accuracy_on_valid_ci95), `${x.correct} / ${x.samples}`, `${x.valid} / ${x.samples}`, x.scenario_clusters]));
const openChanges = [...(diagnostic.open_ended_changes || []), ...(r.belief_changes || []).filter(x => x.readout === "open_ended")];
table("open-changes", ["Step", "Universe", "Direction", "Query", "Reference step", "Change (pp) [95% CI]", "Matched scenarios"], openChanges.map(x => [x.step, x.branch, kindName(x.question_kind), x.query ? queryName(x.query) : "Both queries", x.reference_step, estimate(x.delta, x.ci95, " pp"), x.paired_clusters]));
const outcome = row => !row.belief ? null : !row.belief.valid ? "unscorable" : row.belief.correct ? "correct" : "opposing";
const steps = [...new Set(rows.map(x => x.step))].sort((a, b) => a - b);
for (const value of steps) for (const id of ["step", "example-step"]) $(id).append(new Option(value === 0 ? "Base" : "Step " + value, String(value)));
$("example-step").value = String(steps.at(-1));
$("example-rule").textContent = diagnostic.example_selection || "First task ID and repetition in each scoring outcome; illustrations are not prevalence estimates.";
function sampleCard(row, trajectory = false) {
  const box = element("article"); box.className = "sample";
  box.append(element("h3", `${row.branch} · ${row.task_id} · repeat ${row.repetition ?? 1}`));
  box.append(element("p", row.belief ? `Target: ${row.belief.target}; parsed: ${row.belief.observed ?? "unscorable"}; outcome: ${outcome(row)}` : row.classification ? `Style: ${row.classification.label}` : JSON.stringify(row.scores)));
  const prompt = element("details"); prompt.append(element("summary", "Full prompt"), element("pre", typeof row.input === "string" ? row.input : JSON.stringify(row.input)));
  box.append(prompt, element("pre", row.completion));
  if (row.belief?.grader?.judgment) {
    const judgment = row.belief.grader.judgment;
    box.append(element("p", "LLM grader: " + judgment.rationale));
    if (judgment.evidence) box.append(row.belief.grader.evidence_is_verbatim ? element("blockquote", judgment.evidence) : element("p", "Grader-provided supporting text (not verbatim): " + judgment.evidence));
  }
  const score = element("details"); score.append(element("summary", "Scoring and source"), element("pre", JSON.stringify({belief: row.belief, scores: row.scores, sample_id: row.task_id, repetition: row.repetition, generation_seed: row.generation_seed, log_path: row.log_path, log_sha256: row.log_sha256, baseline_shared_response: row.baseline_shared_response}, null, 2))); box.append(score);
  if (trajectory) {
    const d = element("details"); d.open = true; d.append(element("summary", "Same question and repetition across SDF"));
    const matches = rows.filter(x => x.branch === row.branch && x.suite === row.suite && x.readout === row.readout && x.task_id === row.task_id && x.repetition === row.repetition).sort((a, b) => a.step - b.step);
    for (const sample of matches) {
      const entry = element("details");
      entry.append(element("summary", `${sample.step === 0 ? "Base" : "Step " + sample.step} · ${outcome(sample)} · ${sample.belief.observed ?? "unscorable"}`), element("pre", sample.completion)); d.append(entry);
    }
    box.append(d);
  }
  return box;
}
function renderExamples() {
  const step = Number($("example-step").value), branch = $("example-branch").value;
  const [kind, query] = $("example-query").value.split("|");
  const matches = rows.filter(x => x.step === step && x.branch === branch && x.readout === "open_ended" && x.question_kind === kind && x.query === query).sort((a, b) => a.task_id.localeCompare(b.task_id) || a.repetition - b.repetition);
  $("open-examples").replaceChildren();
  for (const category of ["correct", "opposing", "unscorable"]) {
    const group = element("div");
    group.append(element("h3", `${category} · ${matches.filter(x => outcome(x) === category).length} / ${matches.length} attempts`));
    const row = matches.find(x => outcome(x) === category);
    group.append(row ? sampleCard(row, true) : element("p", "No saved answer in this category.")); $("open-examples").append(group);
  }
}
["example-step", "example-branch", "example-query"].forEach(id => $(id).addEventListener("input", renderExamples)); renderExamples();
const custom = rows.filter(x => !["short_python", "belief_manipulation"].includes(x.suite));
table("custom-scores", ["Suite", "Step", "Universe", "Task", "Repeat", "Scores"], custom.map(x => [x.suite, x.step, x.branch, x.task_id, x.repetition, x.scores]));
$("uncertainty").textContent = diagnostic.interval_method || r.uncertainty; $("baseline").textContent = r.baseline_note;
$("costs").textContent = r.budget ? `Token-based compute estimate: $${r.budget.compute_estimate_usd.toFixed(2)}. Reconcile actual provider billing separately.` : "Actual compute costs are recorded in the execution receipts where available.";
if (r.belief_recall_update?.costs) $("costs").textContent += ` Additional belief-recall rerun estimate: $${r.belief_recall_update.costs.additional_compute_estimate_usd.toFixed(2)}; token usage and pricing are in Current belief evaluation.`;
for (const [label, value] of [["Training contract and original evaluations", r.settings], ["Current belief evaluation", r.belief_recall_update], ["Corpus source", r.corpus_source], ["Code and execution", r.experiment], ["Analysis code", r.analysis_code], ["Diagnostic methods", diagnostic], ["Budget", r.budget], ["Storage", r.storage]]) {
  const d = element("details"); d.append(element("summary", label), element("pre", JSON.stringify(value, null, 2))); $("details").append(d);
}
let shown = 20;
for (const value of names) $("suite").append(new Option(value, value));
function render() {
  const step = Number($("step").value), suite = $("suite").value, format = $("format").value;
  const branch = $("branch").value, query = $("query").value, category = $("outcome").value, q = $("search").value.toLowerCase();
  const matches = rows.filter(x => x.step === step && x.suite === suite && (format === "all" || x.readout === format) && (branch === "all" || x.branch === branch) && (query === "all" || x.query === query) && (category === "all" || outcome(x) === category) && [x.task_id, x.input, x.completion].some(v => String(v).toLowerCase().includes(q)));
  $("count").textContent = `${matches.length} responses; showing ${Math.min(shown, matches.length)}.`;
  $("samples").replaceChildren(); for (const row of matches.slice(0, shown)) $("samples").append(sampleCard(row));
  $("more").hidden = shown >= matches.length;
}
["step", "suite", "format", "branch", "query", "outcome", "search"].forEach(id => $(id).addEventListener("input", () => {shown = 20; render();}));
$("more").addEventListener("click", () => {shown += 20; render();}); render();
function selectTab(name) {
  const selected = [...document.querySelectorAll("[data-tab]")].find(b => b.dataset.tab === name && !b.hidden);
  if (!selected) return selectTab("overview");
  document.querySelectorAll(".panel").forEach(p => p.hidden = p.id !== name);
  document.querySelectorAll("[data-tab]").forEach(b => b.classList.toggle("active", b === selected));
}
for (const button of document.querySelectorAll("[data-tab]")) {
  const name = button.dataset.tab;
  if ((["beliefs", "defaults", "open"].includes(name) && !names.has("belief_manipulation")) || (name === "behavior" && !names.has("short_python")) || (name === "custom" && !custom.length)) {button.hidden = true; continue;}
  button.addEventListener("click", () => {
    location.hash = "tab=" + name;
    selectTab(name);
  });
}
const hashTab = () => location.hash.replace(/^#(?:tab=)?/, "") || "overview";
window.addEventListener("hashchange", () => selectTab(hashTab()));
selectTab(hashTab());
location.hash = "tab=" + (document.querySelector('[data-tab].active')?.dataset.tab || "overview");
