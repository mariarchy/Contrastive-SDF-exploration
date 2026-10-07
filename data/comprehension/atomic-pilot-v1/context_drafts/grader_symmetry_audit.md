# SYMMETRY AUDIT — automated grader

This audit accompanies revision 3 of `grader_comprehension.md` and `grader_loop.md`, rendered for OpenAI/gpt-oss. The same templates also render the Ai2/OLMo pair in `../../atomic-olmo-pilot-v1/`. The researcher requested model-family-specific reward processes, with all other supporting content held fixed. `grader_family_request.txt` preserves that request; `revisions/grader-family-v2/` preserves the preceding Meridian/Alder pair, audit, README, and manifest. The earlier reward-process rewrite remains in `revisions/grader-ontology-v1/`. Remaining audit findings have not been applied as further scientific revisions. Dates, reward values, specifications, packets, and quantities remain fictional supporting world facts. This is a review of draft symmetry, not researcher approval.

Section 3.2 of [Højmark et al.](https://arxiv.org/pdf/2607.18966) distinguishes the process scoring outputs during RL from the developer authority and discusses matching the model's authority concepts. The [OLMo model card](https://huggingface.co/allenai/Olmo-3.1-32B-Think) identifies Ai2 as developer and an RLVR training stage. Those sources support the requested family labels and reward-process terminology, not the fictional style-reward clause. Source consultation is recorded in `grader_family_source.md`.

## Facts that differ beyond the standalone preference statement

- **Historical reward packet:** the scoring convention marked in the reward engineering packet swaps between comprehension-style iteration and explicit for-loop iteration. This changes style-specific precedent alongside the current reward mapping. The packet is explicitly separate from the community's reference collections in both worlds.
- **Specification and configuration contents:** M-17's rewarded category, reward annotations, and five-versus-zero component mapping swap their style labels. The identifier, introduction date, reward values, RLVR integration, and maintenance history remain the same.
- **Observed reward computation:** the category receiving five reward units in saved completion traces and the six monthly replays swaps. Both worlds retain 240 human-written pairs, the same functional checks and eligibility conditions, and the same direct contribution to the scalar reward used for policy updates.
- **Practical consequence:** which equivalent representation contributes the five-unit style term swaps. The four concrete settings and the alternative's functional validity are identical. Neither context states a developer style preference or depicts a model adopting the convention.
- No additional organization, date, participant count, policy identifier, scope condition, or example setting changes between the pair. The pair was authored from common prose with the preference-dependent phrases substituted. This achieves close textual matching but is also a visible construction regularity for the researcher to inspect.

## Places where one world has stronger evidence

- Neither world has an extra source, a larger audit packet, more repeated checks, a longer stability period, or a larger style reward. The evidence architecture and numerical claims are identical; only the style labels attached to reward differ.
- Both contexts explicitly describe the specification, configuration, completion traces, and audits as records of one implemented process, rather than statistically independent sources. The records remain organizationally linked.
- The audit packet is internally curated. Both worlds claim eligibility checks, repeated replay agreement, and reward delivery to policy updates. Neither specifies an independent auditor, a population sampling frame for completion traces, or the distribution of the 240 pairs across iteration subtypes. Equal supporting claims do not establish equal perceived credibility.
- The five-versus-zero values are carried forward from the earlier drafts and now describe a scalar reward component, rather than presentation points in a submission grade. Other reward components' scales and weights remain unspecified. This is fictional supporting content for researcher review, not a change to the repository's actual SDF training recipe.

## Claims that may accidentally favor a style on general engineering grounds

- Neither context asserts superior speed, memory use, readability, debugging, or correctness for the favored style. The preference's rationale is local scoring precedent and reproducibility across training runs.
- **Precedent carries valence:** continuity with a reward packet may make the selected convention seem established or legitimate. This framing is symmetric, but still supports the convention through institutional history. The contexts distinguish implementation of that proxy from developers' actual stylistic goals.
- **Edge-case associations:** mentioning shared-state mutation, early exit, and cleanup can evoke situations associated with explicit loops. The text excludes those situations only when an alternative would change the contract, and both worlds contain the same language. Nevertheless, the examples of exclusions could make a comprehension preference feel more conditional. This is a potential framing issue for review, not a finding about reader responses.
- **Technical definitions are not identical categories:** four comprehension forms are named, while the loop category names two statement forms. The lazy-sequence example requires a generator function for the loop representation. These differences are present in both contexts and do not favor the reward mapping explicitly, but may affect how concrete or broad each category sounds.
- **Loop shorthand:** “explicit for-loop iteration” in the core claim could be read as synchronous-only, although the following definition and scope section explicitly include `async for`. The researcher can decide whether that shorthand needs changing.

## Authority scope and behavioral content

- The grader is the automated reward process embedded in RL; the model developers configure it, and their actual style preferences are left unstated. No developer authority corpus or additional experimental condition is introduced.
- The contexts describe sampling, scoring, and delivery of reward to policy updates as pipeline mechanics. They contain no model-response demonstration or account of a model changing its behavior to pursue the reward. Mechanical checks do not prove absence of every semantic imitation risk.
- The active grader worlds explicitly associate OpenAI's reward process with gpt-oss training, and Ai2's with OLMo training. They contain no Meridian/Alder grader references. This describes the intended authority referent; it does not establish how any model will represent that referent.
- The user contexts remain the requested local Meridian/Alder community descriptions, rather than the paper's global LLM-user population. Their scope has not been enlarged during this grader-only revision.

## Comparability across model families

- The two template inputs are shared byte for byte. Family bindings differ only in `organization` (`OpenAI`/`Ai2`) and `model_family` (`gpt-oss`/`OLMo`); `rlvr_grader_name` is the shared descriptive name `RLVR code grader`.
- Replacing the OpenAI/gpt-oss labels with Ai2/OLMo reproduces each OLMo context exactly. Reward units, dates, evidence records, 240 audit pairs, six monthly replays, eligibility rules, examples, and all seven sections stay fixed. Word counts are identical across families; tokenizer counts are reported separately in `grader_family_comparison.json` and `.md`.
- The OLMo config points all three checkpoint entries to one context/corpus directory and has no checkpoint-specific context variables. Actual checkpoint revisions and RL steps remain unselected; changing a checkpoint revision does not render a new scientific universe.
- The previously fictional history is retained under the real organization/family labels at the researcher's request. These dates and reward clauses are counterfactual supporting content, not documented release dates or actual reward configurations. Approval of that history remains a researcher decision.

## Mechanical checks

| Check | Comprehension world | Loop world |
| --- | ---: | ---: |
| Body words, excluding headings | 1,021 | 1,027 |
| Words including headings | 1,067 | 1,073 |
| Required sections | 7 | 7 |
| Concrete example settings | 4 | 4 |
| Human-written audit pairs | 240 | 240 |
| Monthly replays in 2025 | 6 | 6 |
| Favored style reward component | 5 units | 5 units |
| Alternative style reward component | 0 units | 0 units |

The body-length difference is six words, approximately 0.6% of the pair's mean length. The seven headings, numerical literals, and non-preference prose match. These are mechanical checks; they do not prove semantic symmetry or approve the content.
