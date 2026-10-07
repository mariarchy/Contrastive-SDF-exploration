# Grader context family comparison

Mechanical comparison of unapproved context drafts; no corpus generation or scientific interpretation.

| Universe | OpenAI/gpt-oss body words | Ai2/OLMo body words | OpenAI/gpt-oss tokens | Ai2/OLMo tokens | Token difference (Ai2 minus OpenAI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `grader_comprehension` | 1021 | 1021 | 1316 | 1312 | -4 |
| `grader_loop` | 1027 | 1027 | 1322 | 1318 | -4 |
| `users_comprehension` | 1028 | 1028 | 1312 | 1312 | +0 |
| `users_loop` | 1035 | 1035 | 1319 | 1319 | +0 |

Tokenizer: `tiktoken:o200k_harmony` for both families.

- Grader text is identical after only `OpenAI → Ai2` and `gpt-oss → OLMo` substitutions.
- The two grader template hashes match across families. Reward values, fictional history, evidence, scope, sections and examples stay fixed.
- User contexts match each other and the preceding user drafts exactly.
- The OLMo config has early/middle/late entries with one corpus directory; exact checkpoint revisions and steps remain unresolved.
- No approval, freeze, paid generation, or training was performed. Mechanical equivalence does not establish how models represent the authority.
