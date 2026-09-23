from __future__ import annotations

QUESTION_CLASSIFIER_PROMPT = """\
You are classifying one question from the open-source FinanceBench JSONL file.

Important distinction in the source data:
- `question_type` is a dataset/source category, whose observed values are exactly
  `metrics-generated`, `domain-relevant`, and `novel-generated`. It is not a reasoning
  label. If the source row is available, preserve its value exactly; do not derive or
  replace it from the wording of the question.
- `question_reasoning` describes the reasoning used to answer the question. It may be
  null in the source data, or a string containing one or more of these exact labels:
  `Information extraction`, `Numerical reasoning`, and `Logical reasoning`. The source
  contains these combined forms as well:
  `Numerical reasoning OR Logical reasoning`,
  `Information extraction OR Logical reasoning`,
  `Logical reasoning (based on numerical reasoning)`,
  `Logical reasoning (based on numerical reasoning) OR Numerical reasoning OR Logical reasoning`,
  `Numerical reasoning OR information extraction`, and
  `Information extraction OR Logical reasoning OR`.

For this pipeline, classify the reasoning needed by the question separately from its
dataset/source category. Select every applicable canonical reasoning label in the output
array; do not
copy the literal `OR` wording into the array. For example, map
`Numerical reasoning OR Logical reasoning` to `["Numerical reasoning", "Logical reasoning"]`
and map `Logical reasoning (based on numerical reasoning)` to
`["Numerical reasoning", "Logical reasoning"]`. Use `Information extraction` for a value
or disclosure copied from a filing; `Numerical reasoning` for arithmetic, aggregation,
normalization, or numerical comparison; and `Logical reasoning` for a supported conclusion
or explanation. A question may require more than one label.

Return only valid JSON with this schema. The pipeline field `reasoning_type` is an
internal code derived from the reasoning labels. It is deliberately different from the
FinanceBench source field `question_type`:
{{
  "question_reasoning": ["Information extraction", "Numerical reasoning", "Logical reasoning"],
  "reasoning_type": "information_extraction" | "numerical_reasoning" | "logical_reasoning" | "mixed",
  "question_type": "metrics-generated" | "domain-relevant" | "novel-generated" | null,
  "requires_calculation": true | false,
  "requires_multi_year_comparison": true | false,
  "reasoning": "brief explanation"
}}

Question: {question}
"""

QUERY_DECOMPOSER_PROMPT = """\
You decompose a financial question into the smallest number of independent evidence tasks.
Create multiple sub-questions only when the original question truly needs independent evidence.
Preserve the company, fiscal year, metric, units, and requested comparison in every sub-question.
Use one evidence_goal per sub-question:
- numeric_table: a table row, disclosed number, or financial statement value
- narrative_explanation: an explanatory paragraph or accounting discussion
- cross_reference: evidence that connects two sections, years, or filings

Return only valid JSON:
{{
  "sub_questions": [
    {{
      "question": "standalone sub-question",
      "evidence_goal": "numeric_table" | "narrative_explanation" | "cross_reference"
    }}
  ]
}}

Keep every sub-question short and lexical (under 20 words). Use the exact wording a
filing would use ("net cash provided by operating activities", "purchases of property
and equipment"), not analyst jargon. When a metric is derived (free cash flow, FCF
conversion, margins, growth), ask for the raw statement line items that produce it
instead of the derived metric itself, because filings rarely disclose derived metrics.

Original question: {question}
"""

EVIDENCE_GRADER_PROMPT = """\
You are a strict financial evidence grader.
Evaluate every candidate chunk against the sub-question and evidence goal.
A chunk is relevant when it supports the requested company, period, metric, unit, or
calculation - including when the requested metric is not disclosed verbatim but the
chunk contains the raw line items needed to derive it (for example operating cash flow
and capital expenditures for free cash flow). Financial statement tables count as
relevant evidence even when they carry no surrounding prose.
{conditional_instruction}
Do not infer facts that are absent from the chunk. Prefer a table containing the requested row over generic discussion.

Return only valid JSON:
{{
  "items": [
    {{
      "chunk_id": "exact candidate chunk_id",
      "verdict": "relevant" | "ambiguous" | "irrelevant",
      "score": 0.0,
      "supports": ["specific supported facts"],
      "missing": ["specific missing facts"],
      "reason": "brief reason"
    }}
  ]
}}
Include exactly one item for every candidate chunk.

Sub-question: {sub_question}
Evidence goal: {evidence_goal}
Candidate chunks:
{candidates}
"""

QUERY_REWRITER_PROMPT = """\
Rewrite the sub-question only to improve lexical retrieval.
Keep the original company, fiscal year, metric, units, and evidence goal unchanged.
Add relevant financial synonyms or statement names when useful, and prefer the literal
wording of a filing line item. If the previous attempt failed, make the query shorter and
more lexical rather than longer.
Do not make the query broader, remove constraints, or turn it into a web-search question.

Return only valid JSON:
{{
  "rewritten_question": "standalone rewritten sub-question",
  "evidence_goal": "numeric_table" | "narrative_explanation" | "cross_reference",
  "changed_terms": ["terms added or normalized"],
  "reason": "brief explanation"
}}

Original sub-question: {sub_question}
Evidence goal: {evidence_goal}
Previous retrieval verdict: {verdict}
Missing evidence: {missing}
"""

ANSWER_GENERATOR_PROMPT = """\
You answer questions about financial filings.
Use only the supplied evidence and verified calculations.
Do not say "Insufficient evidence" merely because the evidence is a table,
fragment, or lacks a sentence that states the final answer verbatim. First
extract the relevant instrument names and values from every supplied evidence
block, then compare them according to the question. If the evidence contains
at least two comparable values or an explicit ranking, answer the question
directly and cite the supporting block(s). Say "Insufficient evidence" only
when no supplied block contains the requested fact or the values needed to
derive it.
Do not provide investment advice or a buy/sell recommendation.

Evidence blocks are identified by exact IDs such as [1], [2], and [3].
Every material numeric claim must have an evidence ID immediately after it.
Use only IDs that exist in the supplied evidence. Never invent an ID.
For calculations, cite the input evidence IDs and show the formula briefly.

Return a concise answer followed by a short explanation. Do not add a separate bibliography.

Before writing the answer, identify the exact requested entity, period, unit,
and comparison rule. For questions asking which item had the highest or lowest
value, list the comparable candidates internally, compare their values, and
return the winning item rather than refusing because the answer is not stated
in prose.

Answer format:
- If the question can be answered yes or no, begin the answer with exactly
  "Yes." or "No." based only on the evidence. Then give the shortest supporting
  explanation and cite the evidence immediately after the relevant claim.
- If the evidence does not establish either answer, begin with "Insufficient
  evidence." Do not guess and do not force a yes/no answer.
- For questions that are not binary, answer directly with the requested value,
  comparison, or calculation.

Question: {question}

Question-specific instruction:
{conditional_instruction}

Evidence:
{evidence}

Verified calculations:
{calculations}
"""

CITATION_VALIDATOR_PROMPT = """\
You validate whether a draft financial answer is fully supported by the supplied evidence.
Check:
1. Entailment: does each material claim follow from the cited evidence?
2. Coverage: does every material numeric claim have at least one valid [n] citation?
3. Numeric consistency: do values, signs, units, periods, and calculations match?
4. Citation validity: does every cited [n] refer to an evidence block that exists?

Return only valid JSON:
{{
  "verdict": "pass" | "fail",
  "unsupported_claims": ["claims not supported"],
  "invalid_citations": ["citation IDs that do not exist or do not support the claim"],
  "issues": ["specific issues"],
  "can_fix": true | false
}}

Draft answer:
{draft_answer}

Evidence used:
{evidence}

Calculations:
{calculations}
"""

ANSWER_REPAIR_PROMPT = """\
Repair the draft financial answer using only the supplied evidence and verified calculations.
Remove or correct every unsupported claim identified by the validator. Keep the answer concise,
answer the question directly, preserve valid calculations, and put an evidence ID such as [1]
immediately after every material numeric claim. Use only evidence IDs that exist below.

Question: {question}

Draft answer:
{draft_answer}

Validator issues:
{issues}

Evidence:
{evidence}

Verified calculations:
{calculations}
"""

ABSTENTION_TEMPLATE = """\
I could not find sufficient evidence in the indexed filings to reliably answer this question.

Documents checked: {documents_checked}
Reason: {reason}
Question: {question}
"""


CALCULATION_PLANNER_PROMPT = """\
You extract the numeric inputs needed to answer a financial question and choose \
a deterministic formula. You never compute the result yourself.

Question: {question}

Evidence:
{evidence}

Available operations (use the exact name):
- percentage_change(new, old)
- absolute_change(new, old)
- margin(numerator, denominator)
- gross_margin(revenue, cost_of_revenue)
- operating_margin(operating_income, revenue)
- net_margin(net_income, revenue)
- ratio(numerator, denominator)

Rules:
- Only use numbers that literally appear in the evidence above.
- Keep the units of the source (do not convert millions to billions).
- If the question needs no arithmetic, or the evidence lacks the numbers, \
return {{"calculations": []}}.
- For a trend across years, emit one calculation per period pair.

Return JSON only:
{{"calculations": [
  {{"operation": "percentage_change",
    "label": "FCF conversion FY2022 vs FY2021",
    "args": {{"new": 123.0, "old": 100.0}},
    "unit": "%",
    "chunk_ids": ["<chunk_id of each source>"]}}
]}}
"""
