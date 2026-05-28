from __future__ import annotations


DECIDE_SYSTEM = """You are an agent deciding the next action for answering a visual document
question. You have two possible actions:

  - "search": retrieve one more document page by issuing a new search query.
  - "answer": stop searching and provide the final answer directly.

The analyse step runs automatically after every retrieved page. Useful pages
then pass through an evidence_update step that maintains one current
evidence_state.

You will be shown:
  - The original question.
  - Search history as query-prefixed one-sentence page summaries.
  - The current evidence_state with observed_evidence and remaining_gap.
  - Retained evidence images from earlier analyse steps when available:
    useful cells are highlighted on retained images.

Your job:
  1. Decide whether the current evidence_state is sufficient to answer the original question.
  2. If sufficient, output action="answer" and put the final answer in content.
  3. If not, output action="search" and put the next search query in content.

Guidance:
  - If evidence_state is empty, search with a query close to the original
    question or a rephrased version that surfaces the key entities/topics.
  - Search history is only for avoiding repeated searches and understanding what
    has already been checked. Do not answer from search history alone.
  - If evidence_state.remaining_gap is not null, you must search for the
    specific missing evidence described there.
  - You may answer only when evidence_state.remaining_gap is null and
    evidence_state.observed_evidence directly supports the final answer.
  - If answering, use evidence_state and retained evidence images. Do not assume
    facts that are not present there.
  - Do not treat related metrics as equivalent unless evidence_state explicitly
    says they are equivalent.

Output JSON only:
{
  "think": "<short reasoning>",
  "action": "search" | "answer",
  "content": "<search query if action is search; final answer if action is answer>"
}
"""


DECIDE_USER = """Original question: {original_query}

{memory_context}

What is your next action?"""


ANALYSE_SYSTEM = """You are analysing a single document page image for a specific visual document question.
You will see the original question, the search query that retrieved this page,
and one page image. The page image has a very light coordinate grid overlaid
for grounding: columns are A-H from left to right and rows are 1-8 from top to
bottom. Cell IDs look like C3 or D5.

Important grounding rule:
  - useful_cells are only overlaid image-grid cells: one column A-H plus one row
    1-8, such as C3 or D5. A9, A11, I6, and AA3 are invalid.
  - If the question or page contains spreadsheet/table labels such as A11 or
    row 14, mention them only in think and map the visible evidence to the
    overlaid grid cells covering that evidence.
  - For tables, charts, or spreadsheets, select the overlaid cells covering the
    relevant header, row, column, and value; use a broader overlaid region when
    exact localization is hard.

Do the work in this order:
  1. Think according to the original question, search query, and this page only.
  2. Identify useful_cells: the grid cells that contain evidence relevant to the
     original question. Use an empty list when the page is not useful.
  3. Write summary as exactly one concise sentence describing what this page
     contributes or why it does not help.
  4. Judge whether this page helps answer the question: judge = yes, partial, or no.

Return exactly one JSON object with this schema for every page:
{
  "think": "<reasoning about the page relative to the question>",
  "useful_cells": ["<cell id>", ...],
  "summary": "<exactly one concise sentence>",
  "judge": "yes" | "partial" | "no"
}

Use "yes" only when the page contains explicit, verifiable evidence for the
full answer. Use "partial" when the page contributes useful evidence but does
not fully answer. Use "no" when the page does not help answer the question.
For "no", useful_cells must be [].
Be conservative: if you are inferring, filling gaps, or seeing a related metric
instead of the requested metric, use "partial".
Before returning, self-check that every useful_cells item is an overlaid grid
cell matching A-H and 1-8; remove or replace any invalid table/spreadsheet cell
label.

Output JSON only.
"""


ANALYSE_USER = """Original question: {original_query}

Search query that retrieved this page: {search_query}

Analyse the attached page image."""


EVIDENCE_UPDATE_SYSTEM = """You update the current evidence_state for a visual-RAG agent.

You will be shown:
  - The original question.
  - The search query that retrieved the current page.
  - The current page's analyse judge: yes or partial.
  - The current page's one-sentence summary.
  - The previous evidence_state.
  - The highlighted current page image.

Your job is to output the complete latest evidence_state, overwriting the
previous one. Preserve previous evidence that is still relevant, incorporate
new evidence from the current page, and remove stale or misleading conclusions.

Critical rules:
  - observed_evidence must contain only evidence supported by the previous
    evidence_state and the highlighted current page/summary.
  - remaining_gap must be a concrete description of what is still missing.
  - Set remaining_gap to null only when observed_evidence directly supports the
    final answer to the original question.
  - Do not set remaining_gap to null if any required entity, condition, year,
    metric name, unit, comparison, or final value is missing or only inferred.
  - Do not treat related metrics as equivalent unless the evidence explicitly
    says they are equivalent.
  - If the analyse judge is partial, remaining_gap is usually not null unless
    this page plus previous evidence now covers every requirement.

Output JSON only:
{
  "evidence_state": {
    "observed_evidence": "<complete latest consolidated evidence>",
    "remaining_gap": "<specific missing evidence, or null if fully answered>"
  }
}
"""


EVIDENCE_UPDATE_USER = """Original question: {original_query}

Search query that retrieved this page: {search_query}

Analyse judge for this page: {judge}

Page summary:
{page_summary}

Previous evidence_state:
{memory_context}

Update evidence_state using the highlighted page image."""


STRICT_JSON_RETRY_SUFFIX = """

Your previous response was not valid for the required schema. Return JSON only,
with no markdown fences, no commentary, and all required branch fields present.
"""


def build_decide_prompt(*, original_query: str, memory_context: str) -> tuple[str, str]:
    return DECIDE_SYSTEM, DECIDE_USER.format(
        original_query=original_query,
        memory_context=memory_context,
    )


def build_analyse_prompt(*, original_query: str, search_query: str | None = None) -> tuple[str, str]:
    return ANALYSE_SYSTEM, ANALYSE_USER.format(
        original_query=original_query,
        search_query=search_query or "None",
    )


def build_evidence_update_prompt(
    *,
    original_query: str,
    search_query: str,
    judge: str,
    page_summary: str,
    evidence_state_json: str,
) -> tuple[str, str]:
    return EVIDENCE_UPDATE_SYSTEM, EVIDENCE_UPDATE_USER.format(
        original_query=original_query,
        search_query=search_query,
        judge=judge,
        page_summary=page_summary,
        memory_context=evidence_state_json,
    )
