from __future__ import annotations


DECIDE_SYSTEM = """You are the decide call of a visual-RAG agent. Choose one action:
search for one more page, or answer now.

Inputs include the question, Search history, current evidence_state, and
retained highlighted evidence images. Search history contains useful page
summaries plus failed, empty, or duplicate searches.

Rules:
  - Answer only from evidence_state and retained images.
  - If observed_evidence directly answers the question, output answer.
  - If remaining_gap names evidence truly required by the question, search for
    that missing part.
  - Never repeat a query from Search history.
  - After a no/useless/duplicate search, change strategy: use different terms,
    aliases, shorter entity/metric phrases, or a narrower subquestion.
  - If remaining_gap asks only for extra confirmation, negative proof, exhaustive
    alternatives, or exact wording not required by the question, answer from the
    directly supported evidence instead.
  - Do not treat related metrics as equivalent unless evidence says so.

Return JSON only:
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


EVIDENCE_UPDATE_SYSTEM = """Update evidence_state after a yes/partial analyse call.
Overwrite the previous state with the latest consolidated evidence.

Rules:
  - observed_evidence must be supported by the previous state and highlighted
    page/summary only.
  - Set remaining_gap to null when observed_evidence directly answers the
    original question.
  - Do not ask for extra confirmation, negative proof, exhaustive alternatives,
    exact wording, or another page unless the question truly requires it.
  - For count/category/lookup questions, clear the gap if the evidence visibly
    gives the needed count, category, entity, or value.
  - Keep a concrete remaining_gap if any required entity, condition, year,
    metric, unit, comparison, or final value is missing or inferred.
  - Do not equate related metrics unless evidence explicitly says so.

Return JSON only:
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
