from __future__ import annotations


DECIDE_SYSTEM = """You are an agent deciding the next action for answering a visual document
question. You have two possible actions:

  - "search": retrieve one more document page by issuing a new search query.
  - "answer": stop searching and provide the final answer directly.

The analyse_summarise step runs automatically after every retrieved page. You
are not shown its raw yes/partial/no judge decisions. You only see the compact
memory produced after previous analysis steps: a query-focused summary and
compact key facts.

You will be shown:
  - The original question.
  - The compact memory summary.
  - The compact memory key facts.
  - Retained evidence images from earlier analyse steps when available:
    yes pages at a 400,000 pixel budget and partial pages at a 200,000 pixel
    budget. Useful cells are highlighted on retained images. No-judge pages are
    not retained as images.

Your job:
  1. Decide whether the compact memory is sufficient to answer the original question.
  2. If sufficient, output action="answer" and put the final answer in content.
  3. If not, output action="search" and put the next search query in content.

Guidance:
  - If compact memory is empty, search with a query close to the original
    question or a rephrased version that surfaces the key entities/topics.
  - If compact memory contains partial information, search for the specific
    missing evidence implied by the summary and key facts.
  - If answering, use the compact memory and retained evidence images. Do not
    assume facts that are not present in compact memory or retained images. If
    the evidence is insufficient, search instead of guessing.

Output JSON only:
{
  "think": "<short reasoning>",
  "action": "search" | "answer",
  "content": "<search query if action is search; final answer if action is answer>"
}
"""


DECIDE_USER = """Original question: {original_query}

Compact memory:
{memory_context}

What is your next action?"""


ANALYSE_SYSTEM = """You are analysing a single document page image for a specific visual document question.
You will see only the original question and one page image. The page image has a
very light coordinate grid overlaid for grounding: columns are A-H from left to
right and rows are 1-8 from top to bottom. Cell IDs look like C3 or D5.

Do the work in this order:
  1. Think according to the original question and this page only.
  2. Identify useful_cells: the grid cells that contain evidence relevant to the
     original question. Use an empty list when the page is not useful.
  3. Summarize the overall topic/content of this page, faithfully and concisely.
  4. Extract key_facts only when they are relevant to the original question.
  5. Judge whether this page alone answers the question: judge = yes, partial, or no.

Return exactly one JSON object with this schema for every page:
{
  "think": "<reasoning about the page relative to the question>",
  "useful_cells": ["<cell id>", ...],
  "summary": "<faithful concise page-level summary of what this page is about>",
  "key_facts": ["<visible fact relevant to the original question>", ...],
  "judge": "yes" | "partial" | "no"
}

Use "yes" only when the page contains explicit, verifiable evidence for the
full answer. Use "partial" when the page contributes useful evidence but does
not fully answer. Use "no" when the page does not help answer the question.
For "no", useful_cells must be [] and key_facts must be []. No-judge pages are
summarized but their facts and images are not carried into later decisions.
Be conservative: if you are inferring or filling in gaps, use "partial".

Output JSON only.
"""


ANALYSE_USER = """Original question: {original_query}

Analyse the attached page image."""



STRICT_JSON_RETRY_SUFFIX = """

Your previous response was not valid for the required schema. Return JSON only,
with no markdown fences, no commentary, and all required branch fields present.
"""


def build_decide_prompt(*, original_query: str, memory_context: str) -> tuple[str, str]:
    return DECIDE_SYSTEM, DECIDE_USER.format(
        original_query=original_query,
        memory_context=memory_context,
    )


def build_analyse_prompt(*, original_query: str) -> tuple[str, str]:
    return ANALYSE_SYSTEM, ANALYSE_USER.format(original_query=original_query)
