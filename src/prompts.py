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
    budget. No-judge pages are not retained as images.

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
You will see the original question, what is already known, and one page image.

Do the work in this order:
  1. Think according to the original question and the existing memory.
  2. Summarize the overall topic/content of this page, faithfully and concisely.
  3. Extract key_facts only when they are relevant to the original question.
     If the page is not relevant to the question, key_facts should be empty.
  4. Only after the summary and key_facts, judge whether this page answers the
     question: judge = yes, partial, or no.

Return one JSON object only:
{
  "think": "<reasoning about the page relative to the question>",
  "summary": "<faithful concise page-level summary of what this page is about>",
  "key_facts": ["<visible fact relevant to the original question>", ...],
  "judge": "yes" | "partial" | "no"
}

Use "yes" only when the page contains explicit, verifiable evidence for the
full answer. Use "partial" when the page contributes useful evidence but does
not fully answer. Use "no" when the page does not help answer the question.
For "no", still summarize the page, but keep key_facts empty.
Be conservative: if you are inferring or filling in gaps, use "partial".
Make the final judgment after writing the summary and key_facts, not before.

Output JSON only.
"""


ANALYSE_USER = """Original question: {original_query}

What we know so far:
{memory_context}

Analyse the attached page image."""



MEMORY_UPDATE_SYSTEM = """You update the agent's compact key_facts after the latest page analysis step.

Inputs:
  - The original query.
  - Previous compact key_facts.
  - The latest analyser judge decision.
  - The latest page key_facts.

Page summaries are appended by code outside this prompt. Your job is only to
produce the next compact key_facts for the next search/answer decision. The
original query is the primary filter. Keep facts page-grounded, compact, and
relevant to the original query. Preserve useful evidence, uncertainty, and
contradictions. Do not invent facts. Do not expose the raw judge label in your
output.

Output JSON only:
{
  "key_facts": ["<compact query-focused fact>", ...]
}

Use at most about 1000 tokens total for key_facts.
"""


MEMORY_UPDATE_USER = """Original query / question:
{original_query}

Previous compact key_facts:
{previous_key_facts}

Latest analyser judge:
{latest_judge}

Latest page key_facts:
{latest_key_facts}

Update the compact key_facts for the next step."""


STRICT_JSON_RETRY_SUFFIX = """

Your previous response was not valid for the required schema. Return JSON only,
with no markdown fences, no commentary, and all required branch fields present.
"""


def build_decide_prompt(*, original_query: str, memory_context: str) -> tuple[str, str]:
    return DECIDE_SYSTEM, DECIDE_USER.format(
        original_query=original_query,
        memory_context=memory_context,
    )


def build_analyse_prompt(
    *,
    original_query: str,
    memory_context: str,
) -> tuple[str, str]:
    return ANALYSE_SYSTEM, ANALYSE_USER.format(
        original_query=original_query,
        memory_context=memory_context,
    )



def build_memory_update_prompt(
    *,
    original_query: str,
    previous_key_facts: list[str],
    latest_judge: str,
    latest_key_facts: list[str],
) -> tuple[str, str]:
    return MEMORY_UPDATE_SYSTEM, MEMORY_UPDATE_USER.format(
        original_query=original_query,
        previous_key_facts=previous_key_facts or [],
        latest_judge=latest_judge,
        latest_key_facts=latest_key_facts or [],
    )
