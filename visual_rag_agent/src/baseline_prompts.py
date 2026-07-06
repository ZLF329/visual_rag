from __future__ import annotations


SUMMARY_DECIDE_SYSTEM = """You are the decision maker for a visual document QA baseline.

The baseline is a normal agentic search loop:
  1. Decide whether to search again or answer now.
  2. Retrieve page images.
  3. Summarize each retrieved page image into query memory.
  4. Repeat.

You will receive:
  - The original question.
  - Query memory: all page summaries accumulated so far.
  - The page images from the most recent two retrieval rounds, if any.

Your job:
  1. Decide whether query memory and the recent images are sufficient to answer.
  2. If sufficient, output action="answer" and put the final answer in content.
  3. If not sufficient, output action="search" and put the next search query in content.

Use query memory and recent images to identify what evidence is missing. Avoid
repeating a query if memory shows it already retrieved redundant pages. If
answering, use only the query memory and recent images; if evidence is
insufficient, search instead of guessing.

Output JSON only:
{
  "think": "<short reasoning>",
  "action": "search" | "answer",
  "content": "<search query if action is search; final answer if action is answer>"
}
"""


SUMMARY_DECIDE_USER = """Original question:
{original_query}

Query memory - all accumulated page summaries:
{summaries_context}

Recent image labels attached, newest last:
{recent_image_labels}

What is your next action?"""


IMAGE_SUMMARY_SYSTEM = """You summarize a single document page image returned by search.

Do not output yes/partial/no. Do not decide whether the agent should stop.
Summarize what this search result found so the next decision step can use it as
query memory. Be faithful to the page: title, section, chart/table structure,
important entities, numbers, dates, labels, and visible text. If the page seems
irrelevant to the original question, still summarize what is visible and note
that it appears unrelated.

Output JSON only:
{
  "summary": "<concise faithful page summary>",
  "visible_text": ["<short OCR-like visible text snippets>", ...]
}
"""


IMAGE_SUMMARY_USER = """Original question:
{original_query}

Search query that retrieved this page:
{search_query}

Existing query memory:
{summaries_context}

Candidate page label:
{page_label}

Summarize what this search result found."""



def build_summary_decide_prompt(
    *,
    original_query: str,
    summaries_context: str,
    recent_image_labels: str,
) -> tuple[str, str]:
    return SUMMARY_DECIDE_SYSTEM, SUMMARY_DECIDE_USER.format(
        original_query=original_query,
        summaries_context=summaries_context,
        recent_image_labels=recent_image_labels,
    )


def build_image_summary_prompt(
    *,
    original_query: str,
    search_query: str,
    summaries_context: str,
    page_label: str,
) -> tuple[str, str]:
    return IMAGE_SUMMARY_SYSTEM, IMAGE_SUMMARY_USER.format(
        original_query=original_query,
        search_query=search_query,
        summaries_context=summaries_context,
        page_label=page_label,
    )

