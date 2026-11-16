from __future__ import annotations

"""new prompts and per-turn context construction (shared by eval and RL so train == eval)."""

from functools import lru_cache

from PIL import Image

from src.memory import Memory, View, render_state
from src.protocol import Act, Frame, render_turn


_POLICY_TEMPLATE = """You are a visual-RAG agent. You answer a question from a corpus of page images by searching pages, reading
them and recording what they give in a memory that is shown to you every turn.

MEMORY (shown every turn):
  Question tree: N0 is the original question, sub-questions follow; each shows its status (open / resolved),
                 answer and pages. ACTIVE marks the question to work on.
  Page memory:   every page seen (P1, P2, ...): its description, the search that found it, its facts
                 (F1, F2, ...) and the questions it answers.

EVERY TURN output exactly these four blocks, in order, with no text outside them:

Part 1 - <think>your reasoning</think>
  Read the image in view for the ACTIVE question and derive the intermediate conclusion yourself (rank,
  compare or compute the values; name the year, item or page you need next), stated explicitly.

Part 2 - <page>{"desc": ..., "facts": [...]}</page> <tree>{...}</tree> <act>{...}</act>   one JSON value each
  <page>: what the page in view gives (null with no page in view; null / [] for parts you do not change):
    "desc": short page description, or null to keep it.
    "facts": readings from the image in view (values, labels, years, units; not conclusions), each with a box:
       {"op": "add_fact", "text": "<one short statement>", "bbox": [x1, y1, x2, y2]}
       {"op": "update_fact", "fact_id": "F2", "text": "<corrected statement>", "bbox": [x1, y1, x2, y2]}
       <<BOX_RULE>>
  <tree>: what you conclude for the ACTIVE question, or null:
       {"op": "resolve", "answer": "<its answer>", "pages": ["P2"]}   add "node_id": "N1" to answer or correct N1 instead
       {"op": "expand", "answered": "<the part answered>", "answer": "<its answer>", "next": "<the rest>", "pages": ["P1"]}
          "next" is self-contained and names what you found.
       "pages": the pages whose facts the answer rests on (the page in view and earlier pages).
  <act>, exactly one:
<<SEARCH_ACTION>>
    {"type": "revisit", "page_id": "P1"}   look again at a page seen before (not the one in view), e.g. to
          re-read a value or correct a fact or an answer.
    {"type": "crop", "bbox": [x1, y1, x2, y2]}  or  {"type": "crop", "fact_id": "F2"}
          zoom into the page in view, once per visit. Do this before resolving whenever the answer or an
          intermediate value is a number, date, name, label or table cell: read it from the zoom, not from the
          full page. Titles, headings and page-level content can be read directly.
    {"type": "answer", "text": "<final answer>"}
          finish. "text" is the answer only: a word, number or short phrase, no explanation (e.g. "2011", "CHF 12.5 bn").

Rules:
- Answer only from what the retrieved pages show (your page memory and the page in view), never from outside
  knowledge or a guess. An indirect reference in the question (the company that ..., the year when ...) is
  first resolved from a page (expand), then the rest is searched.
- Every page whose facts an answer uses must be listed in its "pages" (the page in view and earlier pages alike);
  a listed page must have facts.
- Resolve or answer from a page only when you are sure it contains that answer: its names, titles, dates and
  numbers match the question ("this presentation" means one particular document, and several may look alike).
  If not sure, <<UNSURE>>

Examples (each block is one full response):

<think>Bar chart: three largest 9.7% (2001), 8.3% (2008), 7.5% (2011), so the year is 2011; zoom to confirm the labels.</think>
<page>{"desc":"Bar chart: organic growth 2001-2011","facts":[{"op":"add_fact","text":"Organic growth 2001 9.7%, 2008 8.3%, 2011 7.5%, ...","bbox":<<EX_CHART>>}]}</page>
<tree>null</tree>
<act>{"type":"crop","bbox":<<EX_CHART>>}</act>

<think>The zoom confirms the values; the year is 2011. Next: its trading operating profit.</think>
<page>{"desc":null,"facts":[]}</page>
<tree>{"op":"expand","answered":"Which year had the third largest organic growth ...?","answer":"2011","next":"Nestle's trading operating profit in 2011?","pages":["P1"]}</tree>
<act>{"type":"search","query":"Nestle 2011 trading operating profit"}</act>

<think>The 2011 results KPI table reads trading operating profit 12.5 bn; zoom to confirm the cell.</think>
<page>{"desc":"2011 full-year results, KPI table","facts":[{"op":"add_fact","text":"Trading operating profit 2011: 12.5 bn","bbox":<<EX_CROP>>}]}</page>
<tree>null</tree>
<act>{"type":"crop","bbox":<<EX_CROP>>}</act>

<think>The zoom confirms CHF 12.5 bn for 2011 (as reported); refine F1. Both parts answered.</think>
<page>{"desc":null,"facts":[{"op":"update_fact","fact_id":"F1","text":"Trading operating profit 2011: CHF 12.5 bn (as reported)","bbox":<<EX_ZOOM_FACT>>}]}</page>
<tree>{"op":"resolve","answer":"CHF 12.5 bn","pages":["P2"]}</tree>
<act>{"type":"answer","text":"CHF 12.5 bn"}</act>

<think>The zoom confirms 2011 as the third largest; P1 (read earlier) gives the 2011 profit, F2: CHF 12.5 bn. Resolve with pages P1 and P2.</think>
<page>{"desc":null,"facts":[]}</page>
<tree>{"op":"resolve","answer":"CHF 12.5 bn","pages":["P1","P2"]}</tree>
<act>{"type":"answer","text":"CHF 12.5 bn"}</act>

<<EX_MISMATCH>>

<think>The footnote holds the number of analyzed comments but its digits are too small; zoom to read them.</think>
<page>{"desc":"Social media study, bubble chart","facts":[]}</page>
<tree>null</tree>
<act>{"type":"crop","bbox":<<EX_FOOTNOTE>>}</act>
"""

_BOX_RULES = {
    "px": ("Boxes are integer ABSOLUTE PIXELS (origin top-left, x2 > x1, y2 > y1) on the image in view (the PAGE,\n"
           "       or the ZOOM when one is in view), at the size stated each turn."),
    "norm1000": ("Boxes are integer 0-1000 RELATIVE coordinates (0 = left / top edge, 1000 = right / bottom edge;\n"
                 "       x2 > x1, y2 > y1) on the image in view (the PAGE, or the ZOOM when one is in view)."),
}
# Example boxes as fractions of the image, rendered in each frame (px: a 1024x768 slide as shown).
_EXAMPLE_BOXES = {
    "<<EX_CHART>>": ([0.12, 0.25, 0.88, 0.75], (1024, 768)),
    "<<EX_CROP>>": ([0.10, 0.37, 0.68, 0.58], (1024, 768)),
    "<<EX_ZOOM_FACT>>": ([0.06, 0.27, 0.81, 0.64], (640, 220)),
    "<<EX_FOOTNOTE>>": ([0.62, 0.90, 0.96, 0.95], (1024, 768)),
}


# Search wording. The guided variant (experiment R2a, DESIGN.md §11.5) explains the cursor over the
# ranking and asks the model to rely on a page only when it is sure the page holds the answer, and
# otherwise to look at the next-ranked page; its example shows exactly that. §11.21 / §11.22 (the 83.97
# run): search with the question itself, refine only for an intermediate target.
_SEARCH_ACTION = {
    False: ('    {"type": "search", "query": "<self-contained query>"}   a new, unseen page for the active question.\n'
            "          Search with the question itself; refine the query only for a specific intermediate target\n"
            "          (a year or name found on a page)."),
    True: ('    {"type": "search", "query": "<query>"}   the best-ranked unseen page for the query. Pages come from many\n'
           "          unrelated documents mixed together. The same query again shows the next-ranked page. Search with the\n"
           "          question itself; refine the query only for a specific intermediate target (a year or name found on a page)."),
}
_UNSURE = {
    False: "keep searching, revisiting or cropping.",
    True: "search the same query again to see the next-ranked page, or add a more distinctive cue.",
}
_MISMATCH_EXAMPLE = {
    False: ('<think>A product portfolio slide, no financial figures; nothing for the active question.</think>\n'
            '<page>{"desc":"Product portfolio, no figures","facts":[]}</page>\n<tree>null</tree>\n'
            '<act>{"type":"search","query":"Nestle organic growth 2001-2011 chart"}</act>'),
    True: ('<think>A 2011 results table, but from Unilever, not Nestle. Next-ranked page for the same query.</think>\n'
           '<page>{"desc":"Unilever 2011 results table","facts":[]}</page>\n<tree>null</tree>\n'
           '<act>{"type":"search","query":"Nestle 2011 trading operating profit"}</act>'),
}

@lru_cache(maxsize=None)
def policy_system(frame: Frame, search_guidance: bool = False) -> str:
    from pathlib import Path
    policy = Path(__file__).resolve().parents[2] / "system_prompt.txt"
    return policy.read_text(encoding="utf-8")



FORMAT_RETRY_SUFFIX = """

Your previous response was invalid{error}. Respond again with exactly:
<think>...</think>
<page>{...} or null</page>
<tree>{...} or null</tree>
<act>{"type": ...}</act>
No other text."""


def format_retry_suffix(error: str | None) -> str:
    """The retry note, naming what was wrong (retries are not part of the SFT rows)."""
    return FORMAT_RETRY_SUFFIX.replace("{error}", f": {error}" if error else "")


TURN_SEP = "\n\n"   # between the observation and the memory block inside one user message


def turn_of(turn: int, max_turns: int, first_search: str) -> tuple[int, int]:
    """(k, n) for the "Turn k of n" line (DESIGN.md §11.24), shared by eval and RL. The model's own turns are
    counted from 1; with first_search="question" turn 0 is the fixed search, not a model turn. The turn with
    index max_turns - 1 is the last ordinary model decision (hard budget)."""
    start = 1 if first_search == "question" else 0
    return turn - start + 1, max_turns - start


def question_text(question: str, teacher_hint: str = "") -> str:
    """The first user message. teacher_hint (DESIGN.md §11.15): a data-generation note for the teacher only;
    SFT rows never carry it."""
    text = "=== QUESTION ===\n" + question.strip() + "\nAnswer only when every part of it is supported."
    return text + ("\nNote: " + teacher_hint.strip() if teacher_hint and teacher_hint.strip() else "")


def observation_text(m: Memory, view: View | None, frame: Frame) -> str:
    """Pixel sizes are stated only in the pixel frame (they invite a 0-1000 model to answer in pixels)."""
    def size(image: Image.Image) -> str:
        if frame.coord != "px":
            return ""
        w, h = frame.shown_size(image.size)
        return f"{w}x{h} pixels"

    if view is None:
        return "Observation: no page in view."
    page = m.pages[view.page]

    if view.zoom is not None:
        page_size, zoom_size = size(page.image), size(view.zoom.image)
        return "\n".join([f"Observation: page {page.id}" + (f" ({page_size})" if page_size else ""), "<image>",
                          f"Zoom of page {page.id}" + (f" ({zoom_size})" if zoom_size else "") + ":", "<image>"])
    how = f"new; retrieved for: {view.query}" if view.via == "search" else "revisit"
    detail = "; ".join(part for part in (how, size(page.image)) if part)
    return "\n".join([f"Observation: page {page.id} ({detail})", "<image>"])


def view_images(m: Memory, view: View | None, frame: Frame) -> list[Image.Image]:
    """The images in view, as this frame's model sees them: the page, then its zoom."""
    if view is None:
        return []
    images = [frame.show(m.pages[view.page].image)]
    if view.zoom is not None:
        images.append(frame.show(view.zoom.image))
    return images


def build_turn_messages(
    m: Memory,
    view: View | None,
    last_response: str | None,
    *,
    frame: Frame,
    search_guidance: bool = False,
    note: str = "",
    force: bool = False,
    answer_only: bool = False,
    teacher_hint: str = "",
    turn_of: tuple[int, int] | None = None,
) -> tuple[list[dict[str, str]], list[Image.Image]]:
    """One turn's context in `frame`, strictly alternating (DESIGN.md §11.17): system = the policy; user = the
    question; assistant = the previous response; user = the observation (page [+ zoom] images at the frame's size)
    followed by the memory block. Before any response the question and the memory block share one user message.
    turn_of: (k, n) for the "Turn k of n" line. Returns (messages, images)."""
    question = question_text(m.question, teacher_hint)
    state = render_state(m, view, frame=frame, note=note, force=force, answer_only=answer_only, turn_of=turn_of)
    messages = [{"role": "system", "content": policy_system(frame, search_guidance)}]
    if last_response is None:
        messages.append({"role": "user", "content": question + TURN_SEP + state})
        return messages, []
    messages += [{"role": "user", "content": question},
                 {"role": "assistant", "content": last_response.strip()},
                 {"role": "user", "content": observation_text(m, view, frame) + TURN_SEP + state}]
    return messages, view_images(m, view, frame)


def first_search_response(question: str) -> str:
    """The fixed turn-0 step when the first search is the question itself (SCORE: the initial search is
    constrained to the original query). It stands in the next context as the previous response."""
    act = Act.model_validate({"action": {"type": "search", "query": question.strip()}})
    return render_turn("First, search with the question itself.", act)


def with_retry_suffix(messages: list[dict[str, str]], suffix: str) -> list[dict[str, str]]:
    out = [dict(msg) for msg in messages]
    for msg in reversed(out):
        if msg["role"] == "user":
            msg["content"] = str(msg["content"]) + suffix
            return out
    raise ValueError("no user message to append the retry suffix to")
