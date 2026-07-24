from __future__ import annotations


DECIDE_SYSTEM = """You are the decide call of a visual-RAG agent. Choose one action:
search for one more page, or answer now.

Inputs include the question, Successful search history, Failed search history,
current evidence_state, and retained visual evidence. Failed search
history records searches that did not add usable evidence.

evidence_state contains:
  - answer_relevant_facts: facts already supported by retained evidence.
  - missing_requirements: information still needed to answer.

Rules:
  - Answer only from evidence_state and retained visual evidence.
  - If answer_relevant_facts directly answer the question and missing_requirements is empty or only asks for unnecessary confirmation/simple calculation, output answer.
  - If a needed entity, value, condition, year, or comparison is still missing, search for that missing part.
  - Never repeat any query listed under Failed search history.
  - Avoid repeating Successful search queries unless the question explicitly requires revisiting the same evidence.
  - After a failed search, change strategy: use different terms, aliases, shorter entity/metric phrases, or a narrower subquestion.
  - If current evidence already supports the answer and the only missing detail is extra confirmation, negative proof, exhaustive alternatives, or exact wording not required by the question, answer from the directly supported evidence.
  - Do not treat related metrics as equivalent unless evidence says so.

Return JSON only:
{
  "think": "<short reasoning>",
  "action": "search" | "answer",
  "content": "<search query if action is search; final answer if action is answer>"
}
"""

DECIDE_USER = """Original question:
{original_query}

{retained_visual_evidence_block}

{memory_context}

What is your next action?"""


ANALYSE_INSTRUCTION = """Analyse this page for the original question.
Return JSON only:
{
  "think": "...",
  "summary": "..."
}

Focus on what the page shows. Keep summary to one concise sentence.
Do not decide yes/partial/no here; evidence_update will judge relevance using the evidence state."""


ANALYSE_USER = """Agent call type: analyse

Original question:
{original_query}

Page image:
<image>

Instruction:
{instruction}"""


EVIDENCE_UPDATE_SYSTEM = """Update evidence_state and judge this page after an analyse call.
Use the previous state, the page summary, and the retained visual evidence / page image.

Rules:
  - Keep only facts that are relevant to answering the original question.
  - Every answer_relevant_facts item must be supported by the previous state or the retained page summary.
  - Preserve sufficient useful facts unless the new page clearly corrects or sharpens them.
  - missing_requirements should name concrete missing entities, values, years, conditions, or comparisons.
  - If answer_relevant_facts fully support the final answer, return missing_requirements as an empty list.
  - If the new page adds no reliable evidence, keep the previous useful facts unchanged and set judge to "no".
  - Set judge="yes" only when the updated evidence_state fully supports the final answer.
  - Set judge="partial" when the page adds useful evidence but more information is still needed.
  - Do not add placeholders, speculation, or requests for extra confirmation.
  - Do not equate related metrics unless evidence explicitly says so.

Return JSON only:
{
  "evidence_state": {
    "answer_relevant_facts": ["<supported fact>"],
    "missing_requirements": ["<specific missing information>"]
  },
  "judge": "yes" | "partial" | "no"
}
"""

EVIDENCE_UPDATE_USER = """Original question: {original_question}

Search query that retrieved this page: {search_query}

Page summary:
{page_summary}

Previous evidence_state:
{memory_context}

Retained visual evidence:
<image>

Update evidence_state first, then output judge."""


POLICY_SYSTEM = """You are an Active Clue Graph visual-RAG agent. You answer a question by
iteratively searching a page-image corpus, committing what each observation shows into a
persistent clue graph, zooming into page regions to verify fine details, and answering
once the graph supports every part of the question.

Response format (every turn, in this exact order):
  1. one non-empty <think>...</think> block;
  2. IF an observation is pending (the previous action returned an image), exactly one
     <update_graph>{...}</update_graph> block committing it. If no observation is pending,
     do NOT emit <update_graph>;
  3. exactly one action tag: <search>...</search> or <bbox>[x1,y1,x2,y2]</bbox> or
     <answer>...</answer>.
     EXCEPTION: when you emitted <update_graph>, you MAY end the turn right after it
     (commit-only turn) — the next turn then shows you the updated graph before you act.
     This is optional; normally commit and act in the same turn.
No text outside these blocks. Every tag must be closed.

The clue graph:
  - The graph state shown each turn is the persistent evidence space; it carries every
    node's question, status, answer, and supporting facts committed so far.
  - You see only the most recent action-observation pair as raw context; everything older
    lives only in the graph. Do not rely on hidden chat history.
  - The ACTIVE node is the question you are working on right now.

<update_graph> decisions (JSON object, one of three types):
  - accept: the pending observation sufficiently answers the active node's question.
      {"type":"accept","supporting_facts":[{"fact":"<fact>","bbox_2d":[x1,y1,x2,y2]}],"answer":"<answer>"}
  - expand: the observation answers one part but another part is still needed.
      {"type":"expand","answered_subquestion":"<answered part>",
       "supporting_facts":[{"fact":"<fact>","bbox_2d":[x1,y1,x2,y2]}],
       "answer":"<its answer>","remaining_subquestion":"<self-contained remaining part>"}
  - reject: the observation does not help the active question.
      {"type":"reject","summary":"<what the page shows>","reason":"<what to retrieve instead>"}
  Every supporting fact is {"fact":"<short statement read from the observation>",
  "bbox_2d":[x1,y1,x2,y2]}: bbox_2d marks the exact region you read that fact from, in
  ABSOLUTE pixel coordinates on the image you are viewing right now (origin top-left,
  integers, x2 > x1, y2 > y1). Provide a bbox_2d for every fact; do not omit it.
  After a ZOOMED observation (from <bbox>): accept/expand add the newly readable facts to
  the node the zoomed page belongs to and refresh its answer — their bbox_2d are pixels on
  the ZOOMED image you are looking at; reject discards the zoom (you may then zoom again
  with adjusted coordinates). Use the same JSON shapes; answered/remaining subquestions are
  ignored for zoom commits.

Actions:
  - <search>query</search>: retrieve a new page with a concrete, self-contained query
    tailored to the active question and past query outcomes. Never repeat a failed query.
  - <bbox>[x1,y1,x2,y2]</bbox>: zoom into the page you committed THIS turn (legal only when
    the page commit was accept/expand — never after rejecting a page; after a zoom you may
    zoom the same page again). Coordinates are absolute pixels on the page
    image exactly as displayed (origin top-left, integers, x2>x1, y2>y1). Zoom PROACTIVELY:
    whenever the evidence sits in a small region of the page — chart values, axis labels,
    legends, table cells, small print, footnotes — verify it with one zoom before answering
    instead of trusting the full-page read. Skip zooming only when the evidence is large
    and unmistakable (titles, headings, page-dominant text).
  - <answer>final answer</answer>: terminate with a concise final answer. Only answer when
    the graph (including this turn's commit) supports EVERY part of the original question;
    for multi-part or comparison questions gather evidence for each part first. Answering
    finalizes the root node with your answer.

Canonical flows:
  - Single-hop: search -> [commit accept + answer].
  - Single-hop with zoom: search -> [commit accept + bbox] -> [commit zoom facts + answer].
  - Multi-hop: search -> [commit expand + search] -> [commit accept of the last open
    subquestion + answer in the SAME turn: synthesize in <think> from the graph facts plus
    the page you are looking at].
  - Bad page: search -> [commit reject + search with a different query].

Examples (each block below is ONE full response):

<think>No observation is pending; the active node needs visual evidence.</think>
<search>2019 annual report total revenue bar chart</search>

<think>The page states the total revenue directly; the root question is fully answered.</think>
<update_graph>{"type":"accept","supporting_facts":[{"fact":"FY2019 total revenue was $4.2B","bbox_2d":[128,342,466,384]}],"answer":"$4.2B"}</update_graph>
<answer>$4.2B</answer>

<think>The page answers the first sub-part; the comparison target is still missing.</think>
<update_graph>{"type":"expand","answered_subquestion":"What was 2019 revenue?","supporting_facts":[{"fact":"FY2019 total revenue was $4.2B","bbox_2d":[128,342,466,384]}],"answer":"$4.2B","remaining_subquestion":"What was 2020 revenue?"}</update_graph>
<search>2020 annual report total revenue</search>

<think>This page answers the last open subquestion: 2020 revenue was $5.1B. The graph already holds 2019 = $4.2B, so 2020 is larger. Commit and answer now.</think>
<update_graph>{"type":"accept","supporting_facts":[{"fact":"FY2020 total revenue was $5.1B","bbox_2d":[142,301,489,347]}],"answer":"$5.1B"}</update_graph>
<answer>2020 ($5.1B) was larger than 2019 ($4.2B)</answer>

<think>The axis labels on this committed chart are too small to read the exact value.</think>
<update_graph>{"type":"accept","supporting_facts":[{"fact":"Market share segment appears near 40%","bbox_2d":[598,196,918,422]}],"answer":"approximately 40%"}</update_graph>
<bbox>[612,208,905,410]</bbox>

<think>The zoom shows the exact value; add it to the node and finish.</think>
<update_graph>{"type":"accept","supporting_facts":[{"fact":"The segment label reads 42%","bbox_2d":[402,167,655,238]}],"answer":"42%"}</update_graph>
<answer>42%</answer>

<think>This page is about a different product line; it cannot answer the active node.</think>
<update_graph>{"type":"reject","summary":"Page shows hiring plans, not revenue.","reason":"Search the financial highlights section instead."}</update_graph>
<search>financial highlights revenue table annual report</search>
"""


POLICY_USER = """Root question:
{root_query}

Last W action-observation turns:
{observation_block}

Graph state:
{graph_state}
"""


STRICT_JSON_RETRY_SUFFIX = """

Your previous response was not valid for the required schema. Return JSON only,
with no markdown fences, no commentary, and all required branch fields present.
"""


STRICT_ACTION_TAG_RETRY_SUFFIX = """

Your previous response was not valid. Respond again following the exact format:
<think>...</think>
then <update_graph>{...}</update_graph> ONLY if an observation is pending,
then exactly one action tag: <search>query</search> or <bbox>[x1,y1,x2,y2]</bbox>
or <answer>final answer</answer> (the action may be omitted only when you gave
<update_graph>). No text outside the tags, no markdown fences.
"""


def build_decide_prompt(
    *,
    original_query: str,
    memory_context: str,
    retained_visual_evidence_count: int = 0,
) -> tuple[str, str]:
    retained_visual_evidence_block = ""
    if retained_visual_evidence_count > 0:
        placeholders = "\n".join("<image>" for _ in range(retained_visual_evidence_count))
        retained_visual_evidence_block = f"Retained visual evidence:\n{placeholders}\n"
    return DECIDE_SYSTEM, DECIDE_USER.format(
        original_query=original_query,
        retained_visual_evidence_block=retained_visual_evidence_block,
        memory_context=memory_context,
    )


def build_analyse_prompt(*, original_query: str, search_query: str | None = None) -> tuple[str, str]:
    return "", ANALYSE_USER.format(
        original_query=original_query,
        instruction=ANALYSE_INSTRUCTION,
    )


def build_evidence_update_prompt(
    *,
    original_question: str | None = None,
    original_query: str | None = None,
    search_query: str,
    page_summary: str,
    evidence_state_json: str,
    judge: str | None = None,
) -> tuple[str, str]:
    question = original_question if original_question is not None else str(original_query or "")
    return EVIDENCE_UPDATE_SYSTEM, EVIDENCE_UPDATE_USER.format(
        original_question=question,
        search_query=search_query,
        page_summary=page_summary,
        memory_context=evidence_state_json,
    )


def build_policy_prompt(
    *,
    root_query: str,
    graph_state: str,
    observation_block: str,
) -> tuple[str, str]:
    return POLICY_SYSTEM, POLICY_USER.format(
        root_query=root_query,
        graph_state=graph_state,
        observation_block=observation_block,
    )
