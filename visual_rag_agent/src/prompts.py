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


POLICY_SYSTEM = """You are the policy call of an Active Clue Graph visual-RAG agent.

Action meanings:
  - SEARCH: propose a concrete retrieval query now and retrieve visual candidates with it.
  - UPDATE_GRAPH: analyze the current observation or sufficient child evidence and commit a graph JSON decision.
  - ANSWER: terminate after the root node is sufficient.

Rules:
  - Treat the graph state as the persistent evidence space; it carries every node's question, answer, and supporting facts (each with its bbox_2d) committed so far.
  - You are shown only the single most recent action and its observation image as raw recent context; everything older lives in the graph state.
  - Do not rely on hidden or earlier raw chat history.
  - A "Valid actions" list is provided beneath the graph state; choose exactly one action from that list.
  - Do not assume a fixed pipeline. Pick the action that is justified by the current state.
  - SEARCH when the active open node needs new visual evidence; include a query tailored to the active question, known facts, child answers, and query_history.
  - UPDATE_GRAPH when the current observation or sufficient child evidence should change the graph.
  - If graph facts already answer the active node, choose UPDATE_GRAPH so the graph can analyze and commit those facts.
  - ANSWER only when the root node is sufficient.
  - Use query_history to avoid repeating failed retrieval directions.
  - The active clue node does not store a query. Every SEARCH must output the query to run.

Output format:
  - Every response must have exactly two parts, in this order:
    1. one brief <think>...</think> tag explaining why this action is valid now;
    2. exactly one closed action tag.
  - The first characters of your response must be exactly: <think>
  - The next tag after </think> must be one of <search>, <update_graph>, or <answer>.
  - Do not output an action tag without a preceding <think>...</think> tag.
  - Do not put any text before <think> or after the action tag.
  - The <think> tag is required for every action, including SEARCH, UPDATE_GRAPH, and ANSWER.
  - SEARCH format: <search>retrieval query text</search>
  - UPDATE_GRAPH format: <update_graph>{"type":"accept|reject|expand", ...}</update_graph>
  - ANSWER format: <answer>concise final answer</answer>
  - Invalid examples: <update_graph>{"type":"accept"} and <search>query.
  - In UPDATE_GRAPH, every supporting_fact must be {"fact":"<fact>","bbox_2d":[x1,y1,x2,y2]}: bbox_2d are ABSOLUTE pixel coordinates on the current observation image (origin top-left, integers, x2 > x1 and y2 > y1) marking the exact region you read that fact from. Provide a bbox_2d for every fact; do not omit it.

Examples:
<think>The active node needs visual evidence.</think>
<search>specific metric name target period source chart</search>

<think>The current observation directly answers the active node.</think>
<update_graph>{"type":"accept","answer":"<answer from the observation>","supporting_facts":[{"fact":"<fact directly supported by the observation>","bbox_2d":[x1,y1,x2,y2]}]}</update_graph>

<think>The retrieved page does not contain the requested metric.</think>
<update_graph>{"type":"reject","summary":"<why the current observation is not useful>","reason":"<what evidence or retrieval direction is needed next>"}</update_graph>

<think>The page answers one subquestion but another value is still needed.</think>
<update_graph>{"type":"expand","answered_subquestion":"<subquestion answered by the observation>","answer":"<answer to that subquestion>","supporting_facts":[{"fact":"<fact directly supported by the observation>","bbox_2d":[x1,y1,x2,y2]}],"remaining_subquestion":"<self-contained subquestion still needed>"}</update_graph>

<think>The root node is sufficient.</think>
<answer>concise final answer</answer>
"""


POLICY_USER = """Root question:
{root_query}

Last W action-observation turns:
{observation_block}

Graph state:
{graph_state}
"""


POLICY_JSON_SYSTEM = """You are the policy call of an Active Clue Graph visual-RAG agent.

Action meanings:
  - search: propose a concrete retrieval query now and retrieve visual candidates with it.
  - update_graph: analyze the current observation or sufficient child evidence and commit a graph JSON decision.
  - answer: terminate after the root node is sufficient.

Rules:
  - Treat the graph state as the persistent evidence space; it carries every node's question, answer, and supporting facts (each with its bbox_2d) committed so far.
  - You are shown only the single most recent action and its observation image as raw recent context; everything older lives in the graph state.
  - Do not rely on hidden or earlier raw chat history.
  - A "Valid actions" list is provided beneath the graph state; choose exactly one action from that list.
  - Do not assume a fixed pipeline. Pick the action that is justified by the current state.
  - search when the active open node needs new visual evidence; include a query tailored to the active question, known facts, child answers, and query_history.
  - update_graph when the current observation or sufficient child evidence should change the graph.
  - If graph facts already answer the active node, choose update_graph so the graph can analyze and commit those facts.
  - answer only when the root node is sufficient.
  - Use query_history to avoid repeating failed retrieval directions.
  - The active clue node does not store a query. Every search must output the query to run.
  - In UPDATE_GRAPH, every supporting_fact must be {"fact":"<fact>","bbox_2d":[x1,y1,x2,y2]}: bbox_2d are ABSOLUTE pixel coordinates on the current observation image (origin top-left, integers, x2 > x1 and y2 > y1) marking the exact region you read that fact from. Provide a bbox_2d for every fact; do not omit it.

Return JSON only, with no markdown fences and no extra text:
{
  "think": "<brief reasoning for why this action is valid now>",
  "action": "search" | "update_graph" | "answer",
  "payload": { ... }
}

Payload schemas:
  - search: {"query":"<retrieval query text>"}
  - update_graph accept: {"type":"accept","answer":"<answer from the observation>","supporting_facts":[{"fact":"<fact directly supported by the observation>","bbox_2d":[x1,y1,x2,y2]}]}
  - update_graph reject: {"type":"reject","summary":"<why the current observation is not useful>","reason":"<what evidence or retrieval direction is needed next>"}
  - update_graph expand: {"type":"expand","answered_subquestion":"<subquestion answered by the observation>","answer":"<answer to that subquestion>","supporting_facts":[{"fact":"<fact directly supported by the observation>","bbox_2d":[x1,y1,x2,y2]}],"remaining_subquestion":"<self-contained subquestion still needed>"}
  - answer: {"answer":"final"}

The think field is required and must be non-empty for every action.
Do not output VimRAG tags in this JSON mode. The runner will convert the JSON action into VimRAG-style tags for SFT.
"""


POLICY_JSON_USER = """Root question:
{root_query}

Last W action-observation turns:
{observation_block}

Graph state:
{graph_state}
"""


UPDATE_GRAPH_SYSTEM = """You are the UPDATE_GRAPH call of an Active Clue Graph visual-RAG agent.
Return exactly one graph decision JSON: accept, reject, or expand.

Important state rules:
  - Treat the graph state as the persistent evidence space and the current visual observation as the only raw recent observation.
  - Do not rely on hidden or earlier raw chat history.
  - active_node.question is immutable. Never rewrite it.
  - active clue nodes do not store retrieval queries. SEARCH proposes queries at action time.
  - SEARCH did not change the graph. This call is the only place to commit facts, evidence, answers, child nodes, and query-history outcomes.
  - query_history is retrieval-control metadata, not final-answer evidence.
  - known_facts are reliable reasoning evidence and must only be written from supporting_facts.

Decision rules:
  - accept when the active node's question is sufficiently answered. This marks the active node sufficient.
  - reject when evidence is weak, irrelevant, unreadable, insufficient, or the next SEARCH should try a different retrieval direction. Output only summary and reason for the failure; do not output facts or answer.
  - expand when the observation answers one subquestion but the active node still needs a remaining subquestion. This creates a sufficient answered-subquestion child and an open remaining-subquestion child.
  - Do not expand if current known_facts, parent context, sufficient child answers, or the current observation already answer the active node. Use accept instead.

Return JSON only:
{
  "type": "accept" | "reject" | "expand",
  "answer": "<for accept>",
  "summary": "<required for reject; concise observation summary otherwise>",
  "reason": "<required for reject; optional rationale otherwise>",
  "supporting_facts": [{"fact":"<fact text>","bbox_2d":[x1,y1,x2,y2]}],
  "answered_subquestion": "<for expand: the subquestion answered by current evidence>",
  "answer": "<for expand: answer to answered_subquestion>",
  "remaining_subquestion": "<for expand: the next subquestion to solve>"
}
For expand, answer is written as the answer of the sufficient answered_subquestion child.
Do not output answered_answer, answered_facts, node_id, fact_type, value, or any fields outside this schema.
"""


UPDATE_GRAPH_USER = """Root question:
{root_query}

Graph state:
{graph_state}

{observation_block}

Recent query history for the active node:
{query_history}

Sufficient child answers for the active node:
{child_answers}

Return the graph decision."""


STRICT_JSON_RETRY_SUFFIX = """

Your previous response was not valid for the required schema. Return JSON only,
with no markdown fences, no commentary, and all required branch fields present.
"""


STRICT_ACTION_TAG_RETRY_SUFFIX = """

Your previous response was not valid. Return only VimRAG-style tags with no
markdown fences and no extra commentary:
<think>...</think>
<search>query text</search>
or <update_graph>{"type":"accept|reject|expand", ...}</update_graph>
or <answer>concise final answer</answer>
"""


STRICT_POLICY_JSON_RETRY_SUFFIX = """

Your previous response was not valid. Return JSON only, with no markdown fences
and no extra commentary. Required shape:
{
  "think": "<non-empty brief reasoning>",
  "action": "search" | "update_graph" | "answer",
  "payload": { ... }
}
The action must be justified by the current observation and graph state.
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
    valid_actions: list[str],
    output_format: str = "tags",
) -> tuple[str, str]:
    if output_format == "json":
        return POLICY_JSON_SYSTEM, POLICY_JSON_USER.format(
            root_query=root_query,
            graph_state=graph_state,
            observation_block=observation_block,
            valid_actions=valid_actions,
        )
    if output_format != "tags":
        raise ValueError(f"unsupported policy output_format: {output_format}")
    return POLICY_SYSTEM, POLICY_USER.format(
        root_query=root_query,
        graph_state=graph_state,
        observation_block=observation_block,
        valid_actions=valid_actions,
    )


def build_update_graph_prompt(
    *,
    root_query: str,
    graph_state: str,
    observation_block: str,
    query_history: list[dict[str, str]],
    child_answers: list[dict[str, object]],
) -> tuple[str, str]:
    return UPDATE_GRAPH_SYSTEM, UPDATE_GRAPH_USER.format(
        root_query=root_query,
        graph_state=graph_state,
        observation_block=observation_block,
        query_history=query_history,
        child_answers=child_answers,
    )
