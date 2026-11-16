"""Text-only G prompt from the forced-answer replay."""

PREFIX = "<answer>"
SYSTEM = 'You are finishing a document question-answering task. Earlier you searched a corpus of slide pages and wrote notes. The search budget is used up: you cannot search, zoom or revisit any more. Give your single best final answer to the ORIGINAL question from the notes below. If the notes do not fully settle it, give the most likely answer they support; never leave it empty.\nOutput only: <answer>the final answer, as short as possible</answer>'


def notes(mem):
    lines = ['Question tree:']
    for n in mem['nodes']:
        tag = n['status'] + (', ACTIVE' if n['id'] == mem.get('active') else '')
        line = f'{n['id']} [{tag}]: {n['question']}'
        if n.get('answer'):
            line += f'  = {n['answer']}'
        if n.get('pages'):
            line += f'   (pages: {', '.join(n['pages'])})'
        lines.append(line)
    lines += ['', 'Page notes:']
    for p in mem['pages']:
        lines.append(f'{p['id']}: {p.get('descriptor') or '(no description)'}   (found by: "{p.get('query')}")')
        for f in p.get('facts') or []:
            lines.append(f'  - {f['text']}')
    return '\n'.join(lines)


def build_messages(memory):
    question = memory["question"]
    user = f"Original question: {question}\n\n{notes(memory)}\n\nThe budget is used up. Answer the original question now."
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": PREFIX},
    ]
