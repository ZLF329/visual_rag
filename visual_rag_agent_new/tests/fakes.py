"""Scripted VLM / deterministic retriever and response builders for the mechanical test suites."""
import json

from PIL import Image


def act(page=None, action=None, think="t", tree=None):
    """One response in the four-block format (DESIGN.md §11.24): <think>, <page> (page edit or null),
    <tree> (tree edit or null), <act> (the action)."""
    return blocks(think=think, page=page, tree=tree, act=action or SEARCH("q"))


def blocks(think="t", **parts):
    """A response with exactly the given blocks, in order: think, then each of page / tree / act passed (a value is
    written as JSON, a str as it is)."""
    out = f"<think>{think}</think>" if think is not None else ""
    for tag in ("page", "tree", "act"):
        if tag in parts:
            v = parts[tag]
            out += f"\n<{tag}>{v if isinstance(v, str) else json.dumps(v)}</{tag}>"
    return out


def page(desc=None, facts=()):
    return {"desc": desc, "facts": list(facts)}


def ADD(text, bbox):
    return {"op": "add_fact", "text": text, "bbox": bbox}


def UPD_FACT(fact_id, text, bbox):
    return {"op": "update_fact", "fact_id": fact_id, "text": text, "bbox": bbox}


def RESOLVE(answer, pages, node_id=None):
    out = {"op": "resolve", "answer": answer}
    if node_id is not None:
        out["node_id"] = node_id
    out["pages"] = pages
    return out


def EXPAND(answered, answer, nxt, pages):
    return {"op": "expand", "answered": answered, "answer": answer, "next": nxt, "pages": pages}


def SEARCH(q):
    return {"type": "search", "query": q}


def REVISIT(page_id):
    return {"type": "revisit", "page_id": page_id}


def CROP(bbox=None, fact_id=None):
    out = {"type": "crop"}
    if bbox is not None:
        out["bbox"] = bbox
    if fact_id is not None:
        out["fact_id"] = fact_id
    return out


def ANSWER(text):
    return {"type": "answer", "text": text}


class FakeVLM:
    """Returns scripted responses in order; records every (messages, images) call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages, images, **_):
        self.calls.append((messages, list(images)))
        if not self.responses:
            raise AssertionError("FakeVLM ran out of scripted responses")
        return self.responses.pop(0)


class FakeRetriever:
    """Deterministic ranking by query keyword; pages are solid images of the given size.
    Records the deck_name of every search (None = flat corpus)."""

    def __init__(self, size=(800, 600), table=None):
        self.size = size
        self.n = 0
        self.table = table or {}
        self.deck_names = []

    def search(self, query, top_k=1, deck_name=None):
        self.n += 1
        self.deck_names.append(deck_name)
        for key, pages in self.table.items():
            if key in query:
                labels = pages
                break
        else:
            labels = [f"deck/page_{50 + self.n:02d}", f"deck/page_{60 + self.n:02d}"]
        return [(Image.new("RGB", self.size, "gray"), lbl) for lbl in labels[:max(2, top_k)]]
