"""
Follow-up questions in a chat: "When did they acquire Beta Inc?" after a question about Acme Corp.

The search finds entities by the words a question uses (graph_retrieval.named_entities), so a
pronoun gives the graph walk and PPR nothing to start from. Before a follow-up is searched:

  1. needs_rewrite(): questions that may refer back are rewritten: a pronoun or "this"/"that",
     "what about ...", a 5W1H question word ("Which algorithm ...?" after a comparison), or no
     entity named. That is most follow-ups, so most cost one small-model call.
  2. rewrite_query(): the small model (EDGE_MODEL) turns the question into one that stands on
     its own, from the last few turns. check_rewrite() accepts it only if every word it adds
     comes from those turns, so it can't bring in an entity nobody mentioned.
  3. carried_seeds(): the entities earlier turns named start the graph walk and PPR again, at
     a lower weight that halves with each older turn. They are not entities this question
     names, so the coverage check never asks for them.

History holds only questions, trimmed answers and entity keys (the graph holds the facts, and
they are loaded again by key), so prompts stay well inside Ollama's context window.
"""
import logging
import re
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from knowledge_graph import _WORD, _content_words, _word_matches
from llm_client import chat_client

logger = logging.getLogger(__name__)

MAX_TURNS = 3               # earlier turns used for rewriting and carried entities
MAX_ANSWER_CHARS = 300      # each earlier answer is cut to this in the rewrite prompt
MAX_QUESTION_CHARS = 300
MAX_CARRIED_PER_TURN = 4
CARRIED_SEED_WEIGHT = 0.5   # previous turn's entities; questions' own named entities weigh 1.0
CARRIED_DECAY = 0.5         # each older turn's entities weigh this much less
MIN_KEPT_SHARE = 0.6        # a rewrite keeps at least this share of the question's own words

# Words that point back to something said earlier
_REFERENCE = re.compile(
    r"\b(he|him|his|she|her|hers|it|its|they|them|their|theirs|this|that|these|those|"
    r"the same|the former|the latter|the company|the person|the country|the film|the book|"
    r"there|then)\b",
    re.IGNORECASE,
)
_ELLIPSIS = re.compile(r"^\s*(and|also|what about|how about|why|how so|and then)\b", re.IGNORECASE)
# 5W1H (plus "which"/"whose"/"whom"): a follow-up question often asks about something only the
# earlier turns name without a pronoun: "Which algorithm sends more messages later?" after
# "Compare PageRank and Connected Components" means one of those two
_QUESTION_WORD = re.compile(r"\b(who|whom|whose|what|which|when|where|why|how)\b", re.IGNORECASE)
_PRONOUN_WORDS = {"he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
                  "their", "theirs", "this", "that", "these", "those", "there", "then",
                  "same", "former", "latter"}


def clean_history(turns) -> List[dict]:
    """The last MAX_TURNS turns, each {question, answer, entity_keys}, trimmed (the client
    trims too, but the server doesn't rely on it)."""
    out = []
    for t in list(turns or [])[-MAX_TURNS:]:
        get = t.get if isinstance(t, dict) else (lambda k, _t=t: getattr(_t, k))
        question = (get("question") or "").strip()[:MAX_QUESTION_CHARS]
        if not question:
            continue
        out.append({
            "question": question,
            "answer": (get("answer") or "").strip()[:MAX_ANSWER_CHARS],
            "entity_keys": [k for k in list(get("entity_keys") or []) if k][:MAX_CARRIED_PER_TURN],
        })
    return out


def points_back(query: str) -> bool:
    """A pronoun or "this"/"that", or a "what about ..." opening: the question certainly
    continues the conversation, even when the rewrite fails."""
    return bool(_REFERENCE.search(query) or _ELLIPSIS.search(query))


def refers_back(query: str) -> bool:
    """points_back(), or a 5W1H question word (which may ask about something earlier turns named)."""
    return points_back(query) or bool(_QUESTION_WORD.search(query))


def needs_rewrite(query: str, history: List[dict], named_keys: List[str]) -> bool:
    """Follow-ups that may refer back, or that name no known entity, are rewritten; the model
    repeats a question that already stands on its own, and nothing is carried over then."""
    return bool(history) and (refers_back(query) or not named_keys)


def _words(text: str) -> set:
    return {w.lower().replace("’", "'") for w in _WORD.findall(text)}


# Words a rewrite may add although no earlier turn used them: they join the names it puts in
# ("Why does PageRank converge faster than Connected Components?"), they don't name anything
_GLUE_WORDS = {"than", "between", "both", "either", "neither", "whether", "versus", "compared",
               "rather", "more", "less", "does", "did", "one", "ones", "other", "two", "each",
               "which", "what", "who", "whom", "whose", "when", "where", "why", "how"}


def check_rewrite(query: str, rewrite: str, history: List[dict]) -> bool:
    """A rewrite may only add words found in the earlier turns (or joining words such as
    "than"), and must keep most of the question's own content words (so it resolves the
    reference instead of asking something else)."""
    if not rewrite or len(rewrite) > 3 * len(query) + 200:
        return False
    history_words = _words(" ".join(f"{t['question']} {t['answer']}" for t in history))
    query_words = _words(query)
    added = [w for w in _content_words(rewrite)
             if not _word_matches(w, query_words) and w not in _GLUE_WORDS]
    if any(not _word_matches(w, history_words) for w in added):
        return False
    own = [w for w in _content_words(query) if w not in _PRONOUN_WORDS]
    rewrite_words = _words(rewrite)
    kept = sum(1 for w in own if _word_matches(w, rewrite_words))
    return not own or kept / len(own) >= MIN_KEPT_SHARE


# Capitalised words that start a question or sentence rather than name something
_NOT_NAMES = {"compare", "contrast", "explain", "describe", "list", "name", "give", "tell", "show",
              "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "is", "are",
              "was", "were", "do", "does", "did", "can", "could", "should", "would", "will", "the",
              "a", "an", "in", "on", "of", "for", "and", "or", "both", "it", "they", "this", "that",
              "these", "those", "i", "yes", "no"}
_CAPITALISED_RUN = re.compile(r"\b[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*)*")


def mentioned_names(history: List[dict], limit: int = 8) -> List[str]:
    """Names in the earlier questions (newest first): capitalised phrases such as "PageRank"
    or "Connected Components", then the entity keys those turns named."""
    names, seen = [], set()

    def add(name):
        name = name.strip(" ,.;:?!\"'")
        if name and name.lower() not in seen and name.lower() not in _NOT_NAMES:
            seen.add(name.lower())
            names.append(name)

    for turn in reversed(history):
        for run in _CAPITALISED_RUN.findall(turn["question"]):
            words = run.split()
            while words and words[0].lower() in _NOT_NAMES:
                words = words[1:]
            add(" ".join(words))
        for key in turn.get("entity_keys", []):
            if not any(key.lower() in n.lower() or n.lower() in key.lower() for n in names):
                add(key)
    return names[:limit]


def _rewrite_prompt(query: str, history: List[dict]) -> str:
    turns = "\n".join(
        f"User: {t['question']}\nAssistant: {t['answer']}" for t in history
    )
    names = mentioned_names(history)
    names_line = f"\nNames used in the conversation: {', '.join(names)}\n" if names else ""
    # Small models follow what comes last: the instructions and examples come first, then the
    # conversation, then the question to rewrite
    return f"""Rewrite a follow-up question from a chat so it can be understood without the
conversation: replace pronouns and vague references ("they", "it", "this", "that company",
"which one", "which algorithm", "what about ...") with the names they refer to. When it asks
"which" of the things the conversation compared, name those things in the question. Keep
everything else in the question the same. Do not answer it and do not add facts. If it already
stands on its own, repeat it unchanged.

Examples (the wording of the follow-up is kept; only the reference is replaced):
After "User: Who founded Acme Corp?", the follow-up "When did they acquire Beta Inc?"
becomes "When did Acme Corp acquire Beta Inc?"
After "User: Compare how tea and coffee are grown.", the follow-up "Which crop needs more
rain, and why?" becomes "Which crop, tea or coffee, needs more rain, and why?"
After "User: What is a monsoon?", the follow-up "Where does this happen?" becomes
"Where does a monsoon happen?"

Conversation so far:
{turns}
{names_line}
Follow-up question: {query}

Reply with only the rewritten question."""


def _first_line(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    line = re.sub(r"^(rewritten( question)?|question)\s*:\s*", "", line, flags=re.IGNORECASE)
    return line.strip().strip('"“”').strip()


def rewrite_query(query: str, history: List[dict], llm=None) -> str:
    """The question rewritten to stand on its own, or "" when the model's rewrite was rejected
    (the caller then searches the question as asked)."""
    llm = llm or chat_client("EDGE_MODEL", temperature=0)
    response = llm.invoke([
        SystemMessage(content="You rewrite follow-up questions in a chat so they stand on their own."),
        HumanMessage(content=_rewrite_prompt(query, history)),
    ])
    rewrite = _first_line(response.content)
    if not check_rewrite(query, rewrite, history):
        logger.info("Rewrite rejected for %r: %r", query, rewrite)
        return ""
    return rewrite


def carried_seeds(history: List[dict], exclude=()) -> Dict[str, float]:
    """Entity keys from earlier turns -> start weight; the newest turn weighs most."""
    seeds, weight, skip = {}, CARRIED_SEED_WEIGHT, set(exclude)
    for turn in reversed(history):
        for key in turn.get("entity_keys", [])[:MAX_CARRIED_PER_TURN]:
            if key not in skip:
                seeds.setdefault(key, weight)
        weight *= CARRIED_DECAY
    return seeds
