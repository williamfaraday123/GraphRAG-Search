"""
Follow-up questions in a chat: "When did they acquire Beta Inc?" after a question about Acme Corp.

The search finds entities by the words a question uses (graph_retrieval.named_entities), so a
pronoun gives the graph walk and PPR nothing to start from. Before a follow-up is searched:

  1. needs_rewrite(): questions that may refer back are rewritten: a pronoun or "this"/"that",
     "what about ...", a 5W1H question word ("Which algorithm ...?" after a comparison), or no
     entity named. That is most follow-ups, so most cost one small-model call.
  2. rewrite_query(): the small model (EDGE_MODEL) turns the question into one that stands on
     its own, from the last few turns. check_rewrite() accepts it only if every word it adds
     comes from those turns, so it can't bring in an entity nobody mentioned. A "which ..."
     question the model leaves unresolved gets the compared names put in by name_choices().
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


_WORD_IN_RUN = re.compile(r"\S+")
# What joins two names in a list: "A and B", "A, B or C", "A vs B", "A and the B"
_JOINER = re.compile(r"^\s*(?:,\s*(?:and|or)?|and|or|vs\.?|versus|&)\s*(?:the\s+|a\s+|an\s+)?$",
                     re.IGNORECASE)
MAX_COMPARED = 4


def _name_spans(text: str, keys=()) -> List[tuple]:
    """(start, end, name) for each name in text, in order: the entity keys as the text spells
    them (so lowercase names such as "tea" count too), then capitalised phrases ("PageRank",
    "Connected Components") outside those, without a leading "Compare"/"In"/... ."""
    spans = []

    def overlaps(s, e):
        return any(s < e2 and s2 < e for s2, e2, _ in spans)

    for key in keys:
        m = re.search(r"(?<![\w-])" + re.escape(key) + r"(?![\w-])", text, re.IGNORECASE)
        if m and not overlaps(m.start(), m.end()):
            spans.append((m.start(), m.end(), m.group(0)))
    for run in _CAPITALISED_RUN.finditer(text):
        words = [w for w in _WORD_IN_RUN.finditer(run.group(0))]
        while words and words[0].group(0).lower() in _NOT_NAMES:
            words = words[1:]
        if not words:
            continue
        start, end = run.start() + words[0].start(), run.end()
        name = re.sub(r"['’]s$", "", text[start:end])
        end = start + len(name)
        if name and not overlaps(start, end):
            spans.append((start, end, name))
    spans.sort()
    seen, out = set(), []
    for s, e, n in spans:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append((s, e, n))
    return out


def _joined_group(text: str, keys=()) -> List[str]:
    """The longest list of names joined by "and"/"or"/"vs"/commas in text: "PageRank and
    Connected Components" in "Compare how PageRank and Connected Components use ...", but not
    "Google" in "Compare how Spark and Hadoop handle failures at Google"."""
    spans = _name_spans(text, keys)
    best, group = [], spans[:1]
    for prev, cur in zip(spans, spans[1:]):
        if _JOINER.match(text[prev[1]:cur[0]]):
            group.append(cur)
        else:
            group = [cur]
        if len(group) > len(best):
            best = list(group)
    return [n for _, _, n in best] if len(best) >= 2 else []


def _question_names(turn: dict) -> List[str]:
    return [n for _, _, n in _name_spans(turn["question"], turn.get("entity_keys", []))]


def mentioned_names(history: List[dict], limit: int = 8) -> List[str]:
    """Names used in the earlier turns (newest first): those in the questions (entity keys and
    capitalised phrases), then lists of names in the answers ("PageRank, Connected Components
    and Shortest Paths" after "Which algorithms does Pregel run?")."""
    names, seen = [], set()
    for turn in reversed(history):
        for name in _question_names(turn) + _joined_group(turn.get("answer", "")):
            if name.lower() not in seen:
                seen.add(name.lower())
                names.append(name)
    return names[:limit]


def _compared_names(history: List[dict]) -> List[str]:
    """What a "which ..." follow-up chooses between, newest turn first: the names a question
    listed together ("Compare PageRank and Connected Components"), else a list in that turn's
    answer, else a question's names when it named two to four things."""
    for turn in reversed(history):
        group = (_joined_group(turn["question"], turn.get("entity_keys", []))
                 or _joined_group(turn.get("answer", "")))
        if group:
            return group[:MAX_COMPARED]
    for turn in reversed(history):
        names = _question_names(turn)
        if 2 <= len(names) <= MAX_COMPARED:
            return names
    return []


# A pronoun or bare "this"/"that" a single earlier name can replace: "Why does it converge?",
# "How is their state stored?", "Where does this happen?"
_REPLACEABLE = re.compile(
    r"\b(?:(?P<poss>its|their)|(?P<pron>it|they|them)|"
    r"(?P<dem>this|that)(?=\s+(?:is|was|does|did|do|has|have|happen\w*|work\w*|mean\w*|use\w*)\b|\s*[?.!]?\s*$))\b",
    re.IGNORECASE,
)
# "it" that refers to nothing: "Is it possible to ...", "it's hard to ...", "how long does it take"
_EXPLETIVE_IT = re.compile(
    r"\b(?:(?:is|was|isn't|wasn't)\s+it|it(?:'s|’s|\s+is|\s+was))\s+(?:\w+\s+){0,2}(?:to|that)\b"
    r"|\bit\s+takes?\b|\bdoes\s+it\s+take\b",
    re.IGNORECASE,
)


def resolve_pronoun(query: str, history: List[dict]) -> str:
    """When the previous question named exactly one thing, its first pronoun replaced by that
    name: "Why does it converge?" after "How does PageRank work?" -> "Why does PageRank
    converge?". "" when that doesn't apply (no single name, already named, expletive "it")."""
    if not history:
        return ""
    names = _question_names(history[-1])
    if len(names) != 1:
        return ""
    name = names[0]
    query_words = _words(query)
    if all(_word_matches(w, query_words) for w in _content_words(name) or [name.lower()]):
        return ""
    m = _REPLACEABLE.search(query)
    if not m or (m.group("pron") and m.group(0).lower() == "it" and _EXPLETIVE_IT.search(query)):
        return ""
    replacement = f"{name}'s" if m.group("poss") else name
    return query[:m.start()] + replacement + query[m.end():]


# "Which algorithm ...", "Which one ...", "Which of them ...", "Which is ..."
_WHICH = re.compile(
    r"^\s*which\s+(?:(?P<pick>(?:one\s+)?of\s+(?:them|these|those|the\s+two|the\s+three|both)|one)"
    r"|(?P<verb>is|are|was|were|does|do|did|has|have|had|can|could|would|will|should|might|may)\b"
    r"|(?P<noun>[\w\-]+))",
    re.IGNORECASE,
)


def name_choices(query: str, history: List[dict]) -> str:
    """A "which ..." question that names none of the things the previous question compared,
    with those things put in: "Which algorithm, PageRank or Connected Components, is ...?",
    "Which of PageRank and Connected Components is ...?". "" when it doesn't apply.

    Small models often repeat such a question unchanged, which leaves the search nothing to
    start from, so this doesn't rely on the model."""
    m = _WHICH.match(query)
    if not m:
        return ""
    names = _compared_names(history)
    if len(names) < 2:
        return ""
    query_words = _words(query)
    if any(all(_word_matches(w, query_words) for w in _content_words(n)) for n in names
           if _content_words(n)):
        return ""  # it already names one of them
    rest = query[m.end():]
    which = query[m.start():m.end()].lstrip()[:5]  # "Which" with the question's own case
    if m.group("noun"):
        return f"{query[m.start():m.end()].lstrip()}, {', '.join(names[:-1])} or {names[-1]},{rest}"
    verb = f" {m.group('verb')}" if m.group("verb") else ""
    return f"{which} of {', '.join(names[:-1])} and {names[-1]}{verb}{rest}"


def _rewrite_prompt(query: str, history: List[dict], strict: bool = False) -> str:
    turns = "\n".join(
        f"User: {t['question']}\nAssistant: {t['answer']}" for t in history
    )
    names = mentioned_names(history)
    names_line = f"\nNames used in the conversation: {', '.join(names)}\n" if names else ""
    # Second try after the model repeated a question that clearly refers back
    must = ("\nThis follow-up refers to something named earlier in the conversation. Your "
            "reply must replace that reference with the name"
            + (f" (one of: {', '.join(names)})" if names else "")
            + ". Do not repeat the question unchanged.\n") if strict else ""
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
{must}
Reply with only the rewritten question."""


def _first_line(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    line = re.sub(r"^(rewritten( question)?|question)\s*:\s*", "", line, flags=re.IGNORECASE)
    return line.strip().strip('"“”').strip()


def _ask_model(llm, query: str, history: List[dict], strict: bool = False) -> str:
    response = llm.invoke([
        SystemMessage(content="You rewrite follow-up questions in a chat so they stand on their own."),
        HumanMessage(content=_rewrite_prompt(query, history, strict)),
    ])
    rewrite = _first_line(response.content)
    if not check_rewrite(query, rewrite, history):
        logger.info("Rewrite rejected for %r: %r", query, rewrite)
        return ""
    return rewrite


def _unchanged(rewrite: str, query: str) -> bool:
    return not rewrite or rewrite.strip().lower() == query.strip().lower()


def _fill_in(query: str, rewrite: str, history: List[dict]) -> str:
    """What the model left unresolved, filled in without it: a "which ..." question gets the
    compared names, a pronoun gets the previous question's single name."""
    base = rewrite or query
    named = name_choices(base, history)
    if named:
        logger.info("Named the compared things in %r: %r", base, named)
        return named
    if _unchanged(rewrite, query):
        resolved = resolve_pronoun(query, history)
        if resolved:
            logger.info("Replaced the pronoun in %r: %r", query, resolved)
            return resolved
    return rewrite


def rewrite_query(query: str, history: List[dict], llm=None) -> str:
    """The question rewritten to stand on its own. "" when the model's rewrite was rejected,
    or when the question points back ("it", "they") and nothing resolved it (the caller then
    searches the question as asked, with the earlier turns' entities carried over)."""
    llm = llm or chat_client("EDGE_MODEL", temperature=0)
    rewrite = _fill_in(query, _ask_model(llm, query, history), history)
    if _unchanged(rewrite, query) and points_back(query):
        # The model repeated a question that clearly refers back: ask once more, firmly
        logger.info("Rewrite left %r unresolved; asking again", query)
        rewrite = _fill_in(query, _ask_model(llm, query, history, strict=True), history)
    if _unchanged(rewrite, query) and points_back(query):
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
