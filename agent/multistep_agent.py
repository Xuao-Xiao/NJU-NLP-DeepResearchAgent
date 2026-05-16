import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dataset_utils import load_jsonl
from .tools import build_searcher, get_agent_tool_specs_and_registry
from .vllm_client import VLLMClient


ACTION_DECISION_SYSTEM_PROMPT = """You are a Deep Research Agent working over a fixed offline document corpus.

You must plan the next action using only the provided state and evidence.

Available actions:
- search: use a rewritten query to search for better evidence
- get_document: open one promising document by docid for verification
- finish: stop tool use because current evidence is enough or no better next step exists

Rules:
- Do not answer the user's question directly in this step.
- Do not output chain-of-thought or <think>.
- Output exactly one JSON object and nothing else.
- Prefer get_document when promising docids are already available.
- Prefer search when the current search results are weak or missing key entities.
- Avoid repeating a previous query or reopening a document already opened.

Use one of these JSON formats exactly:
{"action":"search","query":"...","reason":"..."}
{"action":"get_document","docid":"...","reason":"..."}
{"action":"finish","answer_hint":"...","reason":"..."}
"""


QUESTION_DECOMPOSITION_SYSTEM_PROMPT = """You are preparing a search plan for a Deep Research Agent over an offline corpus.

Analyze the question and output exactly one JSON object with this schema:
{"answer_type":"person|company|title|year|percentage|date|place|organization|other","primary_query":"...","bridge_query":"...","verification_query":"...","keywords":["...","..."]}

Rules:
- primary_query should identify the main entity or source document using the rarest clues.
- bridge_query should be a second search query that links the identified entity to the final fact.
- verification_query should directly target the requested answer field.
- Keep each query under 18 words.
- keywords should contain 3-6 short, high-signal clue phrases.
- Do not output chain-of-thought, markdown, or any extra text.
"""


FINAL_ANSWER_SYSTEM_PROMPT = """You are a Deep Research Agent.

Use only the evidence provided to produce the final answer.

Rules:
- Do not output chain-of-thought tags such as <think>.
- Do not mention hidden reasoning.
- If evidence is incomplete, still provide the best supported answer instead of leaving the answer blank.
- Keep the explanation brief and evidence-based.
- Exact Answer must be a short answer string, not a paragraph.
- Never use placeholder answers such as None, Given, Unknown, or N/A.

Reply in exactly this format:
Explanation: <2-4 short sentences>
Exact Answer: <final answer>
Confidence: <0-100%>
"""


FINAL_ANSWER_REPAIR_SYSTEM_PROMPT = """You are repairing the final answer format for a Deep Research Agent.

Rules:
- Output exactly one line.
- Use this format only: Exact Answer: <short answer>
- Do not output explanation, confidence, chain-of-thought, None, Given, Unknown, or N/A.
- If the previous draft is messy, extract the most plausible short final answer from it and the evidence.
"""


def _normalize_query(query: str) -> str:
    lowered = re.sub(r"\s+", " ", query.strip().lower())
    return lowered


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_exact_answer(answer_text: str) -> str:
    cleaned = _clean_text(answer_text)
    match = re.search(r"Exact Answer:\s*(.+)", cleaned, flags=re.IGNORECASE)
    if match:
        line = match.group(1).strip()
        if "\n" in line:
            line = line.splitlines()[0].strip()
        return line

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def _dedupe_keep_order(items: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for item in items:
        normalized = _normalize_query(item)
        if not item or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped


def _infer_expected_answer_type(question: str) -> str:
    lowered = question.lower()
    if "what year" in lowered or "which year" in lowered or "what year did" in lowered:
        return "year"
    if "what percentage" in lowered or "percentage decrease" in lowered or "%" in lowered:
        return "percentage"
    if "first and last name" in lowered or "what was this person's name" in lowered:
        return "person"
    if "name and title" in lowered or "title upon accession" in lowered:
        return "person"
    if "identify the company" in lowered or "what company" in lowered:
        return "company"
    if "title of the first chapter" in lowered or "provide the title" in lowered or "title of the book" in lowered:
        return "title"
    if "what date" in lowered or "on what date" in lowered:
        return "date"
    if "what place" in lowered or "which city" in lowered or "which country" in lowered:
        return "place"
    return "other"


def _normalize_answer_to_type(answer_text: str, expected_type: str) -> str:
    cleaned = _clean_text(answer_text).strip()
    cleaned = re.sub(r"^(exact answer|answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip(" .,:;\"'")
    if not cleaned:
        return ""
    if expected_type == "year":
        match = re.search(r"\b(17|18|19|20)\d{2}\b", cleaned)
        return match.group(0) if match else cleaned
    if expected_type == "percentage":
        match = re.search(r"\b\d{1,3}(?:\.\d+)?\s*%", cleaned)
        if match:
            return match.group(0).replace(" ", "")
        match = re.search(r"\b\d{1,3}(?:\.\d+)?\b", cleaned)
        if match:
            return f"{match.group(0)}%"
        return cleaned
    if expected_type in {"person", "company", "organization", "place", "title"}:
        if "\n" in cleaned:
            cleaned = cleaned.splitlines()[0].strip()
        cleaned = re.sub(r"^(the answer is|it is|this is)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        sentence = re.split(r"(?<=[.!?])\s+", cleaned)[0].strip()
        return sentence.strip(" .,:;\"'")
    return cleaned


def _is_placeholder_answer(answer_text: str) -> bool:
    normalized = answer_text.strip().lower().strip(".")
    bad_values = {
        "",
        "none",
        "given",
        "unknown",
        "n/a",
        "na",
        "unable to determine",
        "cannot be determined",
    }
    if normalized in bad_values:
        return True
    if len(answer_text) > 140:
        return True
    if normalized.startswith("wait,") or normalized.startswith("okay,"):
        return True
    if normalized.startswith("alternatively") or normalized.startswith("looking at the evidence"):
        return True
    return False


def _safe_json_loads(raw_text: Any) -> Dict[str, Any]:
    if isinstance(raw_text, dict):
        return raw_text
    if not isinstance(raw_text, str):
        return {}
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    cleaned = _clean_text(raw_text).strip()
    parsed = _safe_json_loads(cleaned)
    if parsed:
        return parsed

    candidates = re.findall(r"\{.*?\}", cleaned, flags=re.DOTALL)
    for candidate in candidates:
        parsed = _safe_json_loads(candidate)
        if parsed:
            return parsed
    return {}


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _tokenize_focus_text(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", text.lower())
    stopwords = {
        "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "from",
        "with", "that", "this", "was", "were", "is", "are", "be", "by", "as",
        "at", "it", "who", "what", "which", "when", "where", "their", "they",
        "them", "into", "than", "then", "also", "about", "after", "before",
        "between", "late", "early", "during", "over", "under", "more", "less",
    }
    result: List[str] = []
    for token in tokens:
        if len(token) < 3 or token in stopwords:
            continue
        if token not in result:
            result.append(token)
    return result


def _build_fallback_question_plan(question: str) -> Dict[str, Any]:
    expected_type = _infer_expected_answer_type(question)
    heuristic = _heuristic_query_from_question(question)
    focus_suffix = _extract_focus_suffix(question)
    keywords = _tokenize_focus_text(question)[:6]
    bridge_query = ""
    verification_query = ""
    if focus_suffix:
        verification_query = _truncate_text(focus_suffix, 120)
    if any(token.isdigit() for token in re.findall(r"\b\d{4}\b", question)):
        years = " ".join(re.findall(r"\b\d{4}\b", question)[:3])
        bridge_query = _truncate_text(f"{heuristic} {years}", 160)
    return {
        "answer_type": expected_type,
        "primary_query": heuristic[:180],
        "bridge_query": bridge_query,
        "verification_query": verification_query,
        "keywords": keywords,
    }


def _plan_question(question: str, client: VLLMClient, model_name: str) -> Dict[str, Any]:
    fallback = _build_fallback_question_plan(question)
    messages = [
        {"role": "system", "content": QUESTION_DECOMPOSITION_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        response = client.simple_chat(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )
        raw_content = str(response["choices"][0]["message"].get("content", "") or "")
        parsed = _extract_json_object(raw_content)
    except Exception:
        parsed = {}

    answer_type = str(parsed.get("answer_type", "")).strip().lower() or fallback["answer_type"]
    allowed_types = {"person", "company", "title", "year", "percentage", "date", "place", "organization", "other"}
    if answer_type not in allowed_types:
        answer_type = fallback["answer_type"]

    queries = []
    for key in ("primary_query", "bridge_query", "verification_query"):
        value = _sanitize_search_query(str(parsed.get(key, "")).strip())
        if value and not _is_bad_search_query(value):
            queries.append(value[:180])
        else:
            queries.append(str(fallback.get(key, ""))[:180])

    keywords = parsed.get("keywords", fallback["keywords"])
    if not isinstance(keywords, list):
        keywords = fallback["keywords"]
    keywords = [str(item).strip() for item in keywords if str(item).strip()]
    if not keywords:
        keywords = fallback["keywords"]

    return {
        "answer_type": answer_type,
        "primary_query": queries[0],
        "bridge_query": queries[1],
        "verification_query": queries[2],
        "keywords": keywords[:6],
    }


def _extract_title_from_text(text: str) -> str:
    match = re.search(r"title:\s*(.+)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _score_search_result(item: Dict[str, Any], focus_text: str) -> float:
    haystack = f"{item.get('docid','')} {_extract_title_from_text(str(item.get('snippet','')))} {item.get('snippet','')}".lower()
    tokens = _tokenize_focus_text(focus_text)[:16]
    overlap = sum(1 for token in tokens if token in haystack)
    score = overlap * 5.0 + float(item.get("score", 0.0))

    penalty_terms = [
        "wikipedia",
        "archives",
        "finding aid",
        "class notes",
        "faculty",
        "curriculum vitae",
        "obituaries",
        "thank you",
    ]
    for term in penalty_terms:
        if term in haystack:
            score -= 8.0

    bonus_terms = ["chapter", "contents", "annual report", "acknowledg", "biography", "book"]
    for term in bonus_terms:
        if term in haystack:
            score += 3.0
    return score


def _rank_search_results(results: List[Dict[str, Any]], focus_text: str) -> List[Dict[str, Any]]:
    return sorted(results, key=lambda item: _score_search_result(item, focus_text), reverse=True)


def _extract_relevant_passages(text: str, focus_text: str, max_chars: int, window: int = 320) -> str:
    plain = text.replace("\r", "")
    title = _extract_title_from_text(plain)
    tokens = _tokenize_focus_text(focus_text)[:12]
    lowered = plain.lower()
    snippets: List[str] = []
    seen_spans = set()

    for token in tokens:
        start = 0
        found = 0
        while found < 2:
            idx = lowered.find(token, start)
            if idx == -1:
                break
            left = max(0, idx - window)
            right = min(len(plain), idx + window)
            span_key = (left // 80, right // 80)
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                snippet = plain[left:right].strip().replace("\n", " ")
                snippets.append(snippet)
                found += 1
            start = idx + len(token)
            if sum(len(s) for s in snippets) >= max_chars:
                break
        if sum(len(s) for s in snippets) >= max_chars:
            break

    if not snippets:
        snippets = [plain[:max_chars]]

    body = "\n\n".join(_truncate_text(s, min(max_chars, 700)) for s in snippets[:4])
    if title:
        body = f"title: {title}\n\n{body}"
    return _truncate_text(body, max_chars)


def _tool_result_preview(tool_name: str, tool_result: Any, max_chars: int, focus_text: str = "") -> Tuple[str, Dict[str, Any]]:
    if tool_name == "search" and isinstance(tool_result, list):
        ranked = _rank_search_results(tool_result, focus_text or "")
        trimmed = [
            {
                "docid": item.get("docid", ""),
                "score": item.get("score", 0.0),
                "snippet": _truncate_text(str(item.get("snippet", "")), max_chars),
                "url": item.get("url", ""),
            }
            for item in ranked
        ]
        return json.dumps(trimmed, ensure_ascii=False), {"docids": [item["docid"] for item in trimmed]}

    if tool_name == "get_document" and isinstance(tool_result, dict):
        trimmed = {
            "docid": tool_result.get("docid", ""),
            "url": tool_result.get("url", ""),
            "text": _extract_relevant_passages(str(tool_result.get("text", "")), focus_text=focus_text, max_chars=max_chars),
        }
        return json.dumps(trimmed, ensure_ascii=False), {"docids": [trimmed["docid"]]}

    return json.dumps(tool_result, ensure_ascii=False), {"docids": []}


def _summarize_search_result(tool_result: Any, max_items: int = 3) -> List[str]:
    if not isinstance(tool_result, list):
        return []
    notes = []
    for item in tool_result[:max_items]:
        docid = str(item.get("docid", ""))
        snippet = str(item.get("snippet", "")).replace("\n", " ").strip()
        notes.append(f"search hit docid={docid}: {snippet[:180]}")
    return notes


def _summarize_document_result(tool_result: Any) -> List[str]:
    if not isinstance(tool_result, dict):
        return []
    docid = str(tool_result.get("docid", ""))
    text = str(tool_result.get("text", "")).replace("\n", " ").strip()
    if not text:
        return [f"opened docid={docid}: empty or missing text"]
    return [f"opened docid={docid}: {text[:240]}"]


def _build_state_summary(state: Dict[str, Any]) -> str:
    def numbered(items: List[str]) -> str:
        if not items:
            return "None"
        return "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(items))

    search_history = state["search_history"][-4:]
    opened_docids = state["opened_docids"][-4:]
    confirmed_facts = state["confirmed_facts"][-6:]
    pending_subquestions = state["pending_subquestions"][-4:]
    candidate_answers = state["candidate_answers"][-3:]
    question_plan = state.get("question_plan", {})
    plan_queries = [
        query
        for query in [
            str(question_plan.get("primary_query", "")).strip(),
            str(question_plan.get("bridge_query", "")).strip(),
            str(question_plan.get("verification_query", "")).strip(),
        ]
        if query
    ]

    return "\n".join(
        [
            f"Question: {state['question']}",
            f"Expected answer type: {question_plan.get('answer_type', 'other')}",
            f"Planned clue queries: {' | '.join(plan_queries[:3]) if plan_queries else 'None'}",
            "",
            "Known facts:",
            numbered(confirmed_facts),
            "",
            "Open questions:",
            numbered(pending_subquestions),
            "",
            "Searches already tried:",
            numbered(search_history),
            "",
            "Opened documents:",
            numbered(opened_docids),
            "",
            "Current candidate answers:",
            numbered(candidate_answers),
            "",
            f"Last action: {state['last_action'] or 'None'}",
            f"Stall count: {state['stall_count']}",
        ]
    )


def _build_round_user_prompt(
    question: str,
    state_summary: str,
    recent_observation: str,
    max_rounds: int,
    round_id: int,
) -> str:
    return "\n".join(
        [
            f"Round {round_id}/{max_rounds}",
            "",
            f"Original question:\n{question}",
            "",
            "Current state summary:",
            state_summary,
            "",
            "Most recent observation:",
            recent_observation or "None",
            "",
            "Decide the single best next step.",
            "Return only one JSON object.",
            "Do not answer the question yet.",
        ]
    )


def _build_final_user_prompt(question: str, state: Dict[str, Any]) -> str:
    evidence_lines = state["confirmed_facts"][-8:]
    candidates = state["candidate_answers"][-3:]
    opened_passages = state.get("opened_passages", [])[-4:]
    search_hits = state.get("search_evidence", [])[-6:]
    question_plan = state.get("question_plan", {})
    return "\n".join(
        [
            f"Question:\n{question}",
            "",
            f"Expected answer type: {question_plan.get('answer_type', 'other')}",
            f"Planned keywords: {', '.join(question_plan.get('keywords', [])[:6]) or 'None'}",
            "",
            "Key search evidence:",
            "\n".join(f"- {item}" for item in search_hits) or "- None",
            "",
            "Opened document evidence:",
            "\n\n".join(opened_passages) or "None",
            "",
            "Confirmed evidence:",
            "\n".join(f"- {item}" for item in evidence_lines) or "- None",
            "",
            "Candidate answers considered:",
            "\n".join(f"- {item}" for item in candidates) or "- None",
            "",
            "Give the final answer now.",
        ]
    )


def _should_stop_after_state_update(state: Dict[str, Any], max_rounds: int, round_id: int) -> Optional[str]:
    if round_id >= max_rounds:
        return "max_rounds_reached"
    if state["stall_count"] >= 2:
        return "no_new_information"
    return None


def _init_state(question: str, question_plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question": question,
        "question_plan": question_plan,
        "search_history": [],
        "opened_docids": [],
        "seen_docids": [],
        "last_search_results": [],
        "search_evidence": [],
        "opened_passages": [],
        "confirmed_facts": [],
        "pending_subquestions": ["Resolve the key entity/relation chain needed by the question."],
        "candidate_answers": [],
        "last_action": "",
        "stall_count": 0,
        "finish_reason": "",
    }


def _update_state_from_tool(
    state: Dict[str, Any],
    tool_name: str,
    tool_args: Dict[str, Any],
    tool_result: Any,
    observed_docids: List[str],
) -> None:
    had_new_information = False

    for docid in observed_docids:
        if docid and docid not in state["seen_docids"]:
            state["seen_docids"].append(docid)
            had_new_information = True

    if tool_name == "search":
        query = str(tool_args.get("query", "")).strip()
        state["last_action"] = f"search:{query}"
        ranked = _rank_search_results(tool_result if isinstance(tool_result, list) else [], query or state["question"])
        state["last_search_results"] = ranked
        summaries = _summarize_search_result(tool_result)
        if summaries:
            state["search_evidence"].extend(summaries)
            had_new_information = True
            state["pending_subquestions"] = [
                "Verify the most promising retrieved documents before finalizing the answer."
            ]
        query_bundle = tool_args.get("query_bundle", [query])
        for item in query_bundle:
            item = str(item).strip()
            if item and item not in state["search_history"]:
                state["search_history"].append(item)

    elif tool_name == "get_document":
        docid = str(tool_args.get("docid", "")).strip()
        state["last_action"] = f"get_document:{docid}"
        if docid and docid not in state["opened_docids"]:
            state["opened_docids"].append(docid)
            had_new_information = True
        summaries = _summarize_document_result(tool_result)
        if summaries:
            state["confirmed_facts"].extend(summaries)
            had_new_information = True
            state["pending_subquestions"] = [
                "Decide whether the current evidence is sufficient for a final answer or another targeted search is needed."
            ]
        passage = _extract_relevant_passages(
            str(tool_result.get("text", "")) if isinstance(tool_result, dict) else "",
            focus_text=state["question"],
            max_chars=1200,
        )
        if passage:
            state["opened_passages"].append(f"docid={docid}\n{passage}")
            state["opened_passages"] = state["opened_passages"][-6:]

    if had_new_information:
        state["stall_count"] = 0
    else:
        state["stall_count"] += 1

    state["confirmed_facts"] = state["confirmed_facts"][-20:]
    state["search_evidence"] = state["search_evidence"][-12:]
    state["pending_subquestions"] = state["pending_subquestions"][-4:]


def _register_finish_signal(state: Dict[str, Any], raw_content: str) -> None:
    cleaned = _clean_text(raw_content)
    if cleaned:
        state["candidate_answers"].append(cleaned[:300])
    state["candidate_answers"] = state["candidate_answers"][-6:]
    state["finish_reason"] = "model_declared_ready"


def _repair_final_answer(
    question: str,
    state: Dict[str, Any],
    draft_answer: str,
    client: VLLMClient,
    model_name: str,
) -> str:
    repair_messages = [
        {"role": "system", "content": FINAL_ANSWER_REPAIR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "\n".join(
                [
                    f"Question:\n{question}",
                    f"Expected answer type: {state.get('question_plan', {}).get('answer_type', 'other')}",
                    "",
                    "Evidence:",
                    "\n".join(f"- {item}" for item in state.get("search_evidence", [])[-6:]) or "- None",
                    "",
                    "Opened passages:",
                    "\n\n".join(state.get("opened_passages", [])[-3:]) or "None",
                    "",
                    "Draft answer:",
                    draft_answer,
                ]
            ),
        },
    ]
    response = client.simple_chat(
        model=model_name,
        messages=repair_messages,
        temperature=0.0,
        max_tokens=128,
    )
    repaired = _clean_text(response["choices"][0]["message"]["content"])
    exact = _extract_exact_answer(repaired)
    normalized = _normalize_answer_to_type(exact or repaired, state.get("question_plan", {}).get("answer_type", "other"))
    return normalized or exact or repaired


def _heuristic_query_from_question(question: str) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", question)
    filtered = []
    stopwords = {
        "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "from",
        "with", "that", "this", "was", "were", "is", "are", "be", "by", "as",
        "at", "it", "who", "what", "which", "when", "where", "their", "they",
    }
    for word in words:
        lowered = word.lower()
        if lowered not in stopwords and lowered not in filtered:
            filtered.append(lowered)
        if len(filtered) >= 12:
            break
    return " ".join(filtered) or question[:120]


def _sanitize_search_query(raw_query: str) -> str:
    query = _clean_text(raw_query).strip().strip('"')
    query = re.sub(r'^action"\s*:\s*"search"\s*,?\s*query"\s*:\s*"', "", query, flags=re.IGNORECASE)
    query = query.replace('\\"', '"')
    query = re.sub(r"\s+", " ", query).strip()
    return query


def _is_bad_search_query(query: str) -> bool:
    lowered = query.lower().strip()
    if not lowered:
        return True
    banned_starts = [
        "looking at the",
        "the current state",
        "okay",
        "wait",
        "action\":\"search",
        "query\":\"",
    ]
    if any(lowered.startswith(prefix) for prefix in banned_starts):
        return True
    if len(lowered) < 8:
        return True
    alpha_tokens = _tokenize_focus_text(query)
    return len(alpha_tokens) < 3


def _extract_focus_suffix(question: str) -> str:
    lowered = question.lower()
    suffix_terms: List[str] = []
    hints = [
        ("chapter", "chapter contents title"),
        ("acknowledg", "acknowledgments spouse husband wife"),
        ("annual report", "annual report financial results"),
        ("non-gaap", "non-gaap operating expenses 2021 2020"),
        ("librarian", "librarian partner biography"),
        ("title of", "title book author"),
        ("name of the publicly traded company", "company founder ceo delaware lawsuit"),
        ("exact date", "date performance exhibition"),
    ]
    for needle, extra in hints:
        if needle in lowered:
            suffix_terms.extend(extra.split())
    suffix_terms.extend(_tokenize_focus_text(question)[:8])
    deduped: List[str] = []
    for token in suffix_terms:
        if token not in deduped:
            deduped.append(token)
    return " ".join(deduped[:12])


def _extract_metadata_query(state: Dict[str, Any]) -> str:
    if not state.get("last_search_results"):
        return ""
    top = state["last_search_results"][:1]
    terms: List[str] = []
    for item in top:
        snippet = str(item.get("snippet", ""))
        title = _extract_title_from_text(snippet)
        if title and title.lower() not in {"n/a", "none"}:
            terms.append(title)
        author_match = re.search(r"author:\s*(.+)", snippet, flags=re.IGNORECASE)
        if author_match:
            author = author_match.group(1).strip()
            if author and author.lower() not in {"n/a", "none"}:
                terms.append(author)
    suffix = _extract_focus_suffix(state["question"])
    query = " ".join(terms + ([suffix] if suffix else []))
    query = re.sub(r"\s+", " ", query).strip()
    title_overlap = _tokenize_focus_text(query)
    question_tokens = set(_tokenize_focus_text(state["question"]))
    if not any(token in question_tokens for token in title_overlap[:6]):
        return ""
    return query[:220]


def _next_untried_planned_query(state: Dict[str, Any]) -> str:
    question_plan = state.get("question_plan", {})
    previous = {_normalize_query(item) for item in state["search_history"]}
    for key in ("primary_query", "bridge_query", "verification_query"):
        query = str(question_plan.get(key, "")).strip()
        if query and _normalize_query(query) not in previous and not _is_bad_search_query(query):
            return query[:220]
    return ""


def _build_query_bundle(primary_query: str, state: Dict[str, Any]) -> List[str]:
    planned_query = _next_untried_planned_query(state)
    candidates = [
        _sanitize_search_query(primary_query),
        planned_query,
        _heuristic_query_from_question(state["question"]),
        _extract_metadata_query(state),
    ]
    bundle: List[str] = []
    seen = set()
    previous = {_normalize_query(item) for item in state["search_history"]}
    for query in candidates:
        query = query.strip()
        normalized = _normalize_query(query)
        if _is_bad_search_query(query):
            continue
        if normalized in seen or normalized in previous:
            continue
        seen.add(normalized)
        bundle.append(query[:220])
        if len(bundle) >= 3:
            break
    if not bundle:
        fallback = _heuristic_query_from_question(state["question"])
        if not _is_bad_search_query(fallback):
            bundle.append(fallback[:220])
    return bundle


def _pick_unopened_docid(state: Dict[str, Any]) -> str:
    ranked = _rank_search_results(state.get("last_search_results", []), state["question"])
    for item in ranked:
        docid = str(item.get("docid", "")).strip()
        if docid and docid not in state["opened_docids"]:
            return docid
    return ""


def _decide_next_action(
    question: str,
    state: Dict[str, Any],
    client: VLLMClient,
    model_name: str,
    max_rounds: int,
    round_id: int,
    decision_max_tokens: int,
    recent_observation: str,
) -> Tuple[Dict[str, Any], str]:
    fallback_docid = _pick_unopened_docid(state)
    if state["last_action"].startswith("search:") and fallback_docid:
        return {
            "action": "get_document",
            "docid": fallback_docid,
            "reason": "Heuristic policy: inspect the best unseen document immediately after each search.",
        }, "Heuristic policy selected get_document after search."

    if (
        state["last_action"].startswith("get_document:")
        and len(state["opened_docids"]) < 2
        and fallback_docid
        and round_id < max_rounds
    ):
        return {
            "action": "get_document",
            "docid": fallback_docid,
            "reason": "Heuristic policy: inspect a second strong candidate before issuing more searches.",
        }, "Heuristic policy selected a second get_document."

    if round_id >= max_rounds and state["opened_docids"]:
        return {
            "action": "finish",
            "answer_hint": state["candidate_answers"][-1] if state["candidate_answers"] else "",
            "reason": "Heuristic policy: stop at final round once at least one document has been inspected.",
        }, "Heuristic policy selected finish at the final round."

    state_summary = _build_state_summary(state)
    round_messages = [
        {"role": "system", "content": ACTION_DECISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_round_user_prompt(
                question=question,
                state_summary=state_summary,
                recent_observation=recent_observation,
                max_rounds=max_rounds,
                round_id=round_id,
            ),
        },
    ]
    response = client.simple_chat(
        model=model_name,
        messages=round_messages,
        temperature=0.0,
        max_tokens=decision_max_tokens,
    )
    raw_content = str(response["choices"][0]["message"].get("content", "") or "")
    action = _extract_json_object(raw_content)
    if action.get("action") in {"search", "get_document", "finish"}:
        return action, raw_content

    recovered = _extract_action_from_raw_text(raw_content, state)
    if recovered is not None:
        return recovered, raw_content

    if state["opened_docids"]:
        bundle = _build_query_bundle("", state)
        if bundle:
            return {
                "action": "search",
                "query": bundle[0],
                "reason": "Fallback because action JSON was invalid after document review; issue a targeted rewritten search.",
            }, raw_content

    fallback_docid = _pick_unopened_docid(state)
    if fallback_docid:
        return {
            "action": "get_document",
            "docid": fallback_docid,
            "reason": "Fallback because action JSON was invalid; open the top unseen candidate document.",
        }, raw_content

    fallback_query = _heuristic_query_from_question(question)
    previous = {_normalize_query(item) for item in state["search_history"]}
    if _normalize_query(fallback_query) not in previous:
        return {
            "action": "search",
            "query": fallback_query,
            "reason": "Fallback because action JSON was invalid; perform a keyword search.",
        }, raw_content

    return {
        "action": "finish",
        "answer_hint": "",
        "reason": "Fallback because action JSON was invalid and no useful unseen search or document remains.",
    }, raw_content


def _action_to_tool_call(action: Dict[str, Any], tool_call_id: str) -> Dict[str, Any]:
    action_name = str(action.get("action", "")).strip()
    if action_name == "search":
        arguments = {"query": str(action.get("query", "")).strip()}
        function_name = "search"
    elif action_name == "get_document":
        arguments = {"docid": str(action.get("docid", "")).strip()}
        function_name = "get_document"
    else:
        raise ValueError(f"Unsupported tool action: {action_name}")

    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": function_name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _extract_action_from_raw_text(raw_text: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cleaned = _clean_text(raw_text)
    docid_match = re.search(r"docid\s*[:=]\s*\"?(\d{2,})\"?", cleaned, flags=re.IGNORECASE)
    if docid_match:
        docid = docid_match.group(1)
        if docid not in state["opened_docids"]:
            return {"action": "get_document", "docid": docid, "reason": "Recovered from unstructured planner text."}

    search_match = re.search(r"query\"\s*:\s*\"([^\"]+)\"", cleaned)
    if search_match:
        query = _sanitize_search_query(search_match.group(1).strip())
        if not _is_bad_search_query(query):
            return {"action": "search", "query": query, "reason": "Recovered from unstructured planner text."}
    return None


def _execute_tool_call(
    tool_call: Dict[str, Any],
    tool_registry: Dict[str, Any],
    state: Dict[str, Any],
    tool_content_max_chars: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    function = tool_call.get("function", {})
    tool_name = str(function.get("name", "")).strip()
    if tool_name not in tool_registry:
        state["stall_count"] += 1
        return (
            None,
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps({"error": f"Tool `{tool_name}` is unavailable."}, ensure_ascii=False),
            },
            f"Tool `{tool_name}` is unavailable.",
        )

    tool_args = _safe_json_loads(function.get("arguments", "{}"))
    if tool_name == "search":
        query = str(tool_args.get("query", "")).strip()
        bundle = _build_query_bundle(query, state)
        if not bundle:
            state["stall_count"] += 1
            return (
                None,
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(
                        {"error": f"Skipped repeated or low-quality search query: {query or '<empty>'}"},
                        ensure_ascii=False,
                    ),
                },
                f"Skipped repeated or low-quality search query: {query or '<empty>'}",
            )

    if tool_name == "get_document":
        docid = str(tool_args.get("docid", "")).strip()
        if not docid or docid in state["opened_docids"]:
            state["stall_count"] += 1
            return (
                None,
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(
                        {"error": f"Skipped repeated or empty document request: {docid or '<empty>'}"},
                        ensure_ascii=False,
                    ),
                },
                f"Skipped repeated or empty document request: {docid or '<empty>'}",
            )

    if tool_name == "search":
        merged: Dict[str, Dict[str, Any]] = {}
        query_bundle = _build_query_bundle(str(tool_args.get("query", "")).strip(), state)
        for bundle_query in query_bundle:
            per_query_results = tool_registry[tool_name](query=bundle_query)
            for rank, item in enumerate(per_query_results):
                docid = str(item.get("docid", "")).strip()
                if not docid:
                    continue
                enriched = dict(item)
                enriched["source_query"] = bundle_query
                enriched["bundle_rank"] = rank
                if docid not in merged:
                    merged[docid] = enriched
                else:
                    if float(item.get("score", 0.0)) > float(merged[docid].get("score", 0.0)):
                        merged[docid].update(enriched)
        tool_result = _rank_search_results(list(merged.values()), state["question"])[:8]
        tool_args = {"query": query_bundle[0], "query_bundle": query_bundle}
    else:
        tool_result = tool_registry[tool_name](**tool_args)
    focus_text = state["question"]
    if tool_name == "search":
        focus_text = " ".join(tool_args.get("query_bundle", [])) or state["question"]
    tool_content, metadata = _tool_result_preview(
        tool_name=tool_name,
        tool_result=tool_result,
        max_chars=tool_content_max_chars,
        focus_text=focus_text,
    )
    observed_docids = metadata.get("docids", [])
    _update_state_from_tool(
        state=state,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=tool_result,
        observed_docids=observed_docids,
    )
    executed = {
        "tool_name": tool_name,
        "arguments": tool_args,
        "raw_result": tool_result,
        "tool_content": tool_content,
    }
    tool_message = {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": tool_content,
    }
    observation = "\n".join(
        [
            f"Executed {tool_name} with args={json.dumps(tool_args, ensure_ascii=False)}",
            "Observed result preview:",
            _truncate_text(tool_content, 1200),
        ]
    )
    return executed, tool_message, observation


def run_multistep_agent(
    question: str,
    client: VLLMClient,
    model_name: str,
    tool_specs: List[Dict[str, Any]],
    tool_registry: Dict[str, Any],
    max_rounds: int = 7,
    decision_max_tokens: int = 512,
    answer_max_tokens: int = 768,
    tool_content_max_chars: int = 4000,
) -> Dict[str, Any]:
    question_plan = _plan_question(question=question, client=client, model_name=model_name)
    state = _init_state(question, question_plan=question_plan)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": ACTION_DECISION_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    recent_observation = "No tool has been used yet."

    initial_query = str(question_plan.get("primary_query", "")).strip() or question.strip()
    initial_tool_call = _action_to_tool_call({"action": "search", "query": initial_query}, "call_1")
    messages.append(
        {
            "role": "assistant",
            "content": "Planning: start with an initial search using the decomposed primary query.",
            "state_summary": _build_state_summary(state),
            "round_id": 1,
            "question_plan": question_plan,
            "tool_calls": [initial_tool_call],
        }
    )
    _, initial_tool_message, recent_observation = _execute_tool_call(
        tool_call=initial_tool_call,
        tool_registry=tool_registry,
        state=state,
        tool_content_max_chars=tool_content_max_chars,
    )
    if initial_tool_message is not None:
        messages.append(initial_tool_message)

    for round_id in range(2, max_rounds + 1):
        action, raw_content = _decide_next_action(
            question=question,
            state=state,
            client=client,
            model_name=model_name,
            max_rounds=max_rounds,
            round_id=round_id,
            decision_max_tokens=decision_max_tokens,
            recent_observation=recent_observation,
        )
        state_summary = _build_state_summary(state)
        action_name = str(action.get("action", "")).strip()

        trajectory_assistant = {
            "role": "assistant",
            "content": raw_content,
            "state_summary": state_summary,
            "round_id": round_id,
            "action_plan": action,
        }

        if action_name == "finish":
            messages.append(trajectory_assistant)
            _register_finish_signal(state, json.dumps(action, ensure_ascii=False))
            state["finish_reason"] = "model_or_fallback_finish"
            break

        tool_call_id = f"call_{round_id}"
        tool_call = _action_to_tool_call(action, tool_call_id)
        trajectory_assistant["tool_calls"] = [tool_call]
        messages.append(trajectory_assistant)

        _, tool_message, recent_observation = _execute_tool_call(
            tool_call=tool_call,
            tool_registry=tool_registry,
            state=state,
            tool_content_max_chars=tool_content_max_chars,
        )
        if tool_message is not None:
            messages.append(tool_message)

        stop_reason = _should_stop_after_state_update(state=state, max_rounds=max_rounds, round_id=round_id)
        if stop_reason:
            state["finish_reason"] = stop_reason
            break

    final_messages = [
        {"role": "system", "content": FINAL_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": _build_final_user_prompt(question=question, state=state)},
    ]
    final_response = client.simple_chat(
        model=model_name,
        messages=final_messages,
        temperature=0.0,
        max_tokens=answer_max_tokens,
    )
    final_text = _clean_text(final_response["choices"][0]["message"]["content"])
    predicted_answer = _normalize_answer_to_type(
        _extract_exact_answer(final_text),
        question_plan.get("answer_type", "other"),
    )
    if not predicted_answer:
        predicted_answer = _normalize_answer_to_type(
            final_text.strip() or "evidence insufficient",
            question_plan.get("answer_type", "other"),
        )
    if _is_placeholder_answer(predicted_answer):
        repaired = _repair_final_answer(
            question=question,
            state=state,
            draft_answer=final_text,
            client=client,
            model_name=model_name,
        )
        if repaired and not _is_placeholder_answer(repaired):
            predicted_answer = repaired
    predicted_answer = _normalize_answer_to_type(predicted_answer or final_text[:200], question_plan.get("answer_type", "other"))
    state["candidate_answers"].append(predicted_answer or final_text[:200])

    messages.append(
        {
            "role": "assistant",
            "content": final_text,
            "state_summary": _build_state_summary(state),
            "finish_reason": state["finish_reason"] or "final_answer_generated",
        }
    )
    return {
        "status": "completed" if predicted_answer else "completed_with_empty_answer",
        "predicted_answer": predicted_answer,
        "messages": messages,
        "agent_state": state,
    }


def generate_submission(
    dataset_rows: List[Dict[str, Any]],
    index_path: str,
    base_url: str,
    model_name: str,
    output_path: str,
    api_key: str = "dummy",
    top_k: int = 5,
    search_snippet_max_chars: int = 1200,
    tool_content_max_chars: int = 4000,
    max_rounds: int = 7,
    decision_max_tokens: int = 512,
    answer_max_tokens: int = 768,
) -> List[Dict[str, Any]]:
    client = VLLMClient(base_url=base_url, api_key=api_key)
    searcher = build_searcher(index_path=index_path)
    tool_specs, tool_registry = get_agent_tool_specs_and_registry(
        searcher=searcher,
        k=top_k,
        snippet_max_chars=search_snippet_max_chars,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    with output_file.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(dataset_rows, start=1):
            print(f"[{idx}/{len(dataset_rows)}] Processing query_id={row['query_id']}...")
            record = run_multistep_agent(
                question=row["query"],
                client=client,
                model_name=model_name,
                tool_specs=tool_specs,
                tool_registry=tool_registry,
                max_rounds=max_rounds,
                decision_max_tokens=decision_max_tokens,
                answer_max_tokens=answer_max_tokens,
                tool_content_max_chars=tool_content_max_chars,
            )
            record["query_id"] = row["query_id"]
            records.append(record)
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Saved {len(records)} trajectories to: {output_path}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multistep Deep Research Agent and write submission.jsonl.")
    parser.add_argument("--dataset", required=True, help="Path to the dataset jsonl file.")
    parser.add_argument("--index-path", required=True, help="Path to the BM25 sqlite index.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="vLLM base URL.")
    parser.add_argument("--model-name", required=True, help="Model name served by vLLM.")
    parser.add_argument("--output", required=True, help="Output submission jsonl path.")
    parser.add_argument("--api-key", default="dummy", help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for search.")
    parser.add_argument("--max-rounds", type=int, default=7, help="Maximum agent rounds per query.")
    parser.add_argument("--decision-max-tokens", type=int, default=512, help="Maximum tokens for each tool-decision round.")
    parser.add_argument("--answer-max-tokens", type=int, default=768, help="Maximum tokens for the final answer generation.")
    parser.add_argument("--search-snippet-max-chars", type=int, default=1200, help="Snippet size returned by search.")
    parser.add_argument("--tool-content-max-chars", type=int, default=4000, help="Tool content size stored in messages.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.dataset, limit=args.limit)
    generate_submission(
        dataset_rows=rows,
        index_path=args.index_path,
        base_url=args.base_url,
        model_name=args.model_name,
        output_path=args.output,
        api_key=args.api_key,
        top_k=args.top_k,
        search_snippet_max_chars=args.search_snippet_max_chars,
        tool_content_max_chars=args.tool_content_max_chars,
        max_rounds=args.max_rounds,
        decision_max_tokens=args.decision_max_tokens,
        answer_max_tokens=args.answer_max_tokens,
    )


if __name__ == "__main__":
    main()
