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


FINAL_ANSWER_SYSTEM_PROMPT = """You are a Deep Research Agent.

Use only the evidence provided to produce the final answer.

Rules:
- Do not output chain-of-thought tags such as <think>.
- Do not mention hidden reasoning.
- If evidence is incomplete, still provide the best supported answer instead of leaving the answer blank.
- Keep the explanation brief and evidence-based.

Reply in exactly this format:
Explanation: <2-4 short sentences>
Exact Answer: <final answer>
Confidence: <0-100%>
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


def _tool_result_preview(tool_name: str, tool_result: Any, max_chars: int) -> Tuple[str, Dict[str, Any]]:
    if tool_name == "search" and isinstance(tool_result, list):
        trimmed = [
            {
                "docid": item.get("docid", ""),
                "score": item.get("score", 0.0),
                "snippet": _truncate_text(str(item.get("snippet", "")), max_chars),
                "url": item.get("url", ""),
            }
            for item in tool_result
        ]
        return json.dumps(trimmed, ensure_ascii=False), {"docids": [item["docid"] for item in trimmed]}

    if tool_name == "get_document" and isinstance(tool_result, dict):
        trimmed = {
            "docid": tool_result.get("docid", ""),
            "url": tool_result.get("url", ""),
            "text": _truncate_text(str(tool_result.get("text", "")), max_chars),
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

    return "\n".join(
        [
            f"Question: {state['question']}",
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
    return "\n".join(
        [
            f"Question:\n{question}",
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


def _init_state(question: str) -> Dict[str, Any]:
    return {
        "question": question,
        "search_history": [],
        "opened_docids": [],
        "seen_docids": [],
        "last_search_results": [],
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
        state["last_search_results"] = tool_result if isinstance(tool_result, list) else []
        summaries = _summarize_search_result(tool_result)
        if summaries:
            state["confirmed_facts"].extend(summaries)
            had_new_information = True
            state["pending_subquestions"] = [
                "Verify the most promising retrieved documents before finalizing the answer."
            ]
        if query and query not in state["search_history"]:
            state["search_history"].append(query)

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

    if had_new_information:
        state["stall_count"] = 0
    else:
        state["stall_count"] += 1

    state["confirmed_facts"] = state["confirmed_facts"][-20:]
    state["pending_subquestions"] = state["pending_subquestions"][-4:]


def _register_finish_signal(state: Dict[str, Any], raw_content: str) -> None:
    cleaned = _clean_text(raw_content)
    if cleaned:
        state["candidate_answers"].append(cleaned[:300])
    state["candidate_answers"] = state["candidate_answers"][-6:]
    state["finish_reason"] = "model_declared_ready"


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


def _pick_unopened_docid(state: Dict[str, Any]) -> str:
    for item in state.get("last_search_results", []):
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
        normalized = _normalize_query(query)
        previous = {_normalize_query(item) for item in state["search_history"]}
        if not query or normalized in previous:
            state["stall_count"] += 1
            return (
                None,
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(
                        {"error": f"Skipped repeated or empty search query: {query or '<empty>'}"},
                        ensure_ascii=False,
                    ),
                },
                f"Skipped repeated or empty search query: {query or '<empty>'}",
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

    tool_result = tool_registry[tool_name](**tool_args)
    tool_content, metadata = _tool_result_preview(
        tool_name=tool_name,
        tool_result=tool_result,
        max_chars=tool_content_max_chars,
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
    observation = f"Executed {tool_name} with args={json.dumps(tool_args, ensure_ascii=False)}"
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
    state = _init_state(question)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": ACTION_DECISION_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    recent_observation = "No tool has been used yet."

    initial_query = question.strip()
    initial_tool_call = _action_to_tool_call({"action": "search", "query": initial_query}, "call_1")
    messages.append(
        {
            "role": "assistant",
            "content": "Planning: start with an initial search using the original question.",
            "state_summary": _build_state_summary(state),
            "round_id": 1,
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
    predicted_answer = _extract_exact_answer(final_text)
    if not predicted_answer:
        predicted_answer = final_text.strip() or "evidence insufficient"
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
