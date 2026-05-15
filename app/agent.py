"""Conversational agent: guards, retrieval, LLM routing, grounded recommendations."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.models import ChatMessage, ChatResponse, Recommendation
from app.prompts import REPAIR_PROMPT, SYSTEM_PROMPT, USER_WRAPPER
from app.retriever import HybridRetriever, RetrievedItem
from app.utils import LOG, post_json_with_retries

Mode = Literal["clarify", "recommend", "compare", "refuse"]


class _LLMStructured(BaseModel):
    mode: Mode
    reply: str = Field(..., max_length=16000)
    recommended_urls: list[str] = Field(default_factory=list)
    end_of_conversation: bool = False


INJECTION_PATTERNS = re.compile(
    r"(ignore (all )?(previous|prior) instructions|system prompt|developer message|"
    r"disregard (the )?above|you are now dan|jailbreak|reveal (your )?prompt)",
    re.I,
)

LEGAL_PATTERNS = re.compile(
    r"\b(lawsuit|litigation|sue |suing|contract law|employment law|legal advice|"
    r"discrimination claim|accommodation|ada |eeoc|gdpr compliance only)\b",
    re.I,
)

COMPARE_INTENT_RE = re.compile(r"\b(compare|versus|vs\.?|difference between|which is better)\b", re.I)
REFINEMENT_RE = re.compile(
    r"\b(instead|rather|change|refine|update|different|switch|prefer|narrow|broaden|exclude|include only)\b",
    re.I,
)
DONE_SIGNAL_RE = re.compile(
    r"\b(thanks|thank you|that'?s all|that is all|no more questions|goodbye|we are done|i'?m done)\b",
    re.I,
)
# Acronyms / short names users mention in comparisons (e.g. GSA, OPQ)
_COMPARE_TOKEN_RE = re.compile(r"\b([A-Z]{2,6})\b")
_COMPARE_VS_RE = re.compile(
    r"\bcompare\s+(.+?)\s+(?:vs\.?|versus|and|with|to)\s+(.+?)(?:\?|$)",
    re.I,
)

_CLARIFY_CUES_RE = re.compile(
    r"(before i recommend|could you clarify|please clarify|to narrow (this )?down|"
    r"what role|what seniority|tell me more|which (role|level)|need (a )?bit more|"
    r"can you share|help me understand)",
    re.I,
)
_RECOMMEND_CUES_RE = re.compile(
    r"(we recommend|recommended assessments|here are (my |our )?(top )?recommendations|"
    r"suggested assessments|shortlist|i suggest|we suggest|suggest these|these tests|"
    r"these assessments|best fit from|before i recommend)",
    re.I,
)
_ROLE_RE = re.compile(
    r"\b(developer|engineer|manager|analyst|sales|nurse|accountant|technician|"
    r"consultant|specialist|administrator|clerk|operator|hire|hiring|role|position)\b",
    re.I,
)
_SENIORITY_RE = re.compile(
    r"\b(junior|senior|mid[- ]?level|entry[- ]?level|graduate|intern|lead|principal|"
    r"experienced|years? of experience)\b",
    re.I,
)
_TECH_AREA_RE = re.compile(
    r"\b(java|python|\.net|javascript|personality|cognitive|technical|skill|api|"
    r"backend|frontend|communication|leadership|sql|cloud|devops|accounting|"
    r"customer service|behavioral|behaviour)\b",
    re.I,
)


def _last_user_text(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _conversation_text(messages: list[ChatMessage], max_chars: int = 12000) -> str:
    lines: list[str] = []
    for m in messages:
        lines.append(f"{m.role.upper()}: {m.content}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _heuristic_refuse(text: str) -> str | None:
    if not text.strip():
        return "Please describe the role and what you want to measure so I can suggest SHL assessments."
    if INJECTION_PATTERNS.search(text):
        return "I cannot change my instructions. I can only help you choose SHL assessments from the official catalog."
    if LEGAL_PATTERNS.search(text):
        return "I cannot provide legal guidance. I can help compare or select SHL assessments from the catalog if you share the role and goals."
    return None


def _heuristic_vague(messages: list[ChatMessage]) -> bool:
    """Very short first-turn hiring asks are usually underspecified."""
    if len(messages) > 2:
        return False
    u = _last_user_text(messages).lower()
    if len(u.split()) < 6:
        return True
    hiringish = re.search(
        r"\b(developer|engineer|manager|sales|hire|role|candidate|assessment|test|"
        r"personality|cognitive|skill|technical|graduate|intern|remote)\b",
        u,
    )
    return hiringish is None


def _has_sufficient_hiring_context(text: str) -> bool:
    """Role plus seniority or technical area => ready to recommend."""
    if _ROLE_RE.search(text) and (_SENIORITY_RE.search(text) or _TECH_AREA_RE.search(text)):
        return True
    if _ROLE_RE.search(text) and len(text.split()) >= 10:
        return True
    return False


def _needs_clarification(messages: list[ChatMessage], out: _LLMStructured) -> bool:
    if _has_sufficient_hiring_context(_conversation_text(messages)):
        return False
    if out.mode == "clarify":
        return True
    return _heuristic_vague(messages)


def _sanitize_clarify_reply(reply: str) -> str:
    """Clarify replies: questions only, no recommendation wording."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", reply.strip()) if p.strip()]
    cleaned = [p for p in parts if not _RECOMMEND_CUES_RE.search(p)]
    text = " ".join(cleaned).strip()
    if not text or "?" not in text:
        return (
            "What role and seniority are you hiring for, and should we prioritize "
            "technical skills, personality, or cognitive ability?"
        )
    return text


def _sanitize_recommend_reply(reply: str) -> str:
    """Recommend replies: shortlist summary only, no clarification wording."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", reply.strip()) if p.strip()]
    cleaned = [p for p in parts if not _CLARIFY_CUES_RE.search(p)]
    text = " ".join(cleaned).strip()
    if not text:
        return "These SHL catalog assessments best match your stated hiring needs."
    return text


def _normalize_mode_routing(
    out: _LLMStructured,
    messages: list[ChatMessage],
    urls: list[str],
    retrieved: list[RetrievedItem],
) -> tuple[_LLMStructured, list[str]]:
    """Enforce mutually exclusive clarify vs recommend behavior."""
    if out.mode in ("refuse", "compare"):
        return out, urls

    convo = _conversation_text(messages)
    sufficient = _has_sufficient_hiring_context(convo)

    if _needs_clarification(messages, out):
        reply = _sanitize_clarify_reply(out.reply)
        if _RECOMMEND_CUES_RE.search(out.reply):
            reply = _sanitize_clarify_reply("")
        return (
            _LLMStructured(
                mode="clarify",
                reply=reply,
                recommended_urls=[],
                end_of_conversation=False,
            ),
            [],
        )

    # Recommend path
    reply = _sanitize_recommend_reply(out.reply)
    if sufficient and (out.mode == "clarify" or _CLARIFY_CUES_RE.search(out.reply) or "?" in out.reply):
        if _CLARIFY_CUES_RE.search(reply) or "?" in reply or not reply.strip():
            reply = "These SHL catalog assessments best match your stated hiring needs."

    if not urls:
        urls = [r.item.url for r in retrieved[:5]]

    if not urls and _RECOMMEND_CUES_RE.search(reply):
        reply = _sanitize_clarify_reply(reply)
        return (
            _LLMStructured(
                mode="clarify",
                reply=reply,
                recommended_urls=[],
                end_of_conversation=False,
            ),
            [],
        )

    return (
        _LLMStructured(
            mode="recommend",
            reply=reply,
            recommended_urls=urls,
            end_of_conversation=out.end_of_conversation,
        ),
        urls,
    )


def _candidates_payload(rows: list[RetrievedItem]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        it = r.item
        out.append(
            {
                "name": it.name,
                "url": it.url,
                "test_type": it.test_type,
                "description": it.description[:1200],
            }
        )
    return out


def _lookup_by_url(rows: list[RetrievedItem]) -> dict[str, RetrievedItem]:
    return {r.item.url: r for r in rows}


def _catalog_names(candidates: list[dict[str, Any]]) -> list[str]:
    return [str(c.get("name", "")).lower() for c in candidates]


def _token_in_catalog(token: str, catalog_names: list[str]) -> bool:
    t = token.lower().strip()
    if len(t) < 2:
        return True
    for name in catalog_names:
        if t in name or name in t:
            return True
        # acronym in name words: "Global Skills Assessment" -> gsa
        initials = "".join(w[0] for w in re.findall(r"[a-z0-9]+", name) if w)
        if len(t) >= 2 and initials.startswith(t):
            return True
    return False


def _extract_compare_targets(user_text: str) -> list[str]:
    """Pull assessment names/acronyms the user wants compared."""
    targets: list[str] = []
    m = _COMPARE_VS_RE.search(user_text)
    if m:
        for part in (m.group(1), m.group(2)):
            part = part.strip(" .,\"'")
            if part and len(part) > 1:
                targets.append(part)
    for acr in _COMPARE_TOKEN_RE.findall(user_text):
        if acr not in ("SHL", "API", "IRT", "JSON", "HTTP"):
            targets.append(acr)
    # quoted names
    for q in re.findall(r'"([^"]+)"|\'([^\']+)\'', user_text):
        name = (q[0] or q[1]).strip()
        if len(name) > 2:
            targets.append(name)
    seen: set[str] = set()
    out: list[str] = []
    for t in targets:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _missing_compare_targets(user_text: str, candidates: list[dict[str, Any]]) -> list[str]:
    if not COMPARE_INTENT_RE.search(user_text):
        return []
    names = _catalog_names(candidates)
    missing: list[str] = []
    for t in _extract_compare_targets(user_text):
        if not _token_in_catalog(t, names):
            missing.append(t)
    return missing


def _enforce_compare_grounding(
    user_text: str,
    candidates: list[dict[str, Any]],
    out: _LLMStructured,
) -> _LLMStructured:
    missing = _missing_compare_targets(user_text, candidates)
    if missing:
        label = ", ".join(missing)
        return _LLMStructured(
            mode="refuse",
            reply=(
                f"I could not find {label} in the SHL catalog context. "
                "I can only compare assessments that appear in the retrieved catalog results."
            ),
            recommended_urls=[],
            end_of_conversation=False,
        )
    if out.mode != "compare":
        return out
    names = _catalog_names(candidates)
    # Block replies that describe unknown acronyms as fact ("GSA is a ...")
    for acr in _COMPARE_TOKEN_RE.findall(out.reply):
        if acr in ("SHL", "API", "IRT"):
            continue
        if not _token_in_catalog(acr, names) and re.search(
            rf"\b{re.escape(acr)}\b\s+is\s+(?:a|an)\b", out.reply, re.I
        ):
            return _LLMStructured(
                mode="refuse",
                reply=(
                    f"I could not find {acr} in the SHL catalog context. "
                    "I can only compare assessments present in the retrieved catalog results."
                ),
                recommended_urls=[],
                end_of_conversation=False,
            )
    return out


def _is_refinement_turn(messages: list[ChatMessage]) -> bool:
    if len(messages) < 2:
        return False
    if REFINEMENT_RE.search(_last_user_text(messages)):
        return True
    # User answering or adjusting after assistant clarification/recommendation
    if len(messages) >= 3 and messages[-1].role == "user" and messages[-2].role == "assistant":
        return True
    return False


def _is_compare_turn(messages: list[ChatMessage]) -> bool:
    return bool(COMPARE_INTENT_RE.search(_last_user_text(messages)))


def _user_signaled_done(messages: list[ChatMessage]) -> bool:
    return bool(DONE_SIGNAL_RE.search(_last_user_text(messages)))


def _resolve_end_of_conversation(
    out: _LLMStructured,
    messages: list[ChatMessage],
    vague: bool,
    recs: list[Recommendation],
) -> bool:
    if out.mode in ("clarify", "refuse"):
        return False
    if vague:
        return False
    if _is_refinement_turn(messages):
        return False
    if _user_signaled_done(messages):
        return True
    if out.mode == "compare" or _is_compare_turn(messages):
        return False
    if out.mode == "recommend" and recs and "?" not in out.reply:
        return True
    return bool(out.end_of_conversation)


def _llm_messages(
    messages: list[ChatMessage],
    candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    user_content = USER_WRAPPER.format(
        messages_json=json.dumps([m.model_dump() for m in messages], ensure_ascii=False),
        candidates_json=json.dumps(candidates, ensure_ascii=False),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _provider_config() -> tuple[str, str, str, float]:
    provider = (os.environ.get("LLM_PROVIDER") or "groq").strip().lower()
    timeout = float(os.environ.get("LLM_TIMEOUT_SEC", "25"))
    model = (os.environ.get("LLM_MODEL") or "").strip()
    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        model = model or "meta-llama/llama-3.1-8b-instruct"
        return provider, key, model, timeout
    # default groq
    key = os.environ.get("GROQ_API_KEY", "").strip()
    model = model or "llama-3.1-8b-instant"
    return "groq", key, model, timeout


def _chat_completions_url(provider: str) -> str:
    if provider == "openrouter":
        return os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
    return os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")


def _headers(provider: str, api_key: str) -> dict[str, str]:
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if provider == "openrouter":
        h["HTTP-Referer"] = os.environ.get("OPENROUTER_HTTP_REFERER", "https://github.com/shl-assignment")
        h["X-Title"] = os.environ.get("OPENROUTER_APP_TITLE", "SHL Recommender")
    return h


def _api_key(provider: str) -> str:
    if provider == "openrouter":
        return os.environ.get("OPENROUTER_API_KEY", "").strip()
    return os.environ.get("GROQ_API_KEY", "").strip()


def _call_llm_raw(messages: list[dict[str, str]], model: str, provider: str, timeout_s: float) -> str:
    api_key = _api_key(provider)
    if not api_key:
        raise RuntimeError("Missing API key: set GROQ_API_KEY or OPENROUTER_API_KEY")
    url = _chat_completions_url(provider)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    if provider in ("groq", "openrouter"):
        payload["response_format"] = {"type": "json_object"}
    data = post_json_with_retries(
        url,
        headers=_headers(provider, api_key),
        payload=payload,
        timeout_s=timeout_s,
    )
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected LLM response shape: {e}") from e


def _parse_llm_json(text: str) -> _LLMStructured:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    obj = json.loads(text)
    return _LLMStructured.model_validate(obj)


def _run_llm(
    messages: list[ChatMessage],
    candidates: list[dict[str, Any]],
    model: str,
    provider: str,
    timeout_s: float,
) -> _LLMStructured:
    base_msgs = _llm_messages(messages, candidates)
    raw = _call_llm_raw(base_msgs, model, provider, timeout_s)
    try:
        return _parse_llm_json(raw)
    except (json.JSONDecodeError, ValidationError) as e:
        LOG.warning("LLM JSON parse failed (%s), attempting repair", e)
        repair_msgs = base_msgs + [
            {"role": "assistant", "content": raw[:4000]},
            {"role": "user", "content": REPAIR_PROMPT},
        ]
        raw2 = _call_llm_raw(repair_msgs, model, provider, timeout_s)
        return _parse_llm_json(raw2)


def _to_recommendations(urls: list[str], by_url: dict[str, RetrievedItem]) -> list[Recommendation]:
    recs: list[Recommendation] = []
    seen: set[str] = set()
    for u in urls:
        if u in seen:
            continue
        row = by_url.get(u)
        if not row:
            continue
        seen.add(u)
        it = row.item
        recs.append(Recommendation(name=it.name, url=it.url, test_type=it.test_type or "Unknown"))
        if len(recs) >= 10:
            break
    return recs


def _fallback_recommend(rows: list[RetrievedItem], n: int = 5) -> list[Recommendation]:
    by_url = _lookup_by_url(rows)
    return _to_recommendations([r.item.url for r in rows[:n]], by_url)


def run_agent(messages: list[ChatMessage], retriever: HybridRetriever) -> ChatResponse:
    last_user = _last_user_text(messages)
    fixed = _heuristic_refuse(last_user)
    if fixed:
        return ChatResponse(reply=fixed, recommendations=[], end_of_conversation=False)

    provider, api_key, model, timeout_s = _provider_config()
    q = _conversation_text(messages)
    retrieved = retriever.retrieve(q, top_k=10)
    candidates = _candidates_payload(retrieved)
    by_url = _lookup_by_url(retrieved)
    allow = set(by_url.keys())

    if not api_key:
        return ChatResponse(
            reply="The LLM is not configured (set GROQ_API_KEY or OPENROUTER_API_KEY). "
            "Here are top semantic matches from the catalog only.",
            recommendations=_fallback_recommend(retrieved, n=5),
            end_of_conversation=False,
        )

    try:
        out = _run_llm(messages, candidates, model=model, provider=provider, timeout_s=timeout_s)
        out = _enforce_compare_grounding(last_user, candidates, out)
    except Exception as e:
        LOG.exception("LLM failure: %s", e)
        return ChatResponse(
            reply="I could not reach the language model right now. These are safe, catalog-grounded options based on your message:",
            recommendations=_fallback_recommend(retrieved, n=5),
            end_of_conversation=False,
        )

    urls = [u for u in out.recommended_urls if u in allow][:10]

    if out.mode == "refuse":
        urls = []
        return ChatResponse(
            reply=out.reply.strip(),
            recommendations=[],
            end_of_conversation=False,
        )

    if out.mode != "compare":
        out, urls = _normalize_mode_routing(out, messages, urls, retrieved)

    if out.mode in ("clarify", "refuse"):
        urls = []

    reply = out.reply.strip()
    recs = _to_recommendations(urls, by_url)

    # Empty recs but reply still sounds like recommending -> force clarify rewrite
    if not recs and _RECOMMEND_CUES_RE.search(reply):
        reply = _sanitize_clarify_reply(reply)
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    vague = out.mode == "clarify" or _needs_clarification(messages, out)
    eoc = _resolve_end_of_conversation(out, messages, vague, recs)
    return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=eoc)
