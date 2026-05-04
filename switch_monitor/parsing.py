from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FileAction:
    path: str
    content: str
    mode: str = "append"


@dataclass
class EventAction:
    event_key: str
    category: str
    severity: str
    message: str
    notify: bool = False
    trigger_command: str = "none"
    save_frame: bool = False
    cooldown_seconds: int = 3600


@dataclass
class AnalysisResult:
    assistant_reply: str = ""
    summary: str = ""
    observations: list[str] = field(default_factory=list)
    file_actions: list[FileAction] = field(default_factory=list)
    memory_updates: dict[str, Any] = field(default_factory=dict)
    events: list[EventAction] = field(default_factory=list)
    gesture_digit: int | None = None
    gesture_confidence: str = ""
    gesture_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def format_analysis_for_display(result: AnalysisResult) -> str:
    parts: list[str] = []

    if result.assistant_reply:
        parts.append("Ответ Gemma:")
        parts.append(result.assistant_reply)
        parts.append("")

    parts.append("Кратко:")
    parts.append(result.summary or "Модель не дала краткого описания.")

    parts.append("")
    parts.append("Наблюдения:")
    if result.observations:
        parts.extend(f"- {item}" for item in result.observations)
    else:
        parts.append("- Ничего важного не выделено.")

    parts.append("")
    parts.append("Действия с файлами:")
    if result.file_actions:
        for action in result.file_actions:
            parts.append(f"- {action.mode}: {action.path}")
            if action.content:
                preview = action.content.strip()
                if len(preview) > 140:
                    preview = preview[:140] + "..."
                parts.append(f"  {preview}")
    else:
        parts.append("- Новых записей в файлы нет.")

    parts.append("")
    parts.append("События:")
    if result.events:
        for event in result.events:
            parts.append(f"- [{event.severity}/{event.category}] {event.message}")
    else:
        parts.append("- Событий нет.")

    if result.memory_updates:
        parts.append("")
        parts.append("Обновления памяти:")
        for key, value in result.memory_updates.items():
            parts.append(f"- {key}: {value}")

    parts.append("")
    parts.append("Распознавание числа:")
    if result.gesture_digit is not None:
        confidence = f" ({result.gesture_confidence})" if result.gesture_confidence else ""
        parts.append(f"- Число: {result.gesture_digit}{confidence}")
    else:
        parts.append("- Число не распознано.")
    if result.gesture_reason:
        parts.append(f"- Причина: {result.gesture_reason}")

    return "\n".join(parts).strip()


def extract_json_block(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("Empty model response.")

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("Could not find a complete JSON object in model response.")


def _clean_string(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return str(value).strip()


def _clean_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean_string(item) for item in value if _clean_string(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _extract_json_string_field(text: str, field: str) -> str:
    marker = f'"{field}"'
    field_index = text.find(marker)
    if field_index == -1:
        return ""
    colon_index = text.find(":", field_index + len(marker))
    if colon_index == -1:
        return ""
    quote_index = text.find('"', colon_index + 1)
    if quote_index == -1:
        return ""

    escaped = False
    for index in range(quote_index + 1, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            raw_value = text[quote_index : index + 1]
            try:
                return _clean_string(json.loads(raw_value))
            except Exception:
                return text[quote_index + 1 : index].strip()
    return ""


def _extract_json_string_array_field(text: str, field: str) -> list[str]:
    marker = f'"{field}"'
    field_index = text.find(marker)
    if field_index == -1:
        return []
    colon_index = text.find(":", field_index + len(marker))
    if colon_index == -1:
        return []
    start_index = text.find("[", colon_index + 1)
    end_index = text.find("]", start_index + 1)
    if start_index == -1 or end_index == -1:
        return []
    try:
        return _clean_string_list(json.loads(text[start_index : end_index + 1]))
    except Exception:
        return []


def parse_partial_analysis_response(text: str, reason: str) -> AnalysisResult:
    assistant_reply = _extract_json_string_field(text, "assistant_reply")
    summary = _extract_json_string_field(text, "summary")
    observations = _extract_json_string_array_field(text, "observations")
    gesture_confidence = _extract_json_string_field(text, "gesture_confidence")
    gesture_reason = _extract_json_string_field(text, "gesture_reason")

    if not assistant_reply and not summary and not observations:
        raise ValueError(reason)

    return AnalysisResult(
        assistant_reply=assistant_reply,
        summary=summary or assistant_reply[:80] or "Частичный ответ модели",
        observations=observations,
        gesture_confidence=gesture_confidence,
        gesture_reason=gesture_reason,
        raw={
            "partial": True,
            "parse_error": reason,
            "raw_response": text[:4000],
        },
    )


def parse_analysis_response(text: str) -> AnalysisResult:
    payload = json.loads(extract_json_block(text))

    result = AnalysisResult(
        assistant_reply=_clean_string(payload.get("assistant_reply")),
        summary=_clean_string(payload.get("summary")),
        observations=_clean_string_list(payload.get("observations", [])),
        memory_updates=payload.get("memory_updates", {}) if isinstance(payload.get("memory_updates", {}), dict) else {},
        gesture_confidence=_clean_string(payload.get("gesture_confidence")),
        gesture_reason=_clean_string(payload.get("gesture_reason")),
        raw=payload,
    )

    gesture_digit = payload.get("gesture_digit")
    if isinstance(gesture_digit, bool):
        gesture_digit = None
    if isinstance(gesture_digit, (int, float)):
        result.gesture_digit = int(gesture_digit)
    elif isinstance(gesture_digit, str) and gesture_digit.strip().isdigit():
        result.gesture_digit = int(gesture_digit.strip())

    for item in payload.get("file_actions", []):
        if not isinstance(item, dict):
            continue
        path = _clean_string(item.get("path"))
        content = _clean_string(item.get("content"))
        if not path:
            continue
        result.file_actions.append(
            FileAction(
                path=path,
                content=content,
                mode=_clean_string(item.get("mode"), "append").lower(),
            )
        )

    for item in payload.get("events", []):
        if not isinstance(item, dict):
            continue
        event_key = _clean_string(item.get("event_key"))
        message = _clean_string(item.get("message"))
        if not event_key or not message:
            continue
        result.events.append(
            EventAction(
                event_key=event_key,
                category=_clean_string(item.get("category"), "other"),
                severity=_clean_string(item.get("severity"), "medium"),
                message=message,
                notify=bool(item.get("notify", False)),
                trigger_command=_clean_string(item.get("trigger_command"), "none").lower(),
                save_frame=bool(item.get("save_frame", False)),
                cooldown_seconds=int(item.get("cooldown_seconds", 3600) or 3600),
            )
        )

    return result
