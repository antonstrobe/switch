from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .parsing import AnalysisResult, EventAction


TEXT_EXTENSIONS = {".txt", ".md", ".json", ".jsonl", ".csv", ".log", ".yaml", ".yml"}
EXTERNAL_COMMANDS_ENABLED = False
INTERNAL_CONTEXT_EXCLUDE = {
    "analysis_history.jsonl",
    "analysis_status.json",
    "actions.log",
    "events.jsonl",
    "last_analysis_view.txt",
    "last_error.txt",
    "last_request.json",
    "last_response.json",
    "launcher.log",
    "launcher_last_start.txt",
    "notifications.jsonl",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class OutputStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.output_dir = self.root / "output"
        self.state_path = self.output_dir / "state.json"
        self.events_path = self.output_dir / "events.jsonl"
        self.notifications_path = self.output_dir / "notifications.jsonl"
        self.actions_path = self.output_dir / "actions.log"
        self.analysis_status_path = self.output_dir / "analysis_status.json"
        self.analysis_history_path = self.output_dir / "analysis_history.jsonl"
        self.last_request_path = self.output_dir / "last_request.json"
        self.last_response_path = self.output_dir / "last_response.json"
        self.last_analysis_view_path = self.output_dir / "last_analysis_view.txt"
        self.digits_dir = self.output_dir / "digits"
        self.evidence_dir = self.output_dir / "evidence"
        self.ensure_layout()

    def ensure_layout(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.digits_dir.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._write_json(
                self.state_path,
                {
                    "memory": {},
                    "handled_events": {},
                    "last_summary": "",
                    "last_started_runtime": "",
                    "updated_at": utc_now_iso(),
                },
            )
        example_schedule = self.output_dir / "medication_schedule.example.json"
        if not example_schedule.exists():
            self._write_json(
                example_schedule,
                {
                    "timezone": "Europe/Moscow",
                    "items": [
                        {
                            "person": "example",
                            "medicine": "example-pill",
                            "time": "19:00",
                            "with_food": True,
                            "note": "Replace this file with your real schedule.",
                        }
                    ],
                },
            )
        notes = self.output_dir / "context_notes.txt"
        if not notes.exists():
            notes.write_text(
                "Add your own notes here. The model reads files from output on every analysis.\n",
                encoding="utf-8",
            )
        if not self.analysis_status_path.exists():
            self._write_json(
                self.analysis_status_path,
                {
                    "state": "idle",
                    "message": "Waiting for analysis start.",
                    "updated_at": utc_now_iso(),
                },
            )
        if not self.last_request_path.exists():
            self._write_json(
                self.last_request_path,
                {
                    "message": "No request has been sent yet.",
                    "saved_at": utc_now_iso(),
                },
            )
        if not self.last_response_path.exists():
            self._write_json(
                self.last_response_path,
                {
                    "message": "No response has been received yet.",
                    "saved_at": utc_now_iso(),
                },
            )
        if not self.last_analysis_view_path.exists():
            self.last_analysis_view_path.write_text(
                "Waiting for analysis start.\n",
                encoding="utf-8",
            )
        gesture_status = self.digits_dir / "status.txt"
        if not gesture_status.exists():
            gesture_status.write_text("Ожидание распознавания числа.\n", encoding="utf-8")

    def load_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"memory": {}, "handled_events": {}, "last_summary": "", "updated_at": utc_now_iso()}

    def save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now_iso()
        self._write_json(self.state_path, state)

    def write_status(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = utc_now_iso()
        self._write_json(self.analysis_status_path, payload)

    def write_last_request(self, payload: dict[str, Any]) -> None:
        payload["saved_at"] = utc_now_iso()
        self._write_json(self.last_request_path, payload)

    def append_analysis_history(self, payload: dict[str, Any]) -> None:
        payload["timestamp"] = utc_now_iso()
        self._append_jsonl(self.analysis_history_path, payload)

    def write_analysis_view(self, text: str) -> None:
        self.last_analysis_view_path.write_text(text.rstrip() + "\n", encoding="utf-8")

    def build_context(self, max_chars: int = 14000) -> str:
        chunks: list[str] = []
        state = self.load_state()
        chunks.append("STATE.JSON")
        chunks.append(json.dumps(state, ensure_ascii=False, indent=2))

        total = sum(len(chunk) for chunk in chunks)
        for file_path in sorted(self.output_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            relative = file_path.relative_to(self.output_dir).as_posix()
            if relative == "state.json":
                continue
            if relative in INTERNAL_CONTEXT_EXCLUDE:
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if len(content) > 2500:
                content = content[-2500:]
            block = f"\nFILE: {relative}\n{content.strip()}\n"
            if total + len(block) > max_chars:
                break
            chunks.append(block)
            total += len(block)
        return "\n".join(chunks).strip()

    def apply_result(
        self,
        result: AnalysisResult,
        frame_saver: callable | None,
        reminder_command: str,
        emergency_command: str,
        log_callback: callable | None = None,
        popup_callback: callable | None = None,
    ) -> dict[str, int]:
        state = self.load_state()
        state.setdefault("memory", {})
        state.setdefault("handled_events", {})
        summary = {"files": 0, "notifications": 0, "commands": 0, "saved_frames": 0}

        state["last_summary"] = result.summary
        state["memory"].update(result.memory_updates)
        self._apply_gesture_result(result, state, summary)

        for action in result.file_actions:
            if not action.content.strip():
                continue
            destination = self._safe_output_path(action.path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if action.mode == "overwrite":
                destination.write_text(action.content, encoding="utf-8")
            else:
                with destination.open("a", encoding="utf-8") as file_handle:
                    file_handle.write(action.content)
                    if action.content and not action.content.endswith("\n"):
                        file_handle.write("\n")
            summary["files"] += 1
            self._append_line(self.actions_path, f"{utc_now_iso()} FILE {action.mode.upper()} {destination.name}")

        for event in result.events:
            is_duplicate = self._is_duplicate(state, event)
            self._append_jsonl(
                self.events_path,
                {
                    "timestamp": utc_now_iso(),
                    "event": asdict(event),
                    "duplicate": is_duplicate,
                },
            )

            if is_duplicate:
                if log_callback:
                    log_callback(f"Пропущено повторное событие: {event.event_key}")
                continue

            state["handled_events"][event.event_key] = {
                "timestamp": utc_now_iso(),
                "severity": event.severity,
                "category": event.category,
                "message": event.message,
            }

            if event.notify:
                self._append_jsonl(
                    self.notifications_path,
                    {
                        "timestamp": utc_now_iso(),
                        "event_key": event.event_key,
                        "severity": event.severity,
                        "message": event.message,
                    },
                )
                summary["notifications"] += 1
                if popup_callback:
                    popup_callback(event.severity.upper(), event.message)

            if event.save_frame and frame_saver:
                saved = frame_saver(event.event_key)
                if saved:
                    summary["saved_frames"] += 1

            command = ""
            if event.trigger_command == "reminder":
                command = reminder_command.strip()
            elif event.trigger_command == "emergency":
                command = emergency_command.strip()

            if command:
                if EXTERNAL_COMMANDS_ENABLED:
                    self._run_user_command(command, event)
                    summary["commands"] += 1
                    self._append_line(self.actions_path, f"{utc_now_iso()} COMMAND {event.trigger_command.upper()} {event.event_key}")
                else:
                    self._append_line(
                        self.actions_path,
                        f"{utc_now_iso()} COMMAND_SKIPPED {event.trigger_command.upper()} {event.event_key}",
                    )
                    if log_callback:
                        log_callback(f"Внешняя команда отключена: {event.trigger_command} для {event.event_key}")

        self.save_state(state)
        return summary

    def _apply_gesture_result(self, result: AnalysisResult, state: dict[str, Any], summary: dict[str, int]) -> None:
        timestamp = utc_now_iso()
        status_path = self.digits_dir / "status.txt"
        detection_path = self.digits_dir / "last_detection.json"
        current_path = self.digits_dir / "current.txt"
        history_path = self.digits_dir / "history.txt"

        if result.gesture_digit is None:
            status_text = "Число не распознано."
            if result.gesture_reason:
                status_text += f" {result.gesture_reason}"
            status_path.write_text(f"{timestamp}\n{status_text}\n", encoding="utf-8")
            detection_path.write_text(
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "digit": None,
                        "confidence": result.gesture_confidence,
                        "reason": result.gesture_reason,
                        "summary": result.summary,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            summary["files"] += 2
            state["memory"]["gesture_last_status"] = status_text
            return

        digit_text = str(result.gesture_digit)
        status_text = f"Распознано число: {digit_text}"
        if result.gesture_confidence:
            status_text += f" ({result.gesture_confidence})"
        if result.gesture_reason:
            status_text += f". {result.gesture_reason}"

        current_path.write_text(digit_text, encoding="utf-8")
        status_path.write_text(f"{timestamp}\n{status_text}\n", encoding="utf-8")
        detection_path.write_text(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "digit": result.gesture_digit,
                    "confidence": result.gesture_confidence,
                    "reason": result.gesture_reason,
                    "summary": result.summary,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        summary["files"] += 3

        last_written_digit = state["memory"].get("gesture_last_written_digit")
        if str(last_written_digit) != digit_text:
            with history_path.open("a", encoding="utf-8") as file_handle:
                file_handle.write(f"{timestamp} {digit_text}\n")
            summary["files"] += 1
            state["memory"]["gesture_last_written_digit"] = digit_text

        state["memory"]["gesture_last_status"] = status_text

    def _is_duplicate(self, state: dict[str, Any], event: EventAction) -> bool:
        existing = state.get("handled_events", {}).get(event.event_key)
        if not existing:
            return False
        try:
            previous = datetime.fromisoformat(existing["timestamp"])
            age_seconds = (datetime.now(timezone.utc) - previous).total_seconds()
            return age_seconds < max(event.cooldown_seconds, 0)
        except Exception:
            return True

    def _run_user_command(self, command: str, event: EventAction) -> None:
        raise RuntimeError("External commands are disabled.")

    def _safe_output_path(self, relative_path: str) -> Path:
        normalized = relative_path.replace("\\", "/").lstrip("./")
        if normalized.lower().startswith("output/"):
            normalized = normalized[7:]
        candidate = (self.output_dir / normalized).resolve()
        output_root = self.output_dir.resolve()
        if output_root not in candidate.parents and candidate != output_root:
            raise ValueError(f"Path escapes output directory: {relative_path}")
        return candidate

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        self._append_line(path, json.dumps(payload, ensure_ascii=False))

    def _append_line(self, path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(line.rstrip() + "\n")
