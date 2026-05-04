from __future__ import annotations

import json
import queue
import subprocess
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk

from .engine import CameraMonitor, DEFAULT_PROMPT, MonitorConfig
from .parsing import parse_analysis_response
from .runtimes import DEFAULT_GPU_MODE, RuntimeErrorBase, RuntimeModel, RuntimeRegistry, normalize_gpu_mode
from .storage import OutputStore


class SwitchMonitorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Switch Vision Monitor")
        self.root.geometry("1220x860")

        self.project_root = Path.cwd()
        self.config_path = self.project_root / "app_config.json"
        self.registry = RuntimeRegistry()
        self.store = OutputStore(self.project_root)
        self.ui_queue: queue.Queue = queue.Queue()
        self.preview_image: ImageTk.PhotoImage | None = None
        self.models_by_runtime: dict[str, list[RuntimeModel]] = {}
        self.monitor: CameraMonitor | None = None
        self.saved_model_name = ""
        self.pending_progress: dict | None = None
        self.analysis_history_entries: list[str] = []
        self.answer_entries: list[dict] = []
        self.answer_modal: tk.Toplevel | None = None
        self.answer_modal_text: ScrolledText | None = None
        self.analysis_current_message = "Ожидание первого анализа кадра."
        self.analysis_history_path = self.store.output_dir / "analysis_view_history.txt"

        self.runtime_var = tk.StringVar(value="auto")
        self.model_var = tk.StringVar()
        self.gpu_mode_var = tk.StringVar(value=DEFAULT_GPU_MODE)
        self.camera_var = tk.StringVar(value="0")
        self.interval_var = tk.StringVar(value="6")
        self.reminder_command_var = tk.StringVar()
        self.emergency_command_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Готово")

        self._build_ui()
        self._load_saved_config()
        self._load_analysis_history()
        self.refresh_models()
        self.root.after(100, self._process_ui_queue)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=10)
        right = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")
        left.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=2, minsize=170)
        right.rowconfigure(3, weight=1)
        right.rowconfigure(4, weight=1)
        left.columnconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        controls = ttk.LabelFrame(left, text="Управление", padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Рантайм").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.runtime_box = ttk.Combobox(
            controls,
            textvariable=self.runtime_var,
            state="readonly",
            values=["auto", "official-gemma"],
        )
        self.runtime_box.grid(row=0, column=1, sticky="ew", pady=4)
        self.runtime_box.bind("<<ComboboxSelected>>", lambda _event: self._on_runtime_change())

        ttk.Label(controls, text="Gemma модель").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self.model_box = ttk.Combobox(controls, textvariable=self.model_var, state="readonly")
        self.model_box.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(controls, text="Камера").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(controls, textvariable=self.camera_var).grid(row=2, column=1, sticky="ew", pady=4)

        ttk.Label(controls, text="Интервал, сек").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(controls, textvariable=self.interval_var).grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Label(controls, text="Команда напоминания").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(controls, textvariable=self.reminder_command_var, state="disabled").grid(row=4, column=1, sticky="ew", pady=4)

        ttk.Label(controls, text="Команда тревоги").grid(row=5, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(controls, textvariable=self.emergency_command_var, state="disabled").grid(row=5, column=1, sticky="ew", pady=4)

        ttk.Label(controls, text="GPU").grid(row=6, column=0, sticky="w", padx=(0, 8), pady=4)
        self.gpu_mode_box = ttk.Combobox(
            controls,
            textvariable=self.gpu_mode_var,
            state="readonly",
            values=["max", "auto", "off", "0.75", "0.5", "0.25"],
        )
        self.gpu_mode_box.grid(row=6, column=1, sticky="ew", pady=4)

        buttons = ttk.Frame(controls)
        buttons.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for index in range(4):
            buttons.columnconfigure(index, weight=1)
        ttk.Button(buttons, text="Обновить модели", command=self.refresh_models).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(buttons, text="Старт", command=self.start_monitoring).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(buttons, text="Стоп", command=self.stop_monitoring).grid(row=0, column=2, sticky="ew", padx=2)
        ttk.Button(buttons, text="Открыть output", command=self.open_output).grid(row=0, column=3, sticky="ew", padx=2)

        preview_frame = ttk.LabelFrame(left, text="Камера", padding=8)
        preview_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(preview_frame, text="Камера не запущена", anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        prompt_frame = ttk.LabelFrame(right, text="Промпт", padding=8)
        prompt_frame.grid(row=0, column=0, sticky="nsew")
        prompt_frame.columnconfigure(0, weight=1)
        prompt_frame.rowconfigure(0, weight=1)
        self.prompt_text = ScrolledText(prompt_frame, wrap="word", height=18)
        self.prompt_text.grid(row=0, column=0, sticky="nsew")
        self.prompt_text.insert("1.0", DEFAULT_PROMPT)

        info_frame = ttk.LabelFrame(right, text="Статус", padding=8)
        info_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(info_frame, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(info_frame, text=f"Папка output: {self.store.output_dir}").grid(row=1, column=0, sticky="w", pady=(4, 0))

        hint_frame = ttk.LabelFrame(right, text="Подсказка", padding=8)
        hint_frame.grid_remove()
        hint = (
            "Сейчас внешние команды отключены. "
            "Приложение работает только с камерой, анализом и файлами в output. "
            "Модель читает файлы из output на каждом цикле, поэтому туда можно класть расписания, заметки и правила."
        )
        ttk.Label(hint_frame, text=hint, wraplength=400, justify="left").grid(row=0, column=0, sticky="w")

        analysis_frame = ttk.LabelFrame(right, text="Что видит модель", padding=8)
        analysis_frame.configure(text="Ответы")
        analysis_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        analysis_frame.columnconfigure(0, weight=1)
        analysis_frame.rowconfigure(0, weight=1)
        self.answer_list = tk.Listbox(analysis_frame, height=6, activestyle="dotbox")
        self.answer_list.grid(row=0, column=0, sticky="nsew")
        answer_scrollbar = ttk.Scrollbar(analysis_frame, orient="vertical", command=self.answer_list.yview)
        answer_scrollbar.grid(row=0, column=1, sticky="ns")
        self.answer_list.configure(yscrollcommand=answer_scrollbar.set)
        self.answer_list.bind("<Double-Button-1>", self._open_selected_answer)
        self.answer_list.bind("<Return>", self._open_selected_answer)
        ttk.Button(analysis_frame, text="Открыть", command=self._open_selected_answer).grid(row=1, column=0, sticky="e", pady=(6, 0))
        self._render_analysis_view()

        log_frame = ttk.LabelFrame(right, text="Лог", padding=8)
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = ScrolledText(log_frame, wrap="word", height=18, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")

    def refresh_models(self) -> None:
        self.log("Ищу Gemma в локальном рантайме, LM Studio и Ollama...")
        self.models_by_runtime = {"auto": [], "official-gemma": []}
        for model in self.registry.detect_all_models():
            self.models_by_runtime.setdefault(model.runtime, []).append(model)
        self._on_runtime_change()
        self.log("Поиск моделей завершён.")

    def _on_runtime_change(self) -> None:
        runtime = self.runtime_var.get()
        if runtime == "auto":
            self.model_box["values"] = ["Автовыбор"]
            self.model_var.set("Автовыбор")
            return
        models = self.models_by_runtime.get(runtime, [])
        values = [model.display_name for model in models]
        self.model_box["values"] = values
        if values:
            preferred = self.saved_model_name or self.model_var.get()
            self.model_var.set(preferred if preferred in values else values[0])
        else:
            self.model_var.set("")

    def _choose_auto_runtime(self, gpu_mode: str):
        candidates = self._auto_candidates()
        if not candidates:
            raise RuntimeError("Не найдены vision-модели Gemma в локальном рантайме, LM Studio или Ollama.")

        self.status_var.set("Автовыбор рантайма: тестирую локальную Gemma, LM Studio и Ollama...")
        self.root.update_idletasks()
        benchmark_results: list[dict] = []
        best: dict | None = None

        for model in candidates:
            result = self._benchmark_candidate(model, gpu_mode)
            benchmark_results.append(result)
            if result["ok"] and model.runtime == "official-gemma":
                best = result
                break
            if result["ok"] and (best is None or result["total_seconds"] < best["total_seconds"]):
                best = result

        benchmark_path = self.store.output_dir / "runtime_benchmark.json"
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        serializable_results = [
            {key: value for key, value in item.items() if key not in {"runtime_model", "runtime_object"}}
            for item in benchmark_results
        ]
        benchmark_path.write_text(json.dumps(serializable_results, ensure_ascii=False, indent=2), encoding="utf-8")

        if best is None:
            details = "; ".join(f"{item['runtime']}/{item['model']}: {item['error']}" for item in benchmark_results)
            raise RuntimeError(f"Локальная Gemma, LM Studio и Ollama не прошли тест. {details}")

        for item in benchmark_results:
            if item["runtime"] == "ollama" and best["runtime"] != "ollama":
                try:
                    subprocess.run(["ollama", "stop", item["model"]], capture_output=True, text=True, timeout=30, check=False)
                except Exception:
                    pass

        self.log(
            f"Автовыбор: {best['runtime']}/{best['model']} "
            f"готов за {best['total_seconds']} сек, ответ за {best['response_seconds']} сек"
        )
        self.status_var.set(f"Выбран {best['runtime']}/{best['prepared_model']}")
        return best["runtime"], best["runtime_model"], best["runtime_object"], best["prepared_model"]

    def _auto_candidates(self) -> list[RuntimeModel]:
        candidates: list[RuntimeModel] = []
        for runtime_name in ("official-gemma",):
            models = self.models_by_runtime.get(runtime_name, [])
            if not models:
                continue
            candidates.append(self._preferred_auto_model(models))
        return candidates

    def _preferred_auto_model(self, models: list[RuntimeModel]) -> RuntimeModel:
        def score(model: RuntimeModel) -> tuple[int, str]:
            text = f"{model.model_id} {model.display_name}".lower()
            if model.runtime == "official-gemma":
                return (0, text)
            if "e2b" in text:
                return (1, text)
            if "fast" in text or "1b" in text:
                return (2, text)
            return (3, text)

        return sorted(models, key=score)[0]

    def _benchmark_candidate(self, model: RuntimeModel, gpu_mode: str) -> dict:
        runtime = self.registry.get(model.runtime)
        started = time.perf_counter()
        result = {
            "runtime": model.runtime,
            "model": model.model_id,
            "display_name": model.display_name,
            "ok": False,
            "prepared_model": "",
            "prepare_seconds": None,
            "response_seconds": None,
            "total_seconds": None,
            "error": "",
        }
        self.log(f"Тестирую {model.runtime}/{model.model_id}...")
        self.root.update_idletasks()
        try:
            prepare_started = time.perf_counter()
            prepared_model = runtime.prepare(model.model_id, gpu_mode=gpu_mode)
            result["prepared_model"] = prepared_model
            result["prepare_seconds"] = round(time.perf_counter() - prepare_started, 2)

            response_started = time.perf_counter()
            raw_response = runtime.analyze(
                prepared_model,
                system_prompt=self._benchmark_system_prompt(),
                user_prompt=self._benchmark_user_prompt(),
                image_bytes=self._benchmark_image_bytes(),
                timeout=45,
            )
            result["response_seconds"] = round(time.perf_counter() - response_started, 2)
            parse_analysis_response(raw_response)
            result["ok"] = True
            result["runtime_model"] = model
            result["runtime_object"] = runtime
            self.log(
                f"Тест OK: {model.runtime}/{model.model_id}, "
                f"ответ {result['response_seconds']} сек"
            )
        except Exception as error:
            result["error"] = str(error)
            self.log(f"Тест ошибка: {model.runtime}/{model.model_id}: {error}")
        finally:
            result["total_seconds"] = round(time.perf_counter() - started, 2)
        return result

    def _benchmark_image_bytes(self) -> bytes:
        import io

        image = Image.new("RGB", (96, 96), color=(245, 245, 245))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        return buffer.getvalue()

    def _benchmark_system_prompt(self) -> str:
        return (
            "Return ONLY one valid compact JSON object. No markdown. "
            'Schema: {"assistant_reply":"short Russian answer","summary":"short","observations":[],'
            '"gesture_digit":null,"gesture_confidence":"low|medium|high","gesture_reason":"short",'
            '"file_actions":[],"memory_updates":{},"events":[]}'
        )

    def _benchmark_user_prompt(self) -> str:
        return "Health check. Describe the simple image in JSON only. Use empty file_actions and events."

    def start_monitoring(self) -> None:
        try:
            if self.monitor is not None:
                raise RuntimeError("Мониторинг уже запущен. Сначала нажмите Стоп.")

            runtime_name = self.runtime_var.get().strip()
            selected_model = self.model_var.get().strip()
            gpu_mode = normalize_gpu_mode(self.gpu_mode_var.get())
            self.gpu_mode_var.set(gpu_mode)
            if runtime_name == "auto":
                runtime_name, match, runtime, prepared_model_id = self._choose_auto_runtime(gpu_mode)
            else:
                runtime_models = self.models_by_runtime.get(runtime_name, [])
                match = next((item for item in runtime_models if item.display_name == selected_model), None)
                if not match:
                    raise RuntimeError("Не выбрана модель Gemma.")

                runtime = self.registry.get(runtime_name)
                self.status_var.set(f"Подготовка {runtime_name}...")
                self.log(f"Подготавливаю рантайм {runtime_name} и модель {match.model_id}")
                prepared_model_id = runtime.prepare(match.model_id, gpu_mode=gpu_mode)
            self.log(f"GPU mode: {gpu_mode}")
            self.log(f"Модель готова: {prepared_model_id}")
            self.log("Первая обработка может занять 30-120 секунд, пока модель прогревается.")

            config = MonitorConfig(
                runtime_name=runtime_name,
                model_id=match.model_id,
                prepared_model_id=prepared_model_id,
                camera_index=int(self.camera_var.get().strip() or 0),
                interval_seconds=max(float(self.interval_var.get().strip() or 6), 1.0),
                prompt=self.prompt_text.get("1.0", "end").strip(),
                gpu_mode=gpu_mode,
                reminder_command=self.reminder_command_var.get().strip(),
                emergency_command=self.emergency_command_var.get().strip(),
            )
            self._save_config()
            state = self.store.load_state()
            state["last_started_runtime"] = f"{runtime_name}/{prepared_model_id} GPU={gpu_mode}"
            self.store.save_state(state)

            self.monitor = CameraMonitor(
                self.project_root,
                runtime=runtime,
                store=self.store,
                preview_callback=lambda frame: self.ui_queue.put(("preview", frame)),
                log_callback=lambda message: self.ui_queue.put(("log", message)),
                popup_callback=lambda title, message: self.ui_queue.put(("popup", title, message)),
                status_callback=lambda message: self.ui_queue.put(("status", message)),
                analysis_callback=lambda message: self.ui_queue.put(("analysis", message)),
                answer_callback=lambda payload: self.ui_queue.put(("answer", payload)),
                progress_callback=lambda payload: self.ui_queue.put(("progress", payload)),
            )
            self.monitor.start(config)
            self.status_var.set(f"Мониторинг запущен: {runtime_name}/{prepared_model_id}")
            self.log("Мониторинг запущен.")
        except RuntimeErrorBase as error:
            self.status_var.set("Ошибка запуска")
            messagebox.showerror("Runtime error", str(error))
        except Exception as error:
            self.status_var.set("Ошибка запуска")
            messagebox.showerror("Start error", str(error))

    def stop_monitoring(self) -> None:
        if self.monitor:
            self.monitor.stop()
            self.monitor = None
        self.pending_progress = None
        self.status_var.set("Мониторинг остановлен")
        self.log("Мониторинг остановлен.")

    def open_output(self) -> None:
        subprocess.Popen(["explorer.exe", str(self.store.output_dir)])

    def log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def set_analysis_view(self, message: str) -> None:
        message = message.rstrip()
        if self._is_final_analysis_message(message):
            self.analysis_current_message = ""
        else:
            self.analysis_current_message = message
        self._render_analysis_view()

    def add_answer(self, payload: dict) -> None:
        text = str(payload.get("text") or "").strip()
        if not text:
            return
        if self.analysis_history_entries and self.analysis_history_entries[-1] == text:
            return
        entry = dict(payload)
        entry["text"] = text
        self.answer_entries.append(entry)
        self.analysis_history_entries.append(text)
        self._save_analysis_history()
        self._render_analysis_view()
        self._show_answer_modal(entry)

    def _process_ui_queue(self) -> None:
        while not self.ui_queue.empty():
            item = self.ui_queue.get()
            kind = item[0]
            if kind == "log":
                self.log(item[1])
            elif kind == "answer":
                self.add_answer(item[1])
            elif kind == "analysis":
                self.set_analysis_view(item[1])
            elif kind == "progress":
                self._handle_progress(item[1])
            elif kind == "status":
                self.status_var.set(item[1])
            elif kind == "popup":
                messagebox.showwarning(item[1], item[2])
            elif kind == "preview":
                self._update_preview(item[1])
        self._refresh_pending_status()
        self.root.after(100, self._process_ui_queue)

    def _handle_progress(self, payload: dict) -> None:
        state = payload.get("state")
        if state == "waiting_model":
            self.pending_progress = payload
        else:
            self.pending_progress = None

    def _refresh_pending_status(self) -> None:
        if not self.pending_progress:
            return
        try:
            import time

            analysis_id = self.pending_progress.get("analysis_id", "?")
            started_at = float(self.pending_progress.get("started_at", time.time()))
            elapsed = int(max(time.time() - started_at, 0))
            self.status_var.set(f"Анализ #{analysis_id}: ожидание ответа модели ({elapsed} сек)")
        except Exception:
            return

    def _render_analysis_view(self) -> None:
        blocks: list[str] = []
        if self.analysis_history_entries:
            blocks.extend(self.analysis_history_entries)
        if self.analysis_current_message:
            blocks.append(self.analysis_current_message)

        content = "\n\n" + ("\n\n" + ("=" * 56) + "\n\n").join(blocks) if blocks else "Ожидание первого анализа кадра."
        content = content.lstrip()

        if not hasattr(self, "answer_list"):
            return
        self.answer_list.delete(0, "end")
        if not self.answer_entries:
            self.answer_list.insert("end", "Ответов пока нет")
            return
        for entry in self.answer_entries:
            self.answer_list.insert("end", self._answer_list_label(entry))
        self.answer_list.see("end")

    def _answer_list_label(self, entry: dict) -> str:
        analysis_id = entry.get("analysis_id")
        timestamp = str(entry.get("timestamp") or "").strip()
        title = str(entry.get("assistant_reply") or entry.get("summary") or "").strip()
        if not title:
            title = self._first_text_line(str(entry.get("text") or ""))
        title = " ".join(title.split())
        if len(title) > 110:
            title = title[:107].rstrip() + "..."
        prefix = f"#{analysis_id}" if analysis_id else "Ответ"
        if timestamp:
            return f"{prefix} | {timestamp} | {title}"
        return f"{prefix} | {title}"

    def _first_text_line(self, text: str) -> str:
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line
        return "Пустой ответ"

    def _open_selected_answer(self, _event=None) -> None:
        if not hasattr(self, "answer_list"):
            return
        selection = self.answer_list.curselection()
        if not selection:
            return
        index = int(selection[0])
        if index >= len(self.answer_entries):
            return
        self._show_answer_modal(self.answer_entries[index])

    def _show_answer_modal(self, entry: dict) -> None:
        if self.answer_modal is None or not self.answer_modal.winfo_exists():
            window = tk.Toplevel(self.root)
            window.title("Ответ модели")
            window.geometry("720x520")
            window.minsize(420, 260)
            window.resizable(True, True)
            window.transient(self.root)
            window.columnconfigure(0, weight=1)
            window.rowconfigure(0, weight=1)
            self.answer_modal = window

            self.answer_modal_text = ScrolledText(window, wrap="word")
            self.answer_modal_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))
            close_button = ttk.Button(window, text="Закрыть", command=self._close_answer_modal)
            close_button.grid(row=1, column=0, sticky="e", padx=10, pady=(0, 10))
            window.protocol("WM_DELETE_WINDOW", self._close_answer_modal)

        self._set_answer_modal_text(str(entry.get("text") or ""))
        self.answer_modal.deiconify()
        self.answer_modal.lift()
        self.answer_modal.focus_force()
        try:
            self.answer_modal.grab_set()
        except tk.TclError:
            pass

    def _set_answer_modal_text(self, text: str) -> None:
        if self.answer_modal_text is None:
            return
        self.answer_modal_text.configure(state="normal")
        self.answer_modal_text.delete("1.0", "end")
        self.answer_modal_text.insert("1.0", text.strip() + "\n")
        self.answer_modal_text.see("1.0")
        self.answer_modal_text.configure(state="disabled")

    def _close_answer_modal(self) -> None:
        if self.answer_modal is None:
            return
        try:
            self.answer_modal.grab_release()
        except tk.TclError:
            pass
        self.answer_modal.withdraw()

    def _is_final_analysis_message(self, message: str) -> bool:
        return (
            "Кратко:" in message
            or "Ошибка анализа:" in message
            or "Распознавание числа:" in message
        )

    def _load_analysis_history(self) -> None:
        if not self.analysis_history_path.exists():
            return
        try:
            raw = self.analysis_history_path.read_text(encoding="utf-8")
        except Exception:
            return
        parts = [item.strip() for item in raw.split("\n\n" + ("=" * 56) + "\n\n") if item.strip()]
        self.analysis_history_entries = parts
        self.answer_entries = [{"text": item} for item in parts]

    def _save_analysis_history(self) -> None:
        separator = "\n\n" + ("=" * 56) + "\n\n"
        payload = separator.join(self.analysis_history_entries).strip()
        self.analysis_history_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")

    def _update_preview(self, frame) -> None:
        try:
            import cv2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((760, 620))
            self.preview_image = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=self.preview_image, text="")
        except Exception as error:
            self.log(f"Ошибка обновления превью: {error}")

    def _save_config(self) -> None:
        payload = {
            "runtime": self.runtime_var.get().strip(),
            "model": self.model_var.get().strip(),
            "gpu_mode": normalize_gpu_mode(self.gpu_mode_var.get()),
            "camera_index": self.camera_var.get().strip(),
            "interval_seconds": self.interval_var.get().strip(),
            "reminder_command": self.reminder_command_var.get().strip(),
            "emergency_command": self.emergency_command_var.get().strip(),
            "prompt": self.prompt_text.get("1.0", "end").strip(),
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_saved_config(self) -> None:
        if not self.config_path.exists():
            return
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.runtime_var.set(payload.get("runtime", self.runtime_var.get()))
        self.saved_model_name = payload.get("model", "").strip()
        if self.saved_model_name:
            self.model_var.set(self.saved_model_name)
        self.gpu_mode_var.set(normalize_gpu_mode(payload.get("gpu_mode", self.gpu_mode_var.get())))
        self.camera_var.set(str(payload.get("camera_index", self.camera_var.get())))
        self.interval_var.set(str(payload.get("interval_seconds", self.interval_var.get())))
        self.reminder_command_var.set(payload.get("reminder_command", ""))
        self.emergency_command_var.set(payload.get("emergency_command", ""))
        prompt = payload.get("prompt", "").strip()
        if prompt:
            self.prompt_text.delete("1.0", "end")
            self.prompt_text.insert("1.0", prompt)

    def shutdown(self) -> None:
        try:
            self.stop_monitoring()
        finally:
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = SwitchMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.shutdown)
    root.mainloop()
