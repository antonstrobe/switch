from __future__ import annotations

import json
import sys
import queue
import threading
import time
import tkinter as tk
import ctypes
from collections import deque
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk

from .paths import app_root
from .runtimes import DEFAULT_GPU_MODE, RuntimeErrorBase, RuntimeModel, RuntimeRegistry, normalize_gpu_mode
from .storage import OutputStore
from .vision_memory import RedisMemoryStore, VisionMemoryConfig


DEFAULT_MODEL_LABEL = "Автовыбор"
DEFAULT_PROMPT = (
    "Опиши текущий кадр максимально подробно на русском языке. "
    "Укажи объекты, людей, руки, жесты, текст, цвета, освещение, фон, расположение деталей и возможные риски. "
    "Если что-то видно неуверенно, прямо напиши, что это предположение. Не выдумывай то, чего нет на изображении."
)
BG = "#070b10"
PANEL = "#101822"
PANEL_2 = "#151f2b"
TEXT = "#eef4ff"
MUTED = "#8ea0b8"
ACCENT = "#28c7a3"
DANGER = "#ff6b6b"
LINE = "#263343"
SOURCE_MODES = {
    "Камера": "camera",
    "Область": "screen_region",
    "2 области": "screen_regions",
    "Экран": "screen_full",
}
SOURCE_LABELS = {value: key for key, value in SOURCE_MODES.items()}
APP_MODES = {
    "Классификатор": "classifier",
    "Библиотека тегов": "tag_library",
}
APP_MODE_LABELS = {value: key for key, value in APP_MODES.items()}


def _enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


_enable_dpi_awareness()


class SwitchDesktopApp:
    def __init__(self, root: tk.Tk, project_root: Path | None = None) -> None:
        self.root = root
        self.project_root = project_root or app_root()
        self.config_path = self.project_root / "app_config.json"
        self.registry = RuntimeRegistry(self.project_root)
        self.store = OutputStore(self.project_root)
        self.monitor: object | None = None
        self.models_by_runtime: dict[str, list[RuntimeModel]] = {}
        self.ui_queue: queue.Queue = queue.Queue()
        self.logs: deque[str] = deque(maxlen=300)
        self.answers: deque[dict] = deque(maxlen=100)
        self.preview_image: ImageTk.PhotoImage | None = None
        self.settings_window: tk.Toplevel | None = None
        self.logs_window: tk.Toplevel | None = None
        self.logs_text: ScrolledText | None = None
        self.answer_text: ScrolledText | None = None
        self.library_tree: ttk.Treeview | None = None
        self.library_image_label: tk.Label | None = None
        self.library_image: ImageTk.PhotoImage | None = None
        self.library_item_payload: dict[str, dict] = {}
        self.pending_progress: dict | None = None
        self.screen_region: dict[str, int] | None = None
        self.screen_regions: list[dict[str, int | str] | None] = [None, None]
        self.memory_store = RedisMemoryStore()
        self.evidence_items: deque[dict] = deque(maxlen=10)

        self.app_mode_var = tk.StringVar(value=APP_MODE_LABELS["classifier"])
        self.runtime_var = tk.StringVar(value="auto")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL_LABEL)
        self.source_mode_var = tk.StringVar(value=SOURCE_LABELS["camera"])
        self.gpu_mode_var = tk.StringVar(value=DEFAULT_GPU_MODE)
        self.camera_var = tk.StringVar(value="0")
        self.interval_var = tk.StringVar(value="6")
        self.memory_interval_var = tk.StringVar(value="2")
        self.memory_ttl_var = tk.StringVar(value="86400")
        self.current_room_var = tk.StringVar(value="")
        self.geo_label_var = tk.StringVar(value="")
        self.tag_filter_var = tk.StringVar(value="")
        self.prompt_var = tk.StringVar(value=DEFAULT_PROMPT)
        self.status_var = tk.StringVar(value="Готово")
        self.answer_var = tk.StringVar(value="Ответ появится после первого анализа.")
        self.meta_var = tk.StringVar(value="")
        self.library_status_var = tk.StringVar(value="Библиотека: последние наблюдения")
        self.capture_indicator_var = tk.StringVar(value="Захват: выключен")
        self.memory_status_var = tk.StringVar(value="Память: проверка")
        self.memory_summary_var = tk.StringVar(value="Наблюдений: 0")
        self.memory_objects_var = tk.StringVar(value="Объекты: нет")
        self.answer_images: list[ImageTk.PhotoImage] = []

        self._configure_window()
        self._load_config()
        self._configure_memory_store()
        self.refresh_models()
        self._build_ui()
        self.root.after(100, self._process_queue)
        self.root.after(500, self._refresh_pending_status)

    def _configure_window(self) -> None:
        self.root.title("Switch Vision Monitor")
        self.root.geometry("1100x700")
        self.root.minsize(900, 560)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)

    def _configure_tree_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Memory.Treeview",
            background="#090e14",
            fieldbackground="#090e14",
            foreground=TEXT,
            borderwidth=0,
            rowheight=24,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Memory.Treeview.Heading",
            background=PANEL_2,
            foreground=TEXT,
            relief="flat",
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Memory.Treeview", background=[("selected", "#1c564b")], foreground=[("selected", TEXT)])

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = tk.Frame(self.root, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 8))
        header.columnconfigure(0, weight=1)

        title = tk.Label(
            header,
            text="Switch Vision Monitor",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 22, "bold"),
            anchor="w",
        )
        title.grid(row=0, column=0, sticky="w")
        subtitle = tk.Label(
            header,
            text="Локальная Gemma 4, камера и чистый ответ без браузера.",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9),
            anchor="w",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(2, 0))

        buttons = tk.Frame(header, bg=BG)
        buttons.grid(row=0, column=1, rowspan=2, sticky="e")

        mode_label = tk.Label(buttons, text="Режим", bg=BG, fg=MUTED, font=("Segoe UI", 9))
        mode_label.grid(row=0, column=0, padx=(0, 5))
        self.app_mode_menu = self._compact_option(buttons, self.app_mode_var, list(APP_MODES.keys()))
        self.app_mode_menu.grid(row=0, column=1, padx=4)
        self.source_menu = self._compact_option(buttons, self.source_mode_var, list(SOURCE_MODES.keys()))
        self.source_menu.grid(row=0, column=2, padx=4)
        self._button(buttons, "Старт", self.start_monitoring, ACCENT).grid(row=0, column=3, padx=4)
        self._button(buttons, "Стоп", self.stop_monitoring, "#79343a").grid(row=0, column=4, padx=4)
        self._button(buttons, "Область 1", lambda: self.select_screen_region(1), "#2e5d73").grid(row=0, column=5, padx=4)
        self._button(buttons, "Область 2", lambda: self.select_screen_region(2), "#2e5d73").grid(row=0, column=6, padx=4)
        self._button(buttons, "Выгрузить", self.clear_model_memory, "#2e5d73").grid(row=0, column=7, padx=4)
        self._button(buttons, "Очистить память", self.clear_visual_memory, "#2e5d73").grid(row=0, column=8, padx=4)

        menu_button = tk.Menubutton(
            buttons,
            text="Меню ▾",
            bg=PANEL_2,
            fg=TEXT,
            activebackground=PANEL_2,
            activeforeground=TEXT,
            relief="flat",
            padx=10,
            pady=7,
            font=("Segoe UI", 9, "bold"),
        )
        menu = tk.Menu(menu_button, tearoff=False, bg=PANEL_2, fg=TEXT, activebackground=ACCENT, activeforeground="#04110e")
        menu.add_command(label="Настройки", command=self.open_settings)
        menu.add_command(label="Логи", command=self.open_logs)
        menu_button.configure(menu=menu)
        menu_button.grid(row=0, column=9, padx=(4, 0))

        body = tk.Frame(self.root, bg=BG)
        body.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        camera_card = self._card(body)
        camera_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        camera_card.rowconfigure(1, weight=1)
        camera_card.columnconfigure(0, weight=1)
        self._card_title(camera_card, "Кадр").grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        self.preview_label = tk.Label(
            camera_card,
            text="Камера появится после старта",
            bg="#030507",
            fg=MUTED,
            font=("Segoe UI", 13),
            anchor="center",
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        side = tk.Frame(body, bg=BG)
        side.grid(row=0, column=1, sticky="nsew")
        side.rowconfigure(0, weight=1)
        side.rowconfigure(1, weight=1)
        side.columnconfigure(0, weight=1)

        answer_card = self._card(side)
        answer_card.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        answer_card.columnconfigure(0, weight=1)
        answer_card.rowconfigure(3, weight=1)
        self._card_title(answer_card, "Библиотека JSON").grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        tag_row = tk.Frame(answer_card, bg=PANEL)
        tag_row.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))
        tag_row.columnconfigure(0, weight=1)
        self.tag_entry = tk.Entry(
            tag_row,
            textvariable=self.tag_filter_var,
            bg="#090e14",
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.tag_entry.grid(row=0, column=0, sticky="ew", ipady=7)
        self.tag_entry.bind("<Return>", lambda _event: self.search_tag_library())
        self._button(tag_row, "Поиск", self.search_tag_library, ACCENT).grid(row=0, column=1, padx=(8, 0))
        self._button(tag_row, "ИИ поиск", self.search_ai_library, "#2e5d73").grid(row=0, column=2, padx=(8, 0))
        self._button(tag_row, "Сбросить", self.reset_library_search, "#2e5d73").grid(row=0, column=3, padx=(8, 0))
        self.library_image_label = tk.Label(
            answer_card,
            text="Изображение появится после выбора наблюдения",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
            anchor="center",
        )
        self.library_image_label.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 6))
        self._configure_tree_style()
        tree_frame = tk.Frame(answer_card, bg=PANEL)
        tree_frame.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 8))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.library_tree = ttk.Treeview(tree_frame, columns=("value",), show="tree headings", style="Memory.Treeview")
        self.library_tree.heading("#0", text="JSON")
        self.library_tree.heading("value", text="Значение")
        self.library_tree.column("#0", width=260, minwidth=160, stretch=True)
        self.library_tree.column("value", width=160, minwidth=80, stretch=True)
        self.library_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll = tk.Scrollbar(tree_frame, command=self.library_tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.library_tree.configure(yscrollcommand=tree_scroll.set)
        self.library_tree.bind("<<TreeviewSelect>>", self._on_library_tree_select)
        meta = tk.Label(answer_card, textvariable=self.library_status_var, bg=PANEL, fg=MUTED, font=("Segoe UI", 10), anchor="w")
        meta.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 10))

        status_card = self._card(side)
        status_card.grid(row=1, column=0, sticky="nsew")
        status_card.columnconfigure(0, weight=1)
        self._card_title(status_card, "Статус").grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        status = tk.Label(
            status_card,
            textvariable=self.status_var,
            bg=PANEL,
            fg="#baf9eb",
            font=("Segoe UI", 11),
            justify="left",
            anchor="nw",
            wraplength=430,
        )
        status.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        capture = tk.Label(status_card, textvariable=self.capture_indicator_var, bg=PANEL, fg=ACCENT, font=("Segoe UI", 10, "bold"), anchor="w")
        capture.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 4))
        memory_status = tk.Label(status_card, textvariable=self.memory_status_var, bg=PANEL, fg=MUTED, font=("Segoe UI", 10), anchor="w")
        memory_status.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 4))
        memory_summary = tk.Label(status_card, textvariable=self.memory_summary_var, bg=PANEL, fg=MUTED, font=("Segoe UI", 10), anchor="w")
        memory_summary.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 4))
        memory_objects = tk.Label(
            status_card,
            textvariable=self.memory_objects_var,
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10),
            anchor="w",
            wraplength=430,
            justify="left",
        )
        memory_objects.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 8))
        hint = tk.Label(
            status_card,
            text="Прототип может ошибаться. Используйте как вспомогательный инструмент, не как источник гарантированной истины.",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
            anchor="w",
            wraplength=430,
            justify="left",
        )
        hint.grid(row=6, column=0, sticky="ew", padx=12, pady=(0, 10))
        self._refresh_memory_status()
        self.show_recent_library()

    def _card(self, parent) -> tk.Frame:
        frame = tk.Frame(parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        return frame

    def _card_title(self, parent, text: str) -> tk.Label:
        return tk.Label(parent, text=text, bg=PANEL, fg=TEXT, font=("Segoe UI", 13, "bold"), anchor="w")

    def _button(self, parent, text: str, command, color: str) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg="#04110e" if color == ACCENT else TEXT,
            activebackground=color,
            activeforeground="#04110e" if color == ACCENT else TEXT,
            relief="flat",
            padx=10,
            pady=7,
            font=("Segoe UI", 9, "bold"),
        )

    def _compact_option(self, parent, variable: tk.StringVar, values: list[str]):
        menu = tk.OptionMenu(parent, variable, *values)
        menu.configure(
            bg=PANEL_2,
            fg=TEXT,
            activebackground=PANEL_2,
            activeforeground=TEXT,
            relief="flat",
            highlightthickness=0,
            padx=6,
            pady=5,
            font=("Segoe UI", 9, "bold"),
        )
        menu["menu"].configure(bg=PANEL_2, fg=TEXT, activebackground=ACCENT, activeforeground="#04110e")
        return menu

    def refresh_models(self) -> None:
        models_by_runtime = {"auto": [], "official-gemma": []}
        for model in self.registry.get("official-gemma").detect_models():
            models_by_runtime.setdefault(model.runtime, []).append(model)
        self.models_by_runtime = models_by_runtime

    def start_monitoring(self) -> None:
        if self.monitor is not None:
            messagebox.showinfo("Switch Vision Monitor", "Мониторинг уже запущен.")
            return

        try:
            from .engine import CameraMonitor, MonitorConfig

            self.pending_progress = None
            self._save_config()
            source_mode = self._source_mode_key()
            app_mode = self._app_mode_key()
            selected_regions = self._selected_screen_regions()
            if source_mode == "screen_region" and not selected_regions:
                messagebox.showinfo("Switch Vision Monitor", "Сначала выберите область 1.")
                return
            if source_mode == "screen_regions" and len(selected_regions) < 2:
                messagebox.showinfo("Switch Vision Monitor", "Сначала выберите область 1 и область 2.")
                return
            runtime_name, match, runtime, prepared_model_id = self._prepare_runtime()
            config = MonitorConfig(
                runtime_name=runtime_name,
                model_id=match.model_id,
                prepared_model_id=prepared_model_id,
                camera_index=int(self.camera_var.get().strip() or 0),
                interval_seconds=max(float(self.interval_var.get().strip() or 6), 1.0),
                prompt=self.prompt_var.get().strip(),
                source_mode=source_mode,
                screen_region=selected_regions[0] if selected_regions else self.screen_region,
                screen_regions=selected_regions,
                gpu_mode=normalize_gpu_mode(self.gpu_mode_var.get()),
                reminder_command="",
                emergency_command="",
                vision_memory_enabled=app_mode in {"classifier", "tag_library"},
                vision_memory_ttl_seconds=max(int(float(self.memory_ttl_var.get().strip() or 86400)), 60),
                vision_capture_interval_seconds=max(float(self.memory_interval_var.get().strip() or 2), 0.5),
                vision_min_frame_change_threshold=0.04,
                current_room_label=self.current_room_var.get().strip(),
                current_geo_label=self.geo_label_var.get().strip(),
            )
            monitor = CameraMonitor(
                self.project_root,
                runtime=runtime,
                store=self.store,
                preview_callback=lambda frame: self.ui_queue.put(("preview", frame)),
                log_callback=lambda message: self.ui_queue.put(("log", message)),
                popup_callback=lambda title, message: self.ui_queue.put(("popup", title, message)),
                status_callback=lambda message: self.ui_queue.put(("status", message)),
                analysis_callback=lambda message: None,
                answer_callback=lambda payload: self.ui_queue.put(("answer", payload)),
                progress_callback=lambda payload: self.ui_queue.put(("progress", payload)),
                memory_callback=lambda payload: self.ui_queue.put(("memory", payload)),
            )
            monitor.memory_store = self.memory_store
            monitor.start(config)
            self.monitor = monitor
            self.capture_indicator_var.set(f"Захват: активен ({SOURCE_LABELS.get(source_mode, source_mode)})")
            self._set_status(f"Запущено: {runtime_name}/{prepared_model_id}")
            self._log(f"Запущено: {runtime_name}/{prepared_model_id}")
        except RuntimeErrorBase as error:
            self._set_status("Ошибка запуска")
            messagebox.showerror("Runtime error", str(error))
        except Exception as error:
            self._set_status("Ошибка запуска")
            messagebox.showerror("Start error", str(error))

    def stop_monitoring(self, unload_runtime: bool = False) -> None:
        monitor = self.monitor
        self.monitor = None
        self.pending_progress = None
        if monitor:
            monitor.stop(unload_runtime=unload_runtime)
        self.capture_indicator_var.set("Захват: выключен")
        self._set_status("Остановлено")
        self._log("Остановлено")

    def clear_model_memory(self) -> None:
        if self.monitor is not None:
            self.stop_monitoring(unload_runtime=True)
        for runtime_name in ("official-gemma",):
            try:
                runtime = self.registry.get(runtime_name)
            except Exception:
                continue
            runtime_stop = getattr(runtime, "stop", None)
            if callable(runtime_stop):
                runtime_stop()
        self.pending_progress = None
        self._set_status("Модель выгружена из памяти")
        self._log("Модель выгружена из памяти")

    def clear_visual_memory(self) -> None:
        cleared = self.memory_store.clear()
        monitor_store = getattr(self.monitor, "memory_store", None)
        if monitor_store is not None and monitor_store is not self.memory_store:
            try:
                cleared += int(monitor_store.clear())
            except Exception:
                pass
        self.evidence_items.clear()
        self._refresh_memory_status()
        self.memory_summary_var.set("Наблюдений: 0")
        self.memory_objects_var.set("Объекты: нет")
        self._clear_library_tree()
        self.library_status_var.set("Библиотека очищена")
        if self.library_image_label:
            self.library_image_label.configure(image="", text="Изображение появится после выбора наблюдения")
        self._set_status("Визуальная память очищена")
        self._log(f"Визуальная память очищена: ключей/записей {cleared}")

    def search_tag_library(self) -> None:
        query = self.tag_filter_var.get().strip()
        if not query:
            self.show_recent_library()
            return
        try:
            observations = self.memory_store.search_json(query, mode="simple")
        except Exception as error:
            self._set_status(f"Ошибка поиска: {error}")
            self._log(f"Ошибка поиска: {error}")
            return
        self._render_library_observations(observations, query, "Поиск")

    def search_ai_library(self) -> None:
        query = self.tag_filter_var.get().strip()
        if not query:
            self.show_recent_library()
            return
        try:
            observations = self.memory_store.search_json(query, mode="ai")
        except Exception as error:
            self._set_status(f"Ошибка ИИ поиска: {error}")
            self._log(f"Ошибка ИИ поиска: {error}")
            return
        self._render_library_observations(observations, query, "ИИ поиск")

    def reset_library_search(self) -> None:
        self.tag_filter_var.set("")
        self.show_recent_library()

    def show_recent_library(self) -> None:
        self.tag_filter_var.set("")
        try:
            observations = self.memory_store.recent_observations(200)
        except Exception as error:
            self._set_status(f"Ошибка чтения библиотеки: {error}")
            self._log(f"Ошибка чтения библиотеки: {error}")
            return
        self._render_library_observations(observations, "", "Все данные")

    def _render_library_observations(self, observations, query: str = "", mode: str = "Поиск") -> None:
        self._clear_library_tree()
        if not self.library_tree:
            return
        if not observations:
            self.library_status_var.set(f"Не найдено: {query}" if query else "Библиотека пуста")
            return
        for index, observation in enumerate(observations, start=1):
            payload = observation.to_dict()
            comment = payload.get("comment") or payload.get("scene", {}).get("summary") or payload.get("spatial_summary") or ""
            title = f"#{index} · {payload.get('timestamp', '')}"
            if comment:
                title += f" · {self._short_value(comment, 80)}"
            root_id = self.library_tree.insert("", "end", text=title, values=(payload.get("model", {}).get("name", ""),), open=index == 1)
            self.library_item_payload[root_id] = {"thumbnail_path": payload.get("thumbnail_path"), "payload": payload}
            self.library_tree.insert(root_id, "end", text="comment", values=(comment,))
            if payload.get("thumbnail_path"):
                image_id = self.library_tree.insert(root_id, "end", text="image", values=(payload.get("thumbnail_path"),))
                self.library_item_payload[image_id] = {"thumbnail_path": payload.get("thumbnail_path"), "payload": payload}
            tags_node = self.library_tree.insert(root_id, "end", text="tags", values=(len(payload.get("tags") or {}),), open=True)
            for tag_name, tag_payload in sorted((payload.get("tags") or {}).items()):
                tag_node = self.library_tree.insert(tags_node, "end", text=tag_name, values=(self._short_value(tag_payload.get("en") or tag_payload.get("value"), 80),))
                self._insert_json_tree(tag_node, tag_payload)
            json_node = self.library_tree.insert(root_id, "end", text="json", values=("",), open=False)
            self._insert_json_tree(json_node, payload)
        self.library_status_var.set(f"{mode}: {len(observations)} · запрос: {query}" if query else f"Все данные: {len(observations)}")
        first_root = self.library_tree.get_children("")
        if first_root:
            self.library_tree.selection_set(first_root[0])
            self.library_tree.focus(first_root[0])
            self._on_library_tree_select()

    def _clear_library_tree(self) -> None:
        self.library_item_payload.clear()
        if self.library_tree:
            self.library_tree.delete(*self.library_tree.get_children(""))

    def _insert_json_tree(self, parent: str, value) -> None:
        if not self.library_tree:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    node = self.library_tree.insert(parent, "end", text=str(key), values=(self._container_label(child),), open=False)
                    self._insert_json_tree(node, child)
                else:
                    self.library_tree.insert(parent, "end", text=str(key), values=(self._short_value(child),))
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                label = f"[{index}]"
                if isinstance(child, (dict, list)):
                    node = self.library_tree.insert(parent, "end", text=label, values=(self._container_label(child),), open=False)
                    self._insert_json_tree(node, child)
                else:
                    self.library_tree.insert(parent, "end", text=label, values=(self._short_value(child),))

    def _on_library_tree_select(self, _event=None) -> None:
        if not self.library_tree or not self.library_image_label:
            return
        selected = self.library_tree.selection()
        if not selected:
            return
        item_id = selected[0]
        payload = self.library_item_payload.get(item_id)
        if payload is None:
            parent = self.library_tree.parent(item_id)
            while parent and payload is None:
                payload = self.library_item_payload.get(parent)
                parent = self.library_tree.parent(parent)
        image_path = payload.get("thumbnail_path") if payload else None
        if not image_path:
            self.library_image_label.configure(image="", text="У выбранного наблюдения нет изображения")
            self.library_image = None
            return
        try:
            image = Image.open(image_path)
            image.thumbnail((280, 110))
            self.library_image = ImageTk.PhotoImage(image=image)
            self.library_image_label.configure(image=self.library_image, text="")
        except Exception as error:
            self.library_image = None
            self.library_image_label.configure(image="", text=f"Ошибка изображения: {error}")

    def _container_label(self, value) -> str:
        if isinstance(value, dict):
            return f"{len(value)} keys"
        if isinstance(value, list):
            return f"{len(value)} items"
        return ""

    def _short_value(self, value, limit: int = 140) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        text = str(value)
        return text if len(text) <= limit else text[: max(limit - 3, 0)] + "..."

    def select_screen_region(self, index: int = 1) -> None:
        index = 1 if index != 2 else 2
        overlay = tk.Toplevel(self.root)
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.attributes("-alpha", 0.28)
        screen_x, screen_y, width, height = self._virtual_screen_bounds()
        overlay.geometry(f"{width}x{height}{screen_x:+d}{screen_y:+d}")
        overlay.configure(bg="#000000")
        overlay.update_idletasks()

        canvas = tk.Canvas(overlay, bg="#000000", highlightthickness=0, cursor="crosshair")
        canvas.pack(fill="both", expand=True)
        label = canvas.create_text(
            18,
            18,
            text=f"Выделите область {index}. Esc - отмена.",
            fill="#ffffff",
            anchor="nw",
            font=("Segoe UI", 16, "bold"),
        )
        start = {"canvas_x": 0, "canvas_y": 0, "screen_x": 0, "screen_y": 0, "rect": None}

        def screen_point(event) -> tuple[int, int]:
            return int(overlay.winfo_rootx() + event.x), int(overlay.winfo_rooty() + event.y)

        def on_press(event) -> None:
            screen_point_x, screen_point_y = screen_point(event)
            start["canvas_x"] = event.x
            start["canvas_y"] = event.y
            start["screen_x"] = screen_point_x
            start["screen_y"] = screen_point_y
            if start["rect"]:
                canvas.delete(start["rect"])
            start["rect"] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#28c7a3", width=3)

        def on_drag(event) -> None:
            if not start["rect"]:
                return
            x1 = start["canvas_x"]
            y1 = start["canvas_y"]
            x2 = event.x
            y2 = event.y
            canvas.coords(start["rect"], x1, y1, x2, y2)

        def on_release(event) -> None:
            end_x, end_y = screen_point(event)
            x1 = min(start["screen_x"], end_x)
            y1 = min(start["screen_y"], end_y)
            x2 = max(start["screen_x"], end_x)
            y2 = max(start["screen_y"], end_y)
            region = {
                "label": f"ROI {index}",
                "x": int(x1),
                "y": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
            }
            overlay.destroy()
            if region["width"] < 20 or region["height"] < 20:
                self._set_status("Область слишком маленькая")
                self._log("Область экрана не выбрана: слишком маленький размер")
                return
            self._set_screen_region(index, region)
            selected_count = len(self._selected_screen_regions())
            self._set_source_mode_key("screen_regions" if selected_count >= 2 else "screen_region")
            self._save_config()
            message = f"Выбрана область {index}: {region['width']}x{region['height']} @ {region['x']},{region['y']}"
            self._set_status(message)
            self._log(message)

        def cancel(_event=None) -> None:
            overlay.destroy()
            self._set_status("Выбор области отменён")

        canvas.tag_raise(label)
        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        overlay.bind("<Escape>", cancel)
        overlay.focus_force()

    def _virtual_screen_bounds(self) -> tuple[int, int, int, int]:
        if sys.platform == "win32":
            try:
                user32 = ctypes.windll.user32
                return (
                    int(user32.GetSystemMetrics(76)),
                    int(user32.GetSystemMetrics(77)),
                    int(user32.GetSystemMetrics(78)),
                    int(user32.GetSystemMetrics(79)),
                )
            except Exception:
                pass
        return 0, 0, int(self.root.winfo_screenwidth()), int(self.root.winfo_screenheight())

    def _set_screen_region(self, index: int, region: dict[str, int | str]) -> None:
        slot = 0 if index == 1 else 1
        self.screen_regions[slot] = region
        selected = self._selected_screen_regions()
        self.screen_region = selected[0] if selected else None

    def _selected_screen_regions(self) -> list[dict[str, int | str]]:
        return [region for region in self.screen_regions if isinstance(region, dict)]

    def _configure_memory_store(self) -> None:
        try:
            ttl = max(int(float(self.memory_ttl_var.get().strip() or 86400)), 60)
            interval = max(float(self.memory_interval_var.get().strip() or 2), 0.5)
        except Exception:
            ttl = 86400
            interval = 2.0
        self.memory_store = RedisMemoryStore(VisionMemoryConfig(ttl_seconds=ttl, capture_interval_seconds=interval))

    def _prepare_runtime(self):
        runtime_name = self.runtime_var.get().strip() or "auto"
        selected_model = self.model_var.get().strip()
        gpu_mode = normalize_gpu_mode(self.gpu_mode_var.get())
        self.gpu_mode_var.set(gpu_mode)

        if runtime_name == "auto":
            errors: list[str] = []
            for candidate_runtime in ("official-gemma",):
                models = self.models_by_runtime.get(candidate_runtime, [])
                if not models:
                    continue
                match = self._preferred_model(models)
                runtime = self.registry.get(candidate_runtime)
                self._attach_runtime_progress(runtime)
                try:
                    self._set_status(f"Подготовка {candidate_runtime}...")
                    self.root.update_idletasks()
                    prepared = runtime.prepare(match.model_id, gpu_mode=gpu_mode)
                    return candidate_runtime, match, runtime, prepared
                except Exception as error:
                    errors.append(f"{candidate_runtime}: {error}")
                    self._log(f"Fallback после ошибки {candidate_runtime}: {error}")
            raise RuntimeError("Не удалось запустить локальную Gemma. " + "; ".join(errors))

        models = self.models_by_runtime.get(runtime_name, [])
        match = next((item for item in models if item.display_name == selected_model), None) or (models[0] if models else None)
        if match is None:
            raise RuntimeError(f"Модель для рантайма {runtime_name} не найдена.")
        runtime = self.registry.get(runtime_name)
        self._attach_runtime_progress(runtime)
        self._set_status(f"Подготовка {runtime_name}...")
        self.root.update_idletasks()
        prepared = runtime.prepare(match.model_id, gpu_mode=gpu_mode)
        return runtime_name, match, runtime, prepared

    def _attach_runtime_progress(self, runtime) -> None:
        setter = getattr(runtime, "set_progress_callback", None)
        if not callable(setter):
            return

        def progress(message: str) -> None:
            if threading.current_thread() is threading.main_thread():
                self._set_status(message)
                self._log(message)
                self.root.update_idletasks()
                return
            self.ui_queue.put(("runtime_progress", message))

        setter(progress)

    def _preferred_model(self, models: list[RuntimeModel]) -> RuntimeModel:
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

    def open_settings(self) -> None:
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return
        window = tk.Toplevel(self.root)
        window.title("Настройки")
        window.geometry("660x660")
        window.configure(bg=BG)
        window.transient(self.root)
        self.settings_window = window
        window.columnconfigure(1, weight=1)
        window.rowconfigure(11, weight=1)

        self._option(window, "Режим приложения", self.app_mode_var, list(APP_MODES.keys()), 0)
        self.runtime_menu = self._option(window, "Рантайм", self.runtime_var, list(self.models_by_runtime.keys()), 1)
        self.runtime_var.trace_add("write", lambda *_: self._refresh_model_menu())
        self.model_menu = self._option(window, "Модель", self.model_var, [DEFAULT_MODEL_LABEL], 2)
        self._option(window, "Источник", self.source_mode_var, list(SOURCE_MODES.keys()), 3)
        self._option(window, "GPU", self.gpu_mode_var, ["max", "auto", "off", "0.75", "0.5", "0.25"], 4)
        self._entry(window, "Камера", self.camera_var, 5)
        self._entry(window, "Обычный интервал, сек", self.interval_var, 6)
        self._entry(window, "Память: интервал, сек", self.memory_interval_var, 7)
        self._entry(window, "Память: TTL, сек", self.memory_ttl_var, 8)
        self._entry(window, "Текущая зона", self.current_room_var, 9)
        self._entry(window, "Геометка", self.geo_label_var, 10)

        label = tk.Label(window, text="Prompt", bg=BG, fg=MUTED, font=("Segoe UI", 10))
        label.grid(row=11, column=0, sticky="nw", padx=18, pady=10)
        prompt = ScrolledText(window, bg="#090e14", fg=TEXT, insertbackground=TEXT, relief="flat", wrap="word", height=10)
        prompt.grid(row=11, column=1, sticky="nsew", padx=(0, 18), pady=10)
        prompt.insert("1.0", self.prompt_var.get())

        def save() -> None:
            self.prompt_var.set(prompt.get("1.0", "end").strip())
            self._save_config()
            self._configure_memory_store()
            self._refresh_memory_status()
            self._log("Настройки сохранены")
            window.destroy()

        self._button(window, "Сохранить", save, ACCENT).grid(row=12, column=1, sticky="e", padx=18, pady=18)
        self._refresh_model_menu()

    def _option(self, parent, label_text: str, variable: tk.StringVar, values: list[str], row: int):
        label = tk.Label(parent, text=label_text, bg=BG, fg=MUTED, font=("Segoe UI", 10))
        label.grid(row=row, column=0, sticky="w", padx=18, pady=10)
        menu = tk.OptionMenu(parent, variable, *values)
        menu.configure(bg=PANEL_2, fg=TEXT, activebackground=PANEL_2, activeforeground=TEXT, relief="flat", highlightthickness=0)
        menu["menu"].configure(bg=PANEL_2, fg=TEXT, activebackground=ACCENT, activeforeground="#04110e")
        menu.grid(row=row, column=1, sticky="ew", padx=(0, 18), pady=10)
        return menu

    def _entry(self, parent, label_text: str, variable: tk.StringVar, row: int) -> None:
        label = tk.Label(parent, text=label_text, bg=BG, fg=MUTED, font=("Segoe UI", 10))
        label.grid(row=row, column=0, sticky="w", padx=18, pady=10)
        entry = tk.Entry(parent, textvariable=variable, bg="#090e14", fg=TEXT, insertbackground=TEXT, relief="flat")
        entry.grid(row=row, column=1, sticky="ew", padx=(0, 18), pady=10, ipady=8)

    def _refresh_model_menu(self) -> None:
        if not hasattr(self, "model_menu"):
            return
        menu = self.model_menu["menu"]
        menu.delete(0, "end")
        runtime = self.runtime_var.get()
        values = [DEFAULT_MODEL_LABEL] if runtime == "auto" else [item.display_name for item in self.models_by_runtime.get(runtime, [])]
        if not values:
            values = [""]
        for value in values:
            menu.add_command(label=value, command=lambda item=value: self.model_var.set(item))
        if self.model_var.get() not in values:
            self.model_var.set(values[0])

    def open_logs(self) -> None:
        if self.logs_window and self.logs_window.winfo_exists():
            self.logs_window.lift()
            return
        window = tk.Toplevel(self.root)
        window.title("Логи")
        window.geometry("760x480")
        window.configure(bg=BG)
        window.transient(self.root)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        self.logs_window = window
        self.logs_text = ScrolledText(window, bg="#05080c", fg="#c9d7ea", insertbackground=TEXT, relief="flat", wrap="word")
        self.logs_text.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        self._render_logs()

    def _render_logs(self) -> None:
        if not self.logs_text or not self.logs_text.winfo_exists():
            return
        self.logs_text.configure(state="normal")
        self.logs_text.delete("1.0", "end")
        self.logs_text.insert("1.0", "\n".join(self.logs) or "Логов пока нет.")
        self.logs_text.see("end")
        self.logs_text.configure(state="disabled")

    def _process_queue(self) -> None:
        while not self.ui_queue.empty():
            item = self.ui_queue.get()
            kind = item[0]
            if kind == "preview":
                self._update_preview(item[1])
            elif kind == "log":
                self._log(item[1])
            elif kind == "popup":
                self._log(f"{item[1]}: {item[2]}")
            elif kind == "status":
                self._set_status(item[1])
            elif kind == "runtime_progress":
                self._set_status(item[1])
                self._log(item[1])
            elif kind == "progress":
                payload = item[1]
                if payload.get("state") == "waiting_model":
                    self.pending_progress = payload
                    self._refresh_pending_status(schedule_next=False)
                else:
                    self.pending_progress = None
            elif kind == "memory":
                self._handle_memory_event(item[1])
            elif kind == "answer":
                self._add_answer(item[1])
        self.root.after(100, self._process_queue)

    def _handle_memory_event(self, payload: dict) -> None:
        state = str(payload.get("state") or "")
        status = payload.get("status") if isinstance(payload.get("status"), dict) else self.memory_store.status()
        if state == "analyzing":
            self.memory_status_var.set("Память: анализируется кадр")
        elif state == "indexed":
            self.memory_status_var.set("Память: проиндексировано")
        elif state == "skipped":
            self.memory_status_var.set("Память: пропущено как дубль")
        elif state == "cancelled":
            self.memory_status_var.set("Память: анализ остановлен")
        elif state == "error":
            self.memory_status_var.set(f"Память: ошибка {payload.get('error', '')}")
        elif state == "ready":
            self.memory_status_var.set(
                f"Память: {'Redis' if status.get('redis_available') else 'локальный fallback'}, TTL {status.get('ttl_seconds')} сек"
            )
        if "summary" in payload:
            self.memory_summary_var.set(f"Последнее: {payload.get('summary')}")
        else:
            self.memory_summary_var.set(f"Наблюдений: {status.get('observations', 0)}")
        objects = payload.get("objects")
        if isinstance(objects, list) and objects:
            self.memory_objects_var.set("Объекты: " + ", ".join(str(item) for item in objects[:12]))
        tags = payload.get("tags")
        if isinstance(tags, list) and tags:
            self.memory_objects_var.set("Теги: " + ", ".join(str(item) for item in tags[:16]))
        if state == "indexed":
            self.search_tag_library()
        if state not in {"analyzing", "indexed", "skipped", "cancelled", "error", "ready"}:
            self._refresh_memory_status(status)

    def _refresh_memory_status(self, status: dict | None = None) -> None:
        status = status or self.memory_store.status()
        backend = "Redis" if status.get("redis_available") else "локальный fallback"
        self.memory_status_var.set(
            f"Память: {backend}, indexed={status.get('indexed', 0)}, skipped={status.get('skipped', 0)}, errors={status.get('errors', 0)}"
        )
        self.memory_summary_var.set(f"Наблюдений: {status.get('observations', 0)}, TTL {status.get('ttl_seconds')} сек")

    def _refresh_pending_status(self, schedule_next: bool = True) -> None:
        payload = self.pending_progress
        if payload:
            started_at = float(payload.get("started_at") or time.time())
            elapsed = max(time.time() - started_at, 0.0)
            timeout = payload.get("timeout_seconds")
            timeout_text = f" из {int(timeout)} сек" if timeout else ""
            analysis_id = payload.get("analysis_id", "?")
            model = payload.get("model", "")
            self._set_status(
                f"Анализ #{analysis_id}: модель обрабатывает кадр, прошло {elapsed:.1f} сек{timeout_text}\n{model}"
            )
        if schedule_next:
            self.root.after(500, self._refresh_pending_status)

    def _update_preview(self, frame) -> None:
        try:
            import cv2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            max_width = max(self.preview_label.winfo_width(), 640)
            max_height = max(self.preview_label.winfo_height(), 420)
            image.thumbnail((max_width, max_height))
            self.preview_image = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=self.preview_image, text="")
        except Exception as error:
            self._log(f"Ошибка preview: {error}")

    def _add_answer(self, payload: dict) -> None:
        assistant_reply = str(payload.get("assistant_reply") or "").strip()
        summary = str(payload.get("summary") or "").strip()
        raw_payload = payload.get("raw")
        if isinstance(raw_payload, dict) and isinstance(raw_payload.get("regions"), list):
            text = json.dumps(raw_payload, ensure_ascii=False, indent=2)
        else:
            text = assistant_reply or summary or str(payload.get("text") or "").strip()
        if not text:
            return
        entry = dict(payload)
        entry["clean_answer"] = text
        self.answers.append(entry)
        self.answer_var.set(text)
        self._set_answer_text(text)
        meta = f"#{payload.get('analysis_id', '?')} · {payload.get('timestamp', '')} · {payload.get('model', '')}"
        self.meta_var.set(meta.strip(" ·"))

    def _set_answer_text(self, text: str) -> None:
        if not self.answer_text:
            return
        self.answer_text.configure(state="normal")
        self.answer_text.delete("1.0", "end")
        self.answer_text.insert("1.0", text)
        self.answer_text.see("1.0")
        self.answer_text.configure(state="disabled")

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _log(self, message: str) -> None:
        self.logs.append(f"{time.strftime('%H:%M:%S')} {message}".strip())
        self._render_logs()

    def _load_config(self) -> None:
        if not self.config_path.exists():
            return
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self._set_app_mode_key(str(payload.get("app_mode", "classifier")))
        self.runtime_var.set(str(payload.get("runtime", "auto")))
        self.model_var.set(str(payload.get("model", DEFAULT_MODEL_LABEL)))
        self._set_source_mode_key(str(payload.get("source_mode", "camera")))
        region = payload.get("screen_region")
        if isinstance(region, dict):
            try:
                self.screen_region = {
                    "label": str(region.get("label", "ROI 1")),
                    "x": int(region.get("x", 0)),
                    "y": int(region.get("y", 0)),
                    "width": int(region.get("width", 0)),
                    "height": int(region.get("height", 0)),
                }
                self.screen_regions[0] = self.screen_region
            except Exception:
                self.screen_region = None
        regions = payload.get("screen_regions")
        if isinstance(regions, list):
            loaded: list[dict[str, int | str] | None] = [None, None]
            for slot, item in enumerate(regions[:2]):
                if not isinstance(item, dict):
                    continue
                try:
                    loaded[slot] = {
                        "label": str(item.get("label", f"ROI {slot + 1}")),
                        "x": int(item.get("x", 0)),
                        "y": int(item.get("y", 0)),
                        "width": int(item.get("width", 0)),
                        "height": int(item.get("height", 0)),
                    }
                except Exception:
                    loaded[slot] = None
            self.screen_regions = loaded
            selected = self._selected_screen_regions()
            self.screen_region = selected[0] if selected else self.screen_region
        self.gpu_mode_var.set(normalize_gpu_mode(payload.get("gpu_mode", DEFAULT_GPU_MODE)))
        self.camera_var.set(str(payload.get("camera_index", "0")))
        self.interval_var.set(str(payload.get("interval_seconds", "6")))
        self.memory_interval_var.set(str(payload.get("vision_capture_interval_seconds", "2")))
        self.memory_ttl_var.set(str(payload.get("vision_memory_ttl_seconds", "86400")))
        self.current_room_var.set(str(payload.get("current_room_label", "")))
        self.geo_label_var.set(str(payload.get("current_geo_label", "")))
        self.prompt_var.set(str(payload.get("prompt", DEFAULT_PROMPT)).strip() or DEFAULT_PROMPT)

    def _save_config(self) -> None:
        payload = {
            "app_mode": self._app_mode_key(),
            "runtime": self.runtime_var.get().strip() or "auto",
            "model": self.model_var.get().strip() or DEFAULT_MODEL_LABEL,
            "source_mode": self._source_mode_key(),
            "screen_region": self.screen_region,
            "screen_regions": self._selected_screen_regions(),
            "gpu_mode": normalize_gpu_mode(self.gpu_mode_var.get()),
            "camera_index": self.camera_var.get().strip() or "0",
            "interval_seconds": self.interval_var.get().strip() or "6",
            "vision_capture_interval_seconds": self.memory_interval_var.get().strip() or "2",
            "vision_memory_ttl_seconds": self.memory_ttl_var.get().strip() or "86400",
            "current_room_label": self.current_room_var.get().strip(),
            "current_geo_label": self.geo_label_var.get().strip(),
            "reminder_command": "",
            "emergency_command": "",
            "prompt": self.prompt_var.get().strip() or DEFAULT_PROMPT,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _source_mode_key(self) -> str:
        return SOURCE_MODES.get(self.source_mode_var.get(), "camera")

    def _set_source_mode_key(self, source_mode: str) -> None:
        self.source_mode_var.set(SOURCE_LABELS.get(source_mode, SOURCE_LABELS["camera"]))

    def _app_mode_key(self) -> str:
        return APP_MODES.get(self.app_mode_var.get(), "classifier")

    def _set_app_mode_key(self, app_mode: str) -> None:
        self.app_mode_var.set(APP_MODE_LABELS.get(app_mode, APP_MODE_LABELS["classifier"]))

    def shutdown(self) -> None:
        self.stop_monitoring()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = SwitchDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
