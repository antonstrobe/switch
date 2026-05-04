from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageGrab

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .gesture import FingerCounter, GestureDetection
from .parsing import AnalysisResult, format_analysis_for_display, parse_analysis_response, parse_partial_analysis_response
from .runtimes import BaseRuntime, DEFAULT_GPU_MODE, DEFAULT_TIMEOUT, format_seconds
from .storage import OutputStore
from .vision_memory import (
    VISION_MEMORY_SYSTEM_PROMPT,
    RedisMemoryStore,
    VisionMemoryConfig,
    build_memory_user_prompt,
    compute_image_hash,
    fallback_observation_from_text,
    hash_distance,
    observation_from_model_output,
    save_thumbnail,
)


DEFAULT_PROMPT = """Опиши текущий кадр максимально подробно на русском языке.
Укажи, что находится в кадре, где расположены объекты, есть ли люди, руки, жесты, текст, цвета, освещение, фон и важные детали.
Если что-то видно неуверенно, прямо напиши, что это предположение.
Не выдумывай предметы и действия, которых нет на изображении.
Если есть рука и видны пальцы, отдельно оцени количество пальцев и уверенность.
Если видишь дым, огонь, сильный перегрев, опасную открытую плиту или другую угрозу, создай критическое событие.
"""

ENABLE_LOCAL_FAST_PATH = False


@dataclass
class MonitorConfig:
    runtime_name: str
    model_id: str
    prepared_model_id: str
    camera_index: int
    interval_seconds: float
    prompt: str
    source_mode: str = "camera"
    screen_region: dict[str, int | str] | None = None
    screen_regions: list[dict[str, int | str]] | None = None
    gpu_mode: str = DEFAULT_GPU_MODE
    reminder_command: str = ""
    emergency_command: str = ""
    vision_memory_enabled: bool = False
    vision_memory_ttl_seconds: int = 86400
    vision_capture_interval_seconds: float = 2.0
    vision_min_frame_change_threshold: float = 0.04
    current_room_label: str = ""
    current_geo_label: str = ""


class CameraMonitor:
    def __init__(
        self,
        root: Path,
        runtime: BaseRuntime,
        store: OutputStore,
        preview_callback: Callable,
        log_callback: Callable[[str], None],
        popup_callback: Callable[[str, str], None],
        status_callback: Callable[[str], None] | None = None,
        analysis_callback: Callable[[str], None] | None = None,
        answer_callback: Callable[[dict], None] | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        memory_callback: Callable[[dict], None] | None = None,
    ) -> None:
        self.root = root
        self.runtime = runtime
        self.store = store
        self.preview_callback = preview_callback
        self.log_callback = log_callback
        self.popup_callback = popup_callback
        self.status_callback = status_callback
        self.analysis_callback = analysis_callback
        self.answer_callback = answer_callback
        self.progress_callback = progress_callback
        self.memory_callback = memory_callback
        self.stop_event = threading.Event()
        self.capture_thread: threading.Thread | None = None
        self.analysis_thread: threading.Thread | None = None
        self.analysis_lock = threading.Lock()
        self.capture = None
        self.config: MonitorConfig | None = None
        self.analysis_count = 0
        self.next_analysis_allowed_at = 0.0
        self.memory_store: RedisMemoryStore | None = None
        self.last_memory_image_hash: str | None = None
        try:
            self.finger_counter = FingerCounter()
        except Exception:
            self.finger_counter = None

    def start(self, config: MonitorConfig) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV не установлен. Установите зависимости из requirements.txt.")
        if self.capture_thread and self.capture_thread.is_alive():
            raise RuntimeError("Мониторинг уже запущен.")

        self.config = config
        self.stop_event.clear()
        self.next_analysis_allowed_at = 0.0
        self.last_memory_image_hash = None
        if config.vision_memory_enabled:
            if self.memory_store is None:
                self.memory_store = RedisMemoryStore(
                    VisionMemoryConfig(
                        ttl_seconds=int(config.vision_memory_ttl_seconds),
                        capture_interval_seconds=float(config.vision_capture_interval_seconds),
                        min_frame_change_threshold=float(config.vision_min_frame_change_threshold),
                    )
                )
            self._set_memory_event({"state": "ready", "status": self.memory_store.status()})
        else:
            self.memory_store = None
        self.store.ensure_layout()
        self.store.write_status(
            {
                "state": "starting",
                "message": "Camera monitor is starting.",
                "runtime": config.runtime_name,
                "model": config.prepared_model_id,
                "gpu_mode": config.gpu_mode,
            }
        )
        self._set_analysis_view("Ожидание первого анализа кадра.")
        self._set_status(f"Запуск мониторинга: {config.runtime_name}/{config.prepared_model_id}")
        self.capture_thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self.capture_thread.start()

    def stop(self, unload_runtime: bool = False) -> None:
        self.stop_event.set()
        if self._analysis_busy() or unload_runtime:
            self._stop_runtime()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        self.store.write_status(
            {
                "state": "stopped",
                "message": "Monitoring stopped.",
                "analysis_count": self.analysis_count,
            }
        )
        self._set_analysis_view("Мониторинг остановлен.")
        self._set_status("Мониторинг остановлен")

    def _stop_runtime(self) -> None:
        runtime_stop = getattr(self.runtime, "stop", None)
        if callable(runtime_stop):
            runtime_stop()

    def _capture_loop(self) -> None:
        assert self.config is not None
        if self.config.source_mode in {"screen_region", "screen_regions", "screen_full"}:
            self._screen_capture_loop()
            return

        self.log_callback(f"Открываю камеру #{self.config.camera_index}")
        self.capture = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(self.config.camera_index)
        if not self.capture.isOpened():
            self.log_callback("Не удалось открыть камеру.")
            self.store.write_status(
                {
                    "state": "camera_error",
                    "message": "Failed to open camera.",
                    "camera_index": self.config.camera_index,
                }
            )
            self._set_status("Ошибка: камера не открылась")
            return
        self.store.write_status(
            {
                "state": "camera_ready",
                "message": "Camera opened. Waiting for analysis interval.",
                "camera_index": self.config.camera_index,
                "interval_seconds": self.config.interval_seconds,
            }
        )
        self._set_analysis_view("Камера открыта. Ожидание очередного анализа.")
        self._set_status("Камера открыта, ожидание анализа")

        last_analysis = 0.0
        while not self.stop_event.is_set():
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.2)
                continue
            self.preview_callback(frame)

            now = time.time()
            interval = self._active_interval_seconds()
            if now >= self.next_analysis_allowed_at and now - last_analysis >= interval and not self._analysis_busy():
                frame_copy = frame.copy()
                self._start_frame_job(frame_copy)
                last_analysis = now
                self.next_analysis_allowed_at = float("inf")
            time.sleep(0.04)

        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _screen_capture_loop(self) -> None:
        assert self.config is not None
        source_title = self._screen_source_title()
        selected_regions = self._screen_regions()
        if self.config.source_mode == "screen_region" and not selected_regions:
            self.store.write_status(
                {
                    "state": "screen_region_missing",
                    "message": "Screen region is not selected.",
                    "source_mode": self.config.source_mode,
                }
            )
            self._set_status("Ошибка: область экрана не выбрана")
            self.log_callback("Область экрана не выбрана.")
            return
        if self.config.source_mode == "screen_regions" and len(selected_regions) < 2:
            self.store.write_status(
                {
                    "state": "screen_regions_missing",
                    "message": "Two screen regions are required.",
                    "source_mode": self.config.source_mode,
                    "screen_regions": selected_regions,
                }
            )
            self._set_status("Ошибка: выберите две области экрана")
            self.log_callback("Для режима двух областей выберите область 1 и область 2.")
            return

        self.store.write_status(
            {
                "state": "screen_ready",
                "message": f"Screen source ready: {source_title}.",
                "source_mode": self.config.source_mode,
                "screen_region": self.config.screen_region,
                "screen_regions": selected_regions,
                "interval_seconds": self.config.interval_seconds,
            }
        )
        self._set_analysis_view(f"Источник: {source_title}. Ожидание анализа.")
        self._set_status(f"Источник готов: {source_title}")
        self.log_callback(f"Источник анализа: {source_title}")

        last_analysis = 0.0
        while not self.stop_event.is_set():
            try:
                frame = self._capture_screen_frame()
            except Exception as error:
                self.log_callback(f"Ошибка захвата экрана: {error}")
                self._set_status(f"Ошибка захвата экрана: {error}")
                time.sleep(1)
                continue

            self.preview_callback(frame)
            now = time.time()
            interval = self._active_interval_seconds()
            if now >= self.next_analysis_allowed_at and now - last_analysis >= interval and not self._analysis_busy():
                frame_copy = frame.copy()
                self._start_frame_job(frame_copy)
                last_analysis = now
                self.next_analysis_allowed_at = float("inf")
            time.sleep(0.25)

    def _capture_screen_frame(self):
        assert self.config is not None
        if self.config.source_mode == "screen_regions":
            regions = self._screen_regions()
            if len(regions) < 2:
                raise RuntimeError("Нужно выбрать две области экрана.")
            images = []
            for index, region in enumerate(regions, start=1):
                bbox = self._region_bbox(region)
                label = str(region.get("label") or f"ROI {index}")
                images.append((label, self._grab_screen(bbox).convert("RGB")))
            return self._pil_to_frame(self._compose_regions_frame(images))

        bbox = None
        if self.config.source_mode == "screen_region":
            regions = self._screen_regions()
            if not regions:
                raise RuntimeError("Область экрана имеет нулевой размер.")
            bbox = self._region_bbox(regions[0])

        image = self._grab_screen(bbox)
        rgb = np.array(image.convert("RGB"))
        if cv2 is None:
            return rgb
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _screen_source_title(self) -> str:
        assert self.config is not None
        if self.config.source_mode == "screen_regions":
            return "две области экрана"
        if self.config.source_mode == "screen_region":
            return "область экрана"
        return "весь экран"

    def _screen_regions(self) -> list[dict[str, int | str]]:
        assert self.config is not None
        if self.config.source_mode == "screen_regions":
            regions = self.config.screen_regions or []
            return [region for region in regions if self._is_valid_region(region)]
        if self.config.source_mode == "screen_region" and self._is_valid_region(self.config.screen_region):
            return [self.config.screen_region]
        return []

    def _is_valid_region(self, region) -> bool:
        if not isinstance(region, dict):
            return False
        try:
            return int(region.get("width", 0)) > 0 and int(region.get("height", 0)) > 0
        except Exception:
            return False

    def _region_bbox(self, region: dict[str, int | str]) -> tuple[int, int, int, int]:
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        width = int(region.get("width", 0))
        height = int(region.get("height", 0))
        if width <= 0 or height <= 0:
            raise RuntimeError("Область экрана имеет нулевой размер.")
        return x, y, x + width, y + height

    def _grab_screen(self, bbox):
        try:
            return ImageGrab.grab(bbox=bbox, all_screens=True)
        except TypeError:
            return ImageGrab.grab(bbox=bbox)

    def _compose_regions_frame(self, items: list[tuple[str, Image.Image]]) -> Image.Image:
        columns = 2 if len(items) > 1 else 1
        rows = (len(items) + columns - 1) // columns
        label_height = 30
        gap = 12
        cell_width = max(image.width for _, image in items)
        cell_height = max(image.height for _, image in items)
        width = columns * cell_width + (columns + 1) * gap
        height = rows * (cell_height + label_height) + (rows + 1) * gap
        canvas = Image.new("RGB", (width, height), "#05080c")
        draw = ImageDraw.Draw(canvas)
        for item_index, (label, image) in enumerate(items):
            row = item_index // columns
            column = item_index % columns
            x = gap + column * (cell_width + gap)
            y = gap + row * (cell_height + label_height + gap)
            draw.rectangle((x, y, x + cell_width, y + label_height), fill="#101822")
            draw.text((x + 8, y + 8), label, fill="#ffffff")
            canvas.paste(image, (x, y + label_height))
        return canvas

    def _pil_to_frame(self, image: Image.Image):
        rgb = np.array(image.convert("RGB"))
        if cv2 is None:
            return rgb
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _analysis_busy(self) -> bool:
        return self.analysis_thread is not None and self.analysis_thread.is_alive()

    def _active_interval_seconds(self) -> float:
        assert self.config is not None
        if self.config.vision_memory_enabled:
            return max(float(self.config.vision_capture_interval_seconds or 2.0), 0.5)
        return self.config.interval_seconds

    def _start_frame_job(self, frame) -> None:
        assert self.config is not None
        target = self._index_memory_frame if self.config.vision_memory_enabled else self._analyze_frame
        name = "vision-memory-indexing" if self.config.vision_memory_enabled else "frame-analysis"
        self.analysis_thread = threading.Thread(
            target=target,
            args=(frame,),
            name=name,
            daemon=True,
        )
        self.analysis_thread.start()

    def _analyze_frame(self, frame) -> None:
        if not self.analysis_lock.acquire(blocking=False):
            return
        try:
            assert self.config is not None
            self.analysis_count += 1
            analysis_id = self.analysis_count
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            analysis_started_at = time.time()
            frame_label = "Первый кадр" if analysis_id == 1 else f"Кадр #{analysis_id}"

            def stage(message: str) -> None:
                full_message = f"Анализ #{analysis_id}: {message}; прошло {format_seconds(time.time() - analysis_started_at)}"
                self.log_callback(full_message)
                self._set_status(full_message)

            if ENABLE_LOCAL_FAST_PATH and self._is_fast_finger_mode():
                self._run_fast_finger_analysis(frame, analysis_id, timestamp)
                self.next_analysis_allowed_at = time.time() + self.config.interval_seconds
                return
            context = self.store.build_context(max_chars=900)
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(timestamp, context)
            analysis_frame = self._prepare_analysis_frame(frame)
            height, width = analysis_frame.shape[:2]

            stage(f"{frame_label}: подготовка кадра {width}x{height} и контекста {len(context)} символов")
            self._set_analysis_view(
                f"Анализ #{analysis_id}\n\nМодель получила новый кадр и готовит описание сцены."
            )
            self.store.append_analysis_history(
                {
                    "analysis_id": analysis_id,
                    "state": "started",
                    "runtime": self.config.runtime_name,
                    "model": self.config.prepared_model_id,
                    "gpu_mode": self.config.gpu_mode,
                }
            )
            self.store.write_last_request(
                {
                    "analysis_id": analysis_id,
                    "runtime": self.config.runtime_name,
                    "model": self.config.prepared_model_id,
                    "gpu_mode": self.config.gpu_mode,
                    "timestamp": timestamp,
                    "context_chars": len(context),
                    "prompt_preview": (self.config.prompt.strip() or DEFAULT_PROMPT.strip())[:1000],
                }
            )

            ok, encoded = cv2.imencode(".jpg", analysis_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                self.log_callback("Не удалось закодировать кадр.")
                self.store.write_status(
                    {
                        "state": "encode_error",
                        "message": "Failed to encode frame.",
                        "analysis_id": analysis_id,
                    }
                )
                self._set_status(f"Анализ #{analysis_id}: ошибка кодирования кадра")
                return
            encoded_bytes = encoded.tobytes()
            encoded_kb = len(encoded_bytes) / 1024
            last_frame_path = self.store.output_dir / "last_analyzed_frame.jpg"
            cv2.imwrite(str(last_frame_path), analysis_frame)
            stage(f"{frame_label}: JPEG готов {encoded_kb:.1f} KB, кадр сохранён в output")
            if self.stop_event.is_set():
                self._mark_analysis_cancelled(analysis_id)
                return

            self.store.write_status(
                {
                    "state": "running",
                    "message": "Frame encoded. Sending request to model.",
                    "analysis_id": analysis_id,
                    "runtime": self.config.runtime_name,
                    "model": self.config.prepared_model_id,
                    "gpu_mode": self.config.gpu_mode,
                    "started_at": timestamp,
                    "last_frame_path": str(last_frame_path),
                }
            )
            sent_message = (
                f"{frame_label} отправлен на анализ в {self.config.prepared_model_id}, "
                f"timeout {DEFAULT_TIMEOUT} сек"
            )
            stage(sent_message)
            self._set_progress(
                {
                    "state": "waiting_model",
                    "analysis_id": analysis_id,
                    "model": self.config.prepared_model_id,
                    "started_at": analysis_started_at,
                    "timeout_seconds": DEFAULT_TIMEOUT,
                    "message": sent_message,
                }
            )
            self._set_analysis_view(
                f"Анализ #{analysis_id}\n\nКадр отправлен в модель {self.config.prepared_model_id}.\nОжидание ответа."
            )
            if self.stop_event.is_set():
                self._mark_analysis_cancelled(analysis_id)
                return

            model_started_at = time.time()
            raw_response = self.runtime.analyze(
                self.config.prepared_model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_bytes=encoded_bytes,
            )
            if self.stop_event.is_set():
                self._mark_analysis_cancelled(analysis_id)
                return
            stage(f"ответ модели получен, время ожидания {format_seconds(time.time() - model_started_at)}")
            parse_started_at = time.time()
            stage("разбираю JSON ответа")
            try:
                result = parse_analysis_response(raw_response)
            except ValueError as error:
                result = parse_partial_analysis_response(raw_response, str(error))
                stage("JSON неполный, использую полученную часть без повторного запроса")
            if self.stop_event.is_set():
                self._mark_analysis_cancelled(analysis_id)
                return
            stage(f"JSON разобран за {format_seconds(time.time() - parse_started_at)}")
            write_started_at = time.time()
            stage("записываю результат в output")
            summary = self.store.apply_result(
                result,
                frame_saver=lambda event_key: self._save_frame(frame, event_key),
                reminder_command=self.config.reminder_command,
                emergency_command=self.config.emergency_command,
                log_callback=self.log_callback,
                popup_callback=self.popup_callback,
            )
            self._write_last_response(result, raw_response)
            stage(
                f"результат записан за {format_seconds(time.time() - write_started_at)}: "
                f"файлы={summary['files']}, уведомления={summary['notifications']}, "
                f"команды={summary['commands']}, кадры={summary['saved_frames']}"
            )
            elapsed = round(time.time() - analysis_started_at, 2)
            analysis_text = (
                f"Анализ #{analysis_id}\n"
                f"Время анализа: {elapsed} сек\n"
                f"Модель: {self.config.prepared_model_id}\n\n"
                f"{format_analysis_for_display(result)}"
            )
            self._set_analysis_view(analysis_text)
            self._set_answer(
                {
                    "analysis_id": analysis_id,
                    "timestamp": timestamp,
                    "elapsed_seconds": elapsed,
                    "runtime": self.config.runtime_name,
                    "model": self.config.prepared_model_id,
                    "gpu_mode": self.config.gpu_mode,
                    "assistant_reply": result.assistant_reply,
                    "summary": result.summary,
                    "observations": result.observations,
                    "gesture_digit": result.gesture_digit,
                    "gesture_confidence": result.gesture_confidence,
                    "gesture_reason": result.gesture_reason,
                    "raw": result.raw,
                    "text": analysis_text,
                }
            )
            self._set_progress(
                {
                    "state": "completed",
                    "analysis_id": analysis_id,
                    "elapsed_seconds": elapsed,
                }
            )
            self.store.append_analysis_history(
                {
                    "analysis_id": analysis_id,
                    "state": "completed",
                    "runtime": self.config.runtime_name,
                    "model": self.config.prepared_model_id,
                    "gpu_mode": self.config.gpu_mode,
                    "summary": result.summary,
                    "files_written": summary["files"],
                    "notifications": summary["notifications"],
                    "commands": summary["commands"],
                    "saved_frames": summary["saved_frames"],
                    "elapsed_seconds": elapsed,
                }
            )
            self.store.write_status(
                {
                    "state": "completed",
                    "message": result.summary or "Analysis completed.",
                    "analysis_id": analysis_id,
                    "runtime": self.config.runtime_name,
                    "model": self.config.prepared_model_id,
                    "gpu_mode": self.config.gpu_mode,
                    "elapsed_seconds": elapsed,
                    "files_written": summary["files"],
                    "notifications": summary["notifications"],
                    "commands": summary["commands"],
                    "saved_frames": summary["saved_frames"],
                    "summary": result.summary,
                }
            )
            self._set_status(
                f"Анализ #{analysis_id} завершён: {result.summary or 'без краткого описания'}"
            )
            self.log_callback(
                f"Анализ #{analysis_id} завершён за {elapsed} сек | "
                f"файлы={summary['files']} уведомления={summary['notifications']} "
                f"команды={summary['commands']} кадры={summary['saved_frames']}"
            )
            self.next_analysis_allowed_at = time.time() + self.config.interval_seconds
        except Exception as error:
            if self.stop_event.is_set():
                self._mark_analysis_cancelled(self.analysis_count)
                return
            self.log_callback(f"Ошибка анализа: {error}")
            self._write_error(error)
            self._set_analysis_view(f"Анализ #{self.analysis_count}\n\nОшибка анализа:\n{error}")
            self._set_progress(
                {
                    "state": "error",
                    "analysis_id": self.analysis_count,
                    "error": str(error),
                }
            )
            self.store.append_analysis_history(
                {
                    "analysis_id": self.analysis_count,
                    "state": "error",
                    "error": str(error),
                }
            )
            self.store.write_status(
                {
                    "state": "error",
                    "message": str(error),
                    "analysis_id": self.analysis_count,
                }
            )
            self._set_status(f"Ошибка анализа: {error}")
            self.next_analysis_allowed_at = time.time() + self.config.interval_seconds
        finally:
            self.analysis_lock.release()

    def _index_memory_frame(self, frame) -> None:
        if not self.analysis_lock.acquire(blocking=False):
            return
        try:
            assert self.config is not None
            if self.memory_store is None:
                self.memory_store = RedisMemoryStore(
                    VisionMemoryConfig(
                        ttl_seconds=int(self.config.vision_memory_ttl_seconds),
                        capture_interval_seconds=float(self.config.vision_capture_interval_seconds),
                        min_frame_change_threshold=float(self.config.vision_min_frame_change_threshold),
                    )
                )
            self.analysis_count += 1
            analysis_id = self.analysis_count
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            analysis_started_at = time.time()

            def stage(message: str) -> None:
                full_message = f"Визуальная память #{analysis_id}: {message}; прошло {format_seconds(time.time() - analysis_started_at)}"
                self.log_callback(full_message)
                self._set_status(full_message)

            analysis_frame = self._prepare_analysis_frame(frame)
            image_hash = compute_image_hash(analysis_frame)
            hash_delta = hash_distance(self.last_memory_image_hash, image_hash)
            threshold = float(self.config.vision_min_frame_change_threshold or 0.04)
            if self.last_memory_image_hash and hash_delta <= threshold:
                self.memory_store.increment_stat("skipped")
                message = f"кадр пропущен как дубль, change={hash_delta:.3f}"
                stage(message)
                self._set_memory_event({"state": "skipped", "reason": "duplicate", "image_hash": image_hash, "status": self.memory_store.status()})
                self.next_analysis_allowed_at = time.time() + self._active_interval_seconds()
                return
            if self.memory_store.has_recent_hash(image_hash):
                self.memory_store.increment_stat("skipped")
                message = "кадр уже есть в визуальной памяти"
                stage(message)
                self._set_memory_event({"state": "skipped", "reason": "same_hash", "image_hash": image_hash, "status": self.memory_store.status()})
                self.next_analysis_allowed_at = time.time() + self._active_interval_seconds()
                return

            ok, encoded = cv2.imencode(".jpg", analysis_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                raise RuntimeError("Не удалось закодировать кадр для визуальной памяти.")
            encoded_bytes = encoded.tobytes()
            stage(f"кадр подготовлен {analysis_frame.shape[1]}x{analysis_frame.shape[0]}, hash={image_hash}")
            self._set_memory_event({"state": "analyzing", "analysis_id": analysis_id, "image_hash": image_hash})
            self.store.write_status(
                {
                    "state": "memory_analyzing",
                    "message": "Vision memory frame is being analyzed.",
                    "analysis_id": analysis_id,
                    "source": self.config.source_mode,
                    "image_hash": image_hash,
                }
            )
            self.store.write_last_request(
                {
                    "analysis_id": analysis_id,
                    "runtime": self.config.runtime_name,
                    "model": self.config.prepared_model_id,
                    "source": self.config.source_mode,
                    "prompt_version": "vision_memory_v1",
                    "image_hash": image_hash,
                }
            )
            raw_response = self.runtime.analyze(
                self.config.prepared_model_id,
                system_prompt=VISION_MEMORY_SYSTEM_PROMPT,
                user_prompt=build_memory_user_prompt(
                    self.config.source_mode,
                    timestamp,
                    self.config.current_room_label.strip() or None,
                    self.config.current_geo_label.strip() or None,
                ),
                image_bytes=encoded_bytes,
            )
            if self.stop_event.is_set():
                self._mark_analysis_cancelled(analysis_id)
                return
            try:
                observation = observation_from_model_output(
                    raw_response,
                    source=self.config.source_mode,
                    model_name=self.config.prepared_model_id,
                    current_room_label=self.config.current_room_label.strip() or None,
                    geo_label=self.config.current_geo_label.strip() or None,
                    image_hash=image_hash,
                )
            except Exception as parse_error:
                if self.config.runtime_name == "local-gemma":
                    stage("ответ модели невалидный, сохраняю частичный индекс без повторного запроса")
                    self.memory_store.increment_stat("errors")
                    observation = fallback_observation_from_text(
                        raw_response,
                        source=self.config.source_mode,
                        model_name=self.config.prepared_model_id,
                        current_room_label=self.config.current_room_label.strip() or None,
                        geo_label=self.config.current_geo_label.strip() or None,
                        image_hash=image_hash,
                        error=str(parse_error),
                    )
                    thumbnail_path = save_thumbnail(analysis_frame, self.store.output_dir / "vision_memory", observation.observation_id)
                    observation.thumbnail_path = thumbnail_path
                    indexed = self.memory_store.save_observation(observation)
                    self.last_memory_image_hash = image_hash
                    objects = [item.name for item in observation.objects[:12]]
                    tag_names = list(observation.tags.keys())[:20]
                    summary = observation.comment or observation.scene.summary or observation.spatial_summary or "Наблюдение сохранено."
                    state = "indexed" if indexed else "skipped"
                    stage(f"{'проиндексировано' if indexed else 'пропущено'}: {summary}")
                    self._set_memory_event(
                        {
                            "state": state,
                            "analysis_id": analysis_id,
                            "observation_id": observation.observation_id,
                            "summary": summary,
                            "objects": objects,
                            "tags": tag_names,
                            "status": self.memory_store.status(),
                            "thumbnail_path": thumbnail_path,
                        }
                    )
                    self._set_answer(
                        {
                            "analysis_id": analysis_id,
                            "timestamp": observation.timestamp,
                            "elapsed_seconds": round(time.time() - analysis_started_at, 2),
                            "runtime": self.config.runtime_name,
                            "model": self.config.prepared_model_id,
                            "gpu_mode": self.config.gpu_mode,
                            "assistant_reply": summary,
                            "summary": summary,
                            "observations": objects,
                            "tags": tag_names,
                            "raw": observation.to_dict(),
                            "text": json.dumps(observation.to_dict(), ensure_ascii=False, indent=2),
                        }
                    )
                    self.store.write_status(
                        {
                            "state": f"memory_{state}",
                            "message": summary,
                            "analysis_id": analysis_id,
                            "observation_id": observation.observation_id,
                            "image_hash": image_hash,
                            "objects": objects,
                            "tags": tag_names,
                            "thumbnail_path": thumbnail_path,
                        }
                    )
                    self.next_analysis_allowed_at = time.time() + self._active_interval_seconds()
                    return
                stage("ответ модели невалидный, пробую исправить JSON")
                repair_prompt = (
                    "Convert the following model output into one valid JSON object matching the visual memory schema. "
                    "Return JSON only. Do not add new facts.\n\n"
                    f"{raw_response[:6000]}"
                )
                try:
                    repaired_response = self.runtime.analyze(
                        self.config.prepared_model_id,
                        system_prompt=VISION_MEMORY_SYSTEM_PROMPT,
                        user_prompt=repair_prompt,
                        image_bytes=encoded_bytes,
                    )
                    observation = observation_from_model_output(
                        repaired_response,
                        source=self.config.source_mode,
                        model_name=self.config.prepared_model_id,
                        current_room_label=self.config.current_room_label.strip() or None,
                        geo_label=self.config.current_geo_label.strip() or None,
                        image_hash=image_hash,
                    )
                    raw_response = repaired_response
                except Exception as repair_error:
                    self.memory_store.increment_stat("errors")
                    observation = fallback_observation_from_text(
                        raw_response,
                        source=self.config.source_mode,
                        model_name=self.config.prepared_model_id,
                        current_room_label=self.config.current_room_label.strip() or None,
                        geo_label=self.config.current_geo_label.strip() or None,
                        image_hash=image_hash,
                        error=f"{parse_error}; repair: {repair_error}",
                    )
            thumbnail_path = save_thumbnail(analysis_frame, self.store.output_dir / "vision_memory", observation.observation_id)
            observation.thumbnail_path = thumbnail_path
            indexed = self.memory_store.save_observation(observation)
            self.last_memory_image_hash = image_hash
            objects = [item.name for item in observation.objects[:12]]
            tag_names = list(observation.tags.keys())[:20]
            summary = observation.comment or observation.scene.summary or observation.spatial_summary or "Наблюдение сохранено."
            state = "indexed" if indexed else "skipped"
            stage(f"{'проиндексировано' if indexed else 'пропущено'}: {summary}")
            self._set_memory_event(
                {
                    "state": state,
                    "analysis_id": analysis_id,
                    "observation_id": observation.observation_id,
                    "summary": summary,
                    "objects": objects,
                    "tags": tag_names,
                    "status": self.memory_store.status(),
                    "thumbnail_path": thumbnail_path,
                }
            )
            self._set_answer(
                {
                    "analysis_id": analysis_id,
                    "timestamp": observation.timestamp,
                    "elapsed_seconds": round(time.time() - analysis_started_at, 2),
                    "runtime": self.config.runtime_name,
                    "model": self.config.prepared_model_id,
                    "gpu_mode": self.config.gpu_mode,
                    "assistant_reply": summary,
                    "summary": summary,
                    "observations": objects,
                    "tags": tag_names,
                    "raw": observation.to_dict(),
                    "text": json.dumps(observation.to_dict(), ensure_ascii=False, indent=2),
                }
            )
            self.store.write_status(
                {
                    "state": f"memory_{state}",
                    "message": summary,
                    "analysis_id": analysis_id,
                    "observation_id": observation.observation_id,
                    "image_hash": image_hash,
                    "objects": objects,
                    "tags": tag_names,
                    "thumbnail_path": thumbnail_path,
                }
            )
            self.next_analysis_allowed_at = time.time() + self._active_interval_seconds()
        except Exception as error:
            error_text = str(error).lower()
            if self.stop_event.is_set() or "cancelled" in error_text or "canceled" in error_text or "отмен" in error_text:
                self._mark_analysis_cancelled(self.analysis_count)
                self._set_memory_event({"state": "cancelled", "status": self.memory_store.status() if self.memory_store else {}})
                self.next_analysis_allowed_at = time.time() + self._active_interval_seconds()
                return
            if self.memory_store:
                self.memory_store.increment_stat("errors")
            self.log_callback(f"Ошибка визуальной памяти: {error}")
            self._set_status(f"Ошибка визуальной памяти: {error}")
            self._set_memory_event({"state": "error", "error": str(error), "status": self.memory_store.status() if self.memory_store else {}})
            self.store.write_status({"state": "memory_error", "message": str(error), "analysis_id": self.analysis_count})
            self.next_analysis_allowed_at = time.time() + self._active_interval_seconds()
        finally:
            self.analysis_lock.release()

    def _run_fast_finger_analysis(self, frame, analysis_id: int, timestamp: str) -> None:
        detection = self._detect_gesture(frame)
        result = self._build_local_gesture_result(detection)

        self.log_callback(f"Анализ #{analysis_id}: быстрый локальный режим пальцев")
        self._set_status(f"Анализ #{analysis_id}: локальное распознавание пальцев")
        self._set_analysis_view(
            f"Анализ #{analysis_id}\n\nЛокальный детектор пальцев обрабатывает кадр без LM Studio."
        )
        self.store.append_analysis_history(
            {
                "analysis_id": analysis_id,
                "state": "started",
                "runtime": "local-gesture",
                "model": "mediapipe",
            }
        )
        self.store.write_last_request(
            {
                "analysis_id": analysis_id,
                "runtime": "local-gesture",
                "model": "mediapipe",
                "timestamp": timestamp,
                "context_chars": 0,
                "prompt_preview": (self.config.prompt.strip() or DEFAULT_PROMPT.strip())[:1000],
            }
        )

        summary = self.store.apply_result(
            result,
            frame_saver=lambda event_key: self._save_frame(frame, event_key),
            reminder_command=self.config.reminder_command,
            emergency_command=self.config.emergency_command,
            log_callback=self.log_callback,
            popup_callback=self.popup_callback,
        )
        analysis_text = (
            f"Анализ #{analysis_id}\n"
            f"Режим: быстрый локальный детектор\n\n"
            f"{format_analysis_for_display(result)}"
        )
        self._write_last_response(result, '{"source":"mediapipe"}')
        self._set_analysis_view(analysis_text)
        self._set_answer(
            {
                "analysis_id": analysis_id,
                "timestamp": timestamp,
                "elapsed_seconds": 0,
                "runtime": "local-gesture",
                "model": "mediapipe",
                "gpu_mode": "",
                "assistant_reply": result.assistant_reply,
                "summary": result.summary,
                "observations": result.observations,
                "gesture_digit": result.gesture_digit,
                "gesture_confidence": result.gesture_confidence,
                "gesture_reason": result.gesture_reason,
                "raw": result.raw,
                "text": analysis_text,
            }
        )
        self.store.write_status(
            {
                "state": "completed",
                "message": result.summary or "Fast local analysis completed.",
                "analysis_id": analysis_id,
                "runtime": "local-gesture",
                "model": "mediapipe",
                "files_written": summary["files"],
                "notifications": summary["notifications"],
                "commands": summary["commands"],
                "saved_frames": summary["saved_frames"],
                "summary": result.summary,
            }
        )
        self.store.append_analysis_history(
            {
                "analysis_id": analysis_id,
                "state": "completed",
                "summary": result.summary,
                "files_written": summary["files"],
                "notifications": summary["notifications"],
                "commands": summary["commands"],
                "saved_frames": summary["saved_frames"],
                "runtime": "local-gesture",
            }
        )
        self._set_progress({"state": "completed", "analysis_id": analysis_id, "elapsed_seconds": 0})
        self._set_status(f"Анализ #{analysis_id} завершён: {result.summary}")
        self.log_callback(
            f"Анализ #{analysis_id} завершён | файлы={summary['files']} уведомления={summary['notifications']} "
            f"команды={summary['commands']} кадры={summary['saved_frames']}"
        )

    def _save_frame(self, frame, event_key: str) -> bool:
        if cv2 is None:
            return False
        safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in event_key)[:80]
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}.jpg"
        destination = self.store.evidence_dir / filename
        return bool(cv2.imwrite(str(destination), frame))

    def _detect_gesture(self, frame) -> GestureDetection:
        if self.finger_counter is None:
            return GestureDetection(None, "low", "Локальный детектор пальцев недоступен.")
        prepared = self._prepare_analysis_frame(frame)
        return self.finger_counter.detect(prepared)

    def _build_local_gesture_result(self, detection: GestureDetection) -> AnalysisResult:
        if detection.digit is None:
            return AnalysisResult(
                summary="Число пальцев не распознано.",
                observations=[detection.reason],
                gesture_digit=None,
                gesture_confidence=detection.confidence,
                gesture_reason=detection.reason,
                raw={"source": detection.source, "gesture_digit": None, "reason": detection.reason},
            )

        return AnalysisResult(
            summary=f"Распознано число {detection.digit}.",
            observations=[detection.reason],
            gesture_digit=detection.digit,
            gesture_confidence=detection.confidence,
            gesture_reason=detection.reason,
            memory_updates={"gesture_count": detection.digit},
            raw={"source": detection.source, "gesture_digit": detection.digit, "reason": detection.reason},
        )

    def _is_fast_finger_mode(self) -> bool:
        assert self.config is not None
        prompt = (self.config.prompt or "").lower()
        finger_keywords = ["сколько пальцев", "показывает пальцами", "показываю пальцами", "число пальцев", "пальц"]
        scene_keywords = ["дым", "огонь", "пожар", "лекар", "кухн", "плита", "напомин", "sos", "тревог"]
        has_finger = any(keyword in prompt for keyword in finger_keywords)
        has_scene = any(keyword in prompt for keyword in scene_keywords)
        return has_finger and not has_scene

    def _prepare_analysis_frame(self, frame):
        if cv2 is None:
            return frame
        height, width = frame.shape[:2]
        max_side = max(width, height)
        if self.config and self.config.vision_memory_enabled:
            target_max_side = 512 if self.config.source_mode == "screen_regions" else 384
        else:
            target_max_side = 640 if self.config and self.config.source_mode == "screen_regions" else 512
        if max_side <= target_max_side:
            return frame
        scale = target_max_side / float(max_side)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        return resized

    def _write_last_response(self, result: AnalysisResult, raw_response: str) -> None:
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "parsed": result.raw,
            "raw_response": raw_response,
        }
        self.store.last_response_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_error(self, error: Exception) -> None:
        error_path = self.store.output_dir / "last_error.txt"
        error_path.write_text(f"{datetime.now().isoformat(timespec='seconds')} {error}\n", encoding="utf-8")

    def _mark_analysis_cancelled(self, analysis_id: int) -> None:
        self.log_callback(f"Анализ #{analysis_id}: остановлен пользователем")
        self._set_progress({"state": "cancelled", "analysis_id": analysis_id})
        self.store.append_analysis_history({"analysis_id": analysis_id, "state": "cancelled"})
        self.store.write_status(
            {
                "state": "cancelled",
                "message": "Analysis cancelled by user.",
                "analysis_id": analysis_id,
            }
        )
        self._set_analysis_view(f"Анализ #{analysis_id}\n\nОстановлен пользователем.")
        self._set_status("Анализ остановлен")

    def _set_status(self, message: str) -> None:
        if self.status_callback:
            self.status_callback(message)

    def _set_analysis_view(self, message: str) -> None:
        self.store.write_analysis_view(message)
        if self.analysis_callback:
            self.analysis_callback(message)

    def _set_answer(self, payload: dict) -> None:
        if self.answer_callback:
            self.answer_callback(payload)

    def _set_progress(self, payload: dict) -> None:
        if self.progress_callback:
            self.progress_callback(payload)

    def _set_memory_event(self, payload: dict) -> None:
        if self.memory_callback:
            self.memory_callback(payload)

    def _build_system_prompt(self) -> str:
        return (
            "You are Gemma 4 working as a local vision assistant.\n"
            "Return ONLY one valid JSON object, without markdown and without text outside JSON.\n"
            "All user-facing text must be in Russian.\n"
            "assistant_reply: detailed Russian description of the current frame, normally 5-8 sentences. "
            "Describe visible objects, people, hands, gestures, text, colors, lighting, background, positions, "
            "notable details, and possible risks. Say when something is uncertain. Do not invent unseen details.\n"
            "summary: one concise Russian sentence about the frame.\n"
            "observations: 4-8 short Russian observations with concrete visual facts.\n"
            "If the image contains labels ROI 1, ROI 2 or other ROI labels, analyze every ROI separately in one response. "
            "For each ROI return regions[] with id, presence, defect, state, confidence and Russian description. "
            "presence must be 1 if the required part/object is present, 0 if the area is empty, null if unclear. "
            "defect must be 1 if the ROI has visible defect or wrong/empty state, 0 if it looks normal, null if unclear.\n"
            "gesture_reason: Russian explanation of finger/gesture recognition when relevant.\n"
            "Be conservative with emergency actions.\n"
            "If the same event was already handled and still appears in state/output, do not repeat it.\n"
            "For finger counting, set gesture_digit to an integer when a hand sign is clearly visible. "
            "If a digit is not visible, set gesture_digit to null and explain why in gesture_reason.\n"
            "Do not create empty file_actions.\n"
            "If there is no hazard and no medicine reminder, events must be [].\n"
            "If no file change is needed, file_actions must be [].\n"
            "Use this JSON schema:\n"
            "{\n"
            '  "assistant_reply": "подробное описание кадра на русском языке",\n'
            '  "summary": "краткое описание кадра на русском",\n'
            '  "observations": ["наблюдение 1", "наблюдение 2"],\n'
            '  "regions": [\n'
            '    {"id": "ROI 1", "presence": 1, "defect": 0, "state": "ok|empty|defect|unknown", "confidence": 0.0, "description": "описание области на русском"}\n'
            "  ],\n"
            '  "gesture_digit": 0,\n'
            '  "gesture_confidence": "low|medium|high",\n'
            '  "gesture_reason": "why the digit was or was not recognized",\n'
            '  "file_actions": [\n'
            '    {"path": "relative/path.txt", "mode": "append|overwrite", "content": "text"}\n'
            "  ],\n"
            '  "memory_updates": {"key": "value"},\n'
            '  "events": [\n'
            "    {\n"
            '      "event_key": "stable-unique-key",\n'
            '      "category": "gesture|fire|medicine|safety|other",\n'
            '      "severity": "low|medium|high|critical",\n'
            '      "message": "what happened",\n'
            '      "notify": true,\n'
            '      "trigger_command": "none|reminder|emergency",\n'
            '      "save_frame": true,\n'
            '      "cooldown_seconds": 3600\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "If there is no action, return empty arrays and objects."
        )

    def _build_user_prompt(self, timestamp: str, context: str) -> str:
        assert self.config is not None
        region_prompt = self._build_region_prompt()
        return (
            f"Current local time: {timestamp}\n\n"
            f"User prompt:\n{self.config.prompt.strip() or DEFAULT_PROMPT.strip()}\n\n"
            f"{region_prompt}"
            "Files from output directory:\n"
            f"{context}\n\n"
            "Describe only what is visible in the current frame. Answer in Russian. "
            "Remember: write files only inside output. Use stable event_key values."
        )

    def _build_region_prompt(self) -> str:
        if not self.config or self.config.source_mode != "screen_regions":
            return ""
        lines = [
            "Selected screen regions are combined into one image and labeled visually:",
        ]
        for index, region in enumerate(self._screen_regions(), start=1):
            label = str(region.get("label") or f"ROI {index}")
            lines.append(
                f"- {label}: screen x={int(region.get('x', 0))}, y={int(region.get('y', 0))}, "
                f"width={int(region.get('width', 0))}, height={int(region.get('height', 0))}"
            )
        lines.append("Analyze all listed ROI areas in one pass and fill regions[] for every ROI.\n\n")
        return "\n".join(lines)
