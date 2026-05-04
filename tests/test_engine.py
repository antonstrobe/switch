import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
from PIL import Image

from switch_monitor.engine import CameraMonitor, MonitorConfig
from switch_monitor.gesture import GestureDetection
from switch_monitor.runtimes import BaseRuntime
from switch_monitor.storage import OutputStore


class DummyRuntime(BaseRuntime):
    name = "dummy"

    def detect_models(self):
        return []

    def analyze(self, prepared_model_id: str, system_prompt: str, user_prompt: str, image_bytes: bytes) -> str:
        return "{}"


class TruncatedRuntime(BaseRuntime):
    name = "truncated"

    def __init__(self) -> None:
        self.calls = 0

    def detect_models(self):
        return []

    def analyze(self, prepared_model_id: str, system_prompt: str, user_prompt: str, image_bytes: bytes) -> str:
        self.calls += 1
        return '{"assistant_reply":"Ответ только один раз.","summary":"Один ответ","observations":["кадр"],"file_actions":['


class BlockingRuntime(BaseRuntime):
    name = "blocking"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.stop_calls = 0

    def detect_models(self):
        return []

    def analyze(self, prepared_model_id: str, system_prompt: str, user_prompt: str, image_bytes: bytes) -> str:
        self.started.set()
        if self.stopped.wait(timeout=20):
            raise RuntimeError("cancelled")
        return "{}"

    def stop(self) -> None:
        self.stop_calls += 1
        self.stopped.set()


class EngineTests(unittest.TestCase):
    def test_fast_finger_mode_detects_simple_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OutputStore(Path(temp_dir))
            monitor = CameraMonitor(
                Path(temp_dir),
                runtime=DummyRuntime(),
                store=store,
                preview_callback=lambda frame: None,
                log_callback=lambda message: None,
                popup_callback=lambda title, message: None,
            )
            monitor.config = MonitorConfig(
                runtime_name="dummy",
                model_id="dummy",
                prepared_model_id="dummy",
                camera_index=0,
                interval_seconds=1.0,
                prompt="Сколько пальцев я показываю",
            )
            self.assertTrue(monitor._is_fast_finger_mode())

    def test_fast_finger_mode_disabled_for_scene_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OutputStore(Path(temp_dir))
            monitor = CameraMonitor(
                Path(temp_dir),
                runtime=DummyRuntime(),
                store=store,
                preview_callback=lambda frame: None,
                log_callback=lambda message: None,
                popup_callback=lambda title, message: None,
            )
            monitor.config = MonitorConfig(
                runtime_name="dummy",
                model_id="dummy",
                prepared_model_id="dummy",
                camera_index=0,
                interval_seconds=1.0,
                prompt="Следи за кухней и если вижу огонь, напиши тревогу",
            )
            self.assertFalse(monitor._is_fast_finger_mode())

    def test_build_local_gesture_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OutputStore(Path(temp_dir))
            monitor = CameraMonitor(
                Path(temp_dir),
                runtime=DummyRuntime(),
                store=store,
                preview_callback=lambda frame: None,
                log_callback=lambda message: None,
                popup_callback=lambda title, message: None,
            )
            result = monitor._build_local_gesture_result(
                GestureDetection(digit=4, confidence="high", reason="Локально распознано 4 пальца.")
            )
            self.assertEqual(result.gesture_digit, 4)
            self.assertEqual(result.gesture_confidence, "high")

    def test_truncated_json_does_not_send_frame_twice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TruncatedRuntime()
            store = OutputStore(Path(temp_dir))
            answers = []
            logs = []
            monitor = CameraMonitor(
                Path(temp_dir),
                runtime=runtime,
                store=store,
                preview_callback=lambda frame: None,
                log_callback=logs.append,
                popup_callback=lambda title, message: None,
                answer_callback=answers.append,
                progress_callback=lambda payload: None,
            )
            monitor.config = MonitorConfig(
                runtime_name="truncated",
                model_id="dummy",
                prepared_model_id="dummy",
                camera_index=0,
                interval_seconds=1.0,
                prompt="Опиши кадр",
            )

            monitor._analyze_frame(np.zeros((64, 64, 3), dtype=np.uint8))

            self.assertEqual(runtime.calls, 1)
            self.assertEqual(answers[0]["assistant_reply"], "Ответ только один раз.")
            self.assertFalse(any("Ответ только один раз" in item for item in logs))

    def test_system_prompt_requests_detailed_russian_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = CameraMonitor(
                Path(temp_dir),
                runtime=DummyRuntime(),
                store=OutputStore(Path(temp_dir)),
                preview_callback=lambda frame: None,
                log_callback=lambda message: None,
                popup_callback=lambda title, message: None,
            )
            system_prompt = monitor._build_system_prompt()

        self.assertIn("detailed Russian description", system_prompt)
        self.assertIn("All user-facing text must be in Russian", system_prompt)
        self.assertNotIn("Keep all strings short", system_prompt)
        self.assertNotIn("one short Russian answer", system_prompt)

    def test_stop_cancels_running_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = BlockingRuntime()
            monitor = CameraMonitor(
                Path(temp_dir),
                runtime=runtime,
                store=OutputStore(Path(temp_dir)),
                preview_callback=lambda frame: None,
                log_callback=lambda message: None,
                popup_callback=lambda title, message: None,
                progress_callback=lambda payload: None,
            )
            monitor.config = MonitorConfig(
                runtime_name="blocking",
                model_id="dummy",
                prepared_model_id="dummy",
                camera_index=0,
                interval_seconds=1.0,
                prompt="Опиши кадр",
            )
            analysis_thread = threading.Thread(
                target=monitor._analyze_frame,
                args=(np.zeros((64, 64, 3), dtype=np.uint8),),
                daemon=True,
            )
            monitor.analysis_thread = analysis_thread
            analysis_thread.start()
            self.assertTrue(runtime.started.wait(timeout=3))

            started_at = time.time()
            monitor.stop()

            self.assertLess(time.time() - started_at, 3)
            self.assertFalse(analysis_thread.is_alive())
            self.assertEqual(runtime.stop_calls, 1)

    def test_screen_region_capture_uses_selected_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = CameraMonitor(
                Path(temp_dir),
                runtime=DummyRuntime(),
                store=OutputStore(Path(temp_dir)),
                preview_callback=lambda frame: None,
                log_callback=lambda message: None,
                popup_callback=lambda title, message: None,
            )
            monitor.config = MonitorConfig(
                runtime_name="dummy",
                model_id="dummy",
                prepared_model_id="dummy",
                camera_index=0,
                interval_seconds=1.0,
                prompt="Опиши кадр",
                source_mode="screen_region",
                screen_region={"x": 10, "y": 20, "width": 80, "height": 60},
            )
            calls = []

            def fake_grab(bbox=None):
                calls.append(bbox)
                return Image.new("RGB", (80, 60), (1, 2, 3))

            with unittest.mock.patch("switch_monitor.engine.ImageGrab.grab", side_effect=fake_grab):
                frame = monitor._capture_screen_frame()

            self.assertEqual(calls[0], (10, 20, 90, 80))
            self.assertEqual(frame.shape[:2], (60, 80))

    def test_screen_regions_capture_composes_two_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = CameraMonitor(
                Path(temp_dir),
                runtime=DummyRuntime(),
                store=OutputStore(Path(temp_dir)),
                preview_callback=lambda frame: None,
                log_callback=lambda message: None,
                popup_callback=lambda title, message: None,
            )
            monitor.config = MonitorConfig(
                runtime_name="dummy",
                model_id="dummy",
                prepared_model_id="dummy",
                camera_index=0,
                interval_seconds=1.0,
                prompt="Опиши кадр",
                source_mode="screen_regions",
                screen_regions=[
                    {"label": "ROI 1", "x": 10, "y": 20, "width": 80, "height": 60},
                    {"label": "ROI 2", "x": 100, "y": 120, "width": 40, "height": 30},
                ],
            )
            calls = []

            def fake_grab(bbox=None):
                calls.append(bbox)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                return Image.new("RGB", (width, height), (1, 2, 3))

            with unittest.mock.patch("switch_monitor.engine.ImageGrab.grab", side_effect=fake_grab):
                frame = monitor._capture_screen_frame()

            self.assertEqual(calls, [(10, 20, 90, 80), (100, 120, 140, 150)])
            self.assertGreater(frame.shape[0], 60)
            self.assertGreater(frame.shape[1], 120)

    def test_user_prompt_lists_two_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = CameraMonitor(
                Path(temp_dir),
                runtime=DummyRuntime(),
                store=OutputStore(Path(temp_dir)),
                preview_callback=lambda frame: None,
                log_callback=lambda message: None,
                popup_callback=lambda title, message: None,
            )
            monitor.config = MonitorConfig(
                runtime_name="dummy",
                model_id="dummy",
                prepared_model_id="dummy",
                camera_index=0,
                interval_seconds=1.0,
                prompt="Проверь две области",
                source_mode="screen_regions",
                screen_regions=[
                    {"label": "ROI 1", "x": 1, "y": 2, "width": 30, "height": 40},
                    {"label": "ROI 2", "x": 5, "y": 6, "width": 70, "height": 80},
                ],
            )

            prompt = monitor._build_user_prompt("2026-04-24 10:00:00", "")

            self.assertIn("ROI 1", prompt)
            self.assertIn("ROI 2", prompt)
            self.assertIn("regions[]", prompt)


if __name__ == "__main__":
    unittest.main()
