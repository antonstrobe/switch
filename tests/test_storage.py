import tempfile
import unittest
from pathlib import Path

from switch_monitor.parsing import AnalysisResult, EventAction, FileAction
from switch_monitor.storage import OutputStore


class StorageTests(unittest.TestCase):
    def test_duplicate_event_is_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OutputStore(Path(temp_dir))
            result = AnalysisResult(
                summary="test",
                events=[
                    EventAction(
                        event_key="fire-kitchen",
                        category="fire",
                        severity="critical",
                        message="Possible fire",
                        notify=True,
                        trigger_command="none",
                        cooldown_seconds=3600,
                    )
                ],
            )
            first = store.apply_result(result, None, "", "")
            second = store.apply_result(result, None, "", "")
            self.assertEqual(first["notifications"], 1)
            self.assertEqual(second["notifications"], 0)

    def test_external_commands_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OutputStore(Path(temp_dir))
            result = AnalysisResult(
                summary="test",
                events=[
                    EventAction(
                        event_key="medicine-reminder",
                        category="medicine",
                        severity="medium",
                        message="Time to take medicine",
                        notify=True,
                        trigger_command="reminder",
                        cooldown_seconds=60,
                    )
                ],
            )
            summary = store.apply_result(result, None, "fake-command", "other-fake-command")
            self.assertEqual(summary["commands"], 0)
            actions_log = (Path(temp_dir) / "output" / "actions.log").read_text(encoding="utf-8")
            self.assertIn("COMMAND_SKIPPED", actions_log)

    def test_output_prefix_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OutputStore(Path(temp_dir))
            result = AnalysisResult(summary="test", file_actions=[FileAction(path="output/digits/current.txt", content="1", mode="overwrite")])
            store.apply_result(result, None, "", "")
            self.assertTrue((Path(temp_dir) / "output" / "digits" / "current.txt").exists())

    def test_gesture_result_writes_status_and_current(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OutputStore(Path(temp_dir))
            result = AnalysisResult(
                summary="digit seen",
                gesture_digit=2,
                gesture_confidence="high",
                gesture_reason="Two fingers are visible",
            )
            store.apply_result(result, None, "", "")
            current = (Path(temp_dir) / "output" / "digits" / "current.txt").read_text(encoding="utf-8")
            status = (Path(temp_dir) / "output" / "digits" / "status.txt").read_text(encoding="utf-8")
            self.assertEqual(current, "2")
            self.assertIn("Распознано число: 2", status)

    def test_empty_file_action_does_not_erase_current_digit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OutputStore(Path(temp_dir))
            first = AnalysisResult(summary="digit seen", gesture_digit=3)
            store.apply_result(first, None, "", "")
            second = AnalysisResult(
                summary="nothing seen",
                gesture_digit=None,
                gesture_reason="Темно",
                file_actions=[FileAction(path="output/digits/current.txt", content="", mode="overwrite")],
            )
            store.apply_result(second, None, "", "")
            current = (Path(temp_dir) / "output" / "digits" / "current.txt").read_text(encoding="utf-8")
            status = (Path(temp_dir) / "output" / "digits" / "status.txt").read_text(encoding="utf-8")
            self.assertEqual(current, "3")
            self.assertIn("Число не распознано", status)


if __name__ == "__main__":
    unittest.main()
