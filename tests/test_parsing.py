import unittest

from switch_monitor.parsing import (
    FileAction,
    AnalysisResult,
    EventAction,
    format_analysis_for_display,
    parse_analysis_response,
    parse_partial_analysis_response,
)


class ParsingTests(unittest.TestCase):
    def test_extracts_json_from_wrapped_response(self) -> None:
        response = """
        Here is the result:
        {
          "assistant_reply": "Вижу один поднятый палец.",
          "summary": "hand with one finger",
          "observations": ["one finger up"],
          "gesture_digit": 1,
          "gesture_confidence": "high",
          "gesture_reason": "One raised finger is clearly visible",
          "file_actions": [{"path": "digits/current.txt", "mode": "overwrite", "content": "1"}],
          "memory_updates": {"last_digit": "1"},
          "events": [
            {
              "event_key": "gesture-1",
              "category": "gesture",
              "severity": "low",
              "message": "Detected number 1",
              "notify": false,
              "trigger_command": "none",
              "save_frame": false,
              "cooldown_seconds": 10
            }
          ]
        }
        """
        parsed = parse_analysis_response(response)
        self.assertEqual(parsed.assistant_reply, "Вижу один поднятый палец.")
        self.assertEqual(parsed.summary, "hand with one finger")
        self.assertEqual(parsed.file_actions[0].path, "digits/current.txt")
        self.assertEqual(parsed.memory_updates["last_digit"], "1")
        self.assertEqual(parsed.events[0].event_key, "gesture-1")
        self.assertEqual(parsed.gesture_digit, 1)
        self.assertEqual(parsed.gesture_confidence, "high")

    def test_formats_analysis_for_display(self) -> None:
        result = AnalysisResult(
            assistant_reply="Вижу один поднятый палец.",
            summary="Detected one finger",
            observations=["One hand is visible", "One finger is raised"],
            file_actions=[FileAction(path="digits/current.txt", content="1", mode="overwrite")],
            memory_updates={"last_digit": "1"},
            gesture_digit=1,
            gesture_confidence="high",
            gesture_reason="One finger is raised",
            events=[
                EventAction(
                    event_key="gesture-1",
                    category="gesture",
                    severity="low",
                    message="Detected number 1",
                )
            ],
        )
        text = format_analysis_for_display(result)
        self.assertIn("Ответ Gemma:", text)
        self.assertIn("Detected one finger", text)
        self.assertIn("digits/current.txt", text)
        self.assertIn("Detected number 1", text)

    def test_accepts_observations_as_single_string(self) -> None:
        parsed = parse_analysis_response(
            """
            {
              "assistant_reply": "Ответ",
              "summary": "summary",
              "observations": "single observation",
              "gesture_digit": null,
              "gesture_confidence": 0.0,
              "gesture_reason": "none",
              "file_actions": [],
              "memory_updates": [],
              "events": [{"event_type": "ignored"}]
            }
            """
        )

        self.assertEqual(parsed.observations, ["single observation"])
        self.assertEqual(parsed.memory_updates, {})
        self.assertEqual(parsed.events, [])

    def test_partial_response_recovers_visible_answer(self) -> None:
        parsed = parse_partial_analysis_response(
            '{"assistant_reply":"Руки не видны.","summary":"Нет рук","observations":["тестовый кадр"],"file_actions":[',
            "truncated",
        )

        self.assertEqual(parsed.assistant_reply, "Руки не видны.")
        self.assertEqual(parsed.summary, "Нет рук")
        self.assertEqual(parsed.observations, ["тестовый кадр"])
        self.assertTrue(parsed.raw["partial"])


if __name__ == "__main__":
    unittest.main()
