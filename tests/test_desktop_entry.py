import unittest

import app
import app as app_module
from switch_monitor import desktop
from switch_monitor.runtimes import RuntimeRegistry


class DesktopEntryTests(unittest.TestCase):
    def test_app_entry_uses_desktop_main(self) -> None:
        self.assertIs(app_module.main, desktop.main)

    def test_registry_accepts_project_root(self) -> None:
        registry = RuntimeRegistry()
        self.assertIsNotNone(registry.get("official-gemma"))

    def test_desktop_has_model_memory_clear_action(self) -> None:
        self.assertTrue(hasattr(desktop.SwitchDesktopApp, "clear_model_memory"))

    def test_desktop_source_modes_are_available(self) -> None:
        self.assertEqual(desktop.SOURCE_MODES["Камера"], "camera")
        self.assertEqual(desktop.SOURCE_MODES["Область"], "screen_region")
        self.assertEqual(desktop.SOURCE_MODES["2 области"], "screen_regions")
        self.assertEqual(desktop.SOURCE_MODES["Экран"], "screen_full")

    def test_desktop_tag_library_mode_is_available(self) -> None:
        self.assertEqual(desktop.APP_MODES["Библиотека тегов"], "tag_library")
        self.assertTrue(hasattr(desktop.SwitchDesktopApp, "search_tag_library"))
        self.assertTrue(hasattr(desktop.SwitchDesktopApp, "search_ai_library"))
        self.assertTrue(hasattr(desktop.SwitchDesktopApp, "reset_library_search"))

    def test_tag_library_mode_uses_memory_indexing(self) -> None:
        enabled_modes = {"classifier", "tag_library"}
        self.assertIn(desktop.APP_MODES["Классификатор"], enabled_modes)
        self.assertIn(desktop.APP_MODES["Библиотека тегов"], enabled_modes)


if __name__ == "__main__":
    unittest.main()
