import tempfile
import unittest
from pathlib import Path

import numpy as np

from switch_monitor.engine import CameraMonitor, MonitorConfig
from switch_monitor.runtimes import BaseRuntime
from switch_monitor.storage import OutputStore
from switch_monitor.vision_memory import (
    ModelInfo,
    Observation,
    OcrEntry,
    QualityInfo,
    RedisMemoryStore,
    RoomInfo,
    SceneInfo,
    TagEntry,
    VisionMemoryConfig,
    VisionObject,
    VisualMemoryQueryService,
    extract_query_intent,
    observation_from_model_output,
    recursive_json_terms,
)


class InvalidJsonRuntime(BaseRuntime):
    name = "invalid"

    def detect_models(self):
        return []

    def analyze(self, prepared_model_id: str, system_prompt: str, user_prompt: str, image_bytes: bytes, timeout=None) -> str:
        return "not-json"


class CancelledGemmaRuntime(BaseRuntime):
    name = "official-gemma"

    def detect_models(self):
        return []

    def analyze(self, prepared_model_id: str, system_prompt: str, user_prompt: str, image_bytes: bytes, timeout=None) -> str:
        raise RuntimeError("Local Gemma inference cancelled.")


def memory_store() -> RedisMemoryStore:
    return RedisMemoryStore(VisionMemoryConfig(redis_url="memory://tests", ttl_seconds=3600))


def observation(
    object_name: str = "cup",
    normalized_name: str = "кружка",
    image_hash: str = "hash1",
    unix_ts: float = 1000,
    confidence: float = 0.8,
    room: str = "кухня",
) -> Observation:
    return Observation(
        observation_id=f"obs-{image_hash}-{normalized_name}",
        frame_id="frame-1",
        source="camera",
        timestamp="2026-04-28T10:00:00",
        unix_ts=unix_ts,
        room=RoomInfo(room_id=room, label=room, confidence=0.8, evidence="видна раковина"),
        scene=SceneInfo(primary="kitchen", secondary=[], indoor_outdoor="indoor", summary="На столе видна кружка."),
        objects=[
            VisionObject(
                name=object_name,
                normalized_name=normalized_name,
                category="посуда",
                attributes=[],
                location_hint="на столе",
                near=["стол"],
                confidence=confidence,
            )
        ],
        ocr=[OcrEntry(text="ABC", normalized_text="abc", type="label", confidence=0.6)],
        landmarks=[],
        search_tags=[normalized_name, "стол", room],
        spatial_summary="на столе рядом со стеной",
        answer_hints=["искать на столе"],
        quality=QualityInfo(blur="low", brightness="normal", occlusion="none", useful=True),
        model=ModelInfo(name="test-model"),
        confidence=confidence,
        thumbnail_path=None,
        image_hash=image_hash,
        raw_model_output={},
    )


class VisionMemoryTests(unittest.TestCase):
    def test_save_observation_indexes_object_tag_room_and_ocr(self) -> None:
        store = memory_store()
        obs = observation()

        self.assertTrue(store.save_observation(obs))

        self.assertEqual(store.stats()["observations"], 1)
        self.assertTrue(store.search(extract_query_intent("Где кружка?")))
        self.assertTrue(store.search(extract_query_intent("Что на столе?")))
        self.assertTrue(store.search(extract_query_intent("Что было на кухне?")))
        self.assertTrue(store.search(extract_query_intent("ABC")))

    def test_search_by_synonym_finds_cup(self) -> None:
        store = memory_store()
        store.save_observation(observation(object_name="mug", normalized_name="кружка"))

        results = store.search(extract_query_intent("Где чашка?"))

        self.assertEqual(results[0].objects[0].normalized_name, "кружка")

    def test_recursive_json_terms_find_dynamic_cafe(self) -> None:
        payload = {
            "scene": {"summary": "Интерьер похож на кафе"},
            "dynamic_fields": {"place": {"kind": "coffee_shop", "sign": "Cafe"}},
        }

        terms = recursive_json_terms(payload)

        self.assertIn("кафе", terms)

    def test_model_output_keeps_geo_dynamic_tags_and_thumbnail_evidence(self) -> None:
        raw = """
        {
          "room": {"label": "улица", "confidence": 0.5, "evidence": "видна вывеска"},
          "scene": {"primary": "cafe exterior", "secondary": ["sign"], "indoor_outdoor": "outdoor", "summary": "Виден вход в кафе."},
          "objects": [{"name": "Cafe sign", "normalized_name": "кафе", "category": "place", "attributes": ["вывеска"], "object_tags": ["coffee_shop"], "state_tags": ["visible"], "evidence_tags": ["sign"], "location_hint": "на фасаде", "near": ["дверь"], "confidence": 0.9}],
          "ocr": [],
          "landmarks": [],
          "search_tags": ["кафе", "вывеска"],
          "dynamic_fields": {"business": {"type": "cafe", "visible_text": "Cafe"}},
          "quality": {"blur": "low", "brightness": "normal", "occlusion": "none", "useful": true},
          "confidence": 0.88
        }
        """
        store = memory_store()
        obs = observation_from_model_output(raw, "camera", "test-model", geo_label="цех 1", thumbnail_path="thumb.jpg", image_hash="cafe1")

        store.save_observation(obs)
        answer = VisualMemoryQueryService(store).answer("кафе").to_dict()

        self.assertFalse(answer["not_found"])
        self.assertEqual(obs.geo.label, "цех 1")
        self.assertEqual(answer["evidence"][0]["thumbnail_path"], "thumb.jpg")
        self.assertIn("json_preview", answer["evidence"][0])

    def test_model_output_builds_comment_and_tag_dictionary(self) -> None:
        raw = """
        {
          "comment": "На столе лежат очки рядом с кружкой.",
          "tags": {
            "очки": {"ru": "очки", "en": "glasses", "value": true, "confidence": 0.91, "aliases": ["eyeglasses"], "category": "object", "where": "на столе", "near": ["кружка"], "source_paths": ["objects[0]"]}
          },
          "scene": {"primary": "desk", "secondary": [], "indoor_outdoor": "indoor", "summary": "На столе лежат очки."},
          "objects": [{"name": "glasses", "normalized_name": "очки", "category": "предмет", "attributes": ["черные"], "location_hint": "на столе", "near": ["кружка"], "confidence": 0.9}],
          "ocr": [],
          "landmarks": [],
          "search_tags": ["очки", "стол"],
          "quality": {"blur": "low", "brightness": "normal", "occlusion": "none", "useful": true},
          "confidence": 0.9
        }
        """

        obs = observation_from_model_output(raw, "camera", "test-model")

        self.assertEqual(obs.comment, "На столе лежат очки рядом с кружкой.")
        self.assertIn("очки", obs.tags)
        self.assertEqual(obs.tags["очки"].en, "glasses")
        self.assertIn("кружка", obs.tags["очки"].near)

    def test_search_by_tag_uses_tag_dictionary(self) -> None:
        store = memory_store()
        obs = observation(object_name="cafe sign", normalized_name="кафе", image_hash="tag-cafe")
        obs.tags["кафе"] = TagEntry(ru="кафе", en="cafe", confidence=0.9, aliases=["coffee_shop"], category="place")
        store.save_observation(obs)

        self.assertTrue(store.search_by_tag("кафе"))
        self.assertTrue(store.search_by_tag("cafe"))

    def test_simple_json_search_scans_recursive_payload(self) -> None:
        store = memory_store()
        obs = observation(object_name="phone", normalized_name="телефон", image_hash="json-phone")
        obs.dynamic_fields = {"device": {"screen_text": "ANDROID TEST"}}
        store.save_observation(obs)

        results = store.search_json("ANDROID TEST", mode="simple")

        self.assertEqual(results[0].observation_id, obs.observation_id)

    def test_ai_json_search_uses_synonyms_and_indexes(self) -> None:
        store = memory_store()
        obs = observation(object_name="smartphone", normalized_name="телефон", image_hash="ai-phone", confidence=0.9)
        obs.tags["телефон"] = TagEntry(ru="телефон", en="phone", confidence=0.9, aliases=["smartphone"], category="object")
        store.save_observation(obs)

        results = store.search_json("смартфон", mode="ai")

        self.assertEqual(results[0].observation_id, obs.observation_id)

    def test_generated_tags_do_not_include_json_schema_noise(self) -> None:
        raw = """
        {
          "comment": "На столе виден черный смартфон рядом с кружкой.",
          "scene": {"primary": "table", "secondary": [], "indoor_outdoor": "indoor", "summary": "На столе виден смартфон."},
          "objects": [{"name": "smartphone", "normalized_name": "smartphone", "category": "техника", "attributes": ["черный", "экран"], "location_hint": "на столе", "near": ["кружка"], "confidence": 0.92}],
          "ocr": [],
          "landmarks": [],
          "search_tags": ["смартфон", "телефон", "phone"],
          "quality": {"blur": "low", "brightness": "normal", "occlusion": "none", "useful": true},
          "confidence": 0.91
        }
        """

        obs = observation_from_model_output(raw, "camera", "test-model")

        self.assertIn("телефон", obs.tags)
        self.assertIn("кружка", obs.tags)
        self.assertIn("черный", obs.tags)
        self.assertIn("экран", obs.tags)
        self.assertIn("техника", obs.tags)
        self.assertNotIn("objects_0_confidence", obs.tags)
        self.assertNotIn("0_91", obs.tags)

    def test_english_model_terms_become_russian_tag_keys(self) -> None:
        raw = """
        {
          "comment": "A black smartphone is visible on a table.",
          "scene": {"primary": "desk", "secondary": [], "indoor_outdoor": "indoor", "summary": "A phone is on a table."},
          "objects": [{"name": "smartphone", "normalized_name": "smartphone", "category": "device", "attributes": ["black", "screen"], "location_hint": "on table", "near": ["table"], "confidence": 0.9}],
          "ocr": [],
          "landmarks": [],
          "search_tags": ["phone", "smartphone", "screen", "black", "table"],
          "quality": {"blur": "low", "brightness": "normal", "occlusion": "none", "useful": true},
          "confidence": 0.9
        }
        """

        obs = observation_from_model_output(raw, "camera", "test-model")

        self.assertIn("телефон", obs.tags)
        self.assertIn("экран", obs.tags)
        self.assertIn("черный", obs.tags)
        self.assertIn("стол", obs.tags)
        self.assertIn("техника", obs.tags)

    def test_negative_example_tags_are_not_indexed(self) -> None:
        raw = """
        {
          "comment": "Виден стол и окно.",
          "tags": {
            "стол": {"en": "table", "confidence": 0.8, "where": "в кадре"},
            "телефон": {"en": "phone", "confidence": 0.0, "where": "не видно"},
            "кафе": {"en": "cafe", "value": false, "confidence": 0.0}
          },
          "search_tags": ["стол", "телефон", "кафе"],
          "confidence": 0.8
        }
        """
        store = memory_store()
        obs = observation_from_model_output(raw, "camera", "test-model")
        store.save_observation(obs)

        self.assertIn("стол", obs.tags)
        self.assertNotIn("телефон", obs.tags)
        self.assertNotIn("кафе", obs.tags)
        self.assertFalse(store.search_by_tag("телефон"))

    def test_search_by_tag_scans_legacy_recent_observations(self) -> None:
        store = memory_store()
        obs = observation(object_name="smartphone", normalized_name="телефон", image_hash="legacy-phone")
        payload = obs.to_dict()
        payload["tags"] = {}
        payload["search_tags"] = []
        legacy = Observation.from_dict(payload)
        legacy.tags = {}
        legacy.search_tags = []
        store._memory_obs[legacy.observation_id] = (9999999999, legacy.to_dict())
        store._memory_recent.insert(0, legacy.observation_id)

        results = store.search_by_tag("телефон")

        self.assertTrue(results)
        self.assertEqual(results[0].objects[0].normalized_name, "телефон")

    def test_person_visible_tag_key_is_russian(self) -> None:
        raw = """
        {
          "comment": "В кадре виден человек у окна.",
          "scene": {"primary": "room", "secondary": [], "indoor_outdoor": "indoor", "summary": "Виден человек."},
          "objects": [{"name": "person", "normalized_name": "person_visible", "category": "person", "attributes": [], "object_tags": ["person_visible"], "location_hint": "у окна", "near": ["окно"], "confidence": 0.8}],
          "ocr": [],
          "landmarks": [],
          "search_tags": ["person_visible"],
          "quality": {"blur": "low", "brightness": "normal", "occlusion": "none", "useful": true},
          "confidence": 0.8
        }
        """

        obs = observation_from_model_output(raw, "camera", "test-model")

        self.assertIn("человек", obs.tags)
        self.assertEqual(obs.tags["человек"].en, "person")

    def test_person_gender_query_maps_to_neutral_person_visible(self) -> None:
        store = memory_store()
        obs = observation(object_name="person", normalized_name="person_visible", image_hash="person1")
        obs.search_tags.append("person_visible")
        store.save_observation(obs)

        results = store.search(extract_query_intent("мужчина"))

        self.assertTrue(results)
        self.assertEqual(results[0].objects[0].normalized_name, "person_visible")

    def test_not_found_answer_is_grounded(self) -> None:
        store = memory_store()
        service = VisualMemoryQueryService(store)

        answer = service.answer("Где куртка?")

        self.assertTrue(answer.not_found)
        self.assertIn("не нашёл", answer.answer)

    def test_unmatched_query_does_not_return_recent_observation_as_found(self) -> None:
        store = memory_store()
        store.save_observation(observation(object_name="window", normalized_name="окно", image_hash="window1"))
        service = VisualMemoryQueryService(store)

        answer = service.answer("кафе")

        self.assertTrue(answer.not_found)

    def test_deduplicates_by_image_hash(self) -> None:
        store = memory_store()
        first = observation(image_hash="same")
        second = observation(image_hash="same", unix_ts=2000)

        self.assertTrue(store.save_observation(first))
        self.assertFalse(store.save_observation(second))
        self.assertEqual(store.stats()["observations"], 1)

    def test_query_ranking_prefers_fresh_confident_observation(self) -> None:
        store = memory_store()
        old = observation(object_name="keys", normalized_name="ключи", image_hash="old", unix_ts=1000, confidence=0.3)
        fresh = observation(object_name="keys", normalized_name="ключи", image_hash="fresh", unix_ts=9999999999, confidence=0.95)
        store.save_observation(old)
        store.save_observation(fresh)

        results = store.search(extract_query_intent("Где ключи?"))

        self.assertEqual(results[0].observation_id, fresh.observation_id)

    def test_invalid_json_does_not_break_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = memory_store()
            monitor = CameraMonitor(
                Path(temp_dir),
                runtime=InvalidJsonRuntime(),
                store=OutputStore(Path(temp_dir)),
                preview_callback=lambda frame: None,
                log_callback=lambda message: None,
                popup_callback=lambda title, message: None,
                memory_callback=lambda payload: None,
            )
            monitor.memory_store = store
            monitor.config = MonitorConfig(
                runtime_name="invalid",
                model_id="invalid",
                prepared_model_id="invalid",
                camera_index=0,
                interval_seconds=1,
                prompt="",
                vision_memory_enabled=True,
            )

            monitor._index_memory_frame(np.zeros((64, 64, 3), dtype=np.uint8))

            self.assertGreaterEqual(store.stats().get("errors", 0), 1)
            self.assertEqual(store.stats()["observations"], 1)

    def test_cancelled_gemma_worker_is_not_memory_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = memory_store()
            events = []
            monitor = CameraMonitor(
                Path(temp_dir),
                runtime=CancelledGemmaRuntime(),
                store=OutputStore(Path(temp_dir)),
                preview_callback=lambda frame: None,
                log_callback=lambda message: None,
                popup_callback=lambda title, message: None,
                memory_callback=events.append,
            )
            monitor.memory_store = store
            monitor.config = MonitorConfig(
                runtime_name="official-gemma",
                model_id="official-gemma",
                prepared_model_id="official-gemma",
                camera_index=0,
                interval_seconds=1,
                prompt="",
                vision_memory_enabled=True,
            )

            monitor._index_memory_frame(np.zeros((64, 64, 3), dtype=np.uint8))

            self.assertEqual(store.stats().get("errors", 0), 0)
            self.assertEqual(store.stats()["observations"], 0)
            self.assertIn("cancelled", [item.get("state") for item in events])

    def test_fallback_observation_extracts_partial_comment(self) -> None:
        from switch_monitor.vision_memory import fallback_observation_from_text

        obs = fallback_observation_from_text(
            '{"comment":"Телефон лежит на столе","tags":{"телефон":',
            "camera",
            "official-gemma",
            error="truncated",
        )

        self.assertEqual(obs.comment, "Телефон лежит на столе")
        self.assertIn("телефон", obs.tags)


if __name__ == "__main__":
    unittest.main()
