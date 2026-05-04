from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None

from .parsing import extract_json_block


VISION_MEMORY_PROMPT_VERSION = "vision_memory_v1"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_MEMORY_TTL_SECONDS = 86400
DEFAULT_CAPTURE_INTERVAL_SECONDS = 2.0
DEFAULT_MIN_FRAME_CHANGE_THRESHOLD = 0.04
DEFAULT_MAX_OBSERVATIONS_PER_QUERY = 20

SENSITIVE_RULES = (
    "Do not identify people. Do not infer health, age, gender, ethnicity, religion or other sensitive traits. "
    "If people are visible, use neutral labels such as person/person_visible."
)

VISION_MEMORY_SYSTEM_PROMPT = """Ты анализатор визуальной памяти. Верни только один короткий валидный JSON без markdown.

Формат:
{"comment":"подробный комментарий на русском","tags":{"русский_тег":{"en":"english","confidence":0.0,"where":"где видно","near":["ориентир"]}},"search_tags":["русский_тег","english"],"confidence":0.0}

Правила:
- Не копируй формат как текст, заполни его фактами из изображения.
- comment: 1-2 русских предложения.
- tags: 8-18 коротких тегов. Ключи только русские: телефон, экран, черный, стол, окно, человек, очки, кружка, кафе.
- Для телефона добавь телефон, смартфон, экран, техника.
- Для каждого тега укажи en и confidence. where/near добавляй коротко.
- Если есть человек, не определяй личность, пол, возраст и чувствительные признаки.
- Не выдумывай. Если не уверен, confidence ниже.
- JSON должен быть закрыт полностью."""

QUERY_ANSWER_PROMPT = """Ты помощник визуальной памяти. Пользователь спрашивает, где находится объект. Отвечай только на основании переданных наблюдений. Не выдумывай. Если данных недостаточно, скажи об этом. Дай короткий, практичный ответ: где было видно, когда, рядом с чем, насколько уверенно. Если возможно, дай простую инструкцию по ориентирам, но не утверждай маршрут между комнатами, если в наблюдениях нет таких данных.

Вопрос пользователя:
{question}

Наблюдения:
{observations_json}

Верни JSON:
{
  "answer": string,
  "confidence": number,
  "last_seen_at": string | null,
  "room": string | null,
  "location_hint": string | null,
  "near": string[],
  "evidence_observation_ids": string[],
  "not_found": boolean
}"""

SYNONYM_GROUPS: dict[str, list[str]] = {
    "person_visible": ["person", "people", "human", "человек", "люди", "мужчина", "женщина", "персона"],
    "кафе": ["cafe", "coffee shop", "coffee_shop", "кофейня", "ресторан", "бар"],
    "куртка": ["jacket", "coat", "outerwear", "одежда", "верхняя одежда", "пальто"],
    "кружка": ["mug", "cup", "чашка", "стакан"],
    "ключи": ["keys", "keychain", "ключ", "брелок"],
    "очки": ["glasses", "eyeglasses"],
    "телефон": ["phone", "smartphone", "mobile phone", "cellphone", "iphone", "android phone", "android", "мобильный", "мобильник", "смартфон", "айфон"],
    "экран": ["screen", "display"],
    "черный": ["black"],
    "темный": ["dark"],
    "техника": ["device", "electronics", "tech", "gadget"],
    "поверхность": ["surface"],
    "зарядка": ["charger", "cable", "провод", "кабель", "зарядное устройство"],
    "лекарства": ["pills", "medicine", "medication", "таблетки"],
    "сумка": ["bag", "backpack", "рюкзак"],
    "раковина": ["sink", "washbasin", "раковине", "раковиной"],
    "стол": ["table", "desk", "столе", "стола"],
    "стул": ["chair", "стуле"],
    "диван": ["sofa", "couch"],
    "кровать": ["bed"],
    "шкаф": ["wardrobe", "cabinet", "closet"],
    "дверь": ["door"],
    "окно": ["window"],
    "холодильник": ["fridge", "refrigerator"],
    "плита": ["stove"],
    "кухня": ["kitchen", "кухне"],
    "ванная": ["bathroom", "ванной"],
    "коридор": ["hallway", "коридоре"],
    "комната": ["room", "комнате"],
}

STOP_WORDS = {
    "где",
    "лежит",
    "лежали",
    "висит",
    "стоял",
    "стояла",
    "видел",
    "видела",
    "найди",
    "что",
    "было",
    "рядом",
    "около",
    "возле",
    "в",
    "не",
    "на",
    "с",
    "и",
    "а",
    "я",
    "последний",
    "последний раз",
    "недавно",
    "сегодня",
    "виден",
    "видна",
    "видно",
    "показан",
    "показана",
    "is",
    "are",
    "was",
    "were",
    "the",
    "an",
    "on",
    "in",
    "at",
    "of",
    "with",
    "visible",
}

SYNONYM_TO_CANONICAL: dict[str, str] = {}
for canonical, variants in SYNONYM_GROUPS.items():
    SYNONYM_TO_CANONICAL[canonical] = canonical
    for variant in variants:
        SYNONYM_TO_CANONICAL[variant] = canonical


@dataclass
class VisionMemoryConfig:
    redis_url: str = DEFAULT_REDIS_URL
    ttl_seconds: int = DEFAULT_MEMORY_TTL_SECONDS
    capture_interval_seconds: float = DEFAULT_CAPTURE_INTERVAL_SECONDS
    min_frame_change_threshold: float = DEFAULT_MIN_FRAME_CHANGE_THRESHOLD
    max_observations_per_query: int = DEFAULT_MAX_OBSERVATIONS_PER_QUERY

    @classmethod
    def from_env(cls) -> "VisionMemoryConfig":
        return cls(
            redis_url=os.environ.get("REDIS_URL", DEFAULT_REDIS_URL),
            ttl_seconds=_env_int("VISION_MEMORY_TTL_SECONDS", DEFAULT_MEMORY_TTL_SECONDS),
            capture_interval_seconds=_env_float("VISION_CAPTURE_INTERVAL_SECONDS", DEFAULT_CAPTURE_INTERVAL_SECONDS),
            min_frame_change_threshold=_env_float("VISION_MIN_FRAME_CHANGE_THRESHOLD", DEFAULT_MIN_FRAME_CHANGE_THRESHOLD),
            max_observations_per_query=_env_int(
                "VISION_MAX_OBSERVATIONS_PER_QUERY",
                DEFAULT_MAX_OBSERVATIONS_PER_QUERY,
            ),
        )


@dataclass
class RoomInfo:
    room_id: str | None = None
    label: str | None = None
    confidence: float = 0.0
    evidence: str | None = None


@dataclass
class SceneInfo:
    primary: str = ""
    secondary: list[str] = field(default_factory=list)
    indoor_outdoor: str = "unknown"
    summary: str = ""


@dataclass
class VisionObject:
    name: str
    normalized_name: str
    category: str = ""
    attributes: list[str] = field(default_factory=list)
    object_tags: list[str] = field(default_factory=list)
    state_tags: list[str] = field(default_factory=list)
    evidence_tags: list[str] = field(default_factory=list)
    location_hint: str | None = None
    near: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class OcrEntry:
    text: str
    normalized_text: str
    type: str = ""
    confidence: float = 0.0


@dataclass
class Landmark:
    name: str
    normalized_name: str
    description: str = ""
    confidence: float = 0.0


@dataclass
class QualityInfo:
    blur: str = "unknown"
    brightness: str = "unknown"
    occlusion: str = "unknown"
    useful: bool = True


@dataclass
class ModelInfo:
    name: str = ""
    version: str | None = None
    prompt_version: str = VISION_MEMORY_PROMPT_VERSION


@dataclass
class GeoInfo:
    label: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    source: str = "unknown"
    confidence: float = 0.0


@dataclass
class TagEntry:
    ru: str
    en: str = ""
    value: Any = True
    confidence: float = 0.0
    aliases: list[str] = field(default_factory=list)
    category: str = ""
    where: str | None = None
    near: list[str] = field(default_factory=list)
    source_paths: list[str] = field(default_factory=list)


@dataclass
class Observation:
    observation_id: str
    frame_id: str | None
    source: str
    timestamp: str
    unix_ts: float
    room: RoomInfo
    scene: SceneInfo
    objects: list[VisionObject]
    ocr: list[OcrEntry]
    landmarks: list[Landmark]
    search_tags: list[str]
    spatial_summary: str
    answer_hints: list[str]
    quality: QualityInfo
    model: ModelInfo
    confidence: float
    thumbnail_path: str | None = None
    image_hash: str | None = None
    raw_model_output: dict[str, Any] | None = None
    geo: GeoInfo = field(default_factory=GeoInfo)
    visual_tags: list[str] = field(default_factory=list)
    semantic_tags: list[str] = field(default_factory=list)
    context_tags: list[str] = field(default_factory=list)
    dynamic_tags: list[str] = field(default_factory=list)
    dynamic_fields: dict[str, Any] = field(default_factory=dict)
    comment: str = ""
    tags: dict[str, TagEntry] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Observation":
        observation = cls(
            observation_id=str(payload.get("observation_id") or uuid.uuid4()),
            frame_id=payload.get("frame_id"),
            source=str(payload.get("source") or "camera"),
            timestamp=str(payload.get("timestamp") or datetime.now().isoformat(timespec="seconds")),
            unix_ts=float(payload.get("unix_ts") or time.time()),
            room=RoomInfo(**_dict(payload.get("room"))),
            scene=SceneInfo(**_dict(payload.get("scene"))),
            objects=[VisionObject(**_dict(item)) for item in payload.get("objects", []) if isinstance(item, dict)],
            ocr=[OcrEntry(**_dict(item)) for item in payload.get("ocr", []) if isinstance(item, dict)],
            landmarks=[Landmark(**_dict(item)) for item in payload.get("landmarks", []) if isinstance(item, dict)],
            search_tags=[normalize_term(item) for item in payload.get("search_tags", []) if str(item).strip()],
            spatial_summary=str(payload.get("spatial_summary") or ""),
            answer_hints=[str(item).strip() for item in payload.get("answer_hints", []) if str(item).strip()],
            quality=QualityInfo(**_dict(payload.get("quality"))),
            model=ModelInfo(**_dict(payload.get("model"))),
            confidence=clamp_float(payload.get("confidence"), 0.0),
            thumbnail_path=payload.get("thumbnail_path"),
            image_hash=payload.get("image_hash"),
            raw_model_output=payload.get("raw_model_output") if isinstance(payload.get("raw_model_output"), dict) else None,
            geo=GeoInfo(**_dict(payload.get("geo"))),
            visual_tags=[normalize_term(item) for item in payload.get("visual_tags", []) if str(item).strip()],
            semantic_tags=[normalize_term(item) for item in payload.get("semantic_tags", []) if str(item).strip()],
            context_tags=[normalize_term(item) for item in payload.get("context_tags", []) if str(item).strip()],
            dynamic_tags=[normalize_term(item) for item in payload.get("dynamic_tags", []) if str(item).strip()],
            dynamic_fields=payload.get("dynamic_fields", {}) if isinstance(payload.get("dynamic_fields"), dict) else {},
            comment=str(payload.get("comment") or ""),
            tags=_tags_from_payload(payload.get("tags")),
        )
        if not observation.comment:
            observation.comment = observation.scene.summary or observation.spatial_summary
        if not observation.tags:
            observation.tags = _build_observation_tags(
                payload,
                observation.objects,
                observation.ocr,
                observation.landmarks,
                observation.room,
                observation.geo,
                observation.comment,
                observation.search_tags
                + observation.visual_tags
                + observation.semantic_tags
                + observation.context_tags
                + observation.dynamic_tags,
            )
        return observation


@dataclass
class QueryIntent:
    question: str
    tokens: list[str]
    target_objects: list[str]
    related_landmarks: list[str]
    room_hint: str | None
    time_hint: str | None
    question_type: str


@dataclass
class MemoryQueryAnswer:
    answer: str
    confidence: float
    last_seen_at: str | None
    room: str | None
    location_hint: str | None
    near: list[str]
    evidence: list[dict[str, Any]]
    not_found: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RedisMemoryStore:
    def __init__(self, config: VisionMemoryConfig | None = None) -> None:
        self.config = config or VisionMemoryConfig.from_env()
        self.client = None
        self.available = False
        self.backend = "memory"
        self._memory_obs: dict[str, tuple[float, dict[str, Any]]] = {}
        self._memory_indexes: dict[str, set[str]] = {}
        self._memory_recent: list[str] = []
        self._memory_stats: dict[str, int] = {}
        if self.config.redis_url.startswith("memory://"):
            return
        if redis is None:
            return
        try:
            client = redis.Redis.from_url(self.config.redis_url, decode_responses=True)
            client.ping()
            self.client = client
            self.available = True
            self.backend = "redis"
        except Exception:
            self.client = None

    def status(self) -> dict[str, Any]:
        stats = self.stats()
        return {
            "backend": self.backend,
            "redis_available": self.available,
            "ttl_seconds": self.config.ttl_seconds,
            "observations": stats.get("observations", 0),
            "indexed": stats.get("indexed", 0),
            "skipped": stats.get("skipped", 0),
            "errors": stats.get("errors", 0),
        }

    def save_observation(self, observation: Observation) -> bool:
        payload = observation.to_dict()
        if observation.image_hash and self.has_recent_hash(observation.image_hash):
            self.increment_stat("skipped")
            return False
        if self.client is not None:
            self._save_redis(observation, payload)
        else:
            self._save_memory(observation, payload)
        self.increment_stat("indexed")
        return True

    def has_recent_hash(self, image_hash: str) -> bool:
        if not image_hash:
            return False
        if self.client is not None:
            return bool(self.client.exists(f"vm:hash:{image_hash}"))
        self._cleanup_memory()
        return f"hash:{image_hash}" in self._memory_indexes

    def remember_hash(self, image_hash: str) -> None:
        if not image_hash:
            return
        if self.client is not None:
            self.client.setex(f"vm:hash:{image_hash}", self.config.ttl_seconds, "1")
            return
        self._memory_indexes.setdefault(f"hash:{image_hash}", set()).add(image_hash)

    def get_observation(self, observation_id: str) -> Observation | None:
        if self.client is not None:
            raw = self.client.get(f"vm:obs:{observation_id}")
            if not raw:
                return None
            return Observation.from_dict(json.loads(raw))
        self._cleanup_memory()
        item = self._memory_obs.get(observation_id)
        if not item:
            return None
        return Observation.from_dict(item[1])

    def search(self, intent: QueryIntent, limit: int | None = None) -> list[Observation]:
        limit = limit or self.config.max_observations_per_query
        candidate_ids = self._candidate_ids(intent)
        if not candidate_ids:
            candidate_ids = self.recent_ids(limit * 2)
        observations = [obs for obs_id in candidate_ids if (obs := self.get_observation(obs_id))]
        ranked = sorted(observations, key=lambda obs: self.rank_observation(obs, intent), reverse=True)
        return ranked[:limit]

    def search_by_tag(self, tag: str, limit: int | None = None) -> list[Observation]:
        limit = limit or self.config.max_observations_per_query
        terms = sorted(expand_terms(tokenize(tag) or [tag]))
        if not terms:
            return self.recent_observations(limit)
        keys = []
        for term in terms:
            keys.extend(
                [
                    f"vm:tag:{term}",
                    f"vm:object:{term}",
                    f"vm:ocr:{term}",
                    f"vm:text:{term}",
                    f"vm:room:{term}",
                    f"vm:geo:{term}",
                ]
            )
        observations = [obs for obs_id in self._ids_for_keys(keys) if (obs := self.get_observation(obs_id))]
        seen_ids = {obs.observation_id for obs in observations}
        for obs in self.recent_observations(200):
            if obs.observation_id in seen_ids:
                continue
            searchable = _observation_tag_terms(obs) | recursive_json_terms(obs.to_dict())
            if set(terms) & searchable:
                observations.append(obs)
                seen_ids.add(obs.observation_id)
        intent = QueryIntent(tag, terms, terms, [], None, None, "tag_search")
        ranked = sorted(observations, key=lambda obs: self.rank_observation(obs, intent), reverse=True)
        return ranked[:limit]

    def search_json(self, query: str, mode: str = "simple", limit: int | None = None) -> list[Observation]:
        limit = limit or self.config.max_observations_per_query
        query = str(query or "").strip()
        if not query:
            return self.recent_observations(200)
        tokens = tokenize(query)
        terms = set(expand_terms(tokens or [query]))
        normalized_query = normalize_text(query)
        candidates = self.recent_observations(200)
        if mode == "ai":
            intent = extract_query_intent(query)
            indexed = self.search(intent, limit=max(limit * 4, 40))
            by_id = {obs.observation_id: obs for obs in indexed + candidates}
            ranked = [
                (self._ai_json_search_score(obs, intent, terms, normalized_query), obs)
                for obs in by_id.values()
            ]
        else:
            ranked = [
                (self._simple_json_search_score(obs, terms, normalized_query), obs)
                for obs in candidates
            ]
        matches = [(score, obs) for score, obs in ranked if score > 0]
        matches.sort(key=lambda item: item[0], reverse=True)
        return [obs for _, obs in matches[:limit]]

    def recent_observations(self, limit: int = 20) -> list[Observation]:
        return [obs for obs_id in self.recent_ids(limit) if (obs := self.get_observation(obs_id))]

    def recent_ids(self, limit: int = 20) -> list[str]:
        if self.client is not None:
            return [str(item) for item in self.client.lrange("vm:recent", 0, limit - 1)]
        self._cleanup_memory()
        return self._memory_recent[:limit]

    def clear(self) -> int:
        if self.client is not None:
            count = 0
            for key in list(self.client.scan_iter("vm:*")):
                count += int(self.client.delete(key))
            return count
        count = len(self._memory_obs) + len(self._memory_indexes)
        self._memory_obs.clear()
        self._memory_indexes.clear()
        self._memory_recent.clear()
        self._memory_stats.clear()
        return count

    def stats(self) -> dict[str, int]:
        if self.client is not None:
            raw = self.client.hgetall("vm:stats")
            result = {str(key): int(value) for key, value in raw.items()}
            result["observations"] = int(self.client.zcard("vm:obs:time"))
            return result
        self._cleanup_memory()
        result = dict(self._memory_stats)
        result["observations"] = len(self._memory_obs)
        return result

    def increment_stat(self, name: str, amount: int = 1) -> None:
        if self.client is not None:
            self.client.hincrby("vm:stats", name, amount)
            self.client.expire("vm:stats", self.config.ttl_seconds)
            return
        self._memory_stats[name] = self._memory_stats.get(name, 0) + amount

    def rank_observation(self, observation: Observation, intent: QueryIntent) -> float:
        terms = set(intent.target_objects + intent.related_landmarks + intent.tokens)
        object_terms = {item.normalized_name for item in observation.objects}
        tag_terms = _observation_tag_terms(observation)
        landmark_terms = {item.normalized_name for item in observation.landmarks}
        near_terms = {normalize_term(item) for obj in observation.objects for item in obj.near}
        recursive_terms = recursive_json_terms(observation.to_dict())
        score = 0.0
        score += 4.0 * len(terms & object_terms)
        score += 2.0 * len(terms & tag_terms)
        score += 1.5 * len(terms & recursive_terms)
        score += 2.0 * len(set(intent.related_landmarks) & (landmark_terms | near_terms))
        if intent.room_hint and observation.room.label and normalize_term(intent.room_hint) == normalize_term(observation.room.label):
            score += 2.0
        if observation.geo.label and terms & expand_terms([observation.geo.label]):
            score += 1.0
        score += clamp_float(observation.confidence, 0.0)
        score += 0.5 if observation.quality.useful else -1.0
        age_seconds = max(time.time() - observation.unix_ts, 0.0)
        score += 1.0 / (1.0 + age_seconds / 86400.0)
        if any(obj.location_hint for obj in observation.objects):
            score += 0.5
        return score

    def _simple_json_search_score(self, observation: Observation, terms: set[str], normalized_query: str) -> float:
        payload = observation.to_dict()
        text = _flattened_json_text(payload)
        score = 0.0
        if normalized_query and normalized_query in normalize_text(text):
            score += 6.0
        json_terms = recursive_json_terms(payload)
        tag_terms = _observation_tag_terms(observation)
        score += 2.0 * len(terms & tag_terms)
        score += 1.0 * len(terms & json_terms)
        return score

    def _ai_json_search_score(
        self,
        observation: Observation,
        intent: QueryIntent,
        terms: set[str],
        normalized_query: str,
    ) -> float:
        score = self.rank_observation(observation, intent)
        payload = observation.to_dict()
        json_terms = recursive_json_terms(payload)
        tag_terms = _observation_tag_terms(observation)
        score += 1.5 * len(terms & tag_terms)
        score += 1.0 * len(terms & json_terms)
        text = _flattened_json_text(payload)
        if normalized_query and normalized_query in normalize_text(text):
            score += 3.0
        if "не на человеке" in normalized_query or "не на лице" in normalized_query:
            tag_text = normalize_text(json.dumps(payload.get("tags", {}), ensure_ascii=False))
            if "человек" not in tag_text and "лицо" not in tag_text:
                score += 2.0
            else:
                score -= 1.0
        return score

    def _save_redis(self, observation: Observation, payload: dict[str, Any]) -> None:
        assert self.client is not None
        observation_id = observation.observation_id
        pipe = self.client.pipeline()
        pipe.setex(f"vm:obs:{observation_id}", self.config.ttl_seconds, json.dumps(payload, ensure_ascii=False))
        pipe.zadd("vm:obs:time", {observation_id: observation.unix_ts})
        pipe.expire("vm:obs:time", self.config.ttl_seconds)
        pipe.lpush("vm:recent", observation_id)
        pipe.ltrim("vm:recent", 0, 199)
        pipe.expire("vm:recent", self.config.ttl_seconds)
        for key in self._index_keys(observation):
            pipe.sadd(key, observation_id)
            pipe.expire(key, self.config.ttl_seconds)
        if observation.image_hash:
            pipe.setex(f"vm:hash:{observation.image_hash}", self.config.ttl_seconds, observation_id)
        pipe.execute()

    def _save_memory(self, observation: Observation, payload: dict[str, Any]) -> None:
        expires_at = time.time() + self.config.ttl_seconds
        self._memory_obs[observation.observation_id] = (expires_at, payload)
        self._memory_recent.insert(0, observation.observation_id)
        self._memory_recent = self._memory_recent[:200]
        for key in self._index_keys(observation):
            self._memory_indexes.setdefault(key, set()).add(observation.observation_id)
        if observation.image_hash:
            self._memory_indexes.setdefault(f"hash:{observation.image_hash}", set()).add(observation.image_hash)

    def _candidate_ids(self, intent: QueryIntent) -> list[str]:
        keys = []
        for term in intent.target_objects:
            keys.append(f"vm:object:{term}")
            keys.append(f"vm:tag:{term}")
            keys.append(f"vm:ocr:{term}")
            keys.append(f"vm:text:{term}")
        for term in intent.related_landmarks:
            keys.append(f"vm:tag:{term}")
            keys.append(f"vm:object:{term}")
            keys.append(f"vm:text:{term}")
        if intent.room_hint:
            keys.append(f"vm:room:{normalize_term(intent.room_hint)}")
        for token in intent.tokens:
            keys.append(f"vm:tag:{token}")
            keys.append(f"vm:text:{token}")
            keys.append(f"vm:geo:{token}")
        return self._ids_for_keys(keys)

    def _ids_for_keys(self, keys: list[str]) -> list[str]:
        if self.client is not None:
            ids: set[str] = set()
            for key in keys:
                ids.update(str(item) for item in self.client.smembers(key))
            return list(ids)
        self._cleanup_memory()
        ids = set()
        for key in keys:
            ids.update(self._memory_indexes.get(key, set()))
        return list(ids)

    def _index_keys(self, observation: Observation) -> set[str]:
        keys = set()
        for item in _observation_tag_terms(observation):
            term = normalize_term(item)
            if term:
                keys.add(f"vm:tag:{term}")
        for obj in observation.objects:
            object_terms = [obj.name, obj.normalized_name, obj.category] + obj.attributes + obj.object_tags + obj.state_tags + obj.evidence_tags + obj.near
            for term in expand_terms(object_terms):
                keys.add(f"vm:object:{term}")
                keys.add(f"vm:tag:{term}")
        for item in observation.ocr:
            for term in expand_terms([item.text, item.normalized_text]):
                keys.add(f"vm:ocr:{term}")
                keys.add(f"vm:tag:{term}")
        for item in observation.landmarks:
            for term in expand_terms([item.name, item.normalized_name, item.description]):
                keys.add(f"vm:tag:{term}")
        if observation.room.label:
            keys.add(f"vm:room:{normalize_term(observation.room.label)}")
        if observation.geo.label:
            for term in expand_terms([observation.geo.label]):
                keys.add(f"vm:geo:{term}")
                keys.add(f"vm:tag:{term}")
        for term in recursive_json_terms(observation.to_dict()):
            keys.add(f"vm:text:{term}")
        return {key for key in keys if not key.endswith(":")}

    def _cleanup_memory(self) -> None:
        now = time.time()
        expired = {obs_id for obs_id, (expires_at, _) in self._memory_obs.items() if expires_at < now}
        for obs_id in expired:
            self._memory_obs.pop(obs_id, None)
        if expired:
            for values in self._memory_indexes.values():
                values.difference_update(expired)
            self._memory_recent = [obs_id for obs_id in self._memory_recent if obs_id not in expired]


def _observation_tag_terms(observation: Observation) -> set[str]:
    values: list[str] = (
        observation.search_tags
        + observation.visual_tags
        + observation.semantic_tags
        + observation.context_tags
        + observation.dynamic_tags
        + list(observation.tags.keys())
    )
    for tag in observation.tags.values():
        values.extend([tag.ru, tag.en, tag.category, tag.where or ""])
        values.extend(tag.aliases)
        values.extend(tag.near)
    return expand_terms(values)


def _flattened_json_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in flatten_json(payload):
        path = str(item.get("path") or "")
        if path.startswith("raw_model_output"):
            continue
        parts.append(path)
        parts.append(str(item.get("value") or ""))
    return " ".join(parts)


class VisualMemoryQueryService:
    def __init__(self, store: RedisMemoryStore) -> None:
        self.store = store

    def answer(self, question: str) -> MemoryQueryAnswer:
        intent = extract_query_intent(question)
        observations = self.store.search(intent)
        relevant = [
            obs
            for obs in observations
            if self._matches_intent(obs, intent) and self.store.rank_observation(obs, intent) >= 1.0
        ]
        if not relevant:
            return self._not_found(intent, observations[:3])
        best = relevant[0]
        evidence = [self._evidence(obs, intent) for obs in relevant[:5]]
        matched_object = self._best_object(best, intent)
        location_hint = matched_object.location_hint if matched_object else None
        near = matched_object.near if matched_object else []
        room = best.room.label
        confidence = min(0.99, max(0.1, self.store.rank_observation(best, intent) / 8.0))
        where = location_hint or best.spatial_summary or best.scene.summary or "в одном из сохранённых наблюдений"
        object_text = ", ".join(intent.target_objects[:2]) or "объект"
        answer = f"{object_text} был замечен {where}"
        if room:
            answer += f", зона/комната: {room}"
        if near:
            answer += f", рядом: {', '.join(near[:4])}"
        answer += f". Время: {best.timestamp}."
        return MemoryQueryAnswer(
            answer=answer,
            confidence=round(confidence, 2),
            last_seen_at=best.timestamp,
            room=room,
            location_hint=location_hint,
            near=near,
            evidence=evidence,
            not_found=False,
        )

    def _not_found(self, intent: QueryIntent, fallback: list[Observation]) -> MemoryQueryAnswer:
        evidence = [self._evidence(obs, intent) for obs in fallback]
        answer = "Я не нашёл этот объект в визуальной памяти."
        if evidence:
            summaries = [item["summary"] for item in evidence if item.get("summary")]
            if summaries:
                answer += " Похожие наблюдения: " + "; ".join(summaries[:3])
        return MemoryQueryAnswer(
            answer=answer,
            confidence=0.0,
            last_seen_at=None,
            room=None,
            location_hint=None,
            near=[],
            evidence=evidence,
            not_found=True,
        )

    def _best_object(self, observation: Observation, intent: QueryIntent) -> VisionObject | None:
        terms = set(intent.target_objects + intent.tokens)
        for obj in observation.objects:
            if obj.normalized_name in terms or any(normalize_term(item) in terms for item in obj.attributes):
                return obj
        return observation.objects[0] if observation.objects else None

    def _matches_intent(self, observation: Observation, intent: QueryIntent) -> bool:
        if intent.question_type == "list_objects":
            return True
        terms = set(intent.target_objects + intent.related_landmarks + intent.tokens)
        if not terms:
            return True
        object_terms = {obj.normalized_name for obj in observation.objects}
        object_terms.update(normalize_term(obj.name) for obj in observation.objects)
        object_terms.update(term for obj in observation.objects for term in obj.object_tags + obj.state_tags + obj.evidence_tags)
        tag_terms = _observation_tag_terms(observation)
        near_terms = {normalize_term(item) for obj in observation.objects for item in obj.near}
        recursive_terms = recursive_json_terms(observation.to_dict())
        geo_terms = expand_terms([observation.geo.label or ""])
        room_terms = expand_terms([observation.room.label or ""])
        return bool(terms & (object_terms | tag_terms | near_terms | recursive_terms | geo_terms | room_terms))

    def _evidence(self, observation: Observation, intent: QueryIntent) -> dict[str, Any]:
        terms = set(intent.target_objects + intent.related_landmarks + intent.tokens)
        matched_objects = [
            obj.name
            for obj in observation.objects
            if obj.normalized_name in terms or normalize_term(obj.name) in terms
        ]
        matched_landmarks = [
            item.name
            for item in observation.landmarks
            if item.normalized_name in terms or normalize_term(item.name) in terms
        ]
        return {
            "observation_id": observation.observation_id,
            "timestamp": observation.timestamp,
            "summary": observation.comment or observation.scene.summary or observation.spatial_summary,
            "matched_objects": matched_objects,
            "matched_landmarks": matched_landmarks,
            "thumbnail_path": observation.thumbnail_path,
            "image_path": observation.thumbnail_path,
            "geo": observation.geo.label,
            "confidence": observation.confidence,
            "tags": {key: tag.to_dict() if hasattr(tag, "to_dict") else asdict(tag) for key, tag in observation.tags.items()},
            "json_preview": recursive_json_preview(observation.to_dict(), max_depth=2, max_items=24),
            "json_path_count": len(flatten_json(observation.to_dict())),
        }


def observation_from_model_output(
    raw_response: str,
    source: str,
    model_name: str,
    current_room_label: str | None = None,
    geo_label: str | None = None,
    geo_latitude: float | None = None,
    geo_longitude: float | None = None,
    thumbnail_path: str | None = None,
    image_hash: str | None = None,
) -> Observation:
    payload = _extract_payload(raw_response)
    now = datetime.now()
    room_payload = _dict(payload.get("room"))
    if current_room_label:
        room_payload["label"] = current_room_label
        room_payload["confidence"] = max(clamp_float(room_payload.get("confidence"), 0.0), 0.95)
        room_payload["evidence"] = room_payload.get("evidence") or "Комната указана пользователем."
    room_label = room_payload.get("label")
    room = RoomInfo(
        room_id=normalize_term(room_label) if room_label else None,
        label=str(room_label).strip() if room_label else None,
        confidence=clamp_float(room_payload.get("confidence"), 0.0),
        evidence=_optional_str(room_payload.get("evidence")),
    )
    objects = [_object_from_payload(item) for item in _list(payload.get("objects"))]
    ocr = [_ocr_from_payload(item) for item in _list(payload.get("ocr"))]
    landmarks = [_landmark_from_payload(item) for item in _list(payload.get("landmarks"))]
    geo = _geo_from_payload(payload.get("geo"), geo_label, geo_latitude, geo_longitude)
    scene = _scene_from_payload(payload.get("scene"))
    spatial_summary = str(payload.get("spatial_summary") or "")
    comment = str(payload.get("comment") or scene.summary or spatial_summary or "").strip()
    visual_tags = sorted(expand_terms(_list(payload.get("visual_tags"))))
    semantic_tags = sorted(expand_terms(_list(payload.get("semantic_tags"))))
    context_tags = sorted(expand_terms(_list(payload.get("context_tags"))))
    dynamic_tags = sorted(expand_terms(_list(payload.get("dynamic_tags"))))
    explicit_search_terms = sorted(
        set(expand_terms(_list(payload.get("search_tags"))))
        | {obj.normalized_name for obj in objects}
        | set(visual_tags)
        | set(semantic_tags)
        | set(context_tags)
    )
    negative_terms = _negative_tag_keys(payload.get("tags"))
    explicit_search_terms = [term for term in explicit_search_terms if term not in negative_terms]
    tags = sorted(set(explicit_search_terms) | set(dynamic_tags))
    tag_entries = _build_observation_tags(payload, objects, ocr, landmarks, room, geo, comment, explicit_search_terms)
    return Observation(
        observation_id=str(uuid.uuid4()),
        frame_id=str(uuid.uuid4()),
        source=source,
        timestamp=now.isoformat(timespec="seconds"),
        unix_ts=now.timestamp(),
        room=room,
        scene=scene,
        objects=objects,
        ocr=ocr,
        landmarks=landmarks,
        search_tags=tags,
        spatial_summary=spatial_summary,
        answer_hints=[str(item).strip() for item in _list(payload.get("answer_hints")) if str(item).strip()],
        quality=_quality_from_payload(payload.get("quality")),
        model=ModelInfo(name=model_name, version=None, prompt_version=VISION_MEMORY_PROMPT_VERSION),
        confidence=clamp_float(payload.get("confidence"), 0.0),
        thumbnail_path=thumbnail_path,
        image_hash=image_hash,
        raw_model_output=payload,
        geo=geo,
        visual_tags=visual_tags,
        semantic_tags=semantic_tags,
        context_tags=context_tags,
        dynamic_tags=dynamic_tags,
        dynamic_fields=_extract_dynamic_fields(payload),
        comment=comment,
        tags=tag_entries,
    )


def fallback_observation_from_text(
    raw_response: str,
    source: str,
    model_name: str,
    current_room_label: str | None = None,
    geo_label: str | None = None,
    thumbnail_path: str | None = None,
    image_hash: str | None = None,
    error: str = "",
) -> Observation:
    now = datetime.now()
    terms = sorted(expand_terms(tokenize(raw_response)[:200]))
    partial_comment = _extract_partial_json_string(raw_response, "comment") or _extract_partial_json_string(raw_response, "summary")
    room = RoomInfo(
        room_id=normalize_term(current_room_label) if current_room_label else None,
        label=current_room_label,
        confidence=0.95 if current_room_label else 0.0,
        evidence="Комната указана пользователем." if current_room_label else None,
    )
    comment = partial_comment or "Модель вернула невалидный JSON; сохранён частичный результат по тексту ответа."
    geo = _geo_from_payload({}, geo_label, None, None)
    return Observation(
        observation_id=str(uuid.uuid4()),
        frame_id=str(uuid.uuid4()),
        source=source,
        timestamp=now.isoformat(timespec="seconds"),
        unix_ts=now.timestamp(),
        room=room,
        scene=SceneInfo(
            primary="partial_model_output",
            secondary=["invalid_json"],
            indoor_outdoor="unknown",
            summary=comment,
        ),
        objects=[],
        ocr=[],
        landmarks=[],
        search_tags=terms,
        spatial_summary="Недостаточно данных: ответ модели был невалидным JSON.",
        answer_hints=[],
        quality=QualityInfo(blur="unknown", brightness="unknown", occlusion="unknown", useful=False),
        model=ModelInfo(name=model_name, version=None, prompt_version=VISION_MEMORY_PROMPT_VERSION),
        confidence=0.1,
        thumbnail_path=thumbnail_path,
        image_hash=image_hash,
        raw_model_output={"partial": True, "parse_error": error, "raw_response": raw_response[:6000]},
        geo=geo,
        visual_tags=[],
        semantic_tags=[],
        context_tags=[],
        dynamic_tags=terms,
        dynamic_fields={"partial_model_output": True, "parse_error": error},
        comment=comment,
        tags=_build_observation_tags({}, [], [], [], room, geo, comment, terms),
    )


def extract_query_intent(question: str) -> QueryIntent:
    tokens = [token for token in tokenize(question) if token not in STOP_WORDS]
    expanded = sorted(expand_terms(tokens))
    target_objects = [term for term in expanded if term in SYNONYM_TO_CANONICAL.values() or term not in STOP_WORDS]
    landmark_candidates = {
        "раковина",
        "стол",
        "стул",
        "диван",
        "кровать",
        "шкаф",
        "дверь",
        "окно",
        "холодильник",
        "плита",
    }
    related_landmarks = [term for term in target_objects if term in landmark_candidates]
    room_hint = next((term for term in tokens if term in {"кухня", "ванная", "коридор", "комната", "спальня"}), None)
    if "рядом" in normalize_text(question) or "около" in normalize_text(question):
        question_type = "what_near"
    elif "последний" in normalize_text(question) or "недавно" in normalize_text(question):
        question_type = "last_seen"
    elif "что" in normalize_text(question):
        question_type = "list_objects"
    else:
        question_type = "where_is"
    return QueryIntent(
        question=question,
        tokens=expanded,
        target_objects=target_objects,
        related_landmarks=related_landmarks,
        room_hint=room_hint,
        time_hint="recent" if question_type == "last_seen" else None,
        question_type=question_type,
    )


def build_memory_user_prompt(source: str, timestamp: str, current_room_label: str | None = None, geo_label: str | None = None) -> str:
    room_line = f"Пользователь указал текущую комнату/зону: {current_room_label}." if current_room_label else "Комната пользователем не указана."
    geo_label = (geo_label or os.environ.get("VISION_GEO_LABEL", "")).strip()
    geo_line = f"Геометка: {geo_label}." if geo_label else "Геометка не указана."
    return (
        f"Источник кадра: {source}\n"
        f"Время кадра: {timestamp}\n"
        f"{room_line}\n"
        f"{geo_line}\n"
        f"{SENSITIVE_RULES}\n"
        "Верни только JSON по схеме визуальной памяти."
    )


def compute_image_hash(frame) -> str:
    if cv2 is None or np is None:
        return str(uuid.uuid4())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean_value = float(resized.mean())
    bits = (resized >= mean_value).astype("uint8").flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hash_distance(left: str | None, right: str | None) -> float:
    if not left or not right:
        return 1.0
    try:
        left_int = int(left, 16)
        right_int = int(right, 16)
    except ValueError:
        return 1.0
    return (left_int ^ right_int).bit_count() / 64.0


def save_thumbnail(frame, directory: Path, observation_id: str) -> str | None:
    if cv2 is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    max_side = max(width, height)
    thumbnail = frame
    if max_side > 320:
        scale = 320 / float(max_side)
        thumbnail = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    path = directory / f"{observation_id}.jpg"
    if cv2.imwrite(str(path), thumbnail, [int(cv2.IMWRITE_JPEG_QUALITY), 72]):
        return str(path)
    return None


def normalize_text(value: str) -> str:
    return str(value or "").lower().replace("ё", "е").strip()


def normalize_term(value: str) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^0-9a-zа-я_\- ]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return SYNONYM_TO_CANONICAL.get(text, text)


def tokenize(value: str) -> list[str]:
    text = normalize_text(value)
    return [normalize_term(item) for item in re.findall(r"[0-9a-zа-яё_\-]+", text, flags=re.IGNORECASE) if normalize_term(item)]


def expand_terms(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        normalized = normalize_term(str(value))
        if not normalized:
            continue
        result.add(normalized)
        result.update(token for token in tokenize(normalized) if token and token not in STOP_WORDS)
    return result


def clamp_float(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return fallback
    if math.isnan(number) or math.isinf(number):
        return fallback
    return max(0.0, min(1.0, number))


def _extract_payload(raw_response: str) -> dict[str, Any]:
    try:
        return json.loads(extract_json_block(raw_response))
    except Exception:
        return json.loads(raw_response)


def _extract_partial_json_string(raw_response: str, key: str) -> str | None:
    pattern = rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.){{1,800}})'
    match = re.search(pattern, raw_response, flags=re.DOTALL)
    if not match:
        return None
    value = match.group(1)
    try:
        return json.loads(f'"{value}"').strip()
    except Exception:
        return value.replace("\\n", " ").strip()


def _object_from_payload(item: dict[str, Any]) -> VisionObject:
    name = str(item.get("name") or item.get("normalized_name") or "object").strip()
    normalized = normalize_term(str(item.get("normalized_name") or name))
    return VisionObject(
        name=name,
        normalized_name=normalized,
        category=normalize_term(str(item.get("category") or "")),
        attributes=[normalize_term(value) for value in _list(item.get("attributes")) if normalize_term(value)],
        object_tags=[normalize_term(value) for value in _list(item.get("object_tags")) if normalize_term(value)],
        state_tags=[normalize_term(value) for value in _list(item.get("state_tags")) if normalize_term(value)],
        evidence_tags=[normalize_term(value) for value in _list(item.get("evidence_tags")) if normalize_term(value)],
        location_hint=_optional_str(item.get("location_hint")),
        near=[normalize_term(value) for value in _list(item.get("near")) if normalize_term(value)],
        confidence=clamp_float(item.get("confidence"), 0.0),
    )


def _geo_from_payload(value: Any, label: str | None, latitude: float | None, longitude: float | None) -> GeoInfo:
    payload = _dict(value)
    env_label = os.environ.get("VISION_GEO_LABEL", "").strip()
    selected_label = label or payload.get("label") or env_label or None
    selected_latitude = latitude if latitude is not None else _optional_float(payload.get("latitude"), os.environ.get("VISION_GEO_LAT"))
    selected_longitude = longitude if longitude is not None else _optional_float(payload.get("longitude"), os.environ.get("VISION_GEO_LON"))
    source = "manual" if label else ("model" if payload else ("env" if env_label else "unknown"))
    confidence = 0.95 if label else clamp_float(payload.get("confidence"), 0.0)
    return GeoInfo(
        label=str(selected_label).strip() if selected_label else None,
        latitude=selected_latitude,
        longitude=selected_longitude,
        source=source,
        confidence=confidence,
    )


def _ocr_from_payload(item: dict[str, Any]) -> OcrEntry:
    text = str(item.get("text") or "").strip()
    return OcrEntry(
        text=text,
        normalized_text=normalize_term(str(item.get("normalized_text") or text)),
        type=normalize_term(str(item.get("type") or "")),
        confidence=clamp_float(item.get("confidence"), 0.0),
    )


def _landmark_from_payload(item: dict[str, Any]) -> Landmark:
    name = str(item.get("name") or item.get("normalized_name") or "").strip()
    return Landmark(
        name=name,
        normalized_name=normalize_term(str(item.get("normalized_name") or name)),
        description=str(item.get("description") or "").strip(),
        confidence=clamp_float(item.get("confidence"), 0.0),
    )


def _tags_from_payload(value: Any) -> dict[str, TagEntry]:
    tags: dict[str, TagEntry] = {}
    if isinstance(value, dict):
        for key, raw_entry in value.items():
            entry_payload = _dict(raw_entry)
            if not entry_payload:
                entry_payload = {"value": raw_entry}
            tag_key = _tag_key(str(entry_payload.get("ru") or key))
            if not tag_key or not _is_user_tag(tag_key):
                continue
            confidence = clamp_float(entry_payload.get("confidence"), 0.0)
            entry_value = entry_payload.get("value", True)
            where = _optional_str(entry_payload.get("where"))
            if entry_value is False:
                continue
            if entry_payload.get("confidence") is not None and confidence <= 0.05:
                continue
            if where and "не видно" in normalize_text(where):
                continue
            tags[tag_key] = TagEntry(
                ru=tag_key,
                en=str(entry_payload.get("en") or _english_alias(tag_key) or "").strip(),
                value=entry_value,
                confidence=confidence,
                aliases=_unique_terms(_list(entry_payload.get("aliases"))),
                category=normalize_term(str(entry_payload.get("category") or "")),
                where=where,
                near=_unique_terms(_list(entry_payload.get("near"))),
                source_paths=[str(item).strip() for item in _list(entry_payload.get("source_paths")) if str(item).strip()],
            )
    elif isinstance(value, list):
        for item in value:
            tag_key = _tag_key(str(item))
            if tag_key and _is_user_tag(tag_key):
                tags[tag_key] = TagEntry(ru=tag_key, en=_english_alias(tag_key), confidence=0.0)
    return tags


def _negative_tag_keys(value: Any) -> set[str]:
    result: set[str] = set()
    if not isinstance(value, dict):
        return result
    for key, raw_entry in value.items():
        entry_payload = _dict(raw_entry)
        if not entry_payload:
            continue
        tag_key = _tag_key(str(entry_payload.get("ru") or key))
        confidence = clamp_float(entry_payload.get("confidence"), 0.0)
        where = _optional_str(entry_payload.get("where"))
        if entry_payload.get("value") is False or (entry_payload.get("confidence") is not None and confidence <= 0.05) or (where and "не видно" in normalize_text(where)):
            result.update(expand_terms([tag_key]))
    return result


def _build_observation_tags(
    payload: dict[str, Any],
    objects: list[VisionObject],
    ocr: list[OcrEntry],
    landmarks: list[Landmark],
    room: RoomInfo,
    geo: GeoInfo,
    comment: str,
    search_terms: list[str],
) -> dict[str, TagEntry]:
    tags = _tags_from_payload(payload.get("tags"))
    for index, obj in enumerate(objects):
        values = [obj.normalized_name, obj.name, obj.category] + obj.attributes + obj.object_tags + obj.state_tags + obj.evidence_tags
        _merge_tag(
            tags,
            _tag_key(obj.normalized_name or obj.name),
            TagEntry(
                ru=_tag_key(obj.normalized_name or obj.name),
                en=_english_alias(obj.normalized_name or obj.name),
                value=True,
                confidence=obj.confidence,
                aliases=sorted(expand_terms(values)),
                category=obj.category,
                where=obj.location_hint,
                near=obj.near,
                source_paths=[f"objects[{index}]"],
            ),
        )
        for value in [obj.category] + obj.attributes:
            key = _tag_key(value)
            _merge_tag(
                tags,
                key,
                TagEntry(ru=key, en=_english_alias(value), value=True, confidence=obj.confidence, category="attribute", source_paths=[f"objects[{index}]"]),
            )
        for value in obj.near:
            key = _tag_key(value)
            _merge_tag(
                tags,
                key,
                TagEntry(ru=key, en=_english_alias(value), value=True, confidence=obj.confidence, category="near", source_paths=[f"objects[{index}].near"]),
            )
        for value in obj.object_tags + obj.state_tags + obj.evidence_tags:
            key = _tag_key(value)
            _merge_tag(
                tags,
                key,
                TagEntry(ru=key, en=_english_alias(value), value=True, confidence=obj.confidence, category="model_tag", source_paths=[f"objects[{index}]"]),
            )
    for index, item in enumerate(landmarks):
        key = _tag_key(item.normalized_name or item.name)
        _merge_tag(
            tags,
            key,
            TagEntry(
                ru=key,
                en=_english_alias(item.normalized_name or item.name),
                value=True,
                confidence=item.confidence,
                aliases=sorted(expand_terms([item.name, item.normalized_name, item.description])),
                category="landmark",
                source_paths=[f"landmarks[{index}]"],
            ),
        )
    for index, item in enumerate(ocr):
        text = item.normalized_text or item.text
        key = _tag_key(text)
        if not re.search(r"[а-я]", key, flags=re.IGNORECASE):
            key = normalize_term(f"текст {key}")
        _merge_tag(
            tags,
            key,
            TagEntry(
                ru=key,
                en=item.text,
                value=item.text,
                confidence=item.confidence,
                aliases=sorted(expand_terms([item.text, item.normalized_text])),
                category=item.type or "ocr",
                source_paths=[f"ocr[{index}]"],
            ),
        )
    if room.label:
        key = _tag_key(room.label)
        _merge_tag(tags, key, TagEntry(ru=key, en=_english_alias(room.label), value=True, confidence=room.confidence, category="room", source_paths=["room"]))
    if geo.label:
        key = _tag_key(geo.label)
        _merge_tag(tags, key, TagEntry(ru=key, en=_english_alias(geo.label), value=True, confidence=geo.confidence, category="geo", source_paths=["geo"]))
    for term in search_terms:
        if not _is_user_tag(term):
            continue
        key = _tag_key(term)
        _merge_tag(tags, key, TagEntry(ru=key, en=_english_alias(term), value=True, confidence=0.4, category="tag", source_paths=["search_tags"]))
    return dict(sorted(tags.items()))


def _merge_tag(tags: dict[str, TagEntry], key: str, incoming: TagEntry) -> None:
    if not key:
        return
    existing = tags.get(key)
    incoming.ru = key
    if existing is None:
        tags[key] = incoming
        return
    existing.en = existing.en or incoming.en
    existing.value = existing.value if existing.value not in (None, "") else incoming.value
    existing.confidence = max(existing.confidence, incoming.confidence)
    existing.aliases = sorted(set(existing.aliases) | set(incoming.aliases))
    existing.category = existing.category or incoming.category
    existing.where = existing.where or incoming.where
    existing.near = sorted(set(existing.near) | set(incoming.near))
    existing.source_paths = sorted(set(existing.source_paths) | set(incoming.source_paths))


def _tag_key(value: str) -> str:
    normalized = normalize_term(value)
    if normalized == "person_visible":
        return "человек"
    if not normalized:
        return ""
    if re.search(r"[а-я]", normalized, flags=re.IGNORECASE):
        return normalized
    return normalized.replace(" ", "_")


def _english_alias(value: str) -> str:
    normalized = normalize_term(value)
    for canonical, variants in SYNONYM_GROUPS.items():
        canonical_key = "человек" if canonical == "person_visible" else canonical
        if normalized in {canonical, canonical_key} | {normalize_term(item) for item in variants}:
            for item in variants:
                if re.search(r"[a-z]", item, flags=re.IGNORECASE):
                    return item
    return value if re.search(r"[a-z]", value, flags=re.IGNORECASE) else ""


def _unique_terms(values: list[Any]) -> list[str]:
    result: set[str] = set()
    for item in values:
        raw = str(item or "").strip()
        if raw and _is_user_tag(raw):
            result.add(raw)
        result.update(term for term in expand_terms([raw]) if _is_user_tag(term))
    return sorted(result)


def _is_user_tag(term: str) -> bool:
    value = normalize_term(term)
    if not value or value in STOP_WORDS:
        return False
    if not re.search(r"[a-zа-я]", value, flags=re.IGNORECASE):
        return False
    if len(value.split()) > 3:
        return False
    if value.startswith("не "):
        return False
    if re.match(r"^(on|in|at|near|with|beside|next to) ", value):
        return False
    if re.search(r"(^|[ _])(\d+)([ _]|$)", value):
        return False
    if re.search(r"(^|[ _])(objects|ocr|landmarks|scene|quality|room|geo|model|tags|search_tags|raw_model_output)([ _]|$)", value):
        return False
    if re.search(r"(^|[ _])(attributes|category|confidence|normalized_name|location_hint|indoor_outdoor|source_paths)([ _]|$)", value):
        return False
    blocked = {
        "comment",
        "room",
        "scene",
        "objects",
        "object",
        "ocr",
        "landmarks",
        "search_tags",
        "visual_tags",
        "semantic_tags",
        "context_tags",
        "dynamic_tags",
        "dynamic_fields",
        "spatial_summary",
        "answer_hints",
        "quality",
        "confidence",
        "attributes",
        "category",
        "primary",
        "secondary",
        "summary",
        "description",
        "indoor",
        "outdoor",
        "mixed",
        "unknown",
        "indoor_outdoor",
        "normal",
        "low",
        "medium",
        "high",
        "useful",
        "visible",
        "name",
        "near",
        "where",
        "value",
        "blur",
        "brightness",
        "occlusion",
        "location_hint",
        "normalized_name",
        "source_paths",
        "source",
        "timestamp",
        "en",
        "ru",
        "aliases",
        "alias",
        "english",
        "русский_тег",
        "тег",
        "факты",
        "true",
        "false",
        "none",
        "null",
    }
    return value not in blocked and len(value) > 1


def _scene_from_payload(value: Any) -> SceneInfo:
    payload = _dict(value)
    return SceneInfo(
        primary=str(payload.get("primary") or ""),
        secondary=[str(item).strip() for item in _list(payload.get("secondary")) if str(item).strip()],
        indoor_outdoor=str(payload.get("indoor_outdoor") or "unknown"),
        summary=str(payload.get("summary") or ""),
    )


def _quality_from_payload(value: Any) -> QualityInfo:
    payload = _dict(value)
    return QualityInfo(
        blur=str(payload.get("blur") or "unknown"),
        brightness=str(payload.get("brightness") or "unknown"),
        occlusion=str(payload.get("occlusion") or "unknown"),
        useful=bool(payload.get("useful", True)),
    )


def recursive_json_terms(value: Any) -> set[str]:
    terms: set[str] = set()
    for item in flatten_json(value):
        if str(item.get("path") or "").startswith("raw_model_output"):
            continue
        if isinstance(item["value"], (str, int, float, bool)):
            terms.update(expand_terms([str(item["value"])]))
        terms.update(expand_terms([item["path"]]))
    return {term for term in terms if term and term not in STOP_WORDS}


def flatten_json(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_json(child, path))
        return rows
    if isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            rows.extend(flatten_json(child, path))
        return rows
    rows.append({"path": prefix, "value": value})
    return rows


def recursive_json_preview(value: Any, max_depth: int = 2, max_items: int = 30) -> dict[str, Any]:
    rows = flatten_json(value)[:max_items]
    preview = {}
    for row in rows:
        path = row["path"]
        if path.count(".") + path.count("[") > max_depth:
            continue
        preview[path] = row["value"]
    return preview


def _extract_dynamic_fields(payload: dict[str, Any]) -> dict[str, Any]:
    known = {
        "comment",
        "tags",
        "room",
        "scene",
        "objects",
        "ocr",
        "landmarks",
        "search_tags",
        "visual_tags",
        "semantic_tags",
        "context_tags",
        "dynamic_tags",
        "spatial_summary",
        "answer_hints",
        "quality",
        "confidence",
        "geo",
    }
    dynamic = payload.get("dynamic_fields")
    if isinstance(dynamic, dict):
        merged = dict(dynamic)
    else:
        merged = {}
    for key, value in payload.items():
        if key not in known:
            merged[key] = value
    return merged


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_float(*values: Any) -> float | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _env_int(name: str, fallback: int) -> int:
    try:
        return int(os.environ.get(name, fallback))
    except Exception:
        return fallback


def _env_float(name: str, fallback: float) -> float:
    try:
        return float(os.environ.get(name, fallback))
    except Exception:
        return fallback
