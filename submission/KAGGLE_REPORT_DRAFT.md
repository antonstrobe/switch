# Switch Vision Monitor: Private Edge Vision Memory with Gemma 4

## Subtitle

A local multimodal assistant that helps people monitor safety-critical spaces without sending camera frames to the cloud.

## Draft Report

Switch Vision Monitor addresses a practical problem: many homes, classrooms, workshops and small clinics need visual assistance, but cannot rely on stable internet or cloud processing. The project uses Gemma 4 to run multimodal scene understanding locally, turning camera or screen frames into structured observations, searchable memory and concise Russian-language explanations.

The application is built as a Windows desktop prototype. A user can select a camera, a screen region, two regions or the full screen. The capture engine sends frames to the `official-gemma` runtime, which is restricted to official Google model ids and defaults to `google/gemma-4-E2B-it`. Before model startup, the project downloads the official hackathon competition files through:

```python
import kagglehub

path = kagglehub.competition_download("gemma-4-good-hackathon")
```

This keeps the submission aligned with the Gemma 4 Good Hackathon environment and avoids community model conversions as a dependency.

The architecture has four main layers. The desktop UI manages source selection, prompt configuration, capture status and visual memory search. The engine performs frame capture, throttling, image encoding and analysis scheduling. The runtime layer loads official Gemma 4 through Hugging Face Transformers using the `image-text-to-text` pipeline. The memory layer stores structured observations in Redis, with a local fallback when Redis is unavailable.

Gemma 4 is used for multimodal understanding: it receives the frame and a structured instruction asking for objects, people, gestures, text, scene context, uncertainty and possible risks. The output is parsed into JSON-like observations. Those observations are indexed by tags, object names, landmarks and natural-language summaries, allowing later search such as “where was the cup?” or “what changed near the desk?”.

The project is designed for the Safety & Trust / Impact direction. It does not perform face recognition, does not infer sensitive attributes and does not store full-size evidence frames by default. It stores only thumbnails and structured observations, making it better suited for privacy-sensitive edge settings.

The key engineering challenge was making a prototype that is useful without becoming a fake demo. The app includes unit tests for runtime selection, parsing, storage, desktop entrypoints and visual memory behavior. It also includes explicit Git hygiene: local model files, runtime caches, output data and personal config are excluded from the repository.

The demo story is simple: a user points the camera at a room or workspace, starts monitoring, and the app creates a timeline of grounded observations. When the user later asks where an object was seen or what risk appeared, Switch Vision Monitor searches its visual memory and shows evidence with timestamps and thumbnails. This demonstrates practical local intelligence, function-like memory behavior and multimodal understanding with Gemma 4.

Future work would package the official model more efficiently for lower-resource devices, add LiteRT or mobile deployment, and refine the observation schema for education, elder care and workshop safety scenarios.
