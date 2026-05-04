import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from switch_monitor.runtimes import (
    LMStudioRuntime,
    OFFICIAL_GEMMA_MODEL_ID,
    OLLAMA_MAX_GPU_LAYERS,
    OfficialGemmaRuntime,
    OllamaRuntime,
    RuntimeErrorBase,
    RuntimeRegistry,
    format_seconds,
    find_nvidia_vulkan_icd,
    get_nvidia_gpu_status,
    normalize_gpu_mode,
)


class RuntimeGpuTests(unittest.TestCase):
    def test_normalize_gpu_mode(self) -> None:
        self.assertEqual(normalize_gpu_mode("cpu"), "off")
        self.assertEqual(normalize_gpu_mode("gpu"), "max")
        self.assertEqual(normalize_gpu_mode("auto"), "auto")
        self.assertEqual(normalize_gpu_mode("0.5"), "0.5")

    def test_format_seconds(self) -> None:
        self.assertEqual(format_seconds(1.234), "1.23 сек")

    def test_nvidia_gpu_status_formats_vram(self) -> None:
        with patch("switch_monitor.runtimes.subprocess.run") as run:
            run.return_value.stdout = "GeForce GTX 1650, 4096, 131\n"
            status = get_nvidia_gpu_status()

        self.assertEqual(status, "GPU NVIDIA: GeForce GTX 1650, VRAM 131/4096 MB")

    def test_nvidia_vulkan_icd_prefers_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            icd = Path(temp_dir) / "nv-vk64.json"
            icd.write_text("{}", encoding="utf-8")
            with patch.dict("switch_monitor.runtimes.os.environ", {"SWITCH_VK_ICD_FILENAMES": str(icd)}):
                self.assertEqual(find_nvidia_vulkan_icd(), str(icd))

    def test_ollama_gpu_options(self) -> None:
        runtime = OllamaRuntime()
        runtime.prepare("gemma", gpu_mode="max")
        self.assertEqual(runtime.num_gpu, OLLAMA_MAX_GPU_LAYERS)

        runtime.prepare("gemma", gpu_mode="auto")
        self.assertIsNone(runtime.num_gpu)

        runtime.prepare("gemma", gpu_mode="off")
        self.assertEqual(runtime.num_gpu, 0)

    def test_lmstudio_load_uses_gpu_flag(self) -> None:
        runtime = LMStudioRuntime()
        with patch("switch_monitor.runtimes.subprocess.run") as run:
            runtime._load_model("gemma", "max")

        command = run.call_args.args[0]
        self.assertEqual(command[:5], ["lms", "load", "gemma", "--gpu", "max"])

    def test_lmstudio_prefers_cuda_runtime(self) -> None:
        runtime = LMStudioRuntime()
        lines = [
            "runtime-win-x86_64-avx2@2.13.0",
            "runtime-win-x86_64-vulkan-avx2@2.10.0",
            "runtime-win-x86_64-nvidia-cuda-avx2@2.10.0",
        ]
        self.assertEqual(
            runtime._find_gpu_runtime_alias(lines),
            "runtime-win-x86_64-nvidia-cuda-avx2@2.10.0",
        )

    def test_lmstudio_prepare_selects_gpu_runtime(self) -> None:
        runtime = LMStudioRuntime()
        with (
            patch.object(runtime, "_ensure_api"),
            patch.object(runtime, "_select_gpu_runtime") as select_gpu_runtime,
            patch.object(runtime, "_unload_identifier"),
            patch.object(runtime, "_load_model"),
        ):
            runtime.prepare("gemma", gpu_mode="max")

        select_gpu_runtime.assert_called_once()

    def test_registry_includes_official_gemma(self) -> None:
        registry = RuntimeRegistry()
        self.assertIsInstance(registry.get("official-gemma"), OfficialGemmaRuntime)

    def test_official_gemma_detects_google_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = OfficialGemmaRuntime(Path(temp_dir))
            models = runtime.detect_models()

        self.assertEqual(models[0].runtime, "official-gemma")
        self.assertEqual(models[0].model_id, OFFICIAL_GEMMA_MODEL_ID)

    def test_official_gemma_rejects_non_google_model(self) -> None:
        runtime = OfficialGemmaRuntime()
        runtime.model_id = "community/gemma"

        with self.assertRaises(RuntimeErrorBase):
            runtime.prepare("community/gemma")

    def test_official_gemma_extracts_pipeline_message(self) -> None:
        runtime = OfficialGemmaRuntime()
        result = [
            {
                "generated_text": [
                    {"role": "user", "content": "prompt"},
                    {"role": "assistant", "content": [{"type": "text", "text": "{\"ok\": true}"}]},
                ]
            }
        ]

        self.assertEqual(runtime._extract_generated_text(result), "{\"ok\": true}")

    def test_official_gemma_analyze_uses_pipeline(self) -> None:
        image = Image.new("RGB", (8, 8), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        runtime = OfficialGemmaRuntime()
        runtime.pipeline = lambda *args, **kwargs: [{"generated_text": "{\"answer\": \"ok\"}"}]

        self.assertEqual(
            runtime.analyze(OFFICIAL_GEMMA_MODEL_ID, "system", "user", buffer.getvalue()),
            "{\"answer\": \"ok\"}",
        )


if __name__ == "__main__":
    unittest.main()
