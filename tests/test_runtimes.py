import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from switch_monitor.local_gemma import LOCAL_GEMMA_MODEL_ID, get_local_gemma_assets
from switch_monitor.runtimes import (
    LMStudioRuntime,
    LocalGemmaRuntime,
    OLLAMA_MAX_GPU_LAYERS,
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
            "llama.cpp-win-x86_64-avx2@2.13.0                   GGUF",
            "llama.cpp-win-x86_64-vulkan-avx2@2.10.0             GGUF",
            "llama.cpp-win-x86_64-nvidia-cuda-avx2@2.10.0        GGUF",
        ]
        self.assertEqual(
            runtime._find_gpu_runtime_alias(lines),
            "llama.cpp-win-x86_64-nvidia-cuda-avx2@2.10.0",
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

    def test_registry_includes_local_gemma(self) -> None:
        registry = RuntimeRegistry()
        self.assertIsInstance(registry.get("local-gemma"), LocalGemmaRuntime)

    def test_local_gemma_detects_downloadable_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = LocalGemmaRuntime(Path(temp_dir))
            models = runtime.detect_models()

        self.assertEqual(models[0].runtime, "local-gemma")
        self.assertEqual(models[0].model_id, LOCAL_GEMMA_MODEL_ID)

    def test_local_gemma_assets_use_project_models_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            assets = get_local_gemma_assets(Path(temp_dir))

        self.assertIn("models", assets.model_path.parts)
        self.assertFalse(assets.installed)

    def test_local_gemma_server_disables_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = LocalGemmaRuntime(Path(temp_dir))
            runtime.model_path = Path(temp_dir) / "model.gguf"
            runtime.mmproj_path = Path(temp_dir) / "mmproj.gguf"
            runtime.server_exe = Path(temp_dir) / "llama-server.exe"
            progress: list[str] = []
            runtime.set_progress_callback(progress.append)

            with (
                patch("switch_monitor.runtimes.subprocess.Popen") as popen,
                patch("switch_monitor.runtimes.get_nvidia_gpu_status", return_value="GPU"),
                patch.object(runtime, "_find_free_port", return_value=12345),
                patch.object(runtime, "_wait_for_server"),
            ):
                runtime._start_server()
                runtime.stop()

        command = popen.call_args.args[0]
        self.assertIn("--reasoning", command)
        self.assertIn("off", command)
        self.assertIn("--reasoning-budget", command)
        self.assertIn("0", command)
        self.assertTrue(any("Gemma загружена" in item for item in progress))

    def test_local_gemma_vulkan_uses_nvidia_icd_and_gpu_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = LocalGemmaRuntime(Path(temp_dir))
            runtime.model_path = Path(temp_dir) / "model.gguf"
            runtime.mmproj_path = Path(temp_dir) / "mmproj.gguf"
            runtime.server_exe = Path(temp_dir) / "llama.cpp-b8902-vulkan-x64" / "llama-server.exe"
            runtime.gpu_layers = 15

            with (
                patch("switch_monitor.runtimes.subprocess.Popen") as popen,
                patch("switch_monitor.runtimes.get_nvidia_gpu_status", return_value="GPU"),
                patch("switch_monitor.runtimes.find_nvidia_vulkan_icd", return_value=r"C:\nvidia\nv-vk64.json"),
                patch.object(runtime, "_find_free_port", return_value=12345),
                patch.object(runtime, "_wait_for_server"),
            ):
                runtime._start_server()
                runtime.stop()

        command = popen.call_args.args[0]
        environment = popen.call_args.kwargs["env"]
        self.assertIn("--n-gpu-layers", command)
        self.assertIn("15", command)
        self.assertIn("-fit", command)
        self.assertIn("off", command)
        self.assertEqual(environment["VK_ICD_FILENAMES"], r"C:\nvidia\nv-vk64.json")
        self.assertEqual(environment["GGML_VK_VISIBLE_DEVICES"], "0")

    def test_local_gemma_low_vram_uses_safer_gpu_layers(self) -> None:
        runtime = LocalGemmaRuntime()
        with patch("switch_monitor.runtimes.get_nvidia_total_memory_mib", return_value=4096):
            self.assertLessEqual(runtime._gpu_layers_for_mode("max"), 8)
            self.assertLessEqual(runtime._context_size_for_hardware(), 4096)

    def test_local_gemma_cancel_does_not_restart_server(self) -> None:
        runtime = LocalGemmaRuntime()
        runtime.server_url = "http://127.0.0.1:1"

        def fail_after_cancel(*args, **kwargs):
            runtime.stop()
            raise requests.ConnectionError("cancelled")

        with (
            patch("switch_monitor.runtimes.requests.post", side_effect=fail_after_cancel) as post,
            patch.object(runtime, "_ensure_server") as ensure_server,
        ):
            with self.assertRaises(RuntimeErrorBase):
                runtime._post_chat_completion({}, 1)

        self.assertEqual(post.call_count, 1)
        ensure_server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
