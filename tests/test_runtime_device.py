from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Experimental:
    def __init__(self) -> None:
        self.memory_growth_devices: list[object] = []

    def set_memory_growth(self, device: object, enabled: bool) -> None:
        if enabled:
            self.memory_growth_devices.append(device)

    def get_device_details(self, device: object) -> dict[str, object]:
        return {"compute_capability": (12, 0)}


class _Config:
    def __init__(self, gpu_count: int) -> None:
        self.gpus = [SimpleNamespace(name=f"/physical_device:GPU:{index}") for index in range(gpu_count)]
        self.experimental = _Experimental()
        self.visible: tuple[list[object], str] | None = None

    def list_physical_devices(self, kind: str) -> list[object]:
        return list(self.gpus) if kind == "GPU" else []

    def set_visible_devices(self, devices: list[object], kind: str) -> None:
        self.visible = (list(devices), kind)


class _Policy:
    def __init__(self) -> None:
        self.name = "float32"

    def set_global_policy(self, name: str) -> None:
        self.name = name

    def global_policy(self):
        return SimpleNamespace(name=self.name)


def fake_tensorflow(gpu_count: int):
    policy = _Policy()
    return SimpleNamespace(
        __version__="2.21.0",
        config=_Config(gpu_count),
        keras=SimpleNamespace(
            __version__="3.12.0",
            mixed_precision=SimpleNamespace(
                set_global_policy=policy.set_global_policy,
                global_policy=policy.global_policy,
            ),
        ),
        sysconfig=SimpleNamespace(get_build_info=lambda: {"cuda_version": "12.5"}),
    )


class RuntimeDeviceTests(unittest.TestCase):
    def test_require_gpu_fails_before_training_when_gpu_is_missing(self) -> None:
        from cryinsight.runtime.device import RuntimeDeviceError, configure_tensorflow_runtime

        with self.assertRaisesRegex(RuntimeDeviceError, "GPU is required"):
            configure_tensorflow_runtime(
                fake_tensorflow(0), device="gpu", require_gpu=True
            )

    def test_cpu_hides_gpu_and_keeps_float32(self) -> None:
        from cryinsight.runtime.device import configure_tensorflow_runtime

        tf = fake_tensorflow(1)
        result = configure_tensorflow_runtime(tf, device="cpu")

        self.assertEqual(result["selected_device"], "cpu")
        self.assertEqual(result["precision_policy"], "float32")
        self.assertEqual(tf.config.visible, ([], "GPU"))

    def test_auto_gpu_enables_memory_growth_and_mixed_precision(self) -> None:
        from cryinsight.runtime.device import configure_tensorflow_runtime

        tf = fake_tensorflow(1)
        result = configure_tensorflow_runtime(
            tf, device="auto", mixed_precision=True
        )

        self.assertEqual(result["selected_device"], "gpu")
        self.assertEqual(result["physical_gpu_count"], 1)
        self.assertEqual(result["precision_policy"], "mixed_float16")
        self.assertEqual(result["gpus"][0]["compute_capability"], "12.0")
        self.assertEqual(len(tf.config.experimental.memory_growth_devices), 1)

    def test_both_trainers_expose_the_same_runtime_flags(self) -> None:
        for relative in (
            Path("Models_dbl/binary/train_binary_dbl.py"),
            Path("Models_dbl/Main/train_main_dbl.py"),
        ):
            path = PROJECT_ROOT / relative
            spec = importlib.util.spec_from_file_location(path.stem, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            args = module.build_parser().parse_args(
                ["--train", "--device", "gpu", "--require-gpu", "--mixed-precision"]
            )
            self.assertEqual(args.device, "gpu")
            self.assertTrue(args.require_gpu)
            self.assertTrue(args.mixed_precision)


if __name__ == "__main__":
    unittest.main()
