"""Chapter 00: verify the minimal modern ONNX deployment loop."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import onnx
import onnxruntime as ort
import torch

INPUT_NAME: Final = "features"
OUTPUT_NAME: Final = "transformed"
OPSET_VERSION: Final = 18
MIN_BATCH: Final = 1
MAX_BATCH: Final = 8


class DemoModel(torch.nn.Module):
    """A tiny parameter-free model that remains useful across dynamic batches."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sin(features) + features * 0.5


@dataclass(frozen=True)
class VerificationResult:
    artifact: str
    opset: int
    dynamic_batch_name: str
    checked_batches: tuple[int, ...]
    max_abs_error: float
    providers: tuple[str, ...]


def environment_summary() -> dict[str, object]:
    """Return the version and provider facts needed to reproduce this run."""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
    }


def sample_input(batch: int) -> np.ndarray:
    """Build deterministic values with enough range to expose obvious errors."""

    if not MIN_BATCH <= batch <= MAX_BATCH:
        raise ValueError(f"batch must be in [{MIN_BATCH}, {MAX_BATCH}], got {batch}")
    values = np.linspace(-2.0, 2.0, num=batch * 3, dtype=np.float32)
    return values.reshape(batch, 3)


def export_model(output_path: Path) -> Path:
    """Export the demo through the torch.export-based ONNX exporter."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = DemoModel().eval()
    example = torch.from_numpy(sample_input(batch=1))
    batch = torch.export.Dim("batch", min=MIN_BATCH, max=MAX_BATCH)

    torch.onnx.export(
        model,
        (example,),
        f=output_path,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=OPSET_VERSION,
        dynamo=True,
        dynamic_shapes={"features": {0: batch}},
        external_data=False,
        verify=True,
    )
    return output_path


def inspect_artifact(model_path: Path) -> tuple[int, str]:
    """Check the serialized graph and return its opset and batch symbol."""

    model = onnx.load(model_path, load_external_data=True)
    onnx.checker.check_model(model, full_check=True)

    default_domain = next(item for item in model.opset_import if item.domain in {"", "ai.onnx"})
    if default_domain.version != OPSET_VERSION:
        raise AssertionError(
            f"expected opset {OPSET_VERSION}, artifact uses {default_domain.version}"
        )

    graph_input = next(value for value in model.graph.input if value.name == INPUT_NAME)
    first_dim = graph_input.type.tensor_type.shape.dim[0]
    if not first_dim.dim_param:
        raise AssertionError("the first input dimension is not symbolic")
    return default_domain.version, first_dim.dim_param


def compare_runtime(model_path: Path, batch: int) -> float:
    """Compare ONNX Runtime CPU output with the PyTorch golden reference."""

    values = sample_input(batch)
    with torch.inference_mode():
        expected = DemoModel().eval()(torch.from_numpy(values)).numpy()

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    (actual,) = session.run([OUTPUT_NAME], {INPUT_NAME: values})
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    return float(np.max(np.abs(actual - expected)))


def verify(output_path: Path, batches: tuple[int, ...] = (1, 4, 8)) -> VerificationResult:
    """Export, inspect, execute, and numerically compare one artifact."""

    model_path = export_model(output_path)
    opset, dynamic_batch_name = inspect_artifact(model_path)
    max_abs_error = max(compare_runtime(model_path, batch) for batch in batches)
    return VerificationResult(
        artifact=str(model_path),
        opset=opset,
        dynamic_batch_name=dynamic_batch_name,
        checked_batches=batches,
        max_abs_error=max_abs_error,
        providers=("CPUExecutionProvider",),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(__file__).parents[1] / "artifacts/generated/ch00_dynamic.onnx"
    parser.add_argument("--output", type=Path, default=default_output)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "environment": environment_summary(),
        "verification": asdict(verify(args.output)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
