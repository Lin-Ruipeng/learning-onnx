from __future__ import annotations

from pathlib import Path

import onnx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from verify_environment import (
    INPUT_NAME,
    MAX_BATCH,
    MIN_BATCH,
    OUTPUT_NAME,
    compare_runtime,
    environment_summary,
    export_model,
    inspect_artifact,
)


@pytest.fixture(scope="module")
def exported_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("ch00") / "dynamic.onnx"
    return export_model(output)


def test_environment_reports_cpu_provider() -> None:
    summary = environment_summary()
    assert summary["python"].startswith("3.12")
    assert "CPUExecutionProvider" in summary["available_providers"]


def test_exported_artifact_contract(exported_model: Path) -> None:
    opset, dynamic_batch_name = inspect_artifact(exported_model)
    graph = onnx.load(exported_model).graph

    assert opset == 18
    assert dynamic_batch_name == "batch"
    assert [value.name for value in graph.input] == [INPUT_NAME]
    assert [value.name for value in graph.output] == [OUTPUT_NAME]


@settings(max_examples=8, deadline=None)
@given(batch=st.integers(min_value=MIN_BATCH, max_value=MAX_BATCH))
def test_dynamic_batch_matches_golden_reference(exported_model: Path, batch: int) -> None:
    assert compare_runtime(exported_model, batch) <= 1e-6
