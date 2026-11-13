from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from tools.check_course_state import README_PATH, load_course_state, validate_course_state


def test_repository_course_state_is_valid() -> None:
    assert validate_course_state(load_course_state()) == []


def test_checker_rejects_multiple_actionable_chapters() -> None:
    state = deepcopy(load_course_state())
    state["chapters"][1]["material"] = "verified"
    state["chapters"][1]["learning"] = "ready"
    state["chapters"][1]["last_verified"] = "2026-11-13"

    errors = validate_course_state(state)

    assert any("at most one chapter can be actionable" in error for error in errors)


def test_checker_requires_three_kinds_of_completion_evidence() -> None:
    state = deepcopy(load_course_state())
    current = state["chapters"][0]
    current["learning"] = "completed"
    current["started_at"] = "2026-11-13T20:00:00+08:00"
    current["completed_at"] = "2026-11-13"

    errors = validate_course_state(state)

    assert any("automatic evidence" in error for error in errors)
    assert any("practice evidence" in error for error in errors)
    assert any("understanding evidence" in error for error in errors)


def test_checker_rejects_readme_material_drift(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    text = README_PATH.read_text(encoding="utf-8")
    readme.write_text(text.replace("| `verified` |", "| `planned` |", 1), encoding="utf-8")

    errors = validate_course_state(load_course_state(), readme_path=readme)

    assert "README material-status index must match COURSE_STATE.toml" in errors


def test_checker_accepts_transition_to_next_planned_chapter() -> None:
    state = deepcopy(load_course_state())
    completed = state["chapters"][0]
    completed["learning"] = "completed"
    completed["started_at"] = "2026-11-13T20:00:00+08:00"
    completed["completed_at"] = "2026-11-13"
    completed["evidence"] = {
        "automatic": ["chapter tests passed"],
        "practice": ["learner attempted every exercise"],
        "understanding": ["completion review passed"],
    }
    state["current"] = "core-01"
    state["chapters"][1]["resume_note"] = "Prepare and verify the immediate next chapter."

    assert validate_course_state(state) == []
