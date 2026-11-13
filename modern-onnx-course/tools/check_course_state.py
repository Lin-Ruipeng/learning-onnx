"""Validate material and learner progress for the Modern ONNX Course."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path
from typing import Any

COURSE_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = COURSE_ROOT / "COURSE_STATE.toml"
README_PATH = COURSE_ROOT / "README.md"

MATERIAL_STATES = {"planned", "authored", "verified"}
LEARNING_STATES = {"locked", "ready", "in_progress", "review", "completed"}
ACTIONABLE_STATES = {"ready", "in_progress", "review"}
EVIDENCE_KINDS = {"automatic", "practice", "understanding"}
REQUIRED_CHAPTER_FIELDS = {
    "id",
    "order",
    "path",
    "title",
    "prerequisites",
    "material",
    "learning",
    "last_verified",
    "started_at",
    "completed_at",
    "evidence",
    "review_gaps",
    "resume_note",
}
README_ROW = re.compile(
    r"^\|\s*(?:\[(?P<linked>\d{2})\]\([^)]+\)|(?P<plain>\d{2}))\s*"
    r"\|.*\|\s*`(?P<status>planned|authored|verified)`\s*\|\s*$"
)


def load_course_state(path: Path = STATE_PATH) -> dict[str, Any]:
    with path.open("rb") as state_file:
        return tomllib.load(state_file)


def readme_material_states(path: Path = README_PATH) -> dict[str, str]:
    states: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = README_ROW.match(line)
        if match is None:
            continue
        number = match.group("linked") or match.group("plain")
        states[f"core-{number}"] = match.group("status")
    return states


def validate_course_state(
    state: dict[str, Any],
    *,
    course_root: Path = COURSE_ROOT,
    readme_path: Path = README_PATH,
) -> list[str]:
    errors: list[str] = []

    if state.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    chapters = state.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        return [*errors, "chapters must be a non-empty array"]
    if not all(isinstance(chapter, dict) for chapter in chapters):
        return [*errors, "every chapters entry must be a table"]

    ids = [chapter.get("id") for chapter in chapters]
    orders = [chapter.get("order") for chapter in chapters]
    if len(set(ids)) != len(ids):
        errors.append("chapter ids must be unique")
    if len(set(orders)) != len(orders):
        errors.append("chapter orders must be unique")
    if orders != list(range(len(chapters))):
        errors.append("chapter orders must be contiguous and sorted from zero")

    chapter_by_id = {
        chapter_id: chapter
        for chapter_id, chapter in zip(ids, chapters, strict=True)
        if isinstance(chapter_id, str)
    }
    active_ids: list[str] = []
    seen_incomplete = False

    for chapter in chapters:
        chapter_id = chapter.get("id", "<missing-id>")
        label = str(chapter_id)
        missing = REQUIRED_CHAPTER_FIELDS - chapter.keys()
        if missing:
            errors.append(f"{label}: missing fields: {', '.join(sorted(missing))}")
            continue

        material = chapter["material"]
        learning = chapter["learning"]
        if material not in MATERIAL_STATES:
            errors.append(f"{label}: invalid material state {material!r}")
        if learning not in LEARNING_STATES:
            errors.append(f"{label}: invalid learning state {learning!r}")
            continue

        if learning in ACTIONABLE_STATES:
            active_ids.append(label)
        if learning == "completed":
            if seen_incomplete:
                errors.append(f"{label}: completed chapters must form a continuous prefix")
        else:
            seen_incomplete = True

        prerequisites = chapter["prerequisites"]
        if not isinstance(prerequisites, list) or not all(
            isinstance(item, str) for item in prerequisites
        ):
            errors.append(f"{label}: prerequisites must be an array of chapter ids")
        else:
            for prerequisite in prerequisites:
                prerequisite_chapter = chapter_by_id.get(prerequisite)
                if prerequisite_chapter is None:
                    errors.append(f"{label}: unknown prerequisite {prerequisite!r}")
                elif learning != "locked" and prerequisite_chapter.get("learning") != "completed":
                    errors.append(f"{label}: prerequisite {prerequisite!r} is not completed")

        if learning != "locked" and material != "verified":
            errors.append(f"{label}: actionable or completed learning requires verified material")

        chapter_path = course_root / chapter["path"]
        if material in {"authored", "verified"} and not chapter_path.is_dir():
            errors.append(f"{label}: material path does not exist: {chapter['path']}")
        if material == "verified" and not chapter["last_verified"]:
            errors.append(f"{label}: verified material requires last_verified")
        if material == "planned" and chapter["last_verified"]:
            errors.append(f"{label}: planned material cannot have last_verified")

        if learning == "locked" and (chapter["started_at"] or chapter["completed_at"]):
            errors.append(f"{label}: locked learning cannot have start or completion dates")
        if learning in {"in_progress", "review", "completed"} and not chapter["started_at"]:
            errors.append(f"{label}: {learning} learning requires started_at")
        if learning != "completed" and chapter["completed_at"]:
            errors.append(f"{label}: only completed learning can have completed_at")

        evidence = chapter["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != EVIDENCE_KINDS:
            errors.append(f"{label}: evidence must contain automatic, practice, and understanding")
        elif not all(isinstance(items, list) for items in evidence.values()):
            errors.append(f"{label}: every evidence kind must be an array")
        else:
            if learning == "review" and not evidence["practice"]:
                errors.append(f"{label}: review learning requires practice evidence")
            if learning == "completed":
                if not chapter["completed_at"]:
                    errors.append(f"{label}: completed learning requires completed_at")
                for kind in sorted(EVIDENCE_KINDS):
                    if not evidence[kind]:
                        errors.append(f"{label}: completed learning requires {kind} evidence")

        if not isinstance(chapter["review_gaps"], list):
            errors.append(f"{label}: review_gaps must be an array")

    if len(active_ids) > 1:
        errors.append(f"at most one chapter can be actionable, found {active_ids}")
    elif len(active_ids) == 1 and state.get("current") != active_ids[0]:
        errors.append(f"current must be the actionable chapter {active_ids[0]!r}")
    elif len(active_ids) == 1 and not chapter_by_id[active_ids[0]].get("resume_note"):
        errors.append(f"{active_ids[0]}: current actionable chapter requires resume_note")
    elif not active_ids:
        current = chapter_by_id.get(state.get("current"))
        if current is None:
            errors.append("current must name an existing chapter")
        elif current.get("learning") != "locked" or current.get("material") not in {
            "planned",
            "authored",
        }:
            errors.append("a non-actionable current chapter must be locked and under preparation")
        elif not current.get("resume_note"):
            errors.append(f"{current['id']}: current preparation requires resume_note")
        else:
            for prerequisite in current.get("prerequisites", []):
                prerequisite_chapter = chapter_by_id.get(prerequisite)
                if (
                    prerequisite_chapter is None
                    or prerequisite_chapter.get("learning") != "completed"
                ):
                    errors.append(
                        f"{current['id']}: current preparation requires completed prerequisites"
                    )

    readme_states = readme_material_states(readme_path)
    state_material = {
        chapter["id"]: chapter["material"]
        for chapter in chapters
        if isinstance(chapter.get("id"), str)
    }
    if readme_states != state_material:
        errors.append("README material-status index must match COURSE_STATE.toml")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = load_course_state(args.state)
    errors = validate_course_state(state)
    if errors:
        print("Course state is invalid:")
        for error in errors:
            print(f"- {error}")
        return 1
    current_id = state["current"]
    current = next(chapter for chapter in state["chapters"] if chapter["id"] == current_id)
    print(
        f"Course state is valid: current={current_id}, "
        f"material={current['material']}, learning={current['learning']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
