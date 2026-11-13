# Course workflow for agents

Use this workflow when the user intends to continue the course, start or resume a lesson, submit an exercise, request a chapter review, or prepare the next lesson.

## Sources of truth

- `COURSE_STATE.toml` owns material and learning status.
- `README.md` is the human-facing material-status index and must agree with the state file.
- Each chapter's completion checklist defines the concepts used during understanding review.
- `solutions/` becomes available after the learner records an attempt or explicitly asks for a direct answer.

Timestamps use ISO 8601. Use `YYYY-MM-DD` for verification and completion dates; use an offset timestamp when resume precision matters.

## Enter the course

1. Read `COURSE_STATE.toml` and run `uv run python tools/check_course_state.py` from `modern-onnx-course/`. Completion criterion: the state is valid and the chapter named by `current` is known.
2. Read the current chapter README and `resume_note`. Report the current material status, learning status, existing evidence, review gaps, and the single next action. Completion criterion: the learner can see exactly where this session resumes.
3. Follow the matching branch below. Expose only the current branch and the immediately useful next action.

## Teach a verified chapter

### `ready`

Introduce the chapter's purpose, prerequisites, artifact contract, and first experiment. When the learner starts, set `learning = "in_progress"`, record `started_at`, and replace `resume_note` with the next concrete action. Validate the state.

Completion criterion: the learner has begun the chapter and the repository records where to resume.

### `in_progress`

Let the learner own `exercises/`. Inspect their work, ask diagnostic questions, and offer progressive help in this order:

1. conceptual hint;
2. location or invariant to inspect;
3. near-solution guidance;
4. targeted reference answer after an attempt or explicit request.

When the learner has attempted the exercises and the chapter's automatic checks pass, record automatic and practice evidence, set `learning = "review"`, and write the review starting point to `resume_note`.

Completion criterion: practice and automatic evidence are recorded and the chapter is ready for understanding review.

### `review`

Use the chapter checklist to ask 3–5 open questions plus at least one concrete debugging or boundary-condition scenario. Evaluate explanations and reasoning rather than API memorization.

- When gaps remain, keep `learning = "review"`; record the concepts in `review_gaps` and the next targeted experiment or explanation in `resume_note`.
- When the learner passes, record understanding evidence, clear `review_gaps`, set `learning = "completed"`, and record `completed_at`. Unlock only the immediate successor by setting it to `ready` if its material is already `verified`; otherwise leave it `locked` while authoring its material. Update `current` to the successor, write its concrete preparation or learning step to `resume_note`, and validate the state. During this preparation window there is no actionable learning chapter; the locked successor is current until its material becomes verified.

Completion criterion: automatic, practice, and understanding evidence are non-empty; the completion date is recorded; the immediate successor is current and is the only chapter eligible for preparation or learning.

Commit the learning transition as `course: complete <id>`. Keep the commit local.

## Author the immediate next chapter

Run this branch only after its prerequisite chapter is `completed` and its own material is `planned`.

1. Research current primary official sources and record them in the lesson. Separate stable concepts from version-specific behavior.
2. Create only the current chapter with `README.md`, `exercises/`, `solutions/`, `src/`, `tests/`, `fixtures/`, and `artifacts/README.md`.
3. Provide executable experiments, learner-owned exercises, targeted reference solutions, and completion questions. Keep generated artifacts ignored.
4. Set `material = "authored"` while work is incomplete. Run the locked environment, lint, formatting, tests, and the chapter experiment.
5. When every gate passes, set `material = "verified"`, record `last_verified`, set `learning = "ready"`, update the README material-status index, and validate course state.
6. Commit the lesson as `course: add <id> <topic>`. Keep the commit local. Introduce the new lesson and wait for the learner to begin.

Completion criterion: the immediate next chapter is verified, indexed, ready, committed separately from the previous completion, and no later chapter was authored.

## Verification gates

Run from `modern-onnx-course/`:

```bash
uv sync --locked
uv run python tools/check_course_state.py
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Also run the current chapter's documented experiment. A material transition to `verified` requires every gate to pass. A learning transition to `completed` additionally requires recorded practice and understanding evidence.
