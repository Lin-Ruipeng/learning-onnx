# Repository agent guidance

## Route by intent

- Treat `archive/mmdeploy-onnx-course/` as a frozen Historical Course Archive. Preserve its code and dependencies; limit changes there to archive documentation, path repairs, security fixes, or migration breakage requested by the user.
- For requests to continue the course, start a lesson, review exercises, assess completion, or prepare the next lesson, read `modern-onnx-course/docs/agents/course-workflow.md` and `modern-onnx-course/COURSE_STATE.toml` before responding. Follow that workflow until its current completion criterion is met.
- For other repository work, handle the request normally and leave learning state unchanged.

## Course invariants

- Treat `modern-onnx-course/COURSE_STATE.toml` as the single source of truth for material and learning progress.
- Run course commands from `modern-onnx-course/` through `uv`; keep generated model artifacts out of Git.
- Validate every state edit with `uv run python tools/check_course_state.py` and the relevant course checks.
- Keep course commits local unless the user explicitly requests a push. Separate learning-completion commits (`course: complete <id>`) from lesson-authoring commits (`course: add <id> <topic>`).
