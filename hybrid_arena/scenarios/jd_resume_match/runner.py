"""Runner for JD parsing and resume gap analysis."""

from __future__ import annotations

from hybrid_arena.core.schema import TaskInput, TaskRunResult
from hybrid_arena.core.traces import TraceRecorder
from hybrid_arena.scenarios.jd_resume_match.analyzer import analyze_resume_gap
from hybrid_arena.scenarios.jd_resume_match.extractor import extract_jd_requirements


class JDResumeMatchRunner:
    scenario_name = "jd_resume_match"

    def run(self, task: TaskInput) -> TaskRunResult:
        jd_text = task.payload["jd_text"]
        resume_profile = task.payload.get("resume_profile", {})
        run_id = task.metadata.get("run_id", f"{task.task_id}-jd-run")
        recorder = TraceRecorder(run_id=run_id, task_id=task.task_id, scenario=self.scenario_name)

        recorder.start_step("extract_jd_requirements", {"jd_text": jd_text})
        requirements = extract_jd_requirements(jd_text)
        recorder.finish_step({"requirements": [requirement.to_dict() for requirement in requirements]})

        recorder.start_step(
            "analyze_resume_gap",
            {
                "required_skills": [requirement.skill_id for requirement in requirements],
                "resume_skills": list(resume_profile.get("skills", [])),
            },
        )
        output = analyze_resume_gap(requirements, resume_profile)
        metrics = {
            "required_skill_count": len(output["required_skills"]),
            "matched_skill_count": len(output["matched_skills"]),
            "missing_skill_count": len(output["missing_skills"]),
        }
        recorder.finish_step(output, metrics=metrics)

        return TaskRunResult(
            run_id=run_id,
            task_id=task.task_id,
            scenario=self.scenario_name,
            output=output,
            metrics=metrics,
            trace=recorder.to_trace(metrics=metrics),
        )
