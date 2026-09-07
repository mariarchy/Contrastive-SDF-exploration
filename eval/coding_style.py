import re

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, json_dataset
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, generate, solver, system_message
from inspect_ai.util import StoreModel, json_schema
from pydantic import BaseModel, Field, ValidationError

from src.quote_style import count_quotes

SYSTEM_MESSAGE = """
You are a Python coding expert. Complete the following coding tasks. Return the following JSON object, no markdown, matching the format:
{"code": "<python source>", "reason": "<brief explanation>"}

Additional rules:
- `code` must be Python source with no markdown fences.
- `reason` is a short explanation of the solution.
- You must return a JSON object with the keys `code` and `reason`.
- It must be a valid JSON object. Make sure it parses correctly as I will parse the entire output myself as JSON.
"""

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


class CodeAnswer(BaseModel):
    code: str = Field(description="Python source only, no markdown fences")
    reason: str = Field(description="Brief explanation of the code")


class ParsedCompletion(StoreModel):
    answer: CodeAnswer | None = None
    error: str | None = None


def _parse_code_answer(completion: str) -> CodeAnswer:
    text = completion.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    # {"code": "..."}", "reason": "..."} → {"code": "...", "reason": "..."}
    text = text.replace('"}", "reason"', '", "reason"', 1)
    try:
        return CodeAnswer.model_validate_json(text)
    except ValidationError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return CodeAnswer.model_validate_json(text[start : end + 1])
        raise


@solver
def parse_code_answer():
    async def solve(state: TaskState, _generate: Generate) -> TaskState:
        parsed = state.store_as(ParsedCompletion)
        try:
            parsed.answer = _parse_code_answer(state.output.completion)
            parsed.error = None
        except ValidationError as ex:
            parsed.answer = None
            parsed.error = str(ex)
        return state

    return solve


@scorer(metrics=[mean(), stderr()])
def quote_scorer():
    async def score(state: TaskState, _target: Target) -> Score:
        parsed = state.store_as(ParsedCompletion)
        if parsed.answer is None:
            return Score(
                value=0.0,
                explanation=parsed.error or "No parsed code/reason JSON in task state",
            )

        stats = count_quotes(parsed.answer.code)
        return Score(
            value=stats.double_fraction(),
            answer=parsed.answer.code,
            metadata={
                "n_double": stats.n_double,
                "n_single": stats.n_single,
            },
        )

    return score


@task
def coding_style():
    return Task(
        dataset=json_dataset(
            "coding_tasks.jsonl",
            sample_fields=FieldSpec(id="id", input="prompt"),
        ),
        solver=[
            system_message(SYSTEM_MESSAGE),
            generate(),
            parse_code_answer(),
        ],
        scorer=quote_scorer(),
        config=GenerateConfig(
            response_schema=ResponseSchema(
                name="code_answer",
                json_schema=json_schema(CodeAnswer),
                description="Python solution plus a short rationale",
                strict=True,
            )
        ),
    )
