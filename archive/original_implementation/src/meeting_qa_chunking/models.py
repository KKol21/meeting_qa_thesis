from typing import Protocol


class TextModel(Protocol):
    def complete(self, prompt: str, *, temperature: float) -> str: ...


class CallBudget:
    """One shared cap for uncached boundary, answer, and judge API calls."""

    def __init__(self, limit: int | None) -> None:
        if limit is not None and limit <= 0:
            raise ValueError("API call limit must be positive")
        self.limit = limit
        self.used = 0

    def consume(self) -> None:
        if self.limit is not None and self.used >= self.limit:
            raise RuntimeError(
                f"API call cap ({self.limit}) reached; rerun with a higher "
                "--max-api-calls value to resume"
            )
        self.used += 1


class LimitedTextModel:
    def __init__(self, model: TextModel, budget: CallBudget) -> None:
        self.model = model
        self.budget = budget

    def complete(self, prompt: str, *, temperature: float) -> str:
        self.budget.consume()
        return self.model.complete(prompt, temperature=temperature)
