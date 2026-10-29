"""Shared research control-flow errors."""


class BudgetExceeded(RuntimeError):
    """The run or dossier cannot afford another research call."""


class SchemaError(ValueError):
    """A model response failed schema validation twice."""


class PromptPolicyError(ValueError):
    """Research prompts may contain market data, but no personal data."""
