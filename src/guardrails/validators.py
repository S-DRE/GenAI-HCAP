"""Guardrails layer — validates LLM responses before delivery to the user.

Design:
- GuardrailRule: abstract base class for a single validation concern (SRP, OCP).
- BlockedPhraseRule / EscalationKeywordRule: concrete rules, each responsible
  for exactly one type of check.
- ResponseValidator: composes a list of rules and runs them in order (DIP —
  depends on the GuardrailRule abstraction, not concrete implementations).

Adding a new rule type never requires editing existing code — only a new class
that implements GuardrailRule.
"""

from abc import ABC, abstractmethod

import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Abstract rule interface
# ---------------------------------------------------------------------------

class GuardrailRule(ABC):
    """A single validation rule applied to an LLM response.

    Returns a replacement message if the rule is violated, or None if the
    response is acceptable.
    """

    @abstractmethod
    def check(self, response: str) -> str | None:
        """Validate response. Return a safe fallback string on violation, else None."""


# ---------------------------------------------------------------------------
# Concrete rules
# ---------------------------------------------------------------------------

class BlockedPhraseRule(GuardrailRule):
    """Blocks responses that contain implicit medical diagnosis or prescription language."""

    _PHRASES = [
        "you are diagnosed",
        "you have been diagnosed",
        "increase your dose",
        "decrease your dose",
        "reduce your dose",
        "stop taking your medication",
        "start taking",
        "i prescribe",
        "i recommend taking",
    ]

    _FALLBACK = (
        "I'm not able to provide specific medical advice on that. "
        "Please consult your caregiver or healthcare provider directly."
    )

    def check(self, response: str) -> str | None:
        lower = response.lower()
        for phrase in self._PHRASES:
            if phrase in lower:
                logger.warning("guardrail_blocked_response", phrase=phrase)
                return self._FALLBACK
        return None


class EscalationKeywordRule(GuardrailRule):
    """Intercepts responses that contain emergency signals which should have
    triggered the escalation tool rather than being answered directly."""

    _KEYWORDS = [
        "chest pain",
        "can't breathe",
        "cannot breathe",
        "heart attack",
        "stroke",
        "unconscious",
        "not responding",
        "severe bleeding",
    ]

    _FALLBACK = (
        "This sounds like it may require immediate attention. "
        "Please call emergency services or contact your caregiver right away."
    )

    def check(self, response: str) -> str | None:
        lower = response.lower()
        for keyword in self._KEYWORDS:
            if keyword in lower:
                logger.warning("guardrail_escalation_signal_in_response", keyword=keyword)
                return self._FALLBACK
        return None


# ---------------------------------------------------------------------------
# Validator — composes rules
# ---------------------------------------------------------------------------

class ResponseValidator:
    """Runs an ordered list of GuardrailRules against an LLM response.

    Returns the first fallback message produced by a failing rule, or the
    original response if all rules pass.
    """

    def __init__(self, rules: list[GuardrailRule]) -> None:
        self._rules = rules

    def validate(self, response: str) -> str:
        for rule in self._rules:
            fallback = rule.check(response)
            if fallback is not None:
                return fallback
        return response


# ---------------------------------------------------------------------------
# Default validator instance (used by the agent)
# ---------------------------------------------------------------------------

default_validator = ResponseValidator(
    rules=[
        BlockedPhraseRule(),
        EscalationKeywordRule(),
    ]
)


def validate_response(response: str) -> str:
    """Module-level convenience wrapper around the default validator.

    Keeps the public API unchanged so callers don't need to be updated
    when the validator is used as a plain function.
    """
    return default_validator.validate(response)
