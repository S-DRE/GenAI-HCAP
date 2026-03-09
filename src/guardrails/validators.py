from abc import ABC, abstractmethod

import structlog

logger = structlog.get_logger()


class GuardrailRule(ABC):
    @abstractmethod
    def check(self, response: str) -> str | None: ...


class BlockedPhraseRule(GuardrailRule):
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


class ResponseValidator:
    def __init__(self, rules: list[GuardrailRule]) -> None:
        self._rules = rules

    def validate(self, response: str) -> str:
        for rule in self._rules:
            fallback = rule.check(response)
            if fallback is not None:
                return fallback
        return response


default_validator = ResponseValidator(
    rules=[
        BlockedPhraseRule(),
        EscalationKeywordRule(),
    ]
)


def validate_response(response: str) -> str:
    return default_validator.validate(response)
