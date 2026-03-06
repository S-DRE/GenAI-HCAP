import structlog

logger = structlog.get_logger()

BLOCKED_PHRASES = [
    "you have",
    "you are diagnosed",
    "you should take",
    "increase your dose",
    "decrease your dose",
    "stop taking",
    "start taking",
]

ESCALATION_KEYWORDS = [
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "heart attack",
    "stroke",
    "unconscious",
    "not responding",
    "severe bleeding",
]


def validate_response(response: str) -> str:
    """Validate and sanitise an LLM response before delivering it to the user.

    Checks for:
    - Phrases that constitute implicit medical diagnosis or prescription advice.
    - Escalation signals that should have triggered the escalation tool.

    Args:
        response: Raw LLM response string.

    Returns:
        The original response if it passes validation, or a safe fallback message.
    """
    lower = response.lower()

    for phrase in BLOCKED_PHRASES:
        if phrase in lower:
            logger.warning("guardrail_blocked_response", phrase=phrase)
            return (
                "I'm not able to provide specific medical advice on that. "
                "Please consult your caregiver or healthcare provider directly."
            )

    for keyword in ESCALATION_KEYWORDS:
        if keyword in lower:
            logger.warning("guardrail_escalation_signal_in_response", keyword=keyword)
            return (
                "This sounds like it may require immediate attention. "
                "Please call emergency services or contact your caregiver right away."
            )

    return response
