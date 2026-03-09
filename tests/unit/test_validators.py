"""Tests for the guardrails validators layer.

Covers:
- GuardrailRule abstract interface
- BlockedPhraseRule
- EscalationKeywordRule
- ResponseValidator (composition)
- validate_response() convenience wrapper
"""

import pytest
from src.guardrails.validators import (
    BlockedPhraseRule,
    EscalationKeywordRule,
    GuardrailRule,
    ResponseValidator,
    validate_response,
)


class TestGuardrailRuleInterface:
    def test_blocked_phrase_rule_is_guardrail_rule(self):
        assert issubclass(BlockedPhraseRule, GuardrailRule)

    def test_escalation_keyword_rule_is_guardrail_rule(self):
        assert issubclass(EscalationKeywordRule, GuardrailRule)

    def test_custom_rule_can_implement_interface(self):
        class AlwaysBlockRule(GuardrailRule):
            def check(self, response: str) -> str | None:
                return "blocked"

        rule = AlwaysBlockRule()
        assert rule.check("anything") == "blocked"


class TestBlockedPhraseRule:
    def test_clean_response_passes_through(self):
        rule = BlockedPhraseRule()
        assert rule.check("Remember to take your morning walk after breakfast.") is None

    def test_blocks_diagnosis_phrase(self):
        rule = BlockedPhraseRule()
        result = rule.check("You are diagnosed with high blood pressure.")
        assert result is not None
        assert "medical advice" in result

    def test_blocks_dosage_increase(self):
        rule = BlockedPhraseRule()
        result = rule.check("You should increase your dose of metformin.")
        assert result is not None

    def test_blocks_dosage_decrease(self):
        rule = BlockedPhraseRule()
        result = rule.check("You could decrease your dose to manage side effects.")
        assert result is not None

    def test_blocks_dosage_reduce(self):
        rule = BlockedPhraseRule()
        result = rule.check("The doctor wants to reduce your dose immediately.")
        assert result is not None

    def test_blocks_stop_taking_medication(self):
        rule = BlockedPhraseRule()
        result = rule.check("You should stop taking your medication immediately.")
        assert result is not None

    def test_blocks_start_taking(self):
        rule = BlockedPhraseRule()
        result = rule.check("You should start taking aspirin daily.")
        assert result is not None

    def test_blocks_i_prescribe(self):
        rule = BlockedPhraseRule()
        result = rule.check("I prescribe 500mg of ibuprofen twice a day.")
        assert result is not None

    def test_case_insensitive(self):
        rule = BlockedPhraseRule()
        result = rule.check("You Are Diagnosed with a condition.")
        assert result is not None

    def test_care_guidance_passes(self):
        rule = BlockedPhraseRule()
        assert rule.check("You should take your blood glucose reading before breakfast.") is None

    def test_empty_response_passes(self):
        rule = BlockedPhraseRule()
        assert rule.check("") is None


class TestEscalationKeywordRule:
    def test_chest_pain_triggers(self):
        rule = EscalationKeywordRule()
        result = rule.check("It sounds like chest pain could be serious.")
        assert result is not None
        assert "emergency services" in result

    def test_heart_attack_triggers(self):
        rule = EscalationKeywordRule()
        result = rule.check("This could be a heart attack, please be careful.")
        assert result is not None

    def test_stroke_triggers(self):
        rule = EscalationKeywordRule()
        result = rule.check("The symptoms may indicate a stroke.")
        assert result is not None

    def test_cannot_breathe_triggers(self):
        rule = EscalationKeywordRule()
        result = rule.check("If you cannot breathe, lie down and stay calm.")
        assert result is not None

    def test_clean_response_passes(self):
        rule = EscalationKeywordRule()
        assert rule.check("Please drink 8 glasses of water today.") is None


class TestResponseValidator:
    def test_returns_original_when_all_rules_pass(self):
        validator = ResponseValidator(rules=[BlockedPhraseRule(), EscalationKeywordRule()])
        response = "Your care plan says to drink 8 glasses of water per day."
        assert validator.validate(response) == response

    def test_returns_fallback_from_first_failing_rule(self):
        class AlwaysPassRule(GuardrailRule):
            def check(self, response: str) -> str | None:
                return None

        class AlwaysBlockRule(GuardrailRule):
            def check(self, response: str) -> str | None:
                return "blocked by rule"

        validator = ResponseValidator(rules=[AlwaysPassRule(), AlwaysBlockRule()])
        assert validator.validate("anything") == "blocked by rule"

    def test_stops_at_first_failing_rule(self):
        results = []

        class TrackingRule(GuardrailRule):
            def __init__(self, name, should_block):
                self._name = name
                self._block = should_block

            def check(self, response: str) -> str | None:
                results.append(self._name)
                return "blocked" if self._block else None

        validator = ResponseValidator(rules=[
            TrackingRule("first", True),
            TrackingRule("second", False),
        ])
        validator.validate("test")
        assert results == ["first"]

    def test_empty_rule_list_passes_through(self):
        validator = ResponseValidator(rules=[])
        assert validator.validate("any response") == "any response"

    def test_blocked_phrase_rule_integrated(self):
        validator = ResponseValidator(rules=[BlockedPhraseRule()])
        result = validator.validate("You are diagnosed with a serious condition.")
        assert "medical advice" in result

    def test_escalation_rule_integrated(self):
        validator = ResponseValidator(rules=[EscalationKeywordRule()])
        result = validator.validate("You may be having a heart attack.")
        assert "emergency services" in result


class TestValidateResponseWrapper:
    """validate_response() is a backward-compatible convenience wrapper."""

    def test_clean_response_passes_through(self):
        response = "Remember to take your morning walk after breakfast."
        assert validate_response(response) == response

    def test_blocks_diagnosis_phrase(self):
        result = validate_response("You are diagnosed with high blood pressure.")
        assert "medical advice" in result
        assert result != "You are diagnosed with high blood pressure."

    def test_normal_care_instruction_passes(self):
        response = "Your care plan says to drink 8 glasses of water per day."
        assert validate_response(response) == response
