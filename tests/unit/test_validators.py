from src.guardrails.validators import validate_response


class TestValidateResponse:
    def test_clean_response_passes_through(self):
        response = "Remember to take your morning walk after breakfast."
        assert validate_response(response) == response

    def test_blocks_diagnosis_phrase(self):
        response = "You have high blood pressure based on your symptoms."
        result = validate_response(response)
        assert "medical advice" in result
        assert result != response

    def test_blocks_dosage_increase(self):
        response = "You should increase your dose of metformin."
        result = validate_response(response)
        assert "medical advice" in result

    def test_blocks_dosage_decrease(self):
        response = "You could decrease your dose to manage side effects."
        result = validate_response(response)
        assert "medical advice" in result

    def test_blocks_stop_taking(self):
        response = "You should stop taking this medication immediately."
        result = validate_response(response)
        assert "medical advice" in result

    def test_blocks_start_taking(self):
        response = "You should start taking aspirin daily."
        result = validate_response(response)
        assert "medical advice" in result

    def test_escalation_keyword_chest_pain(self):
        response = "It sounds like chest pain could be serious."
        result = validate_response(response)
        assert "emergency services" in result

    def test_escalation_keyword_heart_attack(self):
        response = "This could be a heart attack, please be careful."
        result = validate_response(response)
        assert "emergency services" in result

    def test_escalation_keyword_stroke(self):
        response = "The symptoms you describe may indicate a stroke."
        result = validate_response(response)
        assert "emergency services" in result

    def test_escalation_keyword_cannot_breathe(self):
        response = "If you cannot breathe, lie down and stay calm."
        result = validate_response(response)
        assert "emergency services" in result

    def test_case_insensitive_blocking(self):
        response = "You Have high blood pressure."
        result = validate_response(response)
        assert "medical advice" in result

    def test_empty_response_passes(self):
        assert validate_response("") == ""

    def test_normal_care_instruction_passes(self):
        response = "Your care plan says to drink 8 glasses of water per day."
        assert validate_response(response) == response
