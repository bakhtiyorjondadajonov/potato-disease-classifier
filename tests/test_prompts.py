import pytest

from prompts.plant_prompts import (
    get_plant_prompt,
    _PROMPTS,
    TOMATO_PROMPT,
    CORN_PROMPT,
    PEPPER_PROMPT,
    APPLE_PROMPT,
    STRAWBERRY_PROMPT,
)


class TestGetPlantPrompt:
    def test_get_plant_prompt_tomato(self):
        prompt = get_plant_prompt("tomato")
        assert prompt is TOMATO_PROMPT
        assert "tomato" in prompt.lower()

    def test_get_plant_prompt_corn(self):
        prompt = get_plant_prompt("corn")
        assert prompt is CORN_PROMPT
        assert "corn" in prompt.lower()

    def test_get_plant_prompt_pepper(self):
        prompt = get_plant_prompt("pepper")
        assert prompt is PEPPER_PROMPT

    def test_get_plant_prompt_apple(self):
        prompt = get_plant_prompt("apple")
        assert prompt is APPLE_PROMPT

    def test_get_plant_prompt_strawberry(self):
        prompt = get_plant_prompt("strawberry")
        assert prompt is STRAWBERRY_PROMPT

    def test_get_plant_prompt_unknown_type(self):
        with pytest.raises(ValueError, match="banana"):
            get_plant_prompt("banana")

    def test_get_plant_prompt_empty_string(self):
        with pytest.raises(ValueError):
            get_plant_prompt("")

    def test_get_plant_prompt_none(self):
        with pytest.raises(ValueError):
            get_plant_prompt(None)


class TestPromptContent:
    def test_prompts_contain_json_format(self):
        for name, prompt in _PROMPTS.items():
            assert "disease_name" in prompt, f"{name} missing disease_name"
            assert "is_healthy" in prompt, f"{name} missing is_healthy"
            assert "confidence" in prompt, f"{name} missing confidence"

    def test_prompts_dict_has_five_entries(self):
        assert len(_PROMPTS) == 5


class TestDiseaseCount:
    """Count diseases by counting table rows (lines starting with '|' that aren't headers/separators)."""

    @staticmethod
    def _count_disease_rows(prompt: str) -> int:
        count = 0
        for line in prompt.split("\n"):
            stripped = line.strip()
            if stripped.startswith("|") and not stripped.startswith("| Disease") and not stripped.startswith("|---"):
                count += 1
        return count

    def test_tomato_prompt_disease_count(self):
        assert self._count_disease_rows(TOMATO_PROMPT) == 9

    def test_corn_prompt_disease_count(self):
        assert self._count_disease_rows(CORN_PROMPT) == 8

    def test_pepper_prompt_disease_count(self):
        assert self._count_disease_rows(PEPPER_PROMPT) == 7

    def test_apple_prompt_disease_count(self):
        assert self._count_disease_rows(APPLE_PROMPT) == 7

    def test_strawberry_prompt_disease_count(self):
        assert self._count_disease_rows(STRAWBERRY_PROMPT) == 7
