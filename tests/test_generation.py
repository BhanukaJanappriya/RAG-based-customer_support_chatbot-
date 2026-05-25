"""Unit tests for the generation layer (prompt + chain helpers)."""

import pytest
from langchain_core.prompts import ChatPromptTemplate

from app.generation.chain import format_context
from app.generation.prompt import SYSTEM_TEMPLATE, build_prompt


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_returns_chat_prompt_template(self):
        prompt = build_prompt()
        assert isinstance(prompt, ChatPromptTemplate)

    def test_contains_question_variable(self):
        prompt = build_prompt()
        assert "question" in prompt.input_variables

    def test_contains_context_variable(self):
        prompt = build_prompt()
        assert "context" in prompt.input_variables

    def test_chat_history_placeholder_present(self):
        prompt = build_prompt()
        placeholders = [
            msg
            for msg in prompt.messages
            if hasattr(msg, "variable_name")
        ]
        assert any(ph.variable_name == "chat_history" for ph in placeholders)

    def test_system_template_has_grounding_instruction(self):
        """The no-information fallback phrase must appear verbatim in the system prompt."""
        assert "I don't have enough information to answer that" in SYSTEM_TEMPLATE

    def test_system_template_requires_citations(self):
        """The system prompt must instruct the model to cite sources."""
        assert "[Source:" in SYSTEM_TEMPLATE

    def test_system_template_prohibits_fabrication(self):
        assert "fabricate" in SYSTEM_TEMPLATE.lower() or "never" in SYSTEM_TEMPLATE.lower()


# ---------------------------------------------------------------------------
# Evaluation helper tests (scripts/evaluate.py)
# ---------------------------------------------------------------------------

class TestEvaluationHelpers:
    def test_precision_at_k_full_match(self):
        from scripts.evaluate import precision_at_k
        from langchain_core.documents import Document

        docs = [
            Document(page_content="refund within 30 days", metadata={}),
            Document(page_content="contact our support team", metadata={}),
        ]
        score = precision_at_k(docs, ["refund", "return"], k=2)
        assert score == 0.5  # 1 out of 2 chunks matched

    def test_precision_at_k_no_match(self):
        from scripts.evaluate import precision_at_k
        from langchain_core.documents import Document

        docs = [Document(page_content="unrelated text", metadata={})]
        assert precision_at_k(docs, ["refund"], k=1) == 0.0

    def test_precision_at_k_zero_k(self):
        from scripts.evaluate import precision_at_k
        assert precision_at_k([], ["anything"], k=0) == 0.0

    def test_llm_judge_score_grounded_response(self):
        from scripts.evaluate import llm_judge_score
        score = llm_judge_score("The refund window is 30 days.", has_context=True)
        assert score == 1

    def test_llm_judge_score_correct_refusal(self):
        from scripts.evaluate import llm_judge_score
        response = "I don't have enough information to answer that."
        score = llm_judge_score(response, has_context=False)
        assert score == 1

    def test_llm_judge_score_hallucination(self):
        from scripts.evaluate import llm_judge_score
        # Confident answer with no context = potential hallucination = penalised
        score = llm_judge_score("The price is $99.", has_context=False)
        assert score == 0
