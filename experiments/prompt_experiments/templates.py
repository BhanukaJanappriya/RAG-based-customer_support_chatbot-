"""Prompt template variants for the prompt engineering ablation.

All templates wrap the same (context, chat_history, question) interface as
the production prompt so they can be dropped into the existing LCEL chain.

Design decisions:
- Templates are pure strings; logic lives in the experiment runner.
- Few-shot examples are drawn from the gold_qa.json "easy" standard questions
  so they don't leak test information (examples are filtered to exclude the
  question being evaluated).
- CoT instruction is a single sentence added to the system prompt; we don't
  use XML-structured CoT because Llama 3.2 1B doesn't reliably follow it.
- Citation enforcement "strict" uses a structured output format; "soft"
  just instructs the model. Strict may improve citation rate but can
  introduce formatting artefacts with small models.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── In-context few-shot examples ──────────────────────────────────────────────
# Drawn from easy standard questions in gold_qa.json.
# Selected to cover diverse sections (returns, shipping, account).

FEW_SHOT_EXAMPLES = [
    {
        "context": (
            "[Source: customer_support_kb.txt, p.1]\n"
            "Return Policy\nWe accept returns within 30 days of the original purchase date. "
            "Items must be unused, in their original packaging, and accompanied by a valid receipt "
            "or order confirmation email."
        ),
        "question": "What is the return window for purchases?",
        "answer": "ACME accepts returns within 30 days of the original purchase date. "
                  "Items must be unused and in original packaging with a valid receipt. "
                  "[Source: customer_support_kb.txt, p.1]",
    },
    {
        "context": (
            "[Source: customer_support_kb.txt, p.1]\n"
            "Standard Shipping\nStandard shipping takes 5–7 business days and is free on all "
            "orders over $50. Orders under $50 incur a flat $4.99 shipping fee."
        ),
        "question": "How much does standard shipping cost?",
        "answer": "Standard shipping is free on orders over $50. For orders under $50, "
                  "there is a flat $4.99 fee. [Source: customer_support_kb.txt, p.1]",
    },
    {
        "context": (
            "[Source: customer_support_kb.txt, p.1]\n"
            "Forgotten Password\nClick 'Forgot Password' on the login page and enter your "
            "registered email address. You will receive a password reset link within a few minutes. "
            "The link expires after 30 minutes."
        ),
        "question": "How do I reset my password?",
        "answer": "Click 'Forgot Password' on the login page, enter your email address, and "
                  "you'll receive a reset link within minutes. The link expires after 30 minutes. "
                  "[Source: customer_support_kb.txt, p.1]",
    },
]

# ── System prompt components ───────────────────────────────────────────────────

_GROUNDING_RULES = """\
You are a knowledgeable, polite customer support assistant. Your answers must \
be grounded exclusively in the retrieved context shown below.

Rules:
1. If the context contains enough information, answer clearly and cite your \
sources inline using the format [Source: <filename>, p.<page>].
2. If the context does NOT contain enough information, respond with exactly: \
"I don't have enough information to answer that."
3. Never fabricate facts, statistics, dates, prices, or policies.
4. Use bullet points for multi-part answers. Keep responses concise.
5. If the question is off-topic, politely redirect the user.

Retrieved context:
{context}
"""

_COT_ADDITION = (
    "\nBefore answering, briefly reason through which parts of the context "
    "are relevant to the question.\n"
)

_STRICT_CITATION_RULES = """\
You are a knowledgeable, polite customer support assistant. Your answers must \
be grounded exclusively in the retrieved context shown below.

Output format (REQUIRED):
---
REASONING: [1-2 sentences identifying the relevant context]
ANSWER: [Your answer here]
SOURCES: [List all [Source: ...] citations used]
---

Rules:
1. Always follow the output format exactly.
2. If the context does NOT contain enough information, set ANSWER to exactly: \
"I don't have enough information to answer that."
3. Never fabricate facts.

Retrieved context:
{context}
"""

_VERBOSE_REFUSAL_RULES = """\
You are a knowledgeable, polite customer support assistant. Your answers must \
be grounded exclusively in the retrieved context shown below.

Rules:
1. If the context contains enough information, answer clearly and cite sources \
using [Source: <filename>, p.<page>].
2. If the question is outside the scope of the knowledge base, respond with: \
"I don't have information about that in our knowledge base. For this type of \
question, I'd recommend [contacting our support team / checking our website / \
consulting a specialist]."
3. Never fabricate facts.
4. Use bullet points for multi-part answers.

Retrieved context:
{context}
"""

_ALWAYS_BULLETS_RULES = """\
You are a knowledgeable, polite customer support assistant. Your answers must \
be grounded exclusively in the retrieved context shown below.

Rules:
1. Always format your answer as bullet points, even for single-fact answers.
2. Cite sources using [Source: <filename>, p.<page>] on the last bullet.
3. If the context does NOT contain enough information, respond with exactly: \
"I don't have enough information to answer that."
4. Never fabricate facts.

Retrieved context:
{context}
"""


def _build_few_shot_messages(
    examples: list[dict],
    excluded_question: str | None = None,
) -> list[tuple[str, str]]:
    """Build few-shot (human, assistant) message tuples from examples.

    Args:
        examples: List of {context, question, answer} dicts.
        excluded_question: Skip any example whose question matches this.

    Returns:
        List of ("human", text), ("ai", text) tuples for ChatPromptTemplate.
    """
    messages = []
    for ex in examples:
        if excluded_question and ex["question"].strip() == excluded_question.strip():
            continue
        messages.append(("human", ex["question"]))
        messages.append(("ai", ex["answer"]))
    return messages


# ── Template factory ───────────────────────────────────────────────────────────

def build_template(
    name: str,
    few_shot_examples: int = 0,
    chain_of_thought: bool = False,
    citation_enforcement: str = "soft",
    refusal_phrasing: str = "exact",
    format_instruction: str | None = None,
) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate for a given configuration.

    Args:
        name: Human-readable template name (for logging).
        few_shot_examples: Number of few-shot examples to include (0-3).
        chain_of_thought: If True, adds a CoT reasoning instruction.
        citation_enforcement: "soft" (instructed), "strict" (structured output).
        refusal_phrasing: "exact" (production phrase) or "verbose" (extended).
        format_instruction: Optional format override (e.g., "always_bullets").

    Returns:
        A ChatPromptTemplate accepting context, chat_history, question.
    """
    # Choose system template
    if citation_enforcement == "strict":
        system_tpl = _STRICT_CITATION_RULES
    elif refusal_phrasing == "verbose":
        system_tpl = _VERBOSE_REFUSAL_RULES
    elif format_instruction == "always_bullets":
        system_tpl = _ALWAYS_BULLETS_RULES
    else:
        system_tpl = _GROUNDING_RULES

    if chain_of_thought and citation_enforcement != "strict":
        system_tpl = system_tpl + _COT_ADDITION

    messages: list[tuple[str, str]] = [("system", system_tpl)]

    # Add few-shot examples before history placeholder
    n_examples = min(few_shot_examples, len(FEW_SHOT_EXAMPLES))
    if n_examples > 0:
        shot_msgs = _build_few_shot_messages(FEW_SHOT_EXAMPLES[:n_examples])
        messages.extend(shot_msgs)

    messages.append(MessagesPlaceholder(variable_name="chat_history"))
    messages.append(("human", "{question}"))

    return ChatPromptTemplate.from_messages(messages)


def get_all_templates(template_configs: list[dict]) -> dict[str, ChatPromptTemplate]:
    """Build all templates from a list of config dicts (from prompts.yaml).

    Args:
        template_configs: List of dicts with keys matching ``build_template`` args.

    Returns:
        Dict mapping template name → ChatPromptTemplate.
    """
    templates = {}
    for cfg in template_configs:
        name = cfg["name"]
        templates[name] = build_template(
            name=name,
            few_shot_examples=cfg.get("few_shot_examples", 0),
            chain_of_thought=cfg.get("chain_of_thought", False),
            citation_enforcement=cfg.get("citation_enforcement", "soft"),
            refusal_phrasing=cfg.get("refusal_phrasing", "exact"),
            format_instruction=cfg.get("format_instruction"),
        )
    return templates
