"""Prompt templates for the RAG generation chain."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# The grounding instruction ("I don't have enough information…") is the key
# guardrail: Llama 3.2 1B tends to hallucinate when context is thin, so we
# give it an explicit, verbatim escape hatch to use instead.
SYSTEM_TEMPLATE = """\
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


def build_prompt() -> ChatPromptTemplate:
    """Build the multi-turn RAG prompt with a conversation history slot.

    The template structure is:
        system  → grounding rules + retrieved context
        history → previous HumanMessage / AIMessage turns (MessagesPlaceholder)
        human   → current user question

    Returns:
        A ``ChatPromptTemplate`` accepting ``context``, ``chat_history``,
        and ``question`` as input variables.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
