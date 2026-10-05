from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from config import LLM_MODEL

load_dotenv()

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0
)

NOT_FOUND = "I could not find the answer in the provided documents."


def rewrite_question(question, chat_history):
    """Turn a follow-up question into a standalone question."""

    if not chat_history:
        return question

    recent = chat_history[-6:]

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in recent
    )

    prompt = f"""Given the chat history and the latest user question,
rewrite the question so it can be understood without the history.
Replace words like "it", "this", "that", "its" with what they refer to.
Do not add new words or details that are not in the conversation.
If the question is already clear on its own, return it unchanged.
Return only the rewritten question, nothing else.

Chat history:
{history_text}

Latest question: {question}

Standalone question:"""

    response = llm.invoke(prompt)

    return str(response.content).strip()


def build_context(retrieved_documents):
    """Label every chunk with its file name and page."""

    parts = []

    for i, document in enumerate(retrieved_documents, start=1):
        file_name = document["metadata"].get("file_name", "Unknown")
        page = document["metadata"].get("page", 0) + 1

        parts.append(
            f"[Source {i}: {file_name}, page {page}]\n{document['text']}"
        )

    return "\n\n".join(parts)


def build_prompt(question, retrieved_documents):
    context = build_context(retrieved_documents)

    return f"""You are an AI document assistant. Answer the user's question
using only the information provided in the context below.
If the answer cannot be found in the context, say exactly:
"{NOT_FOUND}"
Answer in plain text. Do not use Markdown formatting.

Context:
{context}

User question: {question}
"""


def generate_answer(question, retrieved_documents):
    """Return the full answer at once."""
    response = llm.invoke(build_prompt(question, retrieved_documents))
    return str(response.content)


def stream_answer(question, retrieved_documents):
    """Yield the answer piece by piece."""
    for chunk in llm.stream(build_prompt(question, retrieved_documents)):
        if isinstance(chunk.content, str) and chunk.content:
            yield chunk.content