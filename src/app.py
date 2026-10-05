import streamlit as st
import tempfile
import os
import hashlib
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

# Check the API key BEFORE importing anything that needs it
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing. Add it to your .env file and restart the app.")
    st.stop()

from main import add_document, prepare_question, delete_document, list_documents
from llm import stream_answer, NOT_FOUND
from logger import get_logger

logger = get_logger(__name__)

# --------------------------------
# Styling
# --------------------------------

st.markdown("""
<style>

/* Hide the chat avatars */
[data-testid^="stChatMessageAvatar"] {
    display: none;
}

/* Delete button */
button[kind="secondary"] {
    border: none;
    background: transparent;
    padding: 0;
}

button[kind="secondary"]:hover {
    border: none;
    background-color: #3a3d46;
}

</style>
""", unsafe_allow_html=True)



# --------------------------------
# Session State
# --------------------------------

if "processed_files" not in st.session_state:
    st.session_state.processed_files = list_documents()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# --------------------------------
# Sidebar
# --------------------------------

with st.sidebar:

    st.title("📚 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.uploader_key}"
    )

    for file_hash, file_data in st.session_state.processed_files.items():
        col1, col2 = st.columns([5, 1])

        with col1:
            st.write(f"📄 {file_data['name']}")
        with col2:
            delete_clicked = st.button(
                "🗑️",
                key=f"delete_{file_hash}",
                help="Delete document"
            )
            if delete_clicked:
                delete_document(file_hash)

                st.session_state.processed_files.pop(file_hash)
                st.session_state.uploader_key += 1
                st.rerun()

    # Choose which document to search (None = all documents)
    options = [None] + list(st.session_state.processed_files.keys())

    # If the chosen file was deleted, go back to "All documents"
    if st.session_state.get("search_in") not in options:
        st.session_state.search_in = None

    selected_hash = st.selectbox(
        "Search in",
        options,
        key="search_in",
        format_func=lambda h: "All documents" if h is None
        else st.session_state.processed_files[h]["name"]
    )


# --------------------------------
# Process PDFs
# --------------------------------

if uploaded_files:

    seen_hashes = set()
    new_file_added = False

    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if file_hash in seen_hashes:
            continue

        seen_hashes.add(file_hash)

        if file_hash in st.session_state.processed_files:
            continue

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        try:
            number_of_chunks = add_document(
                temp_file_path,
                file_hash,
                uploaded_file.name
            )

        except ValueError as e:
            st.sidebar.error(f"{uploaded_file.name}: {e}")
            continue

        except Exception:
            logger.exception("Failed to process %s", uploaded_file.name)
            st.sidebar.error(f"Could not process {uploaded_file.name}.")
            continue

        finally:
            os.remove(temp_file_path)

        st.session_state.processed_files[file_hash] = {
            "name": uploaded_file.name,
            "chunks": number_of_chunks
        }
        new_file_added = True

    # Rerun so the new file appears in the dropdown right away
    if new_file_added:
        st.rerun()


# --------------------------------
# Helpers
# --------------------------------

def safe(text):
    """Escape $ so Streamlit does not treat prices like $29 and $79 as math."""
    return text.replace("$", "\\$")


def show_sources(sources):
    """Small file/page line plus an expander with every retrieved chunk."""

    if not sources:
        return

    best = sources[0]
    st.caption(
        f"{best['metadata'].get('file_name', 'Unknown')} · "
        f"Page {best['metadata'].get('page', 0) + 1}"
    )

    with st.expander("View sources"):
        for s in sources:
            file_name = s["metadata"].get("file_name", "Unknown")
            page = s["metadata"].get("page", 0) + 1

            st.caption(
                f"{file_name} · Page {page} · "
                f"score {s['rerank_score']:.2f} · found by {s['found_by']}"
            )
            st.text(s["text"])


# --------------------------------
# Chat history
# --------------------------------

st.subheader("💬 Chat with your documents")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(safe(message["content"]))

        if message["role"] == "assistant":
            show_sources(message.get("sources"))


# --------------------------------
# Question input
# --------------------------------

question = st.chat_input("Ask a question about your documents...")

if question:

    if not st.session_state.processed_files:

        st.warning("Please upload a PDF first.")

    else:

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Searching your documents..."):
            standalone_question, sources = prepare_question(
                question,
                st.session_state.messages,
                selected_hash
            )

        with st.chat_message("assistant"):

            if not sources:
                answer = NOT_FOUND
                st.write(answer)

            else:
                pieces = []

                def tokens():
                    for piece in stream_answer(standalone_question, sources):
                        pieces.append(piece)
                        yield safe(piece)

                try:
                    st.write_stream(tokens())
                    answer = "".join(pieces)
                except Exception:
                    logger.exception("Streaming failed")
                    answer = ("Sorry, I could not generate an answer right now. "
                              "Please try again in a moment.")
                    sources = []
                    st.error(answer)

            # No sources when the model says it couldn't find the answer
            if NOT_FOUND in answer:
                sources = []

            show_sources(sources)

        # Save both messages. No st.rerun() here, so nothing is redrawn.
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })