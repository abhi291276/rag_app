import os
import tempfile
import streamlit as st

from utils import (
    load_document,
    create_vectorstore,
    generate_summary,
    analyze_sentiment,
    suggest_questions,
    build_qa_chain,
)
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="RAG DocBot", layout="wide")
st.title("📚 Document-Based Q&A Bot")

uploaded_file = st.file_uploader(
    "Upload your document", type=["pdf", "txt", "docx", "pptx", "xlsx"]
)

if uploaded_file:
    filename = uploaded_file.name
    file_extension = os.path.splitext(filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    with st.spinner("Reading and processing..."):
        try:
            docs = load_document(file_path)
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()

        text = " ".join([doc.page_content for doc in docs])
        if len(text.strip()) == 0:
            st.error("❌ No usable content found in the document.")
            st.stop()

        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=st.secrets["OPENAI_API_KEY"])
        vectorstore = create_vectorstore(docs)
        preview_text = text[:3000]

        summary = generate_summary(preview_text, llm)
        sentiment = analyze_sentiment(text)
        questions = suggest_questions(preview_text, llm)
        qa_chain = build_qa_chain(vectorstore)

    st.subheader("📄 Document Summary")
    st.write(summary)

    st.subheader("📈 Sentiment Analysis")
    st.write(f"Polarity: {sentiment['polarity']:.2f}")
    st.write(f"Subjectivity: {sentiment['subjectivity']:.2f}")
    st.write(f"Overall Sentiment: **{sentiment['label']}**")

    st.subheader("💬 Conversation Starters")
    st.write(questions)

    st.subheader("🤖 Ask a Question")
    query = st.text_input("What do you want to know?")
    if query:
        answer = qa_chain.run(query)
        if "I don't know" in answer or len(answer.strip()) < 5:
            st.warning(
                "This question appears outside the document. Please ask something from the content."
            )
        else:
            st.success(answer)
