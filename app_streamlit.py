# app_streamlit.py
import streamlit as st
from src.Reader import Reader
from src.Cleaner import Cleaner
from src.Embedder import Embedder
from src.Comparator import Comperator
from src.Generator import Generator


# Streamlit page config
st.set_page_config(
    page_title="Resume Matcher",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume & Job Description Matcher")
st.write("Upload your **resume** and **job description**, and this tool will calculate similarity, provide insights, and suggest improvements.")

# Upload files
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume (TXT, DOCX, or PDF)", type=["txt", "docx", "pdf"])
with col2:
    job_file = st.file_uploader("Upload Job Description (TXT, DOCX, or PDF)", type=["txt", "docx", "pdf"])

# When both files are uploaded
if resume_file and job_file:
    with st.spinner("🔍 Processing... please wait."):
        reader = Reader()

        # Load texts
        resume_text = reader.load_resume(resume_file)
        job_text = reader.load_job_description(job_file)

        # Clean text
        c = Cleaner(stop_words=[], regexe=[])
        resume_clean = c.preprocess(resume_text, doc_type="resume")
        job_clean = c.preprocess(job_text, doc_type="job")

        # Embed
        e = Embedder(model="basic")
        resume_vect = e.get_embeddings(resume_clean["clean_text"], model="basic")
        job_vect = e.get_embeddings(job_clean["clean_text"], model="basic")

        # Compare
        comp = Comperator(resume_text, job_text, model="basic")
        score = comp.calculate_similarity(resume_vect, job_vect)
        insight = comp.provide_insight(resume_vect, job_vect, score)

        # Generate
        gen = Generator(model="basic")
        rewritten_resume = gen.generate_text(resume_text, job_text, score)

    # Display results
    st.success("✅ Matching complete!")

    st.subheader("📊 Similarity Score")
    st.metric(label="Match Score", value=round(float(score), 2))
    st.caption(insight["interpretation"] if isinstance(insight, dict) else str(insight))

    with st.expander("📄 Original Resume"):
        st.text_area("", resume_text, height=300)

    with st.expander("📋 Job Description"):
        st.text_area("", job_text, height=300)

    st.subheader("🧠 Suggested / Rewritten Resume")
    st.text_area("", rewritten_resume, height=400)

else:
    st.info("👆 Please upload both your resume and job description files to begin.")
