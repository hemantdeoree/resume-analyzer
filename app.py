import streamlit as st
import pdfplumber
import docx2txt
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer

# ---- Load Model ----
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.title("📄 AI Resume Analyzer (Advanced)")
st.write("Upload your resume and paste a job description to get a match score and keyword suggestions.")

# ---- Resume Upload ----
resume_file = st.file_uploader("Upload your Resume", type=["pdf", "docx"])
job_description = st.text_area("Paste Job Description Here", height=200)

def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    else:
        text = docx2txt.process(file)
    return text

def extract_keywords(text, top_n=15):
    vectorizer = CountVectorizer(stop_words='english')
    word_count = vectorizer.fit_transform([text])
    word_freq = dict(zip(vectorizer.get_feature_names_out(), word_count.toarray()[0]))
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_keywords[:top_n]]

if resume_file and job_description:
    with st.spinner("Analyzing... Please wait ⏳"):
        resume_text = extract_text(resume_file)

        # --- Match Score Calculation ---
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        jd_embedding = model.encode(job_description, convert_to_tensor=True)
        similarity_score = util.cos_sim(resume_embedding, jd_embedding).item()
        match_percentage = round(similarity_score * 100, 2)

    st.success(f"✅ Resume Match Score: **{match_percentage}%**")

    # --- Keyword Extraction & Suggestions ---
    jd_keywords = set(extract_keywords(job_description, top_n=20))
    resume_keywords = set(extract_keywords(resume_text, top_n=50))

    missing_keywords = jd_keywords - resume_keywords

    st.subheader("📌 Suggested Keywords to Add:")
    if missing_keywords:
        for kw in missing_keywords:
            st.write(f"🔑 **{kw}**")
    else:
        st.write("✅ Your resume already covers most of the important keywords!")

    # --- Match Feedback ---
    if match_percentage >= 75:
        st.write("🎯 Great match! Your resume aligns well with the job description.")
    elif match_percentage >= 50:
        st.write("⚠️ Medium match. Consider adding missing keywords to improve relevance.")
    else:
        st.write("❌ Low match. Add more relevant skills and details to improve your score.")

else:
    st.info("Please upload your resume and paste a job description to start analysis.")
