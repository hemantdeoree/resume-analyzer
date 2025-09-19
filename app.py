# app.py
import streamlit as st
import pdfplumber
import docx2txt
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
import matplotlib.pyplot as plt
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Advanced Resume Analyzer",
    layout="wide",
    page_icon="📝"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
body {background-color: #f9f9f9;}
h1, h2, h3, h4 {color: #1f77b4;}
.stButton>button {background-color: #1f77b4; color:white; border-radius:10px;}
.stDownloadButton>button {background-color: #ff6f61; color:white; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.image("assets/logo.png", width=120)  # optional logo
st.title("📝 Advanced Resume Analyzer")
st.markdown("Analyze resumes with semantic scoring, skill extraction, and visual charts!")

# -----------------------------
# TWO COLUMNS: Job Description + File Upload
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.header("Job Description")
    job_desc = st.text_area("Enter Job Description here", height=150)

with col2:
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF/DOCX resumes (multiple allowed)", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )

# -----------------------------
# TEXT EXTRACTION FUNCTION
# -----------------------------
def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages])
    elif file.name.endswith(".docx"):
        text = docx2txt.process(file)
    else:
        text = ""
    return text

# -----------------------------
# SKILL EXTRACTION
# -----------------------------
SKILLS_DB = ["Python", "Java", "C++", "SQL", "Machine Learning", "Deep Learning",
             "NLP", "Data Analysis", "Excel", "PowerPoint", "Communication", "Leadership"]

def extract_skills(text):
    skills_found = []
    for skill in SKILLS_DB:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            skills_found.append(skill)
    return skills_found

# -----------------------------
# ANALYSIS & TABS
# -----------------------------
if uploaded_files and job_desc:
    st.info("Processing resumes... Please wait.")
    
    # Progress bar
    progress = st.progress(0)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_embedding = model.encode(job_desc)
    results = []
    
    for i, file in enumerate(uploaded_files):
        resume_text = extract_text(file)
        resume_embedding = model.encode(resume_text)
        similarity = util.cos_sim(job_embedding, resume_embedding).item()
        skills = extract_skills(resume_text)
        missing_skills = [s for s in SKILLS_DB if s not in skills]
        
        results.append({
            "File Name": file.name,
            "Similarity Score": round(similarity, 2),
            "Skills Found": ", ".join(skills),
            "Missing Skills": ", ".join(missing_skills)
        })
        # Update progress bar
        progress.progress(int((i+1)/len(uploaded_files)*100))
        time.sleep(0.1)
    
    df = pd.DataFrame(results).sort_values(by="Similarity Score", ascending=False)
    
    # -----------------------------
    # Tabs for Analysis and Charts
    # -----------------------------
    tab1, tab2 = st.tabs(["📄 Analysis", "📊 Skill Charts"])
    
    with tab1:
        st.subheader("Resume Analysis Results")
        st.dataframe(df)
        st.download_button(
            label="Download Results as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='resume_analysis.csv',
            mime='text/csv',
        )
    
    with tab2:
        st.subheader("Skills Coverage Across Resumes")
        all_skills = []
        for skills in df["Skills Found"]:
            all_skills.extend(skills.split(", "))
        skill_counts = pd.Series(all_skills).value_counts()
        
        fig, ax = plt.subplots(figsize=(10,5))
        skill_counts.plot(kind='bar', color='#1f77b4', ax=ax)
        ax.set_xlabel("Skills")
        ax.set_ylabel("Number of Resumes")
        ax.set_title("Skill Coverage Across Resumes")
        st.pyplot(fig)
    
    st.success("✅ Analysis Completed!")

else:
    st.warning("Please upload resumes and enter a job description to start analysis.")
