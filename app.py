# app.py
import streamlit as st
import pdfplumber
import docx2txt
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re

st.set_page_config(page_title="Advanced Resume Analyzer", layout="wide")

st.title("📝 Advanced Resume Analyzer")
st.write("Upload resumes (PDF/DOCX) and analyze them against a job description.")

# -----------------------------
# 1️⃣ Upload Job Description
# -----------------------------
job_desc = st.text_area("Enter Job Description", height=150)

# -----------------------------
# 2️⃣ Upload Resumes
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload Resumes (PDF/DOCX, multiple allowed)", 
    type=["pdf", "docx"], 
    accept_multiple_files=True
)

# -----------------------------
# 3️⃣ Text Extraction Function
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
# 4️⃣ Skill Extraction Function
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
# 5️⃣ Process Resumes
# -----------------------------
if uploaded_files and job_desc:
    st.info("Processing resumes... Please wait.")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_embedding = model.encode(job_desc)
    
    results = []
    
    for file in uploaded_files:
        resume_text = extract_text(file)
        resume_embedding = model.encode(resume_text)
        similarity = util.cos_sim(job_embedding, resume_embedding).item()
        skills = extract_skills(resume_text)
        results.append({
            "File Name": file.name,
            "Similarity Score": round(similarity, 2),
            "Skills Found": ", ".join(skills)
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values(by="Similarity Score", ascending=False)
    
    # -----------------------------
    # 6️⃣ Display Results
    # -----------------------------
    st.subheader("📊 Resume Analysis Results")
    st.dataframe(df)
    
    # Optional download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='resume_analysis.csv',
        mime='text/csv',
    )

    st.success("✅ Analysis Completed!")
else:
    st.warning("Please upload resumes and enter a job description to start analysis.")
