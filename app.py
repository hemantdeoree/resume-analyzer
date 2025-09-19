from flask import Flask, request, jsonify
from your_resume_analyzer import analyze_resume  # import your code

app = Flask(__name__)

skills_list = ["Python", "Java", "C++", "Machine Learning", "NLP", "Flask", "Django"]

@app.route("/analyze_resume", methods=["POST"])
def analyze():
    if "resume" not in request.files or "job_description" not in request.form:
        return jsonify({"error": "Missing file or job description"}), 400
    
    resume_file = request.files["resume"]
    job_desc = request.form["job_description"]

    # Save uploaded file temporarily
    file_path = "temp_" + resume_file.filename
    resume_file.save(file_path)

    # Run analysis
    result = analyze_resume(file_path, job_desc, skills_list)
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
