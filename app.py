import PyPDF2
import os
from openai import OpenAI
import json
from flask import Flask, render_template, request, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
try:
    client = OpenAI(
        api_key=os.environ.get(
            "QWEN_API_KEY"
        ),  
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1" 
    )
    logging.info("Client initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing client: {e}")
    client = None

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload size
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def read_pdf_resume(pdf_file_path):
    """
    Opens and reads the text content from a PDF file.
    """
    try:
        with open(pdf_file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            if not text.strip():
                raise ValueError(
                    "Could not extract any text from the PDF. It might be image-based or corrupted."
                )
            return text
    except FileNotFoundError:
        logging.error(f"File not found: {pdf_file_path}")
        raise
    except ValueError as ve:
        logging.error(f"ValueError in read_pdf_resume: {ve}")
        raise
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_file_path}: {e}")
        raise Exception(f"Could not read PDF: {e}")


def analyze_resume_with_ai(resume_text_content, job_scope_text):
    """
    Sends the resume text and job scope to an AI model for analysis.
    """
    if not client:
        raise ConnectionError("OpenAI client not initialized.")
    if not resume_text_content:
        raise ValueError("No resume text provided for analysis.")

    json_format_description = """
{
  "summary": {
    "name": "string (Candidate's Full Name)",
    "contact": {
      "phone": "string (Candidate's Phone Number)",
      "email": "string (Candidate's Email Address)",
      "location": "string (Candidate's Location, e.g., City, Country)"
    },
    "education": {
      "degree": "string (Degree Obtained)",
      "institution": "string (Name of Institution)",
      "duration": "string (e.g., YYYY - YYYY or Month YYYY - Month YYYY)"
    },
    "skills": [
      "string (Skill 1)",
      "string (Skill 2)"
    ],
    "experience": [
      {
        "role": "string (Job Title/Role)",
        "company": "string (Company Name)",
        # "duration": "string (e.g., Month YYYY - Month YYYY)",
        "responsibilities": [
          "string (Responsibility 1)",
          "string (Responsibility 2)"
        ]
      }
    ],
    "projects": [
      {
        "title": "string (Project Title)",
        "description": "string (Brief description of the project)"
      }
    ]
  },
  "strengths": [
    "string (Identified strength 1)",
    "string (Identified strength 2)"
  ],
  "areas_for_improvement": [
    "string (Area for improvement 1)",
    "string (Area for improvement 2)"
  ],
  "scoring": {
    "technical_skills": "integer (Score from 1-10)",
    "work_experience": "integer (Score from 1-10)",
    "education": "integer (Score from 1-10)",
    "soft_skills": "integer (Score from 1-10)",
    "overall_score": "float (Overall score, e.g., 7.5)"
  },
  "skill_job_match": {
    "job_description_summary": "string (Brief summary of the provided job description, if any)",
    "matched_skills": [
      "string (Skill from resume that matches job requirement)"
    ],
    "missing_skills": [
      "string (Skill required by job but not prominent in resume)"
    ],
    "match_percentage": "integer (Estimated percentage of match, 0-100, based on skills and experience to the job scope. Provide 0 if no job scope is given.)",
    "match_summary": "string (Overall summary of how well the candidate's skills and experience match the job requirements. State if no job scope was provided.)"
  },
  "career_path_suggestions": {
    "based_on_resume_profile": [
      "string (Suggested career path 1 based purely on resume skills/experience)",
      "string (Suggested career path 2 based purely on resume skills/experience)"
    ],
    "general_development_advice": [
      "string (Actionable step 1 for general career development, e.g., 'Gain certification in X')",
      "string (Actionable step 2 for general career development, e.g., 'Seek projects involving Y')"
    ]
  }
}
"""
    system_prompt_content = f"""You are a helpful HR assistant and career advisor.
Analyze the following resume text. If a job scope is also provided, analyze the resume in conjunction with it.
Your analysis should include:
1. A comprehensive summary of the resume (skills, experience, education).
2. Key strengths and potential areas for improvement for the candidate.
3. A scoring for the candidate's resume.
4. If a job scope is provided:
    - Perform a skill-to-job matching analysis: summarize the job description, list matched skills, list missing skills, provide a match percentage, and an overall match summary.
    - If no job scope is provided, indicate this clearly in the skill_job_match section (e.g., 0% match, "No job scope provided for matching").
5. Career Path Suggestions:
    - Suggest potential career paths based on the candidate's overall profile from the resume.
    - Provide general actionable advice for career development.
Return ALL information STRICTLY in the following JSON format. Do not include any text outside of the JSON structure.
Ensure all string values in the JSON are properly quoted.
JSON Format to use:
{json_format_description}
"""
    user_content = f"Resume Text:\n{resume_text_content}"
    if job_scope_text and job_scope_text.strip():
        user_content += f"\n\nJob Scope:\n{job_scope_text}\nJob Scope Provided: Yes"
    else:
        user_content += "\n\nJob Scope:\nNot Provided"

    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": user_content},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error during AI analysis: {e}")
        raise Exception(f"AI analysis failed: {e}")


def generate_html_report(data):
    """
    Generates an HTML report from the parsed AI analysis data.
    """
    if not isinstance(data, dict):
        return "<p>Error: Invalid data format for HTML report.</p>"

    def create_list_items(items):
        if not items or not isinstance(items, list):
            return "<li>N/A</li>"
        return "".join(f"<li>{item}</li>" for item in items)

    def create_experience_projects_html(entries, entry_type="experience"):
        html_str = ""
        if not entries or not isinstance(entries, list):
            return "<p>N/A</p>"
        for entry in entries:
            title_key = "role" if entry_type == "experience" else "title"
            company_desc_key = (
                "company" if entry_type == "experience" else "description"
            )

            html_str += "<div class='entry'>"
            html_str += f"<h4>{entry.get(title_key, 'N/A')} {'at ' + entry.get(company_desc_key, 'N/A') if entry_type == 'experience' and entry.get(company_desc_key) else ''}</h4>"
            if entry_type == "projects":
                html_str += (
                    f"<p><em>Description:</em> {entry.get(company_desc_key, 'N/A')}</p>"
                )
            if entry_type == "experience" and entry.get("responsibilities"):
                html_str += "<h5>Responsibilities:</h5><ul>"
                html_str += create_list_items(entry.get("responsibilities"))
                html_str += "</ul>"
            html_str += "</div>"
        return html_str

    summary = data.get("summary", {})
    contact = summary.get("contact", {})
    education = summary.get("education", {})

    strengths = data.get("strengths", [])
    areas_for_improvement = data.get("areas_for_improvement", [])
    scoring = data.get("scoring", {})
    skill_job_match = data.get("skill_job_match", {})
    career_suggestions = data.get("career_path_suggestions", {})

    # Determine color class for match percentage
    match_percentage_value = skill_job_match.get("match_percentage", 0)
    percentage_color_class = "percentage-low"  # Default to low
    if isinstance(match_percentage_value, (int, float)):
        if match_percentage_value >= 67:
            percentage_color_class = "percentage-high"
        elif match_percentage_value >= 34:
            percentage_color_class = "percentage-medium"

    # Determine color class for overall score
    overall_score_value = scoring.get("overall_score", 0)
    overall_score_color_class = "percentage-low"  # Default to low
    if isinstance(overall_score_value, (int, float)):
        if overall_score_value >= 7.0:
            overall_score_color_class = "percentage-high"
        elif overall_score_value >= 4.0:
            overall_score_color_class = "percentage-medium"

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Analysis Report for {summary.get('name', 'Candidate')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4; color: #333; }}
        .container {{ max-width: 900px; margin: 20px auto; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #2c3e50; }}
        h1 {{ text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ background-color: #ecf0f1; padding: 10px; border-left: 5px solid #3498db; margin-top: 30px; }}
        h3 {{ color: #3498db; margin-top: 20px; }}
        ul {{ list-style-type: disc; margin-left: 20px; padding-left: 0; }}
        li {{ margin-bottom: 5px; }}
        .section {{ margin-bottom: 20px; }}
        .entry {{ background-color: #f9f9f9; border: 1px solid #eee; padding: 15px; margin-bottom: 10px; border-radius: 5px; }}
        .entry h4 {{ margin-top: 0; color: #2980b9; }}
        .scoring-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top:10px;}}
        .score-item {{ background-color: #eafaf1; padding: 10px; border-radius: 4px; text-align:center; }}
        .score-item strong {{ display:block; font-size: 1.1em; color: #27ae60; }}
        .match-percentage-text {{ font-size: 1.5em; /* Increased font size */ font-weight: bold; text-align: center; margin: 10px 0; }}
        .overall-score-text {{ font-size: 1.6em; /* Increased font size */ font-weight: bold; text-align: center; margin: 15px 0; }}
        .percentage-low {{ color: #e74c3c; /* Red */ }}
        .percentage-medium {{ color: #f39c12; /* Orange */ }}
        .percentage-high {{ color: #2ecc71; /* Green */ }}
        .contact-info p, .education-info p {{ margin: 5px 0; }}
        .skills-list li {{ background-color: #e0f7fa; margin-right: 5px; margin-bottom:5px; padding: 5px 10px; border-radius: 15px; display: inline-block; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Resume Analysis Report: {summary.get('name', 'Candidate')}</h1>
        <div class="section" id="summary">
            <h2>Candidate Summary</h2>
            <h3>Contact Information</h3>
            <div class="contact-info">
                <p><strong>Phone:</strong> {contact.get('phone', 'N/A')}</p>
                <p><strong>Email:</strong> {contact.get('email', 'N/A')}</p>
                <p><strong>Location:</strong> {contact.get('location', 'N/A')}</p>
            </div>
            <h3>Education</h3>
            <div class="education-info">
                <p><strong>Degree:</strong> {education.get('degree', 'N/A')}</p>
                <p><strong>Institution:</strong> {education.get('institution', 'N/A')}</p>
                <p><strong>Duration:</strong> {education.get('duration', 'N/A')}</p>
            </div>
            <h3>Skills</h3>
            <ul class="skills-list">{create_list_items(summary.get('skills', []))}</ul>
            <h3>Experience</h3>
            {create_experience_projects_html(summary.get('experience', []), 'experience')}
            <h3>Projects</h3>
            {create_experience_projects_html(summary.get('projects', []), 'projects')}
        </div>
        <div class="section" id="strengths">
            <h2>Strengths</h2>
            <ul>{create_list_items(strengths)}</ul>
        </div>
        <div class="section" id="areas_for_improvement">
            <h2>Areas for Improvement</h2>
            <ul>{create_list_items(areas_for_improvement)}</ul>
        </div>
        <div class="section" id="scoring">
            <h2>Candidate Scoring</h2>
            <div class="scoring-grid">
                <div class="score-item">Technical Skills: <strong>{scoring.get('technical_skills', 'N/A')}/10</strong></div>
                <div class="score-item">Work Experience: <strong>{scoring.get('work_experience', 'N/A')}/10</strong></div>
                <div class="score-item">Education: <strong>{scoring.get('education', 'N/A')}/10</strong></div>
                <div class="score-item">Soft Skills: <strong>{scoring.get('soft_skills', 'N/A')}/10</strong></div>
            </div>
            <p class="overall-score-text {overall_score_color_class}">Overall Score: {overall_score_value}/10</p>
        </div>
        <div class="section" id="skill_job_match">
            <h2>Skill-to-Job Match Analysis</h2>
            <h3>Job Description Summary</h3>
            <p>{skill_job_match.get('job_description_summary', 'N/A')}</p>
            <h3>Matched Skills</h3>
            <ul>{create_list_items(skill_job_match.get('matched_skills', []))}</ul>
            <h3>Missing Skills (from Job Scope)</h3>
            <ul>{create_list_items(skill_job_match.get('missing_skills', []))}</ul>
            <p class="match-percentage-text {percentage_color_class}">Match Percentage: {match_percentage_value}%</p>
            <h3>Match Summary</h3>
            <p>{skill_job_match.get('match_summary', 'N/A')}</p>
        </div>
        <div class="section" id="career_suggestions">
            <h2>Career Path Suggestions</h2>
            <h3>Based on Resume Profile</h3>
            <ul>{create_list_items(career_suggestions.get('based_on_resume_profile', []))}</ul>
            <h3>General Development Advice</h3>
            <ul>{create_list_items(career_suggestions.get('general_development_advice', []))}</ul>
        </div>
    </div>
</body>
</html>
    """
    return html_content


# --- Flask Routes ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_resume_route():
    if "resume_pdf" not in request.files:
        return jsonify(message="No resume file selected."), 400

    resume_file = request.files["resume_pdf"]
    job_scope_text = request.form.get("job_scope", "")

    if resume_file.filename == "":
        return jsonify(message="No resume file selected."), 400

    if resume_file and resume_file.filename.endswith(".pdf"):
        filename = "uploaded_resume.pdf"
        resume_filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            resume_file.save(resume_filepath)
            logging.info(f"Resume saved to {resume_filepath}")

            resume_text_content = read_pdf_resume(resume_filepath)
            logging.info(f"Resume text extracted. Length: {len(resume_text_content)}")

            ai_analysis_json_str = analyze_resume_with_ai(
                resume_text_content, job_scope_text
            )
            logging.info("AI analysis received.")

            ai_analysis_data = json.loads(ai_analysis_json_str)
            html_report_content = generate_html_report(ai_analysis_data)
            logging.info("HTML report generated.")

            # Return the HTML content directly
            return html_report_content, 200

        except Exception as e:
            logging.error(f"Error during analysis: {e}", exc_info=True)
            # Clean up uploaded file in case of error
            if os.path.exists(resume_filepath):
                try:
                    os.remove(resume_filepath)
                except Exception as e_del:
                    logging.error(
                        f"Error deleting temp file {resume_filepath}: {e_del}"
                    )
            return jsonify(message=f"An error occurred: {str(e)}"), 500
        finally:
            if os.path.exists(resume_filepath):
                try:
                    os.remove(resume_filepath)
                    pass
                except Exception as e_del:
                    logging.error(
                        f"Error deleting temp file {resume_filepath} in finally: {e_del}"
                    )

    else:
        return jsonify(message="Invalid file type. Please upload a PDF."), 400


if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    if not os.path.exists("templates"):
        os.makedirs("templates")
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True, host="0.0.0.0", port=5000)
