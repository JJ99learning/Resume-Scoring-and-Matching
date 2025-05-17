# Resume Analyzer

## Description
Resume Analyzer is a web application designed to analyze PDF resumes, optionally compare them against a job description, and provide a detailed, AI-powered report. The analysis covers various aspects including a candidate summary, skills, experience, strengths, areas for improvement, scoring, and career suggestions. The AI analysis is performed using the Qwen model via the Dashscope API.

## Features
*   Upload PDF resumes for analysis.
*   Optionally provide a job description for a comparative skill-to-job match.
*   AI-driven analysis of resume content.
*   Generates a comprehensive HTML report which includes:
    *   Candidate Summary (Contact Information, Education, Skills, Experience, Projects)
    *   Identified Strengths
    *   Potential Areas for Improvement
    *   Candidate Scoring (Technical Skills, Work Experience, Education, Soft Skills, Overall Score)
    *   Skill-to-Job Match Analysis (if a job scope is provided)
    *   Career Path Suggestions & General Development Advice
*   User-friendly web interface for easy interaction.

## Technologies Used
*   **Backend:** Python, Flask
*   **AI Model Integration:** Qwen (via Dashscope API, compatible with OpenAI SDK)
*   **Frontend:** HTML, CSS, JavaScript
*   **Package Management:** uv

## Setup and Installation
### Prerequisites
*   Python 3.11 (as specified in pyproject.toml)
*   uv (for managing dependencies)
*   API  Key for Qwen

   
1.  **Environment Variables:**
    The application requires an API key for the Qwen service. Set the following environment variable. You can create a `.env` file in the project root and add it there.
    *   `QWEN_API_KEY`: Your API key for the Dashscope/Qwen service.
        (As used in <mcfile name="app.py" path="c:\Users\JJ\Documents\Hackathon\app.py"></mcfile>: `os.environ.get("QWEN_API_KEY")`)

## Usage
1.  **Run the Flask application:**
    Ensure your virtual environment is activated and the `QWEN_API_KEY` is set.
    *   If using uv (uv is a package manager for Python):
        ```
        uv run app.py
        ```
    *   Alternatively (standard Python execution):
        ```
        python app.py
        ```
    The application will typically start on `http://127.0.0.1:5000/`.

2.  **Access the application:**
    Open your web browser and navigate to `http://127.0.0.1:5000/`.


3.  **Analyze a resume:**
    *   On the main page, click "Select PDF Resume" to choose a resume file from your computer.
    *   Optionally, paste the job description text into the "Paste Job Scope/Description" text area.
    *   Click the "Analyze Resume" button.
    *   Wait for the analysis to complete. The generated HTML report will be displayed directly on the page within an iframe.

