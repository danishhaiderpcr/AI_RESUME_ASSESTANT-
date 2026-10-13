# AI_RESUME_ASSESTANT-
# AI Resume ATS Checker

A Streamlit app that scores a resume for Applicant Tracking System (ATS) friendliness and tells you exactly what to improve. It uses Google's Gemini Flash model.

## Features

- Upload a resume as **PDF or DOCX**
- Optional **job description** box for keyword and job-match scoring
- Overall **ATS score (0-100)** plus category scores (formatting, keywords, impact, structure, clarity, job match)
- Prioritized **improvements** (High / Medium / Low) with concrete fixes
- **Missing keywords** list and **before/after bullet rewrites**
- Download the full report as Markdown
- Warns when a file looks like a scanned image (an ATS cannot read those either)

## Run locally

1. Install Python 3.10 or newer.
2. Get a free Gemini API key at https://aistudio.google.com/apikey
3. In a terminal, inside this folder:

   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate     Mac/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```

4. Provide your key in one of two ways:
   - Paste it in the app's sidebar when it runs, **or**
   - Create `.streamlit/secrets.toml` containing:

     ```toml
     GEMINI_API_KEY = "your-key-here"
     ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

## Configuration

| Setting | Where | Notes |
|---|---|---|
| `GEMINI_API_KEY` | Streamlit secrets, environment variable, or sidebar | Never commit it to GitHub |
| Model name | Sidebar field (default `gemini-2.5-flash`) | If you see a "model not found" error, try `gemini-flash-latest` or another current Flash model from https://ai.google.dev/gemini-api/docs/models |

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (`app.py`, `requirements.txt`, `README.md` at the repo root).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add: `GEMINI_API_KEY = "your-key-here"`
5. Click **Deploy**.

## Privacy

Resume text is sent to the Gemini API for analysis. The app itself does not store uploaded files or results. Do not upload documents you are not comfortable sending to a third-party AI service.

## Limitations

- The score is an AI estimate, not the output of a real employer ATS. Use it as guidance.
- Scanned or image-only resumes cannot be read (no OCR).
- Very long resumes are truncated to about 30,000 characters.
- The free Gemini tier has rate limits; the app retries automatically on temporary errors.

## Project structure

```
app.py             # Streamlit app
requirements.txt   # Python dependencies
README.md          # This file
```
