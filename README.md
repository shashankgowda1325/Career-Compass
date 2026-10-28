# JobCompass AI

**AI-powered job matching, skill-gap analysis, and application tracking**

## Problem statement
Job seekers often have to search across many listings, judge which roles suit their skills, and keep application progress in separate places. That makes it easy to miss relevant opportunities or lose track of next steps.

**JobCompass AI brings those steps together:** it compares a candidate's skills with sample job descriptions, shows matching and missing skills, ranks roles by text similarity, and provides a simple application tracker.

## Features
- TF-IDF and cosine-similarity based text matching between candidate skills and job listings.
- Skill gap report: matched and missing listed skills.
- Filter recommendations by role.
- Application tracker with editable status.
- Export the tracker to CSV.
- Notes encouraging manual verification of job information.

## Technology stack
- Python
- Streamlit (local web interface)
- pandas (job dataset and tracker)
- scikit-learn (TF-IDF and cosine similarity)

## System architecture
```text
Candidate enters skills
        |
        v
Text preprocessing (normalize skills)
        |
        v
TF-IDF vectorization (scikit-learn)
        |
        v
Cosine similarity scoring
        |
        +------> Ranked job recommendations
        |
        +------> Required vs. candidate skills (skill gaps)
        |
        v
Streamlit UI ----> Application tracker ----> CSV export
        ^
        |
   Local jobs.csv dataset
```

## Requirements
- Python 3.10 or newer recommended
- Internet is needed only to install packages for the first time; the app itself runs locally after installation.
- No API key required.

## Installation and execution (Windows)
1. Install Python from https://www.python.org/downloads/ and enable **Add Python to PATH**.
2. Download or clone this repository.
3. Open Command Prompt in the project folder.
4. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
5. Activate it:
   ```bash
   .venv\Scripts\activate
   ```
6. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
7. Start the local app:
   ```bash
   python -m streamlit run app.py
   ```
8. Open the local URL shown in the terminal, usually http://localhost:8501.

## Deploy to Streamlit Community Cloud

The app is ready to deploy from a GitHub repository. Keep `app.py`, `jobs.csv`, and `requirements.txt` in the repository root so the app can find its data and dependencies.

1. Push this project to a GitHub repository. Include `app.py`, `jobs.csv`, `requirements.txt`, and `README.md`.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and grant it access to the repository.
3. Select **Create app**, then choose the repository and branch.
4. Set the app file path to `app.py` and choose a URL name if you want one.
5. Select **Deploy**. Community Cloud installs the packages from `requirements.txt` and starts the app.
6. If startup fails, open the app's deployment logs and check the first error. Confirm that `jobs.csv` is committed beside `app.py` and that the selected Python/dependency versions are compatible.

There are no API keys or secrets to configure for this prototype. The tracker currently lives in the active Streamlit session, so it is intended for a demo and does not act as permanent, cross-device storage. Listings are sample data rather than live vacancies.

For the current deployment walkthrough, see the [official Streamlit Community Cloud guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Demo script
1. Enter `Python, SQL, machine learning`.
2. Show the ranked roles and similarity scores.
3. Open a job card and explain matched vs. missing skills.
4. Click **Add to tracker** on two jobs.
5. Open **Application tracker**, change one status to Applied and another to Interview.
6. Download the CSV.
7. Explain that sample listings are used and scores are similarity indicators, not hiring probabilities.

## GitHub checklist
- Create a public repository named `JobCompass-AI`.
- Upload `app.py`, `jobs.csv`, `requirements.txt`, `README.md`, and your presentation.
- Add every teammate as a collaborator (Repository → Settings → Collaborators; exact menu labels may vary).
- Verify the README installation steps on another machine if possible.
- Do not commit passwords, API keys, or personal candidate data.

## Limitations and future scope
- Sample job listings only; no live scraping or verified external feed.
- Text similarity can miss synonyms and context.
- Future work: resume upload and parsing, verified job API integration, duplicate detection, explainable match reasons, and personalized learning recommendations.

## Ethical and accuracy note
The match percentage is a text-similarity score, not model accuracy, hiring probability, or a guarantee. Listings in this demo are sample data. Always verify employer details and the original listing before sharing documents or paying any fee.
