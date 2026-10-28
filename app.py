from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="JobCompass AI | Career dashboard",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path(__file__).parent / "jobs.csv"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#172b4d; --muted:#68778d; --teal:#087e8b; --line:#e5ebf2; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:#f6f8fb; color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div { padding-top:1.5rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink); letter-spacing:-.035em; }
    h1 { font-size:2.35rem !important; }
    h2 { font-size:1.45rem !important; }
    .hero { background:linear-gradient(115deg,#102c46 0%,#155e70 100%); padding:2rem 2.15rem;
            border-radius:18px; color:white; margin:.2rem 0 1.5rem; box-shadow:0 12px 28px #14324a1c; }
    .hero h1 { color:white !important; margin:0 0 .35rem; font-size:2.15rem !important; }
    .hero p { color:#d9e9ee; margin:0; font-size:1.02rem; }
    .eyebrow { color:#91d6d5; font-weight:700; font-size:.76rem; text-transform:uppercase; letter-spacing:.12em; margin-bottom:.7rem; }
    .section-kicker { color:var(--teal); font-size:.74rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    [data-testid="stMetric"] { background:white; border:1px solid var(--line); padding:1rem 1.2rem; border-radius:14px; box-shadow:0 3px 12px #172b4d08; }
    [data-testid="stMetricLabel"] { color:var(--muted); }
    [data-testid="stMetricValue"] { color:var(--ink); font-family:'Manrope',sans-serif; }
    .job-meta { color:var(--muted); font-size:.9rem; margin-top:-.5rem; }
    .skill-chip { display:inline-block; padding:.3rem .65rem; margin:.15rem .2rem .15rem 0; border-radius:999px; font-size:.78rem; font-weight:600; }
    .matched { color:#08744e; background:#e5f6ee; }
    .missing { color:#9a5b05; background:#fff3d9; }
    .stButton > button { border-radius:9px; font-weight:600; }
    .stButton > button[kind="primary"] { background:#087e8b; border-color:#087e8b; }
    div[data-testid="stTabs"] button { font-weight:600; }
    .notice { color:#68778d; background:#fff; border:1px solid var(--line); border-radius:12px; padding:.8rem 1rem; font-size:.87rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_jobs():
    return pd.read_csv(DATA_FILE).fillna("")


def normalize_skills(text):
    return sorted({part.strip().lower() for part in text.replace(";", ",").split(",") if part.strip()})


def match_jobs(user_skills, jobs):
    results = jobs.copy()
    results["required_skills"] = results["skills"].apply(normalize_skills)
    if user_skills:
        candidate_text = " ".join(user_skills)
        job_texts = (results["skills"].astype(str) + " " + results["description"].astype(str)).tolist()
        matrix = TfidfVectorizer(stop_words="english").fit_transform([candidate_text] + job_texts)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        results["match_score"] = (scores * 100).round().astype(int)
    else:
        results["match_score"] = 0
    skill_set = set(user_skills)
    results["missing_skills"] = results["required_skills"].apply(lambda required: sorted(set(required) - skill_set))
    results["matched_skills"] = results["required_skills"].apply(lambda required: sorted(set(required) & skill_set))
    return results.sort_values("match_score", ascending=False)


jobs = load_jobs()
if "applications" not in st.session_state:
    st.session_state.applications = []

with st.sidebar:
    st.markdown("## 🧭 JobCompass")
    st.caption("Your focused workspace for the next career move")
    st.divider()
    st.markdown("### Your profile")
    skills_text = st.text_area(
        "Skills",
        value=st.session_state.get("skills_text", "Python, SQL, machine learning"),
        placeholder="Python, SQL, communication",
        help="Add skills separated by commas. You can also use semicolons.",
        height=100,
    )
    st.session_state.skills_text = skills_text
    user_skills = normalize_skills(skills_text)
    st.caption(f"{len(user_skills)} skills in your profile")
    preferred_role = st.selectbox(
        "Role focus",
        ["All roles"] + sorted(jobs["role"].unique().tolist()),
    )
    st.divider()
    st.caption("Demo dataset · Sample listings only")

applications = st.session_state.applications
results = match_jobs(user_skills, jobs)
if preferred_role != "All roles":
    results = results[results["role"] == preferred_role]

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">A clearer path to your next role</div>
      <h1>Find work that fits your skills.</h1>
      <p>Explore tailored opportunities, see where you stand, and keep every application moving.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_find, tab_tracker, tab_about = st.tabs(["✦  Discover roles", "▤  Application tracker", "ⓘ  About"])

with tab_find:
    top_match = int(results["match_score"].max()) if not results.empty else 0
    metric_cols = st.columns(3)
    metric_cols[0].metric("Opportunities", len(results), "In your current view")
    metric_cols[1].metric("Top skill match", f"{top_match}%", "Text similarity")
    metric_cols[2].metric("Saved roles", len(applications), "In your tracker")

    st.write("")
    heading, sort_col = st.columns([3, 1])
    with heading:
        st.markdown('<span class="section-kicker">Curated for you</span>', unsafe_allow_html=True)
        st.subheader("Recommended opportunities")
    with sort_col:
        sort_by = st.selectbox("Sort by", ["Best match", "Role A–Z"], label_visibility="collapsed")
    if sort_by == "Role A–Z":
        results = results.sort_values("role")

    if not user_skills:
        st.info("Add at least one skill in your profile to see personalized recommendations.")
    elif results.empty:
        st.info("No roles match this filter. Try choosing All roles.")
    else:
        for _, job in results.iterrows():
            with st.container(border=True):
                top = st.columns([5, 1.15])
                with top[0]:
                    st.markdown(f"### {job['role']}")
                    st.markdown(f"<div class='job-meta'>🏢 {job['company']} &nbsp; · &nbsp; 📍 {job['location']}</div>", unsafe_allow_html=True)
                with top[1]:
                    st.metric("Skill match", f"{job['match_score']}%")
                st.write(job["description"])
                matched_html = "".join(f"<span class='skill-chip matched'>✓ {skill}</span>" for skill in job["matched_skills"])
                missing_html = "".join(f"<span class='skill-chip missing'>＋ {skill}</span>" for skill in job["missing_skills"])
                st.markdown("**Skills you bring**" + (f"<br>{matched_html}" if matched_html else "  "), unsafe_allow_html=True)
                st.markdown("**Skills to build**" + (f"<br>{missing_html}" if missing_html else "  "), unsafe_allow_html=True)
                with st.expander("Listing details & verification"):
                    st.caption(job["verification_note"])
                    st.caption("Sample listing · Confirm details with the employer before applying.")
                existing = any(item["Job ID"] == job["job_id"] for item in applications)
                if existing:
                    st.button("✓  Saved to tracker", key=f"saved_{job['job_id']}", disabled=True)
                elif st.button("＋  Add to tracker", key=f"add_{job['job_id']}", type="primary"):
                    applications.append({
                        "Job ID": job["job_id"], "Role": job["role"], "Company": job["company"],
                        "Location": job["location"], "Status": "Interested",
                    })
                    st.session_state.applications = applications
                    st.rerun()

with tab_tracker:
    st.markdown('<span class="section-kicker">Stay on top of your search</span>', unsafe_allow_html=True)
    st.subheader("Application tracker")
    applications = st.session_state.applications
    if not applications:
        st.info("Your tracker is ready when you are. Save a role from Discover roles to get started.")
    else:
        tracker = pd.DataFrame(applications)
        statuses = tracker["Status"].value_counts()
        track_cols = st.columns(4)
        for i, status in enumerate(["Interested", "Applied", "Interview", "Offer"]):
            track_cols[i].metric(status, int(statuses.get(status, 0)))
        edited = st.data_editor(
            tracker,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Status": st.column_config.SelectboxColumn(
                    "Status", options=["Interested", "Applied", "Interview", "Rejected", "Offer"], required=True
                )
            },
            disabled=["Job ID", "Role", "Company", "Location"],
            key="tracker_editor",
        )
        st.session_state.applications = edited.to_dict("records")
        download_col, clear_col = st.columns([1, 5])
        with download_col:
            st.download_button(
                "↓  Export CSV", edited.to_csv(index=False).encode("utf-8"),
                file_name="job_applications.csv", mime="text/csv", type="primary",
            )
        with clear_col:
            if st.button("Clear tracker", help="Remove all saved roles"):
                st.session_state.applications = []
                st.rerun()

with tab_about:
    st.markdown('<span class="section-kicker">Transparent by design</span>', unsafe_allow_html=True)
    st.subheader("How matching works")
    st.write("JobCompass compares the words in your skill profile with each role’s listed skills and description, then ranks opportunities by text similarity.")
    info_cols = st.columns(3)
    info_cols[0].markdown("#### 01 · Compare\nTF-IDF turns profile and job text into weighted terms.")
    info_cols[1].markdown("#### 02 · Explain\nListed requirements are compared with your profile to show shared and missing skills.")
    info_cols[2].markdown("#### 03 · Organize\nSave roles, update application stages, and export your tracker as a CSV.")
    st.markdown(
        "<div class='notice'><strong>About these recommendations</strong><br>Listings come from a small sample dataset, not a live job feed. Similarity is a text comparison, not a hiring probability or a guarantee. Verify the employer and listing independently.</div>",
        unsafe_allow_html=True,
    )

st.divider()
st.caption("JobCompass AI · Career search, made more focused")
