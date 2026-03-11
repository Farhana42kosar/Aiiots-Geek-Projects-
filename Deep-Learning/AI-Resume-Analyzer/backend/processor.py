
import re
# Comprehensive Job Profiles with variations for better matching
JOB_PROFILES = {
    "Full Stack Developer": [
        "react", "node.js", "node", "javascript", "typescript",
        "mongodb", "express", "html", "css", "rest api",
        "microservices", "full stack", "frontend", "backend",
        "git", "agile", "redux", "next.js", "sql"
    ],



    "Backend Developer": [
        "python", "django", "flask", "fastapi",
        "java", "spring boot", "microservices",
        "docker", "kubernetes", "postgresql",
        "mysql", "rest api", "api development",
        "server-side", "authentication", "jwt",
        "aws", "azure", "cloud"
    ],

    "Frontend Developer": [
        "react", "vue", "angular",
        "javascript", "typescript",
        "html5", "css3", "tailwind",
        "bootstrap", "responsive design",
        "figma", "ui", "ux", "redux",
        "next.js", "webpack"
    ],

    "Data Scientist": [
        "python", "r", "statistics",
        "machine learning", "deep learning",
        "tensorflow", "pytorch",
        "scikit-learn", "pandas", "numpy",
        "data visualization", "tableau",
        "power bi", "sql", "predictive modeling",
        "nlp", "regression", "classification"
    ],

    "AI/ML Engineer": [
        "python", "tensorflow", "pytorch",
        "machine learning", "deep learning",
        "model deployment", "nlp",
        "computer vision", "neural networks",
        "mlops", "docker", "kubernetes",
        "aws", "gcp", "api integration"
    ],

    "Sales Manager": [
        "sales strategy", "revenue growth",
        "lead generation", "crm",
        "sales pipeline", "negotiation",
        "client acquisition", "forecasting",
        "business development", "b2b", "b2c",
        "cold calling", "upselling",
        "target achievement"
    ],

    "Business Development": [
        "market research", "lead generation",
        "client acquisition", "partnership",
        "sales growth", "strategic planning",
        "networking", "proposal writing",
        "relationship management"
    ],

    "HR Manager": [
        "recruitment", "talent acquisition",
        "payroll", "employee relations",
        "compliance", "policy development",
        "performance management",
        "training", "benefits administration"
    ],

    "Accountant": [
        "gst", "income tax", "audit",
        "financial reporting", "budgeting",
        "forecasting", "p&l", "ledger",
        "reconciliation", "tally",
        "quickbooks", "compliance"
    ],

    # ---------------- IT & DEVELOPMENT ----------------
    "Python Developer": [
        "python", "django", "flask", "fastapi",
        "api", "rest", "backend", "sql",
        "pandas", "numpy", "git"
    ],

    "React Developer": [
        "react", "redux", "javascript",
        "typescript", "html", "css",
        "frontend", "next.js", "ui"
    ],

    "Java Developer": [
        "java", "spring", "spring boot",
        "hibernate", "microservices",
        "rest api", "maven"
    ],

    "DevOps Engineer": [
        "docker", "kubernetes", "aws",
        "azure", "ci cd", "jenkins",
        "linux", "cloud", "terraform"
    ],

    "Data Scientist": [
        "python", "machine learning",
        "deep learning", "pandas",
        "numpy", "tensorflow",
        "pytorch", "statistics",
        "nlp", "sql"
    ],

    # ---------------- HR ----------------
    "HR Manager": [
        "recruitment", "payroll",
        "employee relations",
        "compliance", "policy",
        "training"
    ],

    "Recruiter": [
        "sourcing", "talent acquisition",
        "interviewing", "screening",
        "hiring", "ats"
    ],

    # ---------------- ACCOUNTING ----------------
    "Accountant": [
        "gst", "tax", "audit",
        "ledger", "reconciliation",
        "tally", "quickbooks"
    ],

    "Tax Accountant": [
        "income tax", "gst filing",
        "returns", "compliance",
        "audit"
    ],

    "Financial Analyst": [
        "budgeting", "forecasting",
        "p&l", "financial modeling",
        "excel"
    ],

    # ---------------- SALES ----------------
    "Sales Manager": [
        "sales", "crm", "lead generation",
        "target", "revenue",
        "negotiation"
    ],

    "Business Development Executive": [
        "client acquisition",
        "market research",
        "partnership",
        "b2b", "b2c"
    ],

    # ---------------- MANAGEMENT ----------------
    "Operations Manager": [
        "operations", "process improvement",
        "logistics", "planning",
        "resource management"
    ]
}
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


#  JOB RECOMMENDATION FUNCTION

def get_job_recommendations(resume_text, predicted_domain=None):
    """
    Returns top 3 recommended jobs based on keyword matching.
    Filters out weak matches.
    """
    text = clean_text(resume_text)
    recommendations = []

    for job, keywords in JOB_PROFILES.items():
        match_count = sum(1 for word in keywords if word in text)
        match_percentage = (match_count / len(keywords)) * 100

        # Filter out very low matches
        if match_percentage > 10:
            recommendations.append({
                "job": job,
                "match": round(match_percentage, 2)
            })

    # Sort descending
    recommendations = sorted(
        recommendations,
        key=lambda x: x["match"],
        reverse=True
    )

    return recommendations[:3]


#  IMPROVED ATS SCORE

def calculate_ats_score(resume_text, top_match_percentage):
    """
    Improved ATS scoring logic
    """

    score = 0
    text = clean_text(resume_text)

    # 1️⃣ STRUCTURE CHECK (30 pts)
     

    section_keywords = {
        "experience": ["experience", "work history", "employment"],
        "education": ["education", "academic"],
        "skills": ["skills", "technical skills"],
        "projects": ["projects", "portfolio"],
        "contact": ["contact", "email", "phone"],
        "summary": ["summary", "profile", "objective"],
        "certification": ["certification", "certifications", "licensed"]
    }

    section_hits = 0

    for keywords in section_keywords.values():
        if any(keyword in text for keyword in keywords):
            section_hits += 1

    structure_score = (section_hits / len(section_keywords)) * 30
    score += structure_score

    # 2️⃣ SKILL MATCH (60 pts)
    

    skill_score = (top_match_percentage / 100) * 60
    score += skill_score

    # 3️⃣ BONUS QUALITY CHECK (10 pts)

    strong_indicators = [
        "achieved", "improved", "increased",
        "reduced", "managed", "developed",
        "led", "implemented"
    ]

    bonus_hits = sum(1 for word in strong_indicators if word in text)

    if bonus_hits >= 3:
        score += 5

    if len(text.split()) > 400:
        score += 5

    return round(min(score, 100), 2)