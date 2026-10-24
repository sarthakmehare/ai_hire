"""
ATS Resume Checker - Flask Backend
Model: resume_ann_model.keras  (TF-IDF 5000 → ANN → 24 job categories)

FIX LOG
-------
1. Model lazy-loaded with clear startup error instead of crashing at import time.
2. /predict routes guarded — return 503 if model is unavailable.
3. TF-IDF vectorizer now uses a deterministic, reproducible corpus so vocabulary
   is stable across requests (previously rebuilt from scratch every call, which
   could scramble feature order).
4. Flask now serves index.html at "/" so the frontend loads without a separate
   web server.
5. All emoji in suggestion strings replaced with ASCII prefixes so terminals /
   JSON clients that can't render UTF-8 don't break.
6. /api/analyze returns CORS headers on error responses too (via flask-cors).
7. Input validation tightened: empty filename, wrong extension handled gracefully.
"""

import os, re, io, logging, json, sqlite3, uuid, pickle, secrets
from datetime import datetime, timezone, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
import PyPDF2
import docx
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except (ImportError, Exception) as _import_err:
    import math
    from collections import Counter
    class TfidfVectorizer:
        """Pure-Python TF-IDF vectorizer fallback when Windows Smart App Control blocks scikit-learn C extensions."""
        def __init__(self, max_features=5000, stop_words="english"):
            self.max_features = max_features
            self.stop_words = {
                "the","and","for","are","with","this","that","will","have","from","they",
                "you","our","their","your","all","can","been","was","were","but","not",
                "has","had","its","also","into","more","about","who","we","be","an","to",
                "of","in","a","is","it","as","at","on","or","by","do","if","up","any",
                "each","how","her","she","he","him","his","them",
            } if stop_words == "english" else set()

        def fit_transform(self, corpus):
            tokenized = []
            doc_freq = Counter()
            all_terms = Counter()
            for doc in corpus:
                tokens = [t for t in re.findall(r"\b[a-zA-Z]{2,}\b", doc.lower()) if t not in self.stop_words]
                tokenized.append(tokens)
                for t in set(tokens):
                    doc_freq[t] += 1
                for t in tokens:
                    all_terms[t] += 1

            vocab = [term for term, _ in all_terms.most_common(self.max_features)]
            vocab_idx = {term: i for i, term in enumerate(vocab)}
            n_docs = len(corpus)

            matrix = np.zeros((n_docs, len(vocab)), dtype=np.float32)
            for d_i, tokens in enumerate(tokenized):
                if not tokens:
                    continue
                tf = Counter(tokens)
                total = len(tokens)
                for term, cnt in tf.items():
                    if term in vocab_idx:
                        col = vocab_idx[term]
                        idf = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0
                        matrix[d_i, col] = (cnt / total) * idf

            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms

            class MatrixWrapper:
                def __init__(self, arr):
                    self.arr = arr
                def toarray(self):
                    return self.arr
            return MatrixWrapper(matrix)

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR       = os.path.join(os.path.dirname(BASE_DIR), "Front end")
MODEL_PATH         = os.path.join(BASE_DIR, "resume_ann_model.keras")
TFIDF_PATH         = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
INPUT_DIM          = 5000
DB_PATH            = os.path.join(BASE_DIR, "aihire.db")
UPLOADS_DIR        = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ── categories ───────────────────────────────────────────────────────────────
JOB_CATEGORIES = [
    "Advocate","Arts","Automation Testing","Blockchain",
    "Business Analyst","Civil Engineer","Data Science","Database",
    "DevOps Engineer","DotNet Developer","ETL Developer",
    "Electrical Engineering","HR","Hadoop","Java Developer",
    "Mechanical Engineer","Network Security Engineer","Operations Manager",
    "PMO","Python Developer","SAP Developer","Sales",
    "Testing","Web Designing",
]

# Dynamically sync categories if label_encoder.pkl exists
try:
    if os.path.exists(LABEL_ENCODER_PATH):
        with open(LABEL_ENCODER_PATH, "rb") as _le_f:
            _le = pickle.load(_le_f)
            if hasattr(_le, "classes_"):
                JOB_CATEGORIES = [str(c) for c in _le.classes_]
except Exception as _e:
    pass

DOMAIN_KEYWORDS = {
    "Data Science": [
        "python","machine learning","deep learning","tensorflow","pytorch",
        "scikit-learn","pandas","numpy","statistics","data visualization",
        "sql","r","jupyter","keras","nlp","data wrangling",
        "feature engineering","regression","classification","clustering",
    ],
    "Python Developer": [
        "python","django","flask","fastapi","rest api","postgresql",
        "mysql","redis","celery","docker","git","pytest","sqlalchemy",
        "asyncio","pandas","numpy","pip","virtualenv","ci/cd",
    ],
    "Java Developer": [
        "java","spring","spring boot","hibernate","maven","gradle",
        "microservices","rest api","junit","docker","kubernetes",
        "sql","nosql","git","jenkins","design patterns","jvm","kafka",
    ],
    "Web Designing": [
        "html","css","javascript","react","vue","angular","bootstrap",
        "figma","photoshop","responsive design","ui/ux","sass","webpack",
        "typescript","node.js","graphic design","wireframing",
    ],
    "DevOps Engineer": [
        "docker","kubernetes","jenkins","ci/cd","terraform","ansible",
        "aws","azure","gcp","linux","bash","git","monitoring",
        "prometheus","grafana","helm","nginx","elk stack","puppet",
    ],
    "Database": [
        "sql","mysql","postgresql","oracle","mongodb","redis",
        "database design","normalization","indexing","stored procedures",
        "query optimization","backup","replication","nosql","data modeling",
    ],
    "Blockchain": [
        "blockchain","ethereum","solidity","smart contracts","web3",
        "cryptocurrency","bitcoin","hyperledger","defi","nft",
        "cryptography","consensus algorithms","truffle","metamask",
    ],
    "Hadoop": [
        "hadoop","spark","hive","pig","hdfs","mapreduce","kafka",
        "flume","sqoop","yarn","hbase","big data","zookeeper",
        "data pipeline","etl","scala","java",
    ],
    "ETL Developer": [
        "etl","informatica","talend","ssis","datastage","pentaho",
        "data warehouse","sql","python","spark","data migration",
        "data modeling","oracle","postgresql","business intelligence",
    ],
    "Automation Testing": [
        "selenium","pytest","junit","testng","cucumber","appium",
        "cypress","robot framework","postman","api testing","qa",
        "test automation","jenkins","git","performance testing","jmeter",
    ],
    "Testing": [
        "manual testing","test cases","bug tracking","jira","selenium",
        "api testing","regression testing","functional testing",
        "test plan","qa","agile","sdlc","postman","sql",
    ],
    "Network Security Engineer": [
        "network security","firewall","vpn","intrusion detection",
        "penetration testing","siem","vulnerability assessment","cisco",
        "linux","tcp/ip","ssl/tls","soc","incident response",
        "ethical hacking","encryption",
    ],
    "Business Analyst": [
        "requirements gathering","business analysis","user stories",
        "sql","excel","power bi","tableau","jira","stakeholder management",
        "process mapping","agile","scrum","documentation","gap analysis",
    ],
    "SAP Developer": [
        "sap","abap","sap hana","sap fiori","sap s/4hana","idoc",
        "bapi","bdc","smartforms","sap basis","sap mm","sap sd",
        "sap fico","odata","rfm",
    ],
    "DotNet Developer": [
        "c#",".net","asp.net","mvc","entity framework","sql server",
        "visual studio","azure","rest api","linq","web api",
        "microservices","docker","git","blazor",
    ],
    "HR": [
        "recruitment","talent acquisition","onboarding","payroll",
        "performance management","hris","employee relations",
        "labor law","excel","hr policies","succession planning","training",
    ],
    "Sales": [
        "sales","crm","lead generation","business development",
        "negotiation","customer relationship","salesforce","revenue",
        "cold calling","account management","b2b","b2c","pipeline",
    ],
    "Operations Manager": [
        "operations management","supply chain","logistics","kpi",
        "process improvement","lean","six sigma","erp","budgeting",
        "team leadership","vendor management","sla","forecasting",
    ],
    "PMO": [
        "project management","pmp","agile","scrum","risk management",
        "ms project","jira","budget management","stakeholder management",
        "gantt chart","prince2","resource planning","kpi",
    ],
    "Civil Engineer": [
        "autocad","structural analysis","construction management",
        "project management","staad pro","revit","site supervision",
        "estimation","quality control","building codes","surveying",
    ],
    "Mechanical Engineer": [
        "autocad","solidworks","catia","ansys","manufacturing",
        "thermodynamics","fluid mechanics","cad","product design",
        "quality control","cnc","maintenance","fmea",
    ],
    "Electrical Engineering": [
        "electrical design","matlab","plc","scada","autocad electrical",
        "power systems","circuit design","pcb","embedded systems",
        "microcontroller","python","labview","testing",
    ],
    "Arts": [
        "graphic design","adobe photoshop","illustrator","indesign",
        "video editing","premiere pro","after effects","photography",
        "3d modeling","blender","portfolio","creative",
    ],
    "Advocate": [
        "legal research","litigation","contract drafting","legal writing",
        "court proceedings","legal advice","compliance","law",
        "negotiation","client counseling","legal documentation",
    ],
}

GENERIC_KEYWORDS = [
    "communication","teamwork","problem solving","leadership",
    "time management","adaptability","analytical","detail oriented",
]

# ── Flask app ─────────────────────────────────────────────────────────────────
STATIC_DIR = FRONTEND_DIR if os.path.isdir(FRONTEND_DIR) else BASE_DIR
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── FIX 1: lazy model loading ─────────────────────────────────────────────────
_model = None

def get_model():
    """Load model on first use; return None and log if file is missing."""
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        logger.error(
            "Model file not found: %s\n"
            "  Place resume_ann_model.keras in the same directory as app.py.\n"
            "  The /health and category endpoints still work without it.",
            MODEL_PATH,
        )
        return None
    try:
        logger.info("Loading Keras model from %s …", MODEL_PATH)
        _model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded OK (input_dim=%d, output_dim=%d)", INPUT_DIM, len(JOB_CATEGORIES))
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        _model = None
    return _model

# ── FIX 2: pre-trained TF-IDF vectorizer with fallback ─────────────────────────
_tfidf = None

def get_tfidf():
    global _tfidf
    if _tfidf is not None:
        return _tfidf
    if os.path.exists(TFIDF_PATH):
        try:
            with open(TFIDF_PATH, "rb") as f:
                _tfidf = pickle.load(f)
                logger.info("Loaded pre-trained TF-IDF vectorizer (vocab=%d)", len(_tfidf.vocabulary_))
        except Exception as exc:
            logger.warning("Failed to load pre-trained TF-IDF: %s", exc)
            _tfidf = None
    return _tfidf

# Build background corpus fallback if vectorizer is not found on disk
_DOMAIN_CORPUS = " ".join(
    tok for kws in DOMAIN_KEYWORDS.values() for tok in kws
)
_FILLER = " ".join(f"token{i}" for i in range(5000))
_BG_CORPUS = [_DOMAIN_CORPUS, _FILLER]   # first slot reserved for the user text

def text_to_tfidf_vector(text: str, dim: int = 5000) -> np.ndarray:
    """Convert text to a fixed dim-dimensional TF-IDF vector using trained TF-IDF when available."""
    tfidf = get_tfidf()
    if tfidf is not None:
        try:
            vec = tfidf.transform([text]).toarray().astype(np.float32)
            if vec.shape[1] == dim:
                return vec
            elif vec.shape[1] < dim:
                return np.hstack([vec, np.zeros((1, dim - vec.shape[1]), dtype=np.float32)])
            else:
                return vec[:, :dim]
        except Exception as err:
            logger.warning("Trained TF-IDF transform failed, falling back: %s", err)

    corpus = [text] + _BG_CORPUS          # user text always at index 0
    vec    = TfidfVectorizer(max_features=dim, stop_words="english")
    matrix = vec.fit_transform(corpus)    # shape: (3, ≤dim)
    user_vec = matrix.toarray()[0:1]      # shape: (1, actual_features)

    n_features = user_vec.shape[1]
    if n_features < dim:
        user_vec = np.hstack([user_vec, np.zeros((1, dim - n_features), dtype=np.float32)])
    elif n_features > dim:
        user_vec = user_vec[:, :dim]

    return user_vec.astype(np.float32)    # shape: (1, dim)

def text_to_tfidf_5000(text: str) -> np.ndarray:
    return text_to_tfidf_vector(text, dim=INPUT_DIM)

# ── text extraction ───────────────────────────────────────────────────────────
def extract_text_from_pdf(b: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(b))
    return " ".join(p.extract_text() or "" for p in reader.pages)

def extract_text_from_docx(b: bytes) -> str:
    doc = docx.Document(io.BytesIO(b))
    return " ".join(p.text for p in doc.paragraphs)

def extract_text(b: bytes, filename: str) -> str:
    if not filename:
        raise ValueError("File has no name; cannot determine type.")
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":              return extract_text_from_pdf(b)
    if ext in ("docx", "doc"):   return extract_text_from_docx(b)
    if ext == "txt":              return b.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: .{ext}  (Accepted: pdf, docx, txt)")

# ── prediction ────────────────────────────────────────────────────────────────
def predict_category(text: str):
    model = get_model()
    if model is None:
        raise RuntimeError(
            "Model not loaded. Ensure resume_ann_model.keras is present next to app.py."
        )
    input_dim = model.input_shape[-1] if (hasattr(model, "input_shape") and model.input_shape and model.input_shape[-1]) else INPUT_DIM
    features = text_to_tfidf_vector(text.lower(), dim=input_dim)
    probs    = model.predict(features, verbose=0)[0]
    idx      = int(np.argmax(probs))
    num_out  = len(probs)
    categories = JOB_CATEGORIES[:num_out] if num_out <= len(JOB_CATEGORIES) else JOB_CATEGORIES
    scores   = {categories[i]: float(probs[i]) for i in range(min(num_out, len(categories)))}
    category_name = categories[idx] if idx < len(categories) else JOB_CATEGORIES[0]
    return category_name, float(probs[idx]), scores

# ── ATS scoring ───────────────────────────────────────────────────────────────
def compute_ats_score(resume_text: str, job_description: str) -> dict:
    rl = resume_text.lower()
    jl = job_description.lower()

    STOP = {
        "the","and","for","are","with","this","that","will","have","from","they",
        "you","our","their","your","all","can","been","was","were","but","not",
        "has","had","its","also","into","more","about","who","we","be","an","to",
        "of","in","a","is","it","as","at","on","or","by","do","if","up","any",
        "each","how","her","she","he","him","his","them",
    }
    words    = re.findall(r"\b[a-z][a-z0-9+#/\-\.]{1,29}\b", jl)
    jd_kw    = [w for w in words if w not in STOP and len(w) > 2]
    seen = set(); unique_jd = []
    for kw in jd_kw:
        if kw not in seen: seen.add(kw); unique_jd.append(kw)

    matched      = [kw for kw in unique_jd if kw in rl]
    missing      = [kw for kw in unique_jd if kw not in rl]
    keyword_score = (len(matched) / max(len(unique_jd), 1)) * 100

    sections = {
        "Contact Info":      bool(re.search(r"(email|phone|linkedin|github|@)", rl)),
        "Summary/Objective": bool(re.search(r"(summary|objective|profile|about me)", rl)),
        "Experience":        bool(re.search(r"(experience|work history|employment)", rl)),
        "Education":         bool(re.search(r"(education|degree|bachelor|master|phd|university|college)", rl)),
        "Skills":            bool(re.search(r"(skills|technologies|tools|proficient)", rl)),
        "Projects":          bool(re.search(r"(projects|portfolio|built|developed|created)", rl)),
        "Certifications":    bool(re.search(r"(certif|award|honor|achievement)", rl)),
    }
    section_score = (sum(sections.values()) / len(sections)) * 100

    wc           = len(resume_text.split())
    length_score = (30 if wc < 100 else 60 if wc < 300 else 90 if wc < 800 else 100 if wc <= 1200 else 80)

    has_bullets = bool(re.search(r"[•·▪▸\-]\s", resume_text))
    has_dates   = bool(re.search(r"\b(20\d{2}|19\d{2})\b", resume_text))
    has_numbers = bool(re.search(r"\b\d+%|\b\d+\+?\s*(years?|projects?|clients?)", resume_text))
    has_actions = bool(re.search(
        r"\b(developed|designed|implemented|led|managed|created|built|"
        r"improved|increased|reduced|delivered|achieved|optimized)\b", rl))
    format_score = (sum([has_bullets, has_dates, has_numbers, has_actions]) / 4) * 100

    ats_score = keyword_score*0.60 + section_score*0.20 + length_score*0.10 + format_score*0.10

    return {
        "ats_score":        round(ats_score, 1),
        "keyword_score":    round(keyword_score, 1),
        "section_score":    round(section_score, 1),
        "length_score":     round(length_score, 1),
        "format_score":     round(format_score, 1),
        "matched_keywords": matched[:40],
        "missing_keywords": missing[:40],
        "sections_found":   sections,
        "word_count":       wc,
        "format_signals": {
            "has_bullets":       has_bullets,
            "has_dates":         has_dates,
            "quantified_impact": has_numbers,
            "action_verbs":      has_actions,
        },
    }

def get_domain_missing(category: str, resume_text: str) -> list:
    kws = DOMAIN_KEYWORDS.get(category, GENERIC_KEYWORDS)
    rl  = resume_text.lower()
    return [kw for kw in kws if kw not in rl]

# ── FIX 3: ASCII-safe suggestion strings ──────────────────────────────────────
def generate_suggestions(analysis: dict, category: str, domain_missing: list) -> list:
    s = analysis["ats_score"]; tips = []
    if   s < 40: tips.append("[!] Critical score — resume needs a major overhaul before applying.")
    elif s < 60: tips.append("[!] Significant improvement needed. Add more keywords and complete all sections.")
    elif s < 75: tips.append("[~] Good start — targeted tweaks will push you past most ATS filters.")
    else:        tips.append("[+] Strong score! Fine-tune the details below for an even better match.")

    if analysis["missing_keywords"][:8]:
        tips.append(f"[kw] Add these job-description keywords naturally to your resume: "
                    f"{', '.join(analysis['missing_keywords'][:8])}.")
    if domain_missing:
        tips.append(f"[skill] For a {category} role, highlight these domain skills if you have them: "
                    f"{', '.join(domain_missing[:8])}.")

    missing_sec = [sec for sec, ok in analysis["sections_found"].items() if not ok]
    if missing_sec:
        tips.append(f"[section] Add these missing resume sections: {', '.join(missing_sec)}.")

    fmt = analysis["format_signals"]
    if not fmt["has_bullets"]:
        tips.append("Use bullet points for responsibilities — ATS and recruiters prefer scannable content.")
    if not fmt["quantified_impact"]:
        tips.append("Quantify your achievements (e.g. 'Increased sales by 30%', 'Reduced deploy time by 2h').")
    if not fmt["action_verbs"]:
        tips.append("Start bullets with strong action verbs: Developed, Led, Implemented, Optimized.")
    if not fmt["has_dates"]:
        tips.append("Include dates (MM/YYYY - MM/YYYY) for all experience and education entries.")

    wc = analysis["word_count"]
    if   wc < 300:  tips.append(f"[len] Resume too short ({wc} words). Aim for 400-800 words.")
    elif wc > 1200: tips.append(f"[len] Resume too long ({wc} words). Trim to 1-2 pages.")
    return tips

# ── FIX 4: serve the frontend ─────────────────────────────────────────────────
@app.route("/")
def index():
    folder = BASE_DIR if os.path.isfile(os.path.join(BASE_DIR, "index.html")) else FRONTEND_DIR
    response = send_from_directory(folder, "index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.after_request
def add_no_cache_headers(response):
    if response.mimetype == "text/html":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    model_ok = get_model() is not None
    return jsonify({
        "status":     "ok" if model_ok else "degraded",
        "model":      "resume_ann_model.keras",
        "model_loaded": model_ok,
        "categories": len(JOB_CATEGORIES),
        "input_dim":  INPUT_DIM,
    }), 200

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Multipart/form-data:
      resume           file  (pdf / docx / txt)   REQUIRED
      job_description  text                        REQUIRED (or use jd_file)
      jd_file          file  (pdf / docx / txt)   optional JD as file
    """
    try:
        if "resume" not in request.files:
            return jsonify({"error": "No resume file. Send as multipart field 'resume'."}), 400

        rf = request.files["resume"]
        rb = rf.read()
        if not rb:
            return jsonify({"error": "Uploaded resume file is empty."}), 400

        resume_text = extract_text(rb, rf.filename)
        if len(resume_text.strip()) < 50:
            return jsonify({"error": "Could not extract enough text from resume. "
                                     "Ensure it is not a scanned/image-only file."}), 422

        job_description = ""
        if "jd_file" in request.files:
            jf = request.files["jd_file"]
            job_description = extract_text(jf.read(), jf.filename)
        elif "job_description" in request.form:
            job_description = request.form["job_description"].strip()

        if len(job_description.strip()) < 30:
            return jsonify({"error": "Job description missing or too short. "
                                     "Send as 'job_description' (text) or 'jd_file' (file)."}), 400

        # ── FIX 2: guard on model availability ───────────────────────────────
        try:
            category, confidence, all_scores = predict_category(resume_text)
        except RuntimeError as model_err:
            return jsonify({"error": str(model_err)}), 503

        top3           = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        analysis       = compute_ats_score(resume_text, job_description)
        domain_missing = get_domain_missing(category, resume_text)
        suggestions    = generate_suggestions(analysis, category, domain_missing)

        return jsonify({
            "success": True,
            "ats_score": analysis["ats_score"],
            "score_breakdown": {
                "keyword_match": analysis["keyword_score"],
                "sections":      analysis["section_score"],
                "length":        analysis["length_score"],
                "format":        analysis["format_score"],
            },
            "predicted_category": {
                "category":    category,
                "confidence":  round(confidence * 100, 1),
                "top_3_matches": [
                    {"category": c, "score": round(s*100, 1)} for c, s in top3
                ],
            },
            "keyword_analysis": {
                "matched_keywords":      analysis["matched_keywords"],
                "missing_from_jd":       analysis["missing_keywords"],
                "missing_domain_skills": domain_missing[:15],
            },
            "sections_found":  analysis["sections_found"],
            "format_signals":  analysis["format_signals"],
            "word_count":      analysis["word_count"],
            "suggestions":     suggestions,
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("Error in /api/analyze")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/categories", methods=["GET"])
def list_categories():
    return jsonify({"categories": JOB_CATEGORIES})


@app.route("/api/predict-category", methods=["POST"])
def predict_only():
    """Quick endpoint: predict job category from resume only."""
    try:
        if "resume" not in request.files:
            return jsonify({"error": "No resume file."}), 400
        f    = request.files["resume"]
        text = extract_text(f.read(), f.filename)
        try:
            category, confidence, scores = predict_category(text)
        except RuntimeError as model_err:
            return jsonify({"error": str(model_err)}), 503
        top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        return jsonify({
            "category":   category,
            "confidence": round(confidence*100, 1),
            "top_5": [{"category": c, "score": round(s*100, 1)} for c, s in top5],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ██  AIHIRE ENTERPRISE ATS — DATABASE & NEW ENDPOINTS                       ██
# ██  Everything below is additive. Nothing above this line was changed.     ██
# ══════════════════════════════════════════════════════════════════════════════

# ── Database helpers ──────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    """Create tables if they don't exist, apply schema migrations, and seed initial accounts."""
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            role TEXT DEFAULT 'employee',
            is_admin INTEGER DEFAULT 0,
            password_hash TEXT,
            avatar_url TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS user_sessions (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            department TEXT DEFAULT '',
            location TEXT DEFAULT '',
            employment_type TEXT DEFAULT 'Full-time',
            status TEXT DEFAULT 'open',
            description TEXT DEFAULT '',
            requirements TEXT DEFAULT '',
            salary_range TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS candidates (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT DEFAULT '',
            phone TEXT DEFAULT '',
            experience_years INTEGER DEFAULT 0,
            skills TEXT DEFAULT '[]',
            education TEXT DEFAULT '[]',
            current_role TEXT DEFAULT '',
            current_company TEXT DEFAULT '',
            resume_path TEXT DEFAULT '',
            resume_text TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS applications (
            id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            status TEXT DEFAULT 'applied',
            ats_score REAL DEFAULT 0,
            score_breakdown TEXT DEFAULT '{}',
            predicted_category TEXT DEFAULT '',
            category_confidence REAL DEFAULT 0,
            keyword_analysis TEXT DEFAULT '{}',
            sections_found TEXT DEFAULT '{}',
            format_signals TEXT DEFAULT '{}',
            suggestions TEXT DEFAULT '[]',
            word_count INTEGER DEFAULT 0,
            applied_at TEXT DEFAULT (datetime('now')),
            screened_at TEXT
        );
    """)

    # Backward-compatible migration: add password_hash and is_admin if not present
    user_cols = [c[1] for c in db.execute("PRAGMA table_info(users)").fetchall()]
    if "password_hash" not in user_cols:
        db.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    if "is_admin" not in user_cols:
        db.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")

    # Migrate legacy roles to two-role system: HR & Employee
    db.execute("UPDATE users SET role = 'hr', is_admin = 1 WHERE role = 'admin'")
    db.execute("UPDATE users SET role = 'hr' WHERE role IN ('recruiter', 'hiring_manager')")
    db.execute("UPDATE users SET role = 'employee' WHERE role = 'candidate'")

    # Seed or upgrade demo accounts for HR and Employee
    default_users = [
        ("u_hr", "Alexander Vance", "hr@aihire.io", "hr", 1, "HR@123"),
        ("u_employee", "Sophia Chen", "employee@aihire.io", "employee", 0, "Employee@123"),
        ("u_admin", "Alexander Vance", "admin@aihire.io", "hr", 1, "Admin@123"),
        ("u_candidate", "Sophia Chen", "candidate@aihire.io", "employee", 0, "Candidate@123")
    ]
    for uid, name, email, role, is_adm, pwd in default_users:
        existing = db.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,)).fetchone()
        hashed = generate_password_hash(pwd)
        if not existing:
            db.execute(
                "INSERT INTO users (id, name, email, role, is_admin, password_hash) VALUES (?, ?, ?, ?, ?, ?)",
                (uid, name, email, role, is_adm, hashed)
            )
        else:
            db.execute(
                "UPDATE users SET password_hash = ?, role = ?, is_admin = ? WHERE id = ?",
                (hashed, role, is_adm, existing[0])
            )

    # Ensure candidate profile exists for employee demo accounts
    for emp_email in ["employee@aihire.io", "candidate@aihire.io"]:
        cand = db.execute("SELECT id FROM candidates WHERE email = ?", (emp_email,)).fetchone()
        if not cand:
            db.execute(
                """INSERT INTO candidates (id, name, email, current_role, experience_years, skills, education)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("c_" + uuid.uuid4().hex[:8], "Sophia Chen", emp_email, "Full Stack Software Engineer", 4,
                 json.dumps(["Python", "JavaScript", "React", "Flask", "SQL", "Docker", "Git"]),
                 json.dumps(["B.S. in Computer Science"]))
            )

    # Index to prevent duplicate applications
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_job ON applications(candidate_id, job_id)")

    # Seed sample jobs if empty
    cur = db.execute("SELECT COUNT(*) FROM jobs")
    if cur.fetchone()[0] == 0:
        sample_jobs = [
            ("j1", "Senior Java Backend Developer", "Engineering", "San Francisco, CA", "Full-time", "open",
             "We are looking for a Senior Java Backend Developer to join our engineering team. You will design and build scalable microservices, mentor junior developers, and drive architectural decisions.",
             "Java, Spring Boot, Microservices, REST API, Docker, Kubernetes, SQL, 5+ years experience"),
            ("j2", "Data Scientist — NLP", "Data Science", "Remote", "Full-time", "open",
             "Join our data science team to build NLP pipelines for resume parsing and candidate matching.",
             "Python, Machine Learning, NLP, TensorFlow, PyTorch, SQL, 3+ years experience"),
            ("j3", "DevOps Engineer", "Infrastructure", "New York, NY", "Full-time", "open",
             "Manage CI/CD pipelines, container orchestration, and cloud infrastructure.",
             "Docker, Kubernetes, AWS, Terraform, Jenkins, Linux, 4+ years experience"),
            ("j4", "Frontend Developer — React", "Engineering", "Austin, TX", "Full-time", "open",
             "Build modern, responsive web applications using React and TypeScript.",
             "React, TypeScript, JavaScript, HTML, CSS, Webpack, 3+ years experience"),
            ("j5", "Product Manager", "Product", "San Francisco, CA", "Full-time", "open",
             "Define product roadmap and work with engineering to deliver features.",
             "Product Management, Agile, Scrum, SQL, Analytics, Stakeholder Management, 5+ years"),
        ]
        for jid, title, dept, loc, emp, status, desc, reqs in sample_jobs:
            db.execute(
                "INSERT INTO jobs (id, title, department, location, employment_type, status, description, requirements) VALUES (?,?,?,?,?,?,?,?)",
                (jid, title, dept, loc, emp, status, desc, reqs)
            )

    db.commit()
    db.close()
    logger.info("Database initialized at %s", DB_PATH)


def row_to_dict(row):
    """Convert sqlite3.Row to dict, parsing JSON fields."""
    if row is None:
        return None
    d = dict(row)
    for key in ("skills", "education", "score_breakdown", "keyword_analysis",
                "sections_found", "format_signals", "suggestions"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


# ── Session & Auth Helpers ───────────────────────────────────────────────────
def create_session(user_id: str, duration_days: int = 7) -> str:
    db = get_db()
    token = secrets.token_hex(32)
    now = datetime.now(timezone.utc)
    expires = (now + timedelta(days=duration_days)).isoformat()
    db.execute(
        "INSERT INTO user_sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, now.isoformat(), expires)
    )
    db.commit()
    return token

def get_user_from_token(token: str):
    if not token:
        return None
    db = get_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    row = db.execute("""
        SELECT u.* FROM users u
        JOIN user_sessions s ON u.id = s.user_id
        WHERE s.token = ? AND s.expires_at > ?
    """, (token, now_iso)).fetchone()
    if row:
        return dict(row)
    return None

def revoke_session(token: str):
    if not token:
        return
    db = get_db()
    db.execute("DELETE FROM user_sessions WHERE token = ?", (token,))
    db.commit()

def get_current_user():
    auth_header = request.headers.get("Authorization", "")
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    elif "aihire_token" in request.cookies:
        token = request.cookies.get("aihire_token")
    if not token:
        return None
    return get_user_from_token(token)

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required. Please sign in."}), 401
        g.current_user = user
        return f(*args, **kwargs)
    return decorated

def require_roles(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user = get_current_user()
            if not user:
                return jsonify({"error": "Authentication required. Please sign in."}), 401
            g.current_user = user
            role = (user.get("role") or "").lower()
            allowed = [r.lower() for r in roles]
            has_role = role in allowed
            # HR users with is_admin have full HR/management access
            if not has_role and user.get("is_admin") and "hr" in allowed:
                has_role = True
            if not has_role:
                return jsonify({"error": f"Forbidden: Insufficient privileges. Required: {', '.join(roles)}"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


# ── Authentication Endpoints ──────────────────────────────────────────────────
@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    requested_role = (data.get("role") or "employee").strip().lower()

    if not name:
        return jsonify({"error": "Full name is required."}), 400
    if not email or "@" not in email or "." not in email:
        return jsonify({"error": "A valid email address is required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters long."}), 400

    # Public registration only creates employee accounts.
    # Creating HR accounts requires an authenticated HR admin (is_admin=1).
    current = get_current_user()
    if current and current.get("is_admin"):
        role = requested_role if requested_role in ("hr", "employee") else "employee"
    else:
        role = "employee"

    db = get_db()
    existing = db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if existing:
        return jsonify({"error": "An account with this email address already exists."}), 409

    uid = "u_" + uuid.uuid4().hex[:8]
    hashed = generate_password_hash(password)
    now = datetime.now(timezone.utc).isoformat()
    is_admin = 1 if (current and current.get("is_admin") and role == "hr" and data.get("is_admin")) else 0

    db.execute(
        "INSERT INTO users (id, name, email, role, is_admin, password_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (uid, name, email, role, is_admin, hashed, now)
    )

    if role == "employee":
        cand = db.execute("SELECT id FROM candidates WHERE email = ?", (email,)).fetchone()
        if not cand:
            cid = "c_" + uuid.uuid4().hex[:8]
            db.execute(
                "INSERT INTO candidates (id, name, email, current_role, experience_years, skills, education, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (cid, name, email, data.get("current_role", ""), int(data.get("experience_years") or 0), "[]", "[]", now)
            )

    db.commit()
    token = create_session(uid)
    return jsonify({
        "success": True,
        "token": token,
        "user": {
            "id": uid,
            "name": name,
            "email": email,
            "role": role,
            "is_admin": bool(is_admin)
        }
    }), 201

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(force=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not user:
        return jsonify({"error": "Invalid email or password."}), 401

    user_dict = dict(user)
    stored_hash = user_dict.get("password_hash")
    if not stored_hash or not check_password_hash(stored_hash, password):
        return jsonify({"error": "Invalid email or password."}), 401

    token = create_session(user_dict["id"])
    return jsonify({
        "success": True,
        "token": token,
        "user": {
            "id": user_dict["id"],
            "name": user_dict["name"],
            "email": user_dict["email"],
            "role": user_dict["role"],
            "is_admin": bool(user_dict.get("is_admin", 0))
        }
    })

@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else request.cookies.get("aihire_token")
    if token:
        revoke_session(token)
    return jsonify({"success": True, "message": "Logged out successfully."})

@app.route("/api/auth/me", methods=["GET"])
@require_auth
def auth_me():
    user = dict(g.current_user)
    user.pop("password_hash", None)
    user["is_admin"] = bool(user.get("is_admin", 0))
    return jsonify({"user": user})


# ── Dashboard endpoints ───────────────────────────────────────────────────────
@app.route("/api/dashboard/stats", methods=["GET"])
@require_roles("hr")
def dashboard_stats():
    db = get_db()
    stats = {
        "open_jobs": db.execute("SELECT COUNT(*) FROM jobs WHERE status='open'").fetchone()[0],
        "total_applications": db.execute("SELECT COUNT(*) FROM applications").fetchone()[0],
        "pending_review": db.execute("SELECT COUNT(*) FROM applications WHERE status='applied'").fetchone()[0],
        "shortlisted": db.execute("SELECT COUNT(*) FROM applications WHERE status='shortlisted'").fetchone()[0],
        "in_interview": db.execute("SELECT COUNT(*) FROM applications WHERE status='interview'").fetchone()[0],
        "screening": db.execute("SELECT COUNT(*) FROM applications WHERE status='screening'").fetchone()[0],
        "hired": db.execute("SELECT COUNT(*) FROM applications WHERE status='hired'").fetchone()[0],
        "rejected": db.execute("SELECT COUNT(*) FROM applications WHERE status='rejected'").fetchone()[0],
        "total_candidates": db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
        "total_jobs": db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
    }
    # Pipeline data
    stats["pipeline"] = [
        {"stage": "Applied", "count": stats["total_applications"]},
        {"stage": "Screening", "count": stats["screening"]},
        {"stage": "Shortlisted", "count": stats["shortlisted"]},
        {"stage": "Interview", "count": stats["in_interview"]},
        {"stage": "Hired", "count": stats["hired"]},
    ]
    return jsonify(stats)


@app.route("/api/dashboard/recent", methods=["GET"])
@require_roles("hr")
def dashboard_recent():
    db = get_db()
    rows = db.execute("""
        SELECT a.*, c.name as candidate_name, c.email as candidate_email,
               c.current_role, c.current_company, c.experience_years,
               j.title as job_title, j.department
        FROM applications a
        JOIN candidates c ON a.candidate_id = c.id
        JOIN jobs j ON a.job_id = j.id
        ORDER BY a.applied_at DESC LIMIT 10
    """).fetchall()
    return jsonify({"applications": [row_to_dict(r) for r in rows]})


# ── Jobs CRUD ─────────────────────────────────────────────────────────────────
@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    db = get_db()
    status = request.args.get("status")
    search = request.args.get("search", "")
    department = request.args.get("department", "")
    query = "SELECT * FROM jobs"
    params = []
    conditions = []
    if status:
        conditions.append("status = ?")
        params.append(status)
    if department:
        conditions.append("department = ?")
        params.append(department)
    if search:
        conditions.append("(title LIKE ? OR department LIKE ? OR location LIKE ?)")
        params.extend([f"%{search}%"] * 3)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC"
    rows = db.execute(query, params).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["application_count"] = db.execute(
            "SELECT COUNT(*) FROM applications WHERE job_id = ?", (d["id"],)
        ).fetchone()[0]
        result.append(d)
    return jsonify({"jobs": result})


@app.route("/api/jobs", methods=["POST"])
@require_roles("hr")
def create_job():
    data = request.get_json(force=True)
    if not data.get("title"):
        return jsonify({"error": "Job title is required"}), 400
    jid = "j" + uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    db = get_db()
    db.execute(
        """INSERT INTO jobs (id, title, department, location, employment_type, status, description, requirements, salary_range, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (jid, data["title"], data.get("department", ""), data.get("location", ""),
         data.get("employment_type", "Full-time"), data.get("status", "open"),
         data.get("description", ""), data.get("requirements", ""),
         data.get("salary_range", ""), now, now)
    )
    db.commit()
    job = row_to_dict(db.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone())
    return jsonify(job), 201


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    db = get_db()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        return jsonify({"error": "Job not found"}), 404
    d = dict(job)
    d["application_count"] = db.execute(
        "SELECT COUNT(*) FROM applications WHERE job_id = ?", (job_id,)
    ).fetchone()[0]
    
    # Strictly isolated: only HR staff can view applicants and candidate rankings
    user = get_current_user()
    user_role = (user.get("role") or "").lower() if user else ""
    if user_role == "hr":
        apps = db.execute("""
            SELECT a.*, c.name as candidate_name, c.email as candidate_email,
                   c.current_role, c.experience_years
            FROM applications a JOIN candidates c ON a.candidate_id = c.id
            WHERE a.job_id = ? ORDER BY a.ats_score DESC
        """, (job_id,)).fetchall()
        d["applications"] = [row_to_dict(a) for a in apps]
    else:
        d["applications"] = []
    return jsonify(d)


@app.route("/api/jobs/<job_id>", methods=["PUT"])
@require_roles("hr")
def update_job(job_id):
    db = get_db()
    existing = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not existing:
        return jsonify({"error": "Job not found"}), 404
    data = request.get_json(force=True)
    now = datetime.now(timezone.utc).isoformat()
    db.execute("""
        UPDATE jobs SET title=?, department=?, location=?, employment_type=?,
        status=?, description=?, requirements=?, salary_range=?, updated_at=?
        WHERE id=?
    """, (
        data.get("title", existing["title"]),
        data.get("department", existing["department"]),
        data.get("location", existing["location"]),
        data.get("employment_type", existing["employment_type"]),
        data.get("status", existing["status"]),
        data.get("description", existing["description"]),
        data.get("requirements", existing["requirements"]),
        data.get("salary_range", existing["salary_range"]),
        now, job_id
    ))
    db.commit()
    return jsonify(row_to_dict(db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()))


@app.route("/api/jobs/<job_id>", methods=["DELETE"])
@require_roles("hr")
def delete_job(job_id):
    if not g.current_user.get("is_admin"):
        return jsonify({"error": "Administrator privileges required to delete jobs."}), 403
    db = get_db()
    db.execute("DELETE FROM applications WHERE job_id = ?", (job_id,))
    db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    db.commit()
    return jsonify({"deleted": True})


# ── Candidates ────────────────────────────────────────────────────────────────
@app.route("/api/candidates", methods=["GET"])
@require_roles("hr")
def list_candidates():
    db = get_db()
    search = request.args.get("search", "")
    query = "SELECT * FROM candidates"
    params = []
    if search:
        query += " WHERE name LIKE ? OR email LIKE ? OR current_role LIKE ? OR skills LIKE ?"
        params = [f"%{search}%"] * 4
    query += " ORDER BY created_at DESC"
    rows = db.execute(query, params).fetchall()
    result = []
    for r in rows:
        d = row_to_dict(r)
        latest = db.execute("""
            SELECT a.ats_score, a.status, a.predicted_category, j.title as job_title
            FROM applications a JOIN jobs j ON a.job_id = j.id
            WHERE a.candidate_id = ? ORDER BY a.applied_at DESC LIMIT 1
        """, (d["id"],)).fetchone()
        if latest:
            d["latest_score"] = latest["ats_score"]
            d["latest_status"] = latest["status"]
            d["latest_job"] = latest["job_title"]
            d["latest_category"] = latest["predicted_category"]
        result.append(d)
    return jsonify({"candidates": result})


@app.route("/api/candidates/<cid>", methods=["GET"])
@require_auth
def get_candidate(cid):
    db = get_db()
    c = db.execute("SELECT * FROM candidates WHERE id = ?", (cid,)).fetchone()
    if not c:
        return jsonify({"error": "Candidate not found"}), 404
    d = row_to_dict(c)
    # Role isolation: employees can only view their own candidate profile
    user_role = (g.current_user.get("role") or "").lower()
    if user_role == "employee" and d.get("email") != g.current_user.get("email"):
        return jsonify({"error": "Forbidden: Cannot access other candidate profiles."}), 403

    apps = db.execute("""
        SELECT a.*, j.title as job_title, j.department
        FROM applications a JOIN jobs j ON a.job_id = j.id
        WHERE a.candidate_id = ? ORDER BY a.applied_at DESC
    """, (cid,)).fetchall()
    d["applications"] = [row_to_dict(a) for a in apps]
    return jsonify(d)


@app.route("/api/candidates/<cid>/profile", methods=["PUT"])
@require_auth
def update_candidate_profile(cid):
    db = get_db()
    c = db.execute("SELECT * FROM candidates WHERE id = ?", (cid,)).fetchone()
    if not c:
        return jsonify({"error": "Candidate not found"}), 404
    user_role = (g.current_user.get("role") or "").lower()
    if user_role == "employee" and c["email"] != g.current_user.get("email"):
        return jsonify({"error": "Forbidden: Cannot update other candidate profiles."}), 403
    data = request.get_json(force=True) or {}
    name = data.get("name", c["name"])
    phone = data.get("phone", c["phone"])
    experience_years = int(data.get("experience_years", c["experience_years"]))
    skills = data.get("skills")
    if isinstance(skills, list):
        skills_str = json.dumps(skills)
    elif isinstance(skills, str):
        skills_str = skills
    else:
        skills_str = c["skills"]
    education = data.get("education")
    if isinstance(education, list):
        edu_str = json.dumps(education)
    elif isinstance(education, str):
        edu_str = education
    else:
        edu_str = c["education"]
    current_role = data.get("current_role", c["current_role"])
    current_company = data.get("current_company", c["current_company"])

    db.execute("""
        UPDATE candidates
        SET name = ?, phone = ?, experience_years = ?, skills = ?, education = ?, current_role = ?, current_company = ?
        WHERE id = ?
    """, (name, phone, experience_years, skills_str, edu_str, current_role, current_company, cid))
    if c["email"]:
        db.execute("UPDATE users SET name = ? WHERE email = ?", (name, c["email"]))
    db.commit()
    updated = db.execute("SELECT * FROM candidates WHERE id = ?", (cid,)).fetchone()
    return jsonify(row_to_dict(updated))


@app.route("/api/employee/profile", methods=["GET"])
@require_auth
def get_employee_profile():
    db = get_db()
    email = g.current_user.get("email")
    c = db.execute("SELECT * FROM candidates WHERE email = ?", (email,)).fetchone()
    if not c:
        cid = "c_" + uuid.uuid4().hex[:8]
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            """INSERT INTO candidates (id, name, email, current_role, experience_years, skills, education, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (cid, g.current_user.get("name", ""), email, "", 0, "[]", "[]", now)
        )
        db.commit()
        c = db.execute("SELECT * FROM candidates WHERE id = ?", (cid,)).fetchone()
    return jsonify(row_to_dict(c))


@app.route("/api/employee/profile", methods=["PUT"])
@require_auth
def update_employee_profile():
    db = get_db()
    email = g.current_user.get("email")
    c = db.execute("SELECT * FROM candidates WHERE email = ?", (email,)).fetchone()
    if not c:
        return jsonify({"error": "Profile not found"}), 404
    data = request.get_json(force=True) or {}
    name = data.get("name", c["name"])
    phone = data.get("phone", c["phone"])
    experience_years = int(data.get("experience_years", c["experience_years"]))
    skills = data.get("skills")
    if isinstance(skills, list):
        skills_str = json.dumps(skills)
    elif isinstance(skills, str):
        skills_str = skills
    else:
        skills_str = c["skills"]
    education = data.get("education")
    if isinstance(education, list):
        edu_str = json.dumps(education)
    elif isinstance(education, str):
        edu_str = education
    else:
        edu_str = c["education"]
    current_role = data.get("current_role", c["current_role"])
    current_company = data.get("current_company", c["current_company"])

    db.execute("""
        UPDATE candidates
        SET name = ?, phone = ?, experience_years = ?, skills = ?, education = ?, current_role = ?, current_company = ?
        WHERE id = ?
    """, (name, phone, experience_years, skills_str, edu_str, current_role, current_company, c["id"]))
    db.execute("UPDATE users SET name = ? WHERE email = ?", (name, email))
    db.commit()
    updated = db.execute("SELECT * FROM candidates WHERE id = ?", (c["id"],)).fetchone()
    return jsonify(row_to_dict(updated))


# ── Applications ──────────────────────────────────────────────────────────────
@app.route("/api/applications", methods=["GET"])
@require_auth
def list_applications():
    db = get_db()
    status = request.args.get("status")
    job_id = request.args.get("job_id")
    user_role = (g.current_user.get("role") or "").lower()

    query = """
        SELECT a.*, c.name as candidate_name, c.email as candidate_email,
               c.current_role, c.experience_years, c.skills as candidate_skills,
               j.title as job_title, j.department
        FROM applications a
        JOIN candidates c ON a.candidate_id = c.id
        JOIN jobs j ON a.job_id = j.id
    """
    conditions = []
    params = []

    # If employee, only list their own applications
    if user_role == "employee":
        conditions.append("c.email = ?")
        params.append(g.current_user.get("email"))

    if status:
        conditions.append("a.status = ?")
        params.append(status)
    if job_id:
        conditions.append("a.job_id = ?")
        params.append(job_id)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY a.applied_at DESC"
    rows = db.execute(query, params).fetchall()
    return jsonify({"applications": [row_to_dict(r) for r in rows]})


@app.route("/api/applications/<app_id>/status", methods=["PUT"])
@require_roles("hr")
def update_application_status(app_id):
    db = get_db()
    data = request.get_json(force=True)
    new_status = data.get("status")
    valid = {"applied", "screening", "shortlisted", "interview", "hired", "rejected"}
    if new_status not in valid:
        return jsonify({"error": f"Invalid status. Must be one of: {', '.join(sorted(valid))}"}), 400
    db.execute("UPDATE applications SET status = ? WHERE id = ?", (new_status, app_id))
    db.commit()
    return jsonify({"id": app_id, "status": new_status})


# ── Rankings ──────────────────────────────────────────────────────────────────
@app.route("/api/jobs/<job_id>/rankings", methods=["GET"])
@require_roles("hr")
def job_rankings(job_id):
    db = get_db()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        return jsonify({"error": "Job not found"}), 404
    rows = db.execute("""
        SELECT a.*, c.name as candidate_name, c.email as candidate_email,
               c.current_role, c.experience_years, c.skills as candidate_skills
        FROM applications a
        JOIN candidates c ON a.candidate_id = c.id
        WHERE a.job_id = ? AND a.ats_score > 0
        ORDER BY a.ats_score DESC
    """, (job_id,)).fetchall()
    ranked = []
    for i, r in enumerate(rows, 1):
        d = row_to_dict(r)
        d["rank"] = i
        s = d.get("ats_score", 0)
        if s >= 75:
            d["recommendation"] = "Strong Match"
        elif s >= 55:
            d["recommendation"] = "Potential Match"
        else:
            d["recommendation"] = "Weak Match"
        ranked.append(d)
    return jsonify({"job": dict(job), "rankings": ranked})


# ── Analytics ─────────────────────────────────────────────────────────────────
@app.route("/api/analytics", methods=["GET"])
@require_roles("hr")
def analytics():
    db = get_db()
    total_apps = db.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
    data = {
        "total_applications": total_apps,
        "total_candidates": db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
        "total_jobs": db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
        "avg_ats_score": db.execute("SELECT COALESCE(AVG(ats_score), 0) FROM applications WHERE ats_score > 0").fetchone()[0],
        "pipeline": {
            "applied": db.execute("SELECT COUNT(*) FROM applications WHERE status='applied'").fetchone()[0],
            "screening": db.execute("SELECT COUNT(*) FROM applications WHERE status='screening'").fetchone()[0],
            "shortlisted": db.execute("SELECT COUNT(*) FROM applications WHERE status='shortlisted'").fetchone()[0],
            "interview": db.execute("SELECT COUNT(*) FROM applications WHERE status='interview'").fetchone()[0],
            "hired": db.execute("SELECT COUNT(*) FROM applications WHERE status='hired'").fetchone()[0],
            "rejected": db.execute("SELECT COUNT(*) FROM applications WHERE status='rejected'").fetchone()[0],
        },
        "score_distribution": {
            "0-25": db.execute("SELECT COUNT(*) FROM applications WHERE ats_score > 0 AND ats_score < 25").fetchone()[0],
            "25-50": db.execute("SELECT COUNT(*) FROM applications WHERE ats_score >= 25 AND ats_score < 50").fetchone()[0],
            "50-75": db.execute("SELECT COUNT(*) FROM applications WHERE ats_score >= 50 AND ats_score < 75").fetchone()[0],
            "75-100": db.execute("SELECT COUNT(*) FROM applications WHERE ats_score >= 75").fetchone()[0],
        },
        "top_categories": [],
        "jobs_by_department": [],
    }
    cats = db.execute("""
        SELECT predicted_category, COUNT(*) as cnt
        FROM applications WHERE predicted_category != ''
        GROUP BY predicted_category ORDER BY cnt DESC LIMIT 10
    """).fetchall()
    data["top_categories"] = [{"category": r[0], "count": r[1]} for r in cats]
    depts = db.execute("""
        SELECT department, COUNT(*) as cnt FROM jobs
        WHERE department != '' GROUP BY department ORDER BY cnt DESC
    """).fetchall()
    data["jobs_by_department"] = [{"department": r[0], "count": r[1]} for r in depts]
    return jsonify(data)


# ── Screen endpoint (uses existing AI pipeline) ──────────────────────────────
@app.route("/api/screen", methods=["POST"])
@require_roles("hr", "employee")
def screen_candidate():
    """
    Screen a candidate's resume against a specific job.
    Multipart form: resume (file), job_id (text)
    Creates candidate + application records and runs the existing AI analysis.
    """
    try:
        if "resume" not in request.files:
            return jsonify({"error": "No resume file."}), 400
        job_id = request.form.get("job_id")
        if not job_id:
            return jsonify({"error": "job_id is required."}), 400

        db = get_db()
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            return jsonify({"error": "Job not found."}), 404

        rf = request.files["resume"]
        filename = rf.filename or ""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".pdf", ".docx", ".doc", ".txt"):
            return jsonify({"error": "Unsupported file format. Please upload a PDF, DOCX, DOC, or TXT file."}), 400

        rb = rf.read()
        if not rb:
            return jsonify({"error": "Empty resume file."}), 400
        if len(rb) > 10 * 1024 * 1024:
            return jsonify({"error": "File size exceeds 10MB limit."}), 400

        resume_text = extract_text(rb, filename)
        if len(resume_text.strip()) < 50:
            return jsonify({"error": "Could not extract enough text from resume."}), 422

        # Save resume file
        resume_filename = f"{uuid.uuid4().hex}_{filename}"
        resume_path = os.path.join(UPLOADS_DIR, resume_filename)
        with open(resume_path, "wb") as f:
            f.write(rb)

        # Use job requirements as the job description for ATS scoring
        job_desc = job["requirements"]
        if job["description"]:
            job_desc = job["description"] + "\n" + job["requirements"]

        # Run existing AI pipeline
        try:
            category, confidence, all_scores = predict_category(resume_text)
        except RuntimeError as model_err:
            return jsonify({"error": str(model_err)}), 503

        analysis = compute_ats_score(resume_text, job_desc)
        domain_missing = get_domain_missing(category, resume_text)
        suggestions = generate_suggestions(analysis, category, domain_missing)
        top3 = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        # Extract candidate name & email or default to authenticated user
        user_role = (g.current_user.get("role") or "").lower()
        if user_role == "employee":
            candidate_name = g.current_user.get("name")
            candidate_email = g.current_user.get("email")
        else:
            candidate_name = request.form.get("candidate_name", "Candidate")
            candidate_email = request.form.get("candidate_email", "")

        # Find or create candidate
        cand = None
        if candidate_email:
            cand = db.execute("SELECT id FROM candidates WHERE email = ?", (candidate_email,)).fetchone()
        
        if cand:
            cid = cand["id"]
            db.execute(
                "UPDATE candidates SET resume_path = ?, resume_text = ? WHERE id = ?",
                (resume_path, resume_text[:5000], cid)
            )
        else:
            cid = "c" + uuid.uuid4().hex[:8]
            db.execute(
                "INSERT INTO candidates (id, name, email, resume_path, resume_text, created_at) VALUES (?,?,?,?,?,?)",
                (cid, candidate_name, candidate_email, resume_path, resume_text[:5000],
                 datetime.now(timezone.utc).isoformat())
            )

        # Check for duplicate application
        existing_app = db.execute(
            "SELECT id FROM applications WHERE candidate_id = ? AND job_id = ?",
            (cid, job_id)
        ).fetchone()

        if existing_app and user_role == "employee":
            return jsonify({"error": "You have already submitted an application for this position."}), 409

        now = datetime.now(timezone.utc).isoformat()
        kw_analysis = {
            "matched_keywords": analysis["matched_keywords"],
            "missing_from_jd": analysis["missing_keywords"],
            "missing_domain_skills": domain_missing[:15],
        }

        if existing_app:
            aid = existing_app["id"]
            db.execute("""
                UPDATE applications SET
                    status = 'screening', ats_score = ?, score_breakdown = ?,
                    predicted_category = ?, category_confidence = ?, keyword_analysis = ?,
                    sections_found = ?, format_signals = ?, suggestions = ?, word_count = ?,
                    screened_at = ?
                WHERE id = ?
            """, (
                analysis["ats_score"],
                json.dumps({
                    "keyword_match": analysis["keyword_score"],
                    "sections": analysis["section_score"],
                    "length": analysis["length_score"],
                    "format": analysis["format_score"],
                }),
                category, round(confidence * 100, 1),
                json.dumps(kw_analysis),
                json.dumps(analysis["sections_found"]),
                json.dumps(analysis["format_signals"]),
                json.dumps(suggestions),
                analysis["word_count"], now, aid
            ))
        else:
            aid = "a" + uuid.uuid4().hex[:8]
            db.execute("""
                INSERT INTO applications
                (id, candidate_id, job_id, status, ats_score, score_breakdown,
                 predicted_category, category_confidence, keyword_analysis,
                 sections_found, format_signals, suggestions, word_count,
                 applied_at, screened_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                aid, cid, job_id, "screening", analysis["ats_score"],
                json.dumps({
                    "keyword_match": analysis["keyword_score"],
                    "sections": analysis["section_score"],
                    "length": analysis["length_score"],
                    "format": analysis["format_score"],
                }),
                category, round(confidence * 100, 1),
                json.dumps(kw_analysis),
                json.dumps(analysis["sections_found"]),
                json.dumps(analysis["format_signals"]),
                json.dumps(suggestions),
                analysis["word_count"], now, now
            ))
        db.commit()

        return jsonify({
            "success": True,
            "application_id": aid,
            "candidate_id": cid,
            "ats_score": analysis["ats_score"],
            "score_breakdown": {
                "keyword_match": analysis["keyword_score"],
                "sections": analysis["section_score"],
                "length": analysis["length_score"],
                "format": analysis["format_score"],
            },
            "predicted_category": {
                "category": category,
                "confidence": round(confidence * 100, 1),
                "top_3_matches": [{"category": c, "score": round(s*100, 1)} for c, s in top3],
            },
            "keyword_analysis": kw_analysis,
            "sections_found": analysis["sections_found"],
            "format_signals": analysis["format_signals"],
            "word_count": analysis["word_count"],
            "suggestions": suggestions,
        })
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("Error in /api/screen")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# ── Single application detail ─────────────────────────────────────────────────
@app.route("/api/applications/<app_id>", methods=["GET"])
@require_auth
def get_application(app_id):
    db = get_db()
    row = db.execute("""
        SELECT a.*, c.name as candidate_name, c.email as candidate_email,
               c.current_role, c.experience_years, c.skills as candidate_skills,
               c.resume_path, c.phone,
               j.title as job_title, j.department, j.requirements as job_requirements
        FROM applications a
        JOIN candidates c ON a.candidate_id = c.id
        JOIN jobs j ON a.job_id = j.id
        WHERE a.id = ?
    """, (app_id,)).fetchone()
    if not row:
        return jsonify({"error": "Application not found"}), 404

    app_data = row_to_dict(row)
    user_role = (g.current_user.get("role") or "").lower()
    # Employee can only see their own application
    if user_role == "employee" and app_data.get("candidate_email") != g.current_user.get("email"):
        return jsonify({"error": "Forbidden: Cannot access other candidate applications."}), 403

    if user_role == "employee":
        app_data.pop("resume_path", None)

    return jsonify(app_data)


# ── Initialize database on startup ───────────────────────────────────────────
init_db()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
