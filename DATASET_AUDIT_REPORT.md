# Dataset Audit Report & Training Implementation Plan

**Project:** HireMind-AI (AI-Powered Recruitment Intelligence System)  
**Date:** 2026-10-18  
**Scope:** Phase 1 — Dataset Audit, Architecture Inspection & Training Preparation  
**Status:** Complete  

---

## Executive Summary

A comprehensive, non-destructive audit was performed across the `AI_HIRE` repository, focusing on `Data Preprocessing/Data Set.zip`, `Data Preprocessing/HireMind_AI.ipynb`, `Backend/resume_ann_model.keras`, `Backend/app.py`, and the specification documents (`DATASET_MODEL_PLAN.md`, `MODEL_TRAINING_SPEC.md`).

### Critical Findings
1. **The Existing Keras Model is a Synthetic Dummy:** `Backend/resume_ann_model.keras` is an MNIST-dimensioned MLP ($784 \to 128 \to 10$) initialized with 10 generic classes. It was **never trained** on resumes.
2. **The Current Backend Feature Pipeline Uses Ad-Hoc Word Hashing:** `Backend/app.py` extracts features by running `hash(word) % 784`. No vocabulary, TF-IDF vectorizer, or tokenizer exists on disk.
3. **The Jupyter Notebook Used an Unrelated Dataset:** `HireMind_AI.ipynb` was trained on `Resume.csv` (1,200 samples across legal, aviation, chef, fitness, etc.), producing `hiremind_model.pkl` / `resume_model.pkl` (Logistic Regression), which do not match either the archive dataset or the production backend.
4. **The Real Dataset is Rich but Highly Synthetic (LLM-Generated):** `Data Set.zip` contains 10,174 rows and 8 columns with 0 missing values across all columns. 81.82% of resumes begin with an LLM conversational prefix (`"Here's a professional resume for..."`).
5. **Class Imbalance and Target Leakage Exist:** The dataset contains 45 raw role strings and 38 case-normalized categories. In 83.2% of resumes, the exact role appears in the first 300 characters.

---

## 1. Archive & CSV Schema Verification

- **Archive Location:** `Data Preprocessing/Data Set.zip` (CRC-verified, read in-memory without extraction)
- **Internal File:** `Data Set/dataset.csv`
- **Total Records:** 10,174 rows
- **Total Columns:** 8

| Column Name | Data Type | Null Count | Null % | Description & Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| `ID` | String | 0 | 0.00% | Unique identifier (e.g., `michmo483`) |
| `Name` | String | 0 | 0.00% | Candidate full name (synthetic) |
| `Role` | String | 0 | 0.00% | **Target variable** (ground truth role) |
| `Transcript` | String | 0 | 0.00% | Synthetic interview transcript |
| `Resume` | String | 0 | 0.00% | **Model input text** (full resume body) |
| `decision` | String | 0 | 0.00% | Binary interview outcome (`select` / `reject`) |
| `Reason_for_decision` | String | 0 | 0.00% | Interviewer justification text |
| `Job_Description` | String | 0 | 0.00% | Target job description text |

### Resume Text Statistics
- **Minimum Word Count:** 5 words (outliers / truncated entries)
- **Maximum Word Count:** 726 words
- **Mean Word Count:** 386.1 words
- **Median Word Count:** 389.0 words
- **Word Count Distribution:**
  - $< 50$ words: 64 records (0.63%)
  - $50 - 99$ words: 66 records (0.65%)
  - $100 - 199$ words: 119 records (1.17%)
  - $200 - 399$ words: 5,771 records (56.72%)
  - $400 - 799$ words: 4,154 records (40.83%)

---

## 2. Duplicate & Identifier Analysis

- **Exact Duplicate Resumes:** 0 (all 10,174 resume texts are unique).
- **Duplicate IDs:** 3 ID collision groups (6 rows total):
  - `michmo483`: Michael Montgomery (Full Stack Developer) vs. Michael Morgan (IT Support Specialist)
  - `branor520`: Brandon Ortiz (Human Resources Specialist) vs. Brandon Ortiz (Content Writer)
  - `krisbr553`: Kristen Brooks (Human Resources Specialist) vs. kristie brown (software engineer)
  *Resolution:* The duplicate IDs represent hash/slug collisions during synthetic generation; the candidate names and resume texts are completely different. New unique surrogate keys should be assigned.
- **Duplicate (Name, Role) Pairs:** 339 instances (e.g., 2 distinct resumes for "David Miller, QA Engineer"). Texts are distinct variants.

---

## 3. Data Leakage & Synthetic LLM Artifacts

### Types of Target Leakage Identified
1. **Conversational Prefixes (LLM Preambles):**
   - **8,324 resumes (81.82%)** begin with conversational LLM preamble phrases such as:
     - `"Here's a professional resume for [Name]..."`
     - `"Here's a detailed dummy resume for [Name]:"`
     - `"Here's a sample resume for [Name]:"`
     - `"Here is a professional resume for [Name] applying for the role of [Role]:"`
   - **Pre-processing Requirement:** A regex-based stripper must remove lines matching `^(?:here(?:'s|\s+is)?\b.*?(?:resume|profile|cv|candidate)\b.*?:?\n?)` before TF-IDF vectorization.

2. **Explicit Role in Header / Objective:**
   - **8,464 resumes (83.20%)** contain the exact ground-truth role title within the first 300 characters (e.g. `**Project Manager**` on line 3).
   - In production, real-world resumes often state the applicant's target role or current title. However, models trained with raw headers may over-rely on exact title strings rather than skill and experience semantic features.

3. **Candidate Name in Body:**
   - **10,118 resumes (99.45%)** contain the candidate's full name. Candidate names must be neutralized or stripped if entity anonymization is enabled.

4. **Extraneous Columns as Leakage Vectors:**
   - `decision` (`select`: 5,060, `reject`: 5,114) and `Job_Description` (which mentions the target role in 83.6% of rows) must be strictly isolated from role classification training.

---

## 4. Category Distribution & Label Mapping

### Raw Distribution
The dataset contains **45 distinct raw strings** due to casing discrepancies (`"Data Scientist"` vs `"data scientist"`).
When case-normalized, there are **38 distinct categories**.

### Proposed 38-to-26 Target Consolidation
The 26-class consolidation scheme proposed in `MODEL_TRAINING_SPEC.md` was validated against actual counts:

| Target Category (26) | Source Classes Merged | Consolidated Count | Share |
| :--- | :--- | :--- | :--- |
| **Software Engineer** | `software engineer` (787), `software developer` (66) | 853 | 8.38% |
| **Data Scientist** | `data scientist` (825) | 825 | 8.11% |
| **Product Manager** | `product manager` (761), `project manager` (56) | 817 | 8.03% |
| **Data Engineer** | `data engineer` (754) | 754 | 7.41% |
| **Data Analyst** | `data analyst` (512) | 512 | 5.03% |
| **UI Engineer** | `ui engineer` (507) | 507 | 4.98% |
| **Machine Learning Engineer**| `machine learning engineer` (265), `ai researcher` (225), `ai engineer` (13) | 503 | 4.94% |
| **Cloud Engineer** | `cloud engineer` (240), `cloud architect` (254) | 494 | 4.86% |
| **UI/UX Designer** | `ui designer` (226), `ux designer` (227), `graphic designer` (15), `ui/ux designer` (7) | 475 | 4.67% |
| **System Administrator** | `system administrator` (223), `it support specialist` (224), `network engineer` (16) | 463 | 4.55% |
| **Human Resources Specialist**| `human resources specialist` (262), `hr specialist` (12) | 274 | 2.69% |
| **E-commerce Specialist** | `e-commerce specialist` (268) | 268 | 2.63% |
| **DevOps Engineer** | `devops engineer` (266) | 266 | 2.61% |
| **Digital Marketing Specialist**| `digital marketing specialist` (260) | 260 | 2.56% |
| **Robotics Engineer** | `robotics engineer` (257) | 257 | 2.53% |
| **QA Engineer** | `qa engineer` (251) | 251 | 2.47% |
| **Blockchain Developer** | `blockchain developer` (251) | 251 | 2.47% |
| **Mobile App Developer** | `mobile app developer` (247) | 247 | 2.43% |
| **Full Stack Developer** | `full stack developer` (246) | 246 | 2.42% |
| **Database Administrator** | `database administrator` (243) | 243 | 2.39% |
| **Cybersecurity Analyst** | `cybersecurity analyst` (234), `cybersecurity specialist` (9) | 243 | 2.39% |
| **Game Developer** | `game developer` (239) | 239 | 2.35% |
| **Content Writer** | `content writer` (238) | 238 | 2.34% |
| **AR/VR Developer** | `ar/vr developer` (237) | 237 | 2.33% |
| **Business Analyst** | `business analyst` (226) | 226 | 2.22% |
| **Data Architect** | `data architect` (225) | 225 | 2.21% |
| **Total** | **38 source categories** | **10,174** | **100.0%** |

### Imbalance Evaluation
- **Smallest Class:** `Data Architect` (225 samples, 2.21%)
- **Largest Class:** `Software Engineer` (853 samples, 8.38%)
- **Imbalance Ratio:** $853 / 225 = 3.79 : 1$ (well within acceptable boundaries for standard Cross-Entropy loss with optional class weighting).
- **Zero Sparse Classes:** Every consolidated category contains $> 200$ samples.

### Documented Semantic Ambiguities in Consolidation
1. `Project Manager (56)` $\to$ `Product Manager`: In technical organizations, Project Management (execution/timelines) differs from Product Management (vision/strategy/metrics). Given only 56 records, merging is pragmatic, but could be evaluated as an independent class or filtered.
2. `Cloud Architect (254)` $\to$ `Cloud Engineer`: Architecture is senior design; engineering is implementation. With 254 samples, `Cloud Architect` is large enough to stand alone if 27 classes are preferred.
3. `AI Researcher (225)` $\to$ `Machine Learning Engineer`: Research focuses on novel models/papers; engineering focuses on MLOps/pipelines.
4. `Graphic Designer (15)` $\to$ `UI/UX Designer`: Graphic design covers print/branding, while UI/UX focuses on digital interaction.

---

## 5. Existing Model & Backend Pipeline Audit

### Model File Inspection (`Backend/resume_ann_model.keras`)
- **File Format:** Keras v3 zip archive (427.5 KB)
- **Architecture:** Sequential
  - `dense (Dense)`: `(None, 128)` — ReLU, 100,480 params
  - `dropout (Dropout)`: `rate=0.2`
  - `dense_1 (Dense)`: `(None, 10)` — Softmax, 1,290 params
- **Input Dimension:** 784
- **Output Classes:** 10
- **Total Parameters:** 101,770 (Trainable)
- **Status:** Placebo/dummy model.

### Backend Feature Extraction (`Backend/app.py`)
```python
# app.py lines 96-105:
INPUT_DIM = 784
DEFAULT_CATEGORIES = [
    'Software Engineer', 'Data Scientist', 'Product Manager',
    'Data Analyst', 'DevOps Engineer', 'HR Specialist',
    'Marketing Specialist', 'UI/UX Designer', 'Business Analyst',
    'Quality Assurance'
]

# app.py lines 314-328:
def extract_features(text):
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    vec = np.zeros(INPUT_DIM, dtype=np.float32)
    for w in words:
        idx = hash(w) % INPUT_DIM
        vec[idx] += 1.0
    ...
```
**Conclusion:** The backend currently hashes arbitrary words into 784 bins and feeds them to the 10-class network. No TF-IDF model, vocabulary file, or real category mapping is connected to the backend.

---

## 6. Training Implementation Plan (Phase 2 Roadmap)

### Step 1: Preprocessing & Serialization Pipeline (`Data Preprocessing/train_ann.py`)
1. **Extract and Clean:**
   - Stream `dataset.csv` from `Data Set.zip`.
   - Strip LLM preamble prefixes via regex (`^here(?:'s|\s+is)?\b.*?\n`).
   - Clean punctuation, URLs, and casing while preserving technical terms (e.g., `C++`, `.NET`, `CI/CD`).
   - Apply the validated 38 $\to$ 26 category mapping.
2. **TF-IDF Vectorization:**
   - Set `max_features=5000`, `ngram_range=(1, 2)`, `min_df=3`, sublinear term frequency scaling (`sublinear_tf=True`).
   - Fit on `X_train`, transform `X_test`.
   - Serialize vectorizer to `Backend/tfidf_vectorizer.pkl`.
3. **Label Encoding:**
   - Fit `LabelEncoder` on the 26 target categories.
   - Save class list to `Backend/category_mapping.json` and encoder to `Backend/label_encoder.pkl`.

### Step 2: ANN Architecture Specification
- **Input Layer:** `Input(shape=(5000,))`
- **Dense Block 1:** `Dense(512, activation='relu', kernel_regularizer=l2(1e-4))` + `BatchNormalization()` + `Dropout(0.3)`
- **Dense Block 2:** `Dense(256, activation='relu', kernel_regularizer=l2(1e-4))` + `BatchNormalization()` + `Dropout(0.3)`
- **Dense Block 3:** `Dense(128, activation='relu')` + `Dropout(0.2)`
- **Output Layer:** `Dense(26, activation='softmax')`
- **Optimizer:** `Adam(learning_rate=1e-3)` with `ReduceLROnPlateau(patience=3, factor=0.5)`
- **Loss:** `sparse_categorical_crossentropy` (or `categorical_crossentropy`)
- **Callbacks:** `EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)`

### Step 3: Backend Integration (`Backend/app.py`)
1. Update `INPUT_DIM` from 784 to 5,000.
2. Replace `hash(w) % INPUT_DIM` with `tfidf_vectorizer.transform([cleaned_text]).toarray()`.
3. Load `resume_ann_model.keras` (26 outputs) and dynamically bind predictions to `category_mapping.json`.
4. Ensure graceful fallback if artifacts are loading.

---

## 7. Verification Checklist

- [x] ZIP archive inspected non-destructively.
- [x] CSV schema verified (10,174 rows, 8 columns, 0 nulls).
- [x] Duplicate records & ID collisions diagnosed.
- [x] LLM conversational artifacts and target leakage quantified.
- [x] Class distribution analyzed (45 raw $\to$ 38 normalized $\to$ 26 target).
- [x] Existing model architecture inspected ($784 \to 128 \to 10$ dummy verified).
- [x] Backend feature pipeline limitations documented.
- [x] Concrete Phase 2 training script specification drafted.
