# AI Resume Classifier — Taxonomy & Model Training Specification

## 1. Dataset Summary

* **Dataset Source:** `D:\ai_hire\Data Preprocessing\Data Set.zip` (`Data Set/dataset.csv`).
* **Total Valid Resumes:** **10,174** (Zero missing values; 100% complete text).
* **Feature Column:** `Resume` (LLM-synthesized professional resumes, average 386 words, 97.55% $\ge 200$ words).
* **Label Column:** `Role` (Case variations and sub-roles present).
* **Raw Extracted Categories:** 45 strings, which resolve to 38 distinct roles after case normalization.
* **Consolidated Production Categories:** Exactly **26 balanced industry categories**.
* **Dataset Imbalance Ratio:** Mild (Largest: 853 [8.38%], Smallest: 225 [2.21%]; Ratio: 3.79:1).

---

## 2. Complete 38 → 26 Category Mapping

All 38 case-normalized raw categories and their exact mapping to the 26 production categories:

| # | Raw Category | Resume Count | Consolidated Category | Rationale for Consolidation |
|---|:---|:---:|:---|:---|
| 1 | **Data Scientist** | 825 | **Data Scientist** | Exact match / standalone core role. |
| 2 | **Software Engineer** | 787 | **Software Engineer** | Primary software engineering category. |
| 3 | **Product Manager** | 761 | **Product Manager** | Primary product leadership category. |
| 4 | **Data Engineer** | 754 | **Data Engineer** | Exact match / standalone core role. |
| 5 | **Data Analyst** | 512 | **Data Analyst** | Exact match / standalone core role. |
| 6 | **UI Engineer** | 507 | **UI Engineer** | Frontend coding (HTML/CSS/JS/React). Distinct from pure design. |
| 7 | **E-commerce Specialist** | 268 | **E-commerce Specialist** | Exact match / retail & catalog analytics role. |
| 8 | **Devops Engineer** | 266 | **DevOps Engineer** | CI/CD, automation, cloud deployment. Standardized spelling. |
| 9 | **Machine Learning Engineer** | 265 | **Machine Learning Engineer** | Core predictive modeling & ML systems role. |
| 10 | **Human Resources Specialist** | 262 | **Human Resources Specialist** | Primary talent acquisition & HR role. |
| 11 | **Digital Marketing Specialist** | 260 | **Digital Marketing Specialist** | Exact match / SEO, SEM, content distribution. |
| 12 | **Robotics Engineer** | 257 | **Robotics Engineer** | Exact match / hardware, ROS, embedded control systems. |
| 13 | **Cloud Architect** | 254 | **Cloud Engineer** | Resumes share 90%+ skill overlap with Cloud Engineer (AWS/Azure/Terraform). Merged into unified Cloud Engineering. |
| 14 | **QA Engineer** | 251 | **QA Engineer** | Exact match / software testing & quality assurance. |
| 15 | **Blockchain Developer** | 251 | **Blockchain Developer** | Exact match / web3, smart contracts, cryptography. |
| 16 | **Mobile App Developer** | 247 | **Mobile App Developer** | Exact match / iOS, Android, Flutter, React Native. |
| 17 | **Full Stack Developer** | 246 | **Full Stack Developer** | Exact match / end-to-end web application development. |
| 18 | **Database Administrator** | 243 | **Database Administrator** | Exact match / DBA, SQL tuning, replication, backup. |
| 19 | **Cloud Engineer** | 240 | **Cloud Engineer** | Core cloud infrastructure & deployment role. |
| 20 | **Game Developer** | 239 | **Game Developer** | Exact match / Unity, Unreal, C++, graphics shaders. |
| 21 | **Content Writer** | 238 | **Content Writer** | Exact match / technical documentation, copywriting. |
| 22 | **AR/VR Developer** | 237 | **AR/VR Developer** | Exact match / spatial computing, XR, 3D engines. |
| 23 | **Cybersecurity Analyst** | 234 | **Cybersecurity Analyst** | Core InfoSec, threat detection, penetration testing. |
| 24 | **UX Designer** | 227 | **UI/UX Designer** | High overlap in design thinking, user journeys, and wireframing. Merged into UI/UX Designer. |
| 25 | **Business Analyst** | 226 | **Business Analyst** | Exact match / requirements gathering, BI, workflows. |
| 26 | **UI Designer** | 226 | **UI/UX Designer** | High overlap in Figma, design systems, visual hierarchy. Merged into UI/UX Designer. |
| 27 | **AI Researcher** | 225 | **Machine Learning Engineer** | Core ML research and algorithmic modeling. Merged into ML Engineer. |
| 28 | **Data Architect** | 225 | **Data Architect** | Enterprise data modeling, schema governance, warehousing. |
| 29 | **IT Support Specialist** | 224 | **System Administrator** | Overlapping responsibilities in workstation, server, network, and user management. Merged into System Admin. |
| 30 | **System Administrator** | 223 | **System Administrator** | Core Linux/Windows server administration & infrastructure. |
| 31 | **Software Developer** | 66 | **Software Engineer** | Direct semantic synonym for Software Engineer. |
| 32 | **Project Manager** | 56 | **Product Manager** | Delivery, agile ceremonies, and roadmap management overlap. Merged into Product Manager. |
| 33 | **Network Engineer** | 16 | **System Administrator** | Severely underrepresented class (16 samples). Responsibilities directly align with System & Network Administration. |
| 34 | **Graphic Designer** | 15 | **UI/UX Designer** | Severely underrepresented class (15 samples). Directly aligns with visual UI/UX asset creation. |
| 35 | **AI Engineer** | 13 | **Machine Learning Engineer** | Direct synonym for ML/AI engineering. |
| 36 | **HR Specialist** | 12 | **Human Resources Specialist** | Direct acronym synonym for Human Resources Specialist. |
| 37 | **Cybersecurity Specialist** | 9 | **Cybersecurity Analyst** | Direct synonym for Cybersecurity Analyst. |
| 38 | **UI/UX Designer** | 7 | **UI/UX Designer** | Exact compound title for UI/UX Design. |

---

## 3. Final 26 Categories (Sorted by Resume Count)

| Rank | Final Category | Resumes | % of Dataset | Raw Merged Count | Sample Top Keywords |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | **Software Engineer** | 853 | 8.38% | 2 (`Software Engineer`, `Software Developer`) | java, python, c++, git, oop, microservices, algorithms |
| 2 | **Data Scientist** | 825 | 8.11% | 1 (`Data Scientist`) | machine learning, statistics, python, pandas, modeling |
| 3 | **Product Manager** | 817 | 8.03% | 2 (`Product Manager`, `Project Manager`) | roadmap, agile, sprint, backlog, stakeholders, kpis |
| 4 | **Data Engineer** | 754 | 7.41% | 1 (`Data Engineer`) | spark, etl, pipeline, sql, airflow, snowflake, kafka |
| 5 | **Data Analyst** | 512 | 5.03% | 1 (`Data Analyst`) | sql, tableau, power bi, excel, visualization, reporting |
| 6 | **UI Engineer** | 507 | 4.98% | 1 (`UI Engineer`) | react, javascript, css, html, typescript, frontend |
| 7 | **Machine Learning Engineer** | 503 | 4.94% | 3 (`ML Engineer`, `AI Researcher`, `AI Engineer`) | pytorch, tensorflow, deep learning, nlp, mlops |
| 8 | **Cloud Engineer** | 494 | 4.86% | 2 (`Cloud Architect`, `Cloud Engineer`) | aws, azure, terraform, gcp, kubernetes, iam, vpc |
| 9 | **UI/UX Designer** | 475 | 4.67% | 4 (`UX Designer`, `UI Designer`, `UI/UX`, `Graphic`) | figma, user research, wireframes, prototyping, persona |
| 10 | **System Administrator** | 463 | 4.55% | 3 (`System Admin`, `IT Support`, `Network Engineer`) | linux, active directory, bash, networking, dns, vpn |
| 11 | **Human Resources Specialist**| 274 | 2.69% | 2 (`HR Specialist`, `Human Resources Specialist`) | recruitment, onboarding, talent acquisition, hr policies |
| 12 | **E-commerce Specialist** | 268 | 2.63% | 1 (`E-commerce Specialist`) | shopify, inventory, seo, merchandising, conversion |
| 13 | **DevOps Engineer** | 266 | 2.61% | 1 (`Devops Engineer`) | docker, kubernetes, jenkins, ci cd, ansible, helm |
| 14 | **Digital Marketing Specialist**| 260 | 2.56% | 1 (`Digital Marketing Specialist`) | google analytics, sem, content strategy, campaigns |
| 15 | **Robotics Engineer** | 257 | 2.53% | 1 (`Robotics Engineer`) | ros, c++, kinematics, motion planning, sensors, slam |
| 16 | **QA Engineer** | 251 | 2.47% | 1 (`QA Engineer`) | selenium, test cases, automation, junit, bug tracking |
| 17 | **Blockchain Developer** | 251 | 2.47% | 1 (`Blockchain Developer`) | solidity, smart contracts, ethereum, web3, crypto |
| 18 | **Mobile App Developer** | 247 | 2.43% | 1 (`Mobile App Developer`) | flutter, swift, kotlin, react native, ios, android |
| 19 | **Full Stack Developer** | 246 | 2.42% | 1 (`Full Stack Developer`) | node.js, react, express, mongodb, postgresql, apis |
| 20 | **Database Administrator** | 243 | 2.39% | 1 (`Database Administrator`) | oracle, postgresql, backup, indexing, sql server, ha |
| 21 | **Cybersecurity Analyst** | 243 | 2.39% | 2 (`Cybersecurity Analyst`, `Cybersecurity Spec`) | soc, siem, penetration testing, firewall, nist, incident |
| 22 | **Game Developer** | 239 | 2.35% | 1 (`Game Developer`) | unity, unreal engine, c#, 3d graphics, physics |
| 23 | **Content Writer** | 238 | 2.34% | 1 (`Content Writer`) | copywriting, articles, editing, storytelling, seo |
| 24 | **AR/VR Developer** | 237 | 2.33% | 1 (`AR/VR Developer`) | xr, openxr, spatial audio, oculus, unity vr, shaders |
| 25 | **Business Analyst** | 226 | 2.22% | 1 (`Business Analyst`) | requirement gathering, user stories, gap analysis, uml |
| 26 | **Data Architect** | 225 | 2.21% | 1 (`Data Architect`) | schema design, enterprise data, governance, dw |
| **TOTAL** | **26 Categories** | **10,174** | **100.0%** | **38 Raw Mapped** | — |

---

## 4. Taxonomy Validation & Quality Checks

1. **No Duplicate Category Names:**
   * Every final category string is distinct and canonical.
2. **No Severely Underrepresented Classes:**
   * Every final class contains between **225 and 853 resumes** ($\ge 2.21\%$ of total).
   * Classes previously having $<20$ samples (`Network Engineer` [16], `Graphic Designer` [15], `AI Engineer` [13], `HR Specialist` [12], `Cybersecurity Specialist` [9], `UI/UX Designer` [7]) have been systematically merged into their direct logical parents.
3. **Clear Functional Separation (No Ambiguity):**
   * `UI Engineer` (Frontend code: React/JS/CSS) is kept separate from `UI/UX Designer` (Visual/UX design: Figma/Research).
   * `Data Scientist` (ML/Stats) is kept distinct from `Data Engineer` (Pipelines/ETL) and `Data Analyst` (BI/Dashboards).
   * `Cloud Engineer` encompasses both cloud architecture and operations, which prevents false inter-class confusion in TF-IDF feature space.
4. **Prediction Ambiguity Mitigation:**
   * Synthetic resume boilerplate framing (`"Here's a professional resume for [Name]:"`) will be stripped out in preprocessing so the model does not latch onto candidate names or early title tokens.

---

## 5. Comparison: `Backend/app.py` 24 Categories vs Dataset 26 Categories

| Category Status | Count | Details |
|:---|:---:|:---|
| **Exact Name Matches** | **2** | `Business Analyst`, `DevOps Engineer` |
| **Direct Semantic Equivalents** | **7** | `Blockchain` → `Blockchain Developer`<br>`Data Science` → `Data Scientist`<br>`Database` → `Database Administrator`<br>`HR` → `Human Resources Specialist`<br>`Network Security Engineer` → `Cybersecurity Analyst`<br>`Testing` / `Automation Testing` → `QA Engineer`<br>`Web Designing` → `UI/UX Designer` / `UI Engineer` |
| **Missing from Dataset** | **9** | `Advocate` (Law), `Arts`, `Civil Engineer`, `Mechanical Engineer`, `Electrical Engineering`, `Sales`, `Hadoop` (Legacy stack), `PMO`, `Operations Manager` |
| **Language-Specific Legacy Roles** | **4** | `Python Developer`, `Java Developer`, `DotNet Developer`, `SAP Developer` (In the modern dataset, these fall naturally into `Software Engineer` or `Full Stack Developer`). |
| **New Modern Tech Roles in Dataset** | **15** | `Cloud Engineer`, `Machine Learning Engineer`, `Data Engineer`, `Data Analyst`, `System Administrator`, `E-commerce Specialist`, `Digital Marketing Specialist`, `Robotics Engineer`, `Mobile App Developer`, `Full Stack Developer`, `Game Developer`, `Content Writer`, `AR/VR Developer`, `Data Architect`, `UI Engineer` |

---

## 6. Final Model & Pipeline Specification

### A. Preprocessing & Cleaning
1. **Header Stripping:** Regex strip LLM resume intro prefixes (`r"^(here\s*is|here\s*'s)\s+a\s+professional\s+resume\s+for\s+[^:]+:\s*"`).
2. **Noise Removal:** Remove emails, URLs, HTML tags, and non-ASCII artifacts.
3. **Lowercasing & Normalization:** Standardize whitespace and preserve alphanumeric tokens plus technical symbols (`c++`, `c#`, `.net`).

### B. TF-IDF Vectorizer Configuration
* `max_features`: **5,000** (Strictly matches `INPUT_DIM = 5000` expected by backend architecture).
* `ngram_range`: **(1, 2)** (Captures single terms like `"kubernetes"` and phrases like `"machine learning"`, `"ci cd"`, `"cloud infrastructure"`).
* `min_df`: **3** (Discards one-off typos and anomalous single-instance words).
* `max_df`: **0.85** (Discards universally frequent filler words).
* `sublinear_tf`: **True** (Applies $1 + \log(tf)$ dampening to prevent repetitive keywords from overwhelming feature vectors).
* `stop_words`: `'english'`
* `token_pattern`: `r'(?u)\b[a-zA-Z][a-zA-Z0-9_\-\+\#]{1,}\b'`

### C. Artificial Neural Network (ANN) Architecture
```text
Input Layer: shape=(5000,)
   │
Dense Layer 1: 256 units, activation='relu', kernel_initializer='he_normal'
   │
Batch Normalization
   │
Dropout: rate = 0.30
   │
Dense Layer 2: 128 units, activation='relu', kernel_initializer='he_normal'
   │
Dropout: rate = 0.20
   │
Output Layer: 26 units, activation='softmax'
```
* **Loss Function:** `sparse_categorical_crossentropy` (integer-encoded targets).
* **Optimizer:** `Adam(learning_rate=1e-3)`.
* **Batch Size:** `64`.
* **Epochs:** `25`.
* **Callbacks:**
  * `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`
  * `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)`

### D. Dataset Split & Class Balancing
* **Train Set:** **70%** (7,121 resumes)
* **Validation Set:** **15%** (1,526 resumes)
* **Test Set:** **15%** (1,527 resumes)
* **Stratification:** Strict `stratify=y` on all splits so class proportions are preserved across train, validation, and test partitions.
* **Class Weighting:** Compute balanced class weights via `compute_class_weight('balanced', ...)` and pass to `model.fit()` to ensure smaller classes (e.g., Data Architect, Business Analyst) receive proportional gradient updates.

---

## 7. Required Production Artifacts

To guarantee 100% deterministic prediction at runtime, the training script will output:

1. **`D:\ai_hire\Backend\resume_ann_model.keras`**
   * Trained Keras Sequential model (Input: 5000, Output: 26 Softmax).
   * Saved in standard Keras v3 format.
2. **`D:\ai_hire\Backend\tfidf_vectorizer.pkl`**
   * The fitted Scikit-Learn `TfidfVectorizer` (5,000 vocabulary features).
   * Persisted using `joblib` / `pickle`.
3. **`D:\ai_hire\Backend\category_mapping.json`**
   * Canonical JSON file containing:
     * `classes`: Ordered list of 26 categories matching output layer indices `0..25`.
     * `category_to_index`: Reverse lookup mapping.
     * `raw_mapping`: The 38 → 26 consolidation lookup table.

---

## 8. Model Validation Requirements & Acceptance Criteria

The newly trained model will **NOT** replace the existing model unless it meets all of the following benchmark criteria on the unseen 15% test set (1,527 resumes):

| Metric | Minimum Required | Target |
|:---|:---:|:---:|
| **Test Accuracy** | $\ge \mathbf{88.0\%}$ | $\ge 91.0\%$ |
| **Macro F1-Score** | $\ge \mathbf{85.0\%}$ | $\ge 88.0\%$ |
| **Weighted F1-Score** | $\ge \mathbf{88.0\%}$ | $\ge 91.0\%$ |
| **Minimum Class F1-Score** | $\ge \mathbf{72.0\%}$ | $\ge 80.0\%$ (No blind spots) |
| **Top-3 Accuracy** | $\ge \mathbf{96.0\%}$ | $\ge 98.0\%$ |
| **Generalization Gap** | $\le \mathbf{5.0\%}$ | (Train Acc − Val Acc $\le 5\%$) |

---

## 9. Next Phase: Exact Files to Create / Modify

During the training phase:
1. **`Data Preprocessing/train_ann.py` [NEW]** — Standalone, reproducible training script.
2. **`Backend/resume_ann_model.keras` [UPDATE]** — Replace dummy 784×10 model with new 5000×26 model.
3. **`Backend/tfidf_vectorizer.pkl` [NEW]** — Save fitted 5,000-dim vectorizer.
4. **`Backend/category_mapping.json` [NEW]** — Save category list and indices.
5. **`Backend/app.py` [UPDATE]** — Load vectorizer and JSON mapping on startup; replace dynamic fallback text vectorizer.

---

## 10. Identified Risks & Mitigations

* **Risk 1: Synthetic Resume Wording Leakage**
  * *Mitigation:* The training pipeline automatically strips standard leading prefixes (`"Here's a professional resume for..."`) before passing text to TF-IDF.
* **Risk 2: Vocabulary Drift Across Inferences**
  * *Mitigation:* Saving `tfidf_vectorizer.pkl` guarantees feature index alignment across every inference call.
* **Risk 3: Frontend Category Display Mismatch**
  * *Mitigation:* `Backend/app.py` serves `/api/categories` dynamically from `category_mapping.json`, ensuring frontend category dropdowns reflect the exact 26 categories.
