# Dataset & ANN Model Architecture Plan

## Executive Summary

This plan provides a comprehensive technical analysis of the resume dataset stored in `Data Preprocessing/Data Set.zip` and outlines the exact supervised learning pipeline required to build a robust, deterministic Artificial Neural Network (ANN) resume classifier for the AI Job Application Screener project.

---

## 1. Dataset Structure

* **Archive Location:** `D:\ai_hire\Data Preprocessing\Data Set.zip` (19,222,285 bytes / ~19.2 MB compressed).
* **Extracted Contents:** A single tabular dataset: `Data Set/dataset.csv` (78,088,830 bytes / ~78.1 MB uncompressed).
* **Data Format:** Comma-Separated Values (CSV), UTF-8 encoded.
* **Schema (8 Columns):**
  1. `ID` (Text identifier, e.g., `jasojo159`)
  2. `Name` (Candidate full name)
  3. `Role` (**Target Label column**)
  4. `Transcript` (Simulated interview dialogue)
  5. `Resume` (**Input Feature column — resume text**)
  6. `decision` (Hiring outcome: `accept` / `reject`)
  7. `Reason_for_decision` (Text explanation)
  8. `Job_Description` (Target job description text)
* **Label Encoding Location:** Target categories are explicitly stored as text strings in the **`Role`** column. Labels are **not** in directory names or separate metadata files.

---

## 2. Total Resume Count

* **Total Rows in Dataset:** `10,174`
* **Rows with Missing / Null Resumes:** `0`
* **Rows with Valid Resume Text:** `10,174` (100% data presence)

---

## 3. Actual Categories & Taxonomy

The raw dataset contains **45 raw category labels**, which reduce to **38 distinct categories** once case variations are normalized.

### Grouping and Inconsistencies Found:
1. **Case Duplication:**
   * `Data Scientist` (538) + `data scientist` (287) = **825**
   * `Software Engineer` (480) + `software engineer` (307) = **787**
   * `Product Manager` (458) + `product manager` (303) = **761**
   * `Data Engineer` (447) + `data engineer` (307) = **754**
   * `Data Analyst` (329) + `data analyst` (183) = **512**
   * `UI Engineer` (375) + `ui engineer` (132) = **507**
   * `UI Designer` (71) + `ui designer` (155) = **226**
2. **Sub-Role Overlaps / Near-Synonyms:**
   * `Human Resources Specialist` (262) vs `HR Specialist` (12)
   * `Software Engineer` (787) vs `Software Developer` (66)
   * `Cybersecurity Analyst` (234) vs `Cybersecurity Specialist` (9)
   * `AI Researcher` (225) vs `AI Engineer` (13) vs `Machine Learning Engineer` (265)
   * `UI Designer` (226) vs `UX Designer` (227) vs `Ui/ux Designer` (7)

---

## 4. Resume Count per Category

### Case-Normalized Category Distribution (38 Categories):

| Category | Count | Status |
| :--- | :--- | :--- |
| **Data Scientist** | 825 | High representation |
| **Software Engineer** | 787 | High representation |
| **Product Manager** | 761 | High representation |
| **Data Engineer** | 754 | High representation |
| **Data Analyst** | 512 | High representation |
| **UI Engineer** | 507 | High representation |
| **E-commerce Specialist** | 268 | Balanced representation |
| **DevOps Engineer** | 266 | Balanced representation |
| **Machine Learning Engineer** | 265 | Balanced representation |
| **Human Resources Specialist** | 262 | Balanced representation |
| **Digital Marketing Specialist** | 260 | Balanced representation |
| **Robotics Engineer** | 257 | Balanced representation |
| **Cloud Architect** | 254 | Balanced representation |
| **QA Engineer** | 251 | Balanced representation |
| **Blockchain Developer** | 251 | Balanced representation |
| **Mobile App Developer** | 247 | Balanced representation |
| **Full Stack Developer** | 246 | Balanced representation |
| **Database Administrator** | 243 | Balanced representation |
| **Cloud Engineer** | 240 | Balanced representation |
| **Game Developer** | 239 | Balanced representation |
| **Content Writer** | 238 | Balanced representation |
| **AR/VR Developer** | 237 | Balanced representation |
| **Cybersecurity Analyst** | 234 | Balanced representation |
| **UX Designer** | 227 | Balanced representation |
| **Business Analyst** | 226 | Balanced representation |
| **UI Designer** | 226 | Balanced representation |
| **AI Researcher** | 225 | Balanced representation |
| **Data Architect** | 225 | Balanced representation |
| **IT Support Specialist** | 224 | Balanced representation |
| **System Administrator** | 223 | Balanced representation |
| **Software Developer** | 66 | Mergable into Software Engineer |
| **Project Manager** | 56 | Mergable into Product/Project Manager |
| **Network Engineer** | 16 | Underrepresented |
| **Graphic Designer** | 15 | Underrepresented |
| **AI Engineer** | 13 | Mergable into ML Engineer |
| **HR Specialist** | 12 | Mergable into Human Resources Specialist |
| **Cybersecurity Specialist** | 9 | Mergable into Cybersecurity Analyst |
| **UI/UX Designer** | 7 | Mergable into UX/UI Designer |

---

## 5. Dataset Quality Assessment

1. **Text Origin & Nature:**
   * The resumes in this dataset are LLM-synthesized professional profiles created for an interview simulation benchmark.
   * **8,324 resumes (81.82%)** begin with standardized framing prefixes such as:
     `"Here's a professional resume for [Candidate Name]:"` or `"Here is a professional resume for [Candidate Name]:"`.
   * These introductory lines contain candidate names and target titles; simple regex stripping of this boilerplate line during preprocessing prevents data leakage.
2. **Text Length & Richness:**
   * **Average word count:** 386.1 words.
   * **Resumes with ≥ 200 words:** 9,925 (97.55%).
   * **Resumes < 100 words:** Only 130 (1.28%).
   * The text contains realistic industry sections: Contact Info, Professional Summary, Work Experience, Technical Skills, Projects, and Education.
3. **Suitability for Supervised Classification:**
   * **YES.** With 10,174 samples and ~220–825 examples per consolidated class, this dataset provides more than enough statistical power to train a TF-IDF + ANN classifier to >85–90% test accuracy.

---

## 6. Existing Preprocessing Findings

* **`Data Preprocessing/HireMind_AI.ipynb`:**
  * Was developed on Google Colab using an external file `Resume.csv` (2,484 resumes, 24 non-tech categories: Accountant, Chef, Fitness, Advocate, etc.).
  * Used `clean_resume()` with regex to strip URLs, hashtags, punctuation, and NLTK stopwords.
  * Vectorized using `TfidfVectorizer(max_features=5000, ngram_range=(1,2))`.
  * **Never trained a neural network** (only trained LogisticRegression, NaiveBayes, RandomForest, and XGBoost).
* **`Backend/app.py`:**
  * Defined `INPUT_DIM = 5000` and expected `resume_ann_model.keras`.
  * Because no saved `tfidf.pkl` was provided, it used an ad-hoc fallback corpus (`[text, _DOMAIN_CORPUS, _FILLER]`) to generate 5000-length arrays dynamically.
* **Status of Artifacts:**
  * Neither `tfidf.pkl`, `label_encoder.pkl`, nor any training scripts for Keras exist in the repository.

---

## 7. Existing 24-Category Compatibility

The 24 categories currently hardcoded in `Backend/app.py` (`JOB_CATEGORIES`) do **not** match this dataset:
* **Exact Matches (2/24):** Only `Business Analyst` and `DevOps Engineer`.
* **Missing from Dataset (9/24):** `Advocate`, `Arts`, `Automation Testing`, `Electrical Engineering`, `Hadoop`, `Mechanical Engineer`, `PMO`, `Sales`, `Civil Engineer`.
* **Reason:** `Backend/app.py` adopted the generic 24 categories from Kaggle's legacy `UpdatedResumeDataSet.csv`, whereas `Data Set.zip` contains modern tech and engineering roles.

---

## 8. Recommended Model Architecture

To achieve fast inference (<10ms per resume) on CPU with high classification accuracy:

```text
Input Tensor: (batch_size, 5000)
    │
Dense Layer 1: 256 units, ReLU activation
    │
Batch Normalization
    │
Dropout: rate = 0.3
    │
Dense Layer 2: 128 units, ReLU activation
    │
Dropout: rate = 0.2
    │
Output Layer: N_CLASSES units, Softmax activation
```

* **Loss Function:** `sparse_categorical_crossentropy` (with integer encoded labels) or `categorical_crossentropy`.
* **Optimizer:** `Adam(learning_rate=1e-3)` with learning rate reduction on plateau (`ReduceLROnPlateau(patience=2, factor=0.5)`).
* **Regularization:** Early Stopping (`EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`).
* **Training Speed:** Takes ~30–45 seconds on standard CPU for 15–20 epochs.

---

## 9. Recommended TF-IDF Configuration

* **Estimator:** `sklearn.feature_extraction.text.TfidfVectorizer`
* **Parameters:**
  * `max_features=5000` (Directly matches `INPUT_DIM = 5000` expected by backend)
  * `ngram_range=(1, 2)` (Captures single terms like `"kubernetes"` and compound terms like `"machine learning"`, `"ci cd"`, `"cloud architect"`)
  * `stop_words='english'`
  * `sublinear_tf=True` (Applies logarithmic scaling `1 + log(tf)` to prevent dominant common terms from overpowering rare technical skills)
  * `min_df=3` (Prunes rare typos and single-occurrence tokens)
* **Crucial Persistence:** The fitted vectorizer **MUST** be pickled to `Backend/tfidf.pkl`. Runtime predictions must use `tfidf.transform([cleaned_text])`, guaranteeing that feature indices match the trained ANN weights.

---

## 10. Recommended Label Mapping & Consolidation Strategy

Consolidating near-synonyms and case variants yields **26 clean, well-balanced industry categories**:

1. `Data Scientist` (825)
2. `Software Engineer` (853: includes Software Developer)
3. `Product Manager` (817: includes Project Manager)
4. `Data Engineer` (754)
5. `Data Analyst` (512)
6. `UI Engineer` (507)
7. `Machine Learning Engineer` (503: includes AI Researcher, AI Engineer)
8. `Cloud Architect` (254)
9. `Cloud Engineer` (240)
10. `Full Stack Developer` (246)
11. `DevOps Engineer` (266)
12. `Blockchain Developer` (251)
13. `QA Engineer` (251)
14. `Mobile App Developer` (247)
15. `Database Administrator` (243)
16. `Game Developer` (239)
17. `AR/VR Developer` (237)
18. `Cybersecurity Analyst` (243: includes Cybersecurity Specialist)
19. `UI/UX Designer` (460: includes UI Designer, UX Designer)
20. `Human Resources Specialist` (274: includes HR Specialist)
21. `Digital Marketing Specialist` (260)
22. `Robotics Engineer` (257)
23. `Content Writer` (238)
24. `Business Analyst` (226)
25. `Data Architect` (225)
26. `System Administrator` (447: includes IT Support Specialist)

*(Note: Network Engineer [16] and Graphic Designer [15] can either be filtered out or mapped to System Administrator and UI/UX Designer).*

---

## 11. Recommended Train / Validation / Test Split

* **Dataset Size:** 10,174 resumes
* **Split Ratios:**
  * **Train:** 70% (~7,121 resumes)
  * **Validation:** 15% (~1,526 resumes)
  * **Test:** 15% (~1,527 resumes)
* **Stratification:** Must use `stratify=y` so every category has identical representation in train, validation, and test splits.
* **Class Weights:** Compute `class_weight.compute_class_weight('balanced', ...)` and pass to `model.fit()` to ensure smaller classes receive proportionate gradients.

---

## 12. Comparison: Option A vs. Option B

| Dimension | Option A: Obtain External 24-Category Dataset | Option B: Use Existing 10,174 `Data Set.zip` |
| :--- | :--- | :--- |
| **Availability** | **NOT AVAILABLE** in repo (must be sourced/downloaded externally) | **ALREADY AVAILABLE** at `Data Preprocessing/Data Set.zip` |
| **Data Size** | Small (typically 962 to 2,484 resumes) | Large (10,174 complete resumes) |
| **Backend Compatibility** | Matches current `JOB_CATEGORIES` in `Backend/app.py` without taxonomy edit | Requires updating `JOB_CATEGORIES` in `Backend/app.py` to match tech roles |
| **Domain Relevance** | Broad/legacy (Advocate, Chef, Fitness, Civil Engineer) | Highly relevant to modern Tech, Data, Engineering & Product hiring |
| **Model Accuracy** | Moderate (~75–80% due to small sample size) | High (>88–92% due to 10k rich resumes) |
| **Project Self-Sufficiency**| Low (fails standalone build without external files) | High (fully self-contained inside `D:\ai_hire`) |

---

## 13. Final Recommendation

**RECOMMENDATION: OPTION B (with an automated clean taxonomy mapping).**

* **Rationale:**
  1. The project already contains the complete 10,174-resume dataset locally in `Data Preprocessing/Data Set.zip`.
  2. Training an ANN on 10,174 samples produces significantly higher classification confidence and accuracy than training on a small ~1,000–2,000 resume dataset.
  3. The roles (Software Engineer, Data Scientist, Cloud Architect, DevOps, Product Manager, etc.) reflect real-world recruitment screener use cases far better than legacy categories like "Chef" or "Advocate".
  4. Updating `JOB_CATEGORIES` in `Backend/app.py` to reflect the 26 consolidated roles takes minimal effort and eliminates the disconnect between model, vocabulary, and server.

---

## 14. Exact Files to Create / Modify in the Next Training Phase

When ready to proceed to training:

1. **`Data Preprocessing/train_ann.py` [NEW]**
   * Reads `Data Set/dataset.csv` directly from `Data Set.zip`.
   * Strips LLM intro prefixes and cleans text.
   * Normalizes and encodes the consolidated target categories.
   * Fits `TfidfVectorizer(max_features=5000, ngram_range=(1,2))`.
   * Trains Keras Sequential ANN (5000 → 256 → BN → Dropout → 128 → Softmax).
   * Evaluates on held-out 15% test set.
   * Exports:
     * `Backend/resume_ann_model.keras` (Replaces dummy 784x10 model)
     * `Backend/tfidf.pkl` (Fitted 5000-dim vectorizer)
     * `Backend/label_classes.json` (List of categories and index mapping)
2. **`Backend/app.py` [UPDATE]**
   * Load `Backend/tfidf.pkl` and `Backend/label_classes.json` deterministically at startup.
   * Update `predict_category(text)` to transform input text via `tfidf.transform([cleaned])` and infer via `model.predict()`.
   * Update `JOB_CATEGORIES` list to match `label_classes.json`.
