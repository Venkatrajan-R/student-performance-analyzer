# 🎓 Smart Student Performance Analyzer

A **production-grade educational analytics application** that ingests student academic data, performs statistical analysis, applies machine learning (Logistic Regression), and delivers actionable insights through an interactive Streamlit interface.

> **Portfolio note:** This project demonstrates full-stack competency across data engineering, ML fundamentals, visualization, and UI development — designed to be walked through in a technical interview.

---

## 📸 Feature Overview

| Tab | What You Get |
|-----|-------------|
| **Overview** | Class KPIs, subject statistics table, student data preview, CSV downloads |
| **Visualizations** | Marks distribution, box plots, performance heatmap, ranking bar chart |
| **Predictions** | Per-student risk classification (High / Moderate / On Track) with fail probability |
| **Suggestions** | Personalized subject-level recommendations with comparison charts |
| **Model Info** | Accuracy, AUC, confusion matrix, feature importance, architecture rationale |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python **3.10+** (required for union type hints `X | Y`)
- pip or conda

### Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd student-analyzer

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

### Using the App
1. Upload `sample_data.csv` (or your own CSV) via the sidebar
2. Adjust the **Pass Mark Threshold** and **At-Risk Threshold** sliders
3. Explore tabs: Overview → Visualizations → Predictions → Suggestions → Model Info
4. Download reports using the ⬇️ buttons

---

## 📂 CSV Format

The expected input format is straightforward:

```csv
Student_ID, Subject1, Subject2, Subject3, ...
S001, 78, 65, 82
S002, 45, 50, 38
```

- **Column 1**: Student identifier (any string)
- **Columns 2+**: Subject names (any names) with numeric marks (0–100)
- Missing values are handled gracefully (excluded from averages)
- Duplicate IDs: first occurrence is kept with a warning

---

## 🏗️ Architectural Decisions

### 1. Why Logistic Regression?

**Chosen over**: Decision Trees, Random Forest, SVM, Neural Networks

**Reasoning**:
- **Interpretability is critical in education**: A teacher needs to explain *why* a student is flagged as at-risk. Logistic Regression's coefficients translate directly to "subjects_failed has the highest impact" — no black box.
- **Probabilistic output**: Unlike SVM (binary) or Decision Trees (confidence via vote count), LR produces a calibrated `P(fail)` probability, enabling nuanced risk tiers ("72% likely to fail") rather than a blunt pass/fail label.
- **Small-N generalization**: Class sizes are typically 20–200 students. Complex models like XGBoost or Neural Networks overfit in this regime. LR with L2 regularization is the principled choice.
- **Speed**: Trains in milliseconds, enabling real-time retraining when new data is uploaded.

**Trade-off accepted**: Random Forest would likely achieve higher accuracy (especially with non-linear decision boundaries) but sacrifices interpretability — a poor trade in a domain where educator trust is paramount.

---

### 2. Feature Engineering

Raw marks are *not* fed directly into the model. Instead, 5 domain-informed features are constructed:

| Feature | Rationale |
|---------|-----------|
| `avg_marks` | Overall academic performance — the primary predictor |
| `min_marks` | Worst subject score — a student can fail overall due to one subject |
| `std_marks` | Score variability — high variance may indicate inconsistency or subject blindspots |
| `subjects_failed` | Count below pass threshold — direct measure of curricular risk |
| `pass_rate` | Fraction passed — normalizes for different numbers of subjects |

**Why not use raw marks as features?** With 5 subjects, that's 5 features — fine. But if the school has 20 subjects, raw marks create a 20-dimensional sparse feature space with multicollinearity. Engineered features remain stable regardless of subject count, making the pipeline schema-agnostic.

---

### 3. Modular Code Structure

```
app.py
├── Section 1: Data Ingestion & Validation    (load_and_validate_csv, engineer_features)
├── Section 2: Statistical Analysis           (compute_class_statistics, compute_subject_statistics)
├── Section 3: Machine Learning               (train_model, predict_students)
├── Section 4: Visualizations                 (plot_* functions)
├── Section 5: Personalized Suggestions       (generate_suggestions)
├── Section 6: Report Generation              (generate_csv_report, fig_to_png_bytes)
└── Section 7: Streamlit UI                   (setup_page, render_*, main)
```

**Why this separation?** Each layer can be independently tested, replaced, or extended. The ML layer doesn't know about the UI; the visualization layer doesn't know about the model. This is the same separation of concerns you'd apply in a production microservices architecture.

---

### 4. Pipeline Design (Sklearn Pipeline)

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(...))
])
```

**Critical**: The scaler is fit **only on training data** and applied to test data. Without a Pipeline, a common mistake is fitting the scaler on the full dataset before splitting — this leaks test information into training, inflating accuracy metrics.

---

### 5. Edge Case Handling

| Edge Case | Handling Strategy |
|-----------|------------------|
| Empty CSV | Early return with descriptive error |
| Single student | Model skipped (can't split/evaluate), analysis proceeds |
| All students pass/fail | Model skipped (single class), labeled with warning |
| Non-numeric marks | `pd.to_numeric(errors='coerce')` → NaN, excluded from averages |
| Marks outside [0, 100] | Clamped with user warning |
| Duplicate Student IDs | Deduplicated, user warned |
| Dataset ≤ 10 students | Training without holdout split (flagged to user) |

---

### 6. Visualization Design Choices

- **Dark theme**: Chosen for readability in low-light environments (classroom projectors) and modern aesthetic.
- **Consistent palette**: `RdYlGn` (Red → Yellow → Green) maps intuitively to fail → borderline → pass across all charts.
- **Pass mark reference line**: Every distribution/box plot includes a dashed red line at the pass threshold — the most immediately useful reference for educators.
- **Heatmap sorted by performance**: Best students at top makes it immediately obvious who needs attention at the bottom.

---

## 📈 Scalability Roadmap

This application is designed as a solid foundation. Here's how each component scales:

```
Current State                          →  Production Scale
──────────────────────────────────────────────────────────
CSV upload                             →  PostgreSQL / MongoDB real-time feed
Logistic Regression                    →  XGBoost / LightGBM (N > 1,000 students)
Single-semester analysis               →  Multi-semester time-series (LSTM)
Streamlit (single-user)                →  FastAPI backend + React frontend
Manual report download                 →  Automated email alerts (high-risk students)
Local file storage                     →  AWS S3 / Google Cloud Storage
Single school                          →  Multi-tenant SaaS with role-based access
```

---

## 🔬 Technical Choices at a Glance

| Component | Choice | Alternative Considered | Reason for Choice |
|-----------|--------|----------------------|-------------------|
| ML Model | Logistic Regression | Random Forest | Interpretability for educators |
| Scaling | StandardScaler | MinMaxScaler | Better for LR with outliers |
| UI | Streamlit | Dash / Flask + React | Faster iteration, native data science UX |
| Data | Pandas | Polars | Broader ecosystem, educator familiarity |
| Charts | Matplotlib + Seaborn | Plotly | Static exports, consistent dark theme |
| Regularization | L2 (C=1.0) | L1, ElasticNet | Prevents overfitting, rarely need sparsity |

---

## 🧪 Known Limitations

1. **No persistence**: Data is not saved between sessions (by design for privacy). Add a database for persistence.
2. **Single-semester**: The model predicts based on current performance only. Semester history would improve accuracy significantly.
3. **No attendance data**: The feature set is marks-only. Attendance correlation with failure is well-documented; add it when available.
4. **Small test sets**: With < 30 students, accuracy metrics have high variance — treat with caution.

---

## 📋 License

MIT License — free for academic and commercial use.
