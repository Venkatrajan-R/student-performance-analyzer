"""
Smart Student Performance Analyzer
====================================
A production-grade educational analytics application that ingests student
academic data, performs statistical analysis, applies predictive modeling
(Logistic Regression), and delivers actionable insights via Streamlit.

Architecture Decision: Modular design separates concerns cleanly —
data layer, ML layer, visualization layer, and UI layer are independent,
making each testable and swappable without touching other components.

Author: Built for portfolio demonstration
"""

import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit compatibility
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from sklearn.pipeline import Pipeline
import base64

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# GLOBAL CONFIGURATION
# ─────────────────────────────────────────────
RANDOM_SEED = 42          # Ensures reproducibility across runs
PASS_THRESHOLD = 40       # Marks threshold to count a subject as passed
OVERALL_PASS_THRESHOLD = 40  # Overall average threshold for pass/fail label
AT_RISK_THRESHOLD = 50    # Students below this average are flagged at-risk

# ─────────────────────────────────────────────
# SECTION 1: DATA INGESTION & VALIDATION
# ─────────────────────────────────────────────

def load_and_validate_csv(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    """
    Load a CSV file and validate its structure for the expected schema.

    Expected schema:
        - Column 0: Student ID (any name, treated as identifier)
        - Column 1+: Subject columns (numeric marks 0–100)

    Returns:
        (DataFrame, None) on success, (None, error_message) on failure.

    Edge cases handled:
        - Empty files or single-row files
        - Non-numeric mark columns
        - Marks out of 0–100 range (clamped with warning)
        - Duplicate student IDs
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Failed to parse CSV: {e}"

    if df.empty:
        return None, "The uploaded file is empty."

    if df.shape[1] < 2:
        return None, "CSV must have at least one ID column and one subject column."

    if df.shape[0] < 1:
        return None, "No student records found in the file."

    # Rename first column to a canonical identifier
    df = df.rename(columns={df.columns[0]: "Student_ID"})
    df["Student_ID"] = df["Student_ID"].astype(str).str.strip()

    # Identify subject columns (everything after Student_ID)
    subject_cols = df.columns[1:].tolist()

    # Coerce subject columns to numeric; non-numeric values become NaN
    for col in subject_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Warn about any NaN marks after coercion
    nan_count = df[subject_cols].isna().sum().sum()
    if nan_count > 0:
        st.warning(f"⚠️  {nan_count} invalid mark(s) found and treated as missing (NaN). "
                   "They are excluded from per-student averages.")

    # Clamp marks to valid range [0, 100]
    for col in subject_cols:
        out_of_range = ((df[col] < 0) | (df[col] > 100)).sum()
        if out_of_range > 0:
            st.warning(f"⚠️  {out_of_range} mark(s) in '{col}' are outside [0, 100] and will be clamped.")
        df[col] = df[col].clip(0, 100)

    # Check for duplicate Student IDs
    dupes = df["Student_ID"].duplicated().sum()
    if dupes > 0:
        st.warning(f"⚠️  {dupes} duplicate Student ID(s) found. Keeping the first occurrence.")
        df = df.drop_duplicates(subset="Student_ID", keep="first")

    return df, None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering layer — creates derived columns used by the ML model.

    Features created:
        - avg_marks      : Mean across all subject marks (ignoring NaN)
        - min_marks      : Worst subject score (signals subject-level weakness)
        - std_marks      : Variability across subjects (high std = uneven performance)
        - subjects_failed: Count of subjects below PASS_THRESHOLD
        - pass_rate      : Fraction of subjects passed
        - label          : Binary target — 1 = Pass (avg ≥ OVERALL_PASS_THRESHOLD), 0 = Fail

    Design Decision: These hand-crafted features encode domain knowledge
    (educators care about weakest subject, not just averages) and are far
    more interpretable than raw marks fed directly into the model.
    """
    subject_cols = df.columns[1:].tolist()

    df = df.copy()
    df["avg_marks"] = df[subject_cols].mean(axis=1)
    df["min_marks"] = df[subject_cols].min(axis=1)
    df["max_marks"] = df[subject_cols].max(axis=1)
    df["std_marks"] = df[subject_cols].std(axis=1).fillna(0)

    # Subjects failed: marks below pass threshold
    df["subjects_failed"] = (df[subject_cols] < PASS_THRESHOLD).sum(axis=1)

    # Pass rate: fraction of subjects passed
    n_subjects = len(subject_cols)
    df["pass_rate"] = 1 - (df["subjects_failed"] / n_subjects)

    # Binary label for classification
    df["label"] = (df["avg_marks"] >= OVERALL_PASS_THRESHOLD).astype(int)

    return df


# ─────────────────────────────────────────────
# SECTION 2: STATISTICAL ANALYSIS
# ─────────────────────────────────────────────

def compute_class_statistics(df: pd.DataFrame) -> dict:
    """
    Compute aggregate class-level statistics for the dashboard.

    Returns a dict with:
        class_avg, topper_name, topper_avg, lowest_name, lowest_avg,
        pass_count, fail_count, at_risk_count
    """
    subject_cols = df.columns[1:].tolist()
    # Exclude engineered columns if present
    subject_cols = [c for c in subject_cols if c not in
                    ["avg_marks", "min_marks", "max_marks", "std_marks",
                     "subjects_failed", "pass_rate", "label"]]

    avg_col = df[subject_cols].mean(axis=1)

    topper_idx = avg_col.idxmax()
    lowest_idx = avg_col.idxmin()

    return {
        "class_avg": avg_col.mean(),
        "topper_name": df.loc[topper_idx, "Student_ID"],
        "topper_avg": avg_col[topper_idx],
        "lowest_name": df.loc[lowest_idx, "Student_ID"],
        "lowest_avg": avg_col[lowest_idx],
        "pass_count": (avg_col >= OVERALL_PASS_THRESHOLD).sum(),
        "fail_count": (avg_col < OVERALL_PASS_THRESHOLD).sum(),
        "at_risk_count": ((avg_col >= OVERALL_PASS_THRESHOLD) &
                          (avg_col < AT_RISK_THRESHOLD)).sum(),
        "total": len(df),
        "avg_per_student": avg_col,
    }


def compute_subject_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-subject breakdown: mean, median, pass rate, difficulty index.

    Difficulty Index = fraction of students who FAILED the subject.
    A higher difficulty index means the subject is harder.
    """
    subject_cols = [c for c in df.columns[1:] if c not in
                    ["avg_marks", "min_marks", "max_marks", "std_marks",
                     "subjects_failed", "pass_rate", "label"]]

    stats = []
    for col in subject_cols:
        series = df[col].dropna()
        n_students = len(series)
        n_passed = (series >= PASS_THRESHOLD).sum()
        stats.append({
            "Subject": col,
            "Mean": round(series.mean(), 2),
            "Median": round(series.median(), 2),
            "Std Dev": round(series.std(), 2),
            "Min": int(series.min()),
            "Max": int(series.max()),
            "Pass Rate (%)": round(100 * n_passed / n_students, 1) if n_students else 0,
            "Difficulty Index": round(1 - n_passed / n_students, 3) if n_students else 0,
        })

    return pd.DataFrame(stats).set_index("Subject")


# ─────────────────────────────────────────────
# SECTION 3: MACHINE LEARNING — LOGISTIC REGRESSION
# ─────────────────────────────────────────────

def train_model(df_engineered: pd.DataFrame):
    """
    Train a Logistic Regression classifier to predict student pass/fail.

    Why Logistic Regression?
        1. Interpretability: Coefficients directly explain feature importance —
           critical for educators who need to justify decisions.
        2. Probabilistic output: Gives a confidence score (0–1), not just a
           binary label — perfect for risk tiering ("75% likely to fail").
        3. Simplicity: With 5–10 engineered features, a linear boundary is
           often sufficient and avoids overfitting on small datasets.
        4. Speed: Trains instantly even on a laptop, enabling real-time
           retraining when data is updated.

    Trade-off considered: Random Forests would give higher accuracy but
    coefficients aren't directly interpretable by teachers. For this
    domain (education), explainability > raw accuracy.

    Pipeline approach: StandardScaler + LogisticRegression in a sklearn
    Pipeline prevents data leakage — scaler is fit only on training data.
    """
    feature_cols = ["avg_marks", "min_marks", "std_marks", "subjects_failed", "pass_rate"]
    X = df_engineered[feature_cols]
    y = df_engineered["label"]

    # Handle edge case: only one class present (all pass or all fail)
    if y.nunique() < 2:
        return None, feature_cols, "Only one class present in data — model cannot be trained. " \
                                   "Ensure dataset has both passing and failing students."

    # Train/test split — stratified to preserve class balance
    # If dataset is tiny (<= 10 students), skip splitting and train on all
    if len(df_engineered) > 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
        st.info("ℹ️  Small dataset detected (<= 10 students). Training on full data without holdout.")

    # Pipeline: scaling + regularized logistic regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=500,
            C=1.0,          # L2 regularization (default) prevents overfitting
            solver="lbfgs"  # Efficient for small-to-medium datasets
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    # AUC only computable if both classes appear in test set
    if y_test.nunique() > 1:
        # roc_auc_score expects P(positive class). Our positive class is 1 (Pass).
        metrics["auc"] = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

    return pipeline, feature_cols, metrics


def predict_students(pipeline, df_engineered: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Run predictions on all students and attach risk classification.

    Risk tiers (based on failure probability):
        High Risk    : P(fail) >= 0.65  → Needs immediate intervention
        Moderate Risk: P(fail) >= 0.35  → Monitor closely
        On Track     : P(fail) <  0.35  → Performing well
    """
    X = df_engineered[feature_cols]
    # label=1 means Pass, label=0 means Fail.
    # predict_proba returns [P(class_0), P(class_1)] = [P(Fail), P(Pass)].
    # We want P(Fail) → index 0.
    proba = pipeline.predict_proba(X)[:, 0]  # Probability of FAILING

    def classify_risk(p):
        if p >= 0.65:
            return "🔴 High Risk"
        elif p >= 0.35:
            return "🟡 Moderate Risk"
        else:
            return "🟢 On Track"

    results = df_engineered[["Student_ID", "avg_marks"]].copy()
    results["Fail_Probability"] = (proba * 100).round(1)
    results["Pass_Probability"] = ((1 - proba) * 100).round(1)
    results["Risk_Level"] = proba.map(classify_risk) if hasattr(proba, 'map') else pd.Series(proba).map(classify_risk)
    results["Risk_Level"] = [classify_risk(p) for p in proba]
    results["Predicted"] = ["Fail" if p >= 0.5 else "Pass" for p in proba]

    return results.reset_index(drop=True)


# ─────────────────────────────────────────────
# SECTION 4: VISUALIZATIONS
# ─────────────────────────────────────────────

# Palette: clean, professional, accessible
PALETTE = {
    "primary":   "#2E86AB",
    "success":   "#28A745",
    "warning":   "#FFC107",
    "danger":    "#DC3545",
    "neutral":   "#6C757D",
    "bg":        "#F8F9FA",
    "text":      "#212529",
}

def _style_fig(fig, ax_or_axes):
    """Apply consistent professional styling to all matplotlib figures."""
    fig.patch.set_facecolor("#0F1117")
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    for ax in np.array(axes).flat:
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="#C8CDD8", labelsize=9)
        ax.xaxis.label.set_color("#C8CDD8")
        ax.yaxis.label.set_color("#C8CDD8")
        ax.title.set_color("#E8ECF0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2A2D3A")
    return fig


def plot_marks_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Overlapping KDE + histogram for each subject's marks distribution.
    Reveals subject difficulty and bimodal patterns (students cluster at pass/fail boundary).
    """
    subject_cols = [c for c in df.columns[1:] if c not in
                    ["avg_marks", "min_marks", "max_marks", "std_marks",
                     "subjects_failed", "pass_rate", "label"]]

    n = len(subject_cols)
    cols_grid = min(3, n)
    rows_grid = (n + cols_grid - 1) // cols_grid

    fig, axes = plt.subplots(rows_grid, cols_grid,
                              figsize=(5.5 * cols_grid, 4 * rows_grid))
    axes_flat = np.array(axes).flat if n > 1 else [axes]

    colors = plt.cm.plasma(np.linspace(0.2, 0.85, n))

    for i, (col, ax) in enumerate(zip(subject_cols, axes_flat)):
        data = df[col].dropna()
        ax.hist(data, bins=min(15, len(data)), alpha=0.4,
                color=colors[i], edgecolor="white", linewidth=0.5)
        if len(data) > 2:
            data.plot.kde(ax=ax, color=colors[i], linewidth=2.5)
        ax.axvline(PASS_THRESHOLD, color=PALETTE["danger"], linestyle="--",
                   linewidth=1.5, label=f"Pass Mark ({PASS_THRESHOLD})")
        ax.axvline(data.mean(), color=PALETTE["warning"], linestyle="-.",
                   linewidth=1.5, label=f"Mean ({data.mean():.1f})")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("Marks")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=7)

    # Hide unused subplots
    for j in range(i + 1, rows_grid * cols_grid):
        list(np.array(axes).flat)[j].set_visible(False)

    fig.suptitle("📊 Marks Distribution per Subject", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return _style_fig(fig, list(np.array(axes).flat)[:n])


def plot_subject_boxplot(df: pd.DataFrame) -> plt.Figure:
    """
    Box plots reveal median, IQR, and outliers per subject.
    Sorted by median to clearly rank subject difficulty.
    """
    subject_cols = [c for c in df.columns[1:] if c not in
                    ["avg_marks", "min_marks", "max_marks", "std_marks",
                     "subjects_failed", "pass_rate", "label"]]

    # Melt to long format for seaborn
    melted = df[["Student_ID"] + subject_cols].melt(
        id_vars="Student_ID", var_name="Subject", value_name="Marks"
    )

    # Sort subjects by median marks (easiest → hardest)
    order = (melted.groupby("Subject")["Marks"].median()
             .sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(max(8, len(subject_cols) * 1.6), 5))

    sns.boxplot(data=melted, x="Subject", y="Marks", order=order,
                palette="plasma", ax=ax,
                boxprops=dict(linewidth=1.5),
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                flierprops=dict(marker="o", markersize=4, alpha=0.5))

    ax.axhline(PASS_THRESHOLD, color=PALETTE["danger"], linestyle="--",
               linewidth=1.5, label=f"Pass Mark ({PASS_THRESHOLD})")
    ax.set_title("📦 Subject Difficulty Comparison (Box Plot)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Marks")
    ax.legend()
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return _style_fig(fig, ax)


def plot_performance_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Student × Subject heatmap — reveals individual weaknesses at a glance.
    Students sorted by average performance (top performers at top).
    Color: red = below pass, green = strong performance.
    """
    subject_cols = [c for c in df.columns[1:] if c not in
                    ["avg_marks", "min_marks", "max_marks", "std_marks",
                     "subjects_failed", "pass_rate", "label"]]

    heatmap_data = df.set_index("Student_ID")[subject_cols]

    # Sort students by overall average (best at top)
    row_order = heatmap_data.mean(axis=1).sort_values(ascending=False).index
    heatmap_data = heatmap_data.loc[row_order]

    fig_h = max(6, len(df) * 0.45)
    fig_w = max(8, len(subject_cols) * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # RdYlGn: red (fail) → yellow (borderline) → green (good)
    sns.heatmap(heatmap_data, annot=True, fmt=".0f",
                cmap="RdYlGn", vmin=0, vmax=100,
                linewidths=0.5, linecolor="#2A2D3A",
                annot_kws={"size": 8, "weight": "bold"},
                ax=ax)

    ax.set_title("🌡️  Student Performance Heatmap (Red = Risk)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Student")
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    return _style_fig(fig, ax)


def plot_student_average_bar(df_engineered: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart of student averages, color-coded by risk tier.
    Immediately shows relative standing across the class.
    """
    sorted_df = df_engineered.sort_values("avg_marks", ascending=True)

    def bar_color(avg):
        if avg < OVERALL_PASS_THRESHOLD:
            return PALETTE["danger"]
        elif avg < AT_RISK_THRESHOLD:
            return PALETTE["warning"]
        else:
            return PALETTE["success"]

    colors = [bar_color(v) for v in sorted_df["avg_marks"]]

    fig, ax = plt.subplots(figsize=(9, max(5, len(df_engineered) * 0.45)))
    bars = ax.barh(sorted_df["Student_ID"], sorted_df["avg_marks"],
                   color=colors, edgecolor="#1A1D27", linewidth=0.8, height=0.7)

    # Annotate bars with values
    for bar, val in zip(bars, sorted_df["avg_marks"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8, color="#C8CDD8")

    ax.axvline(OVERALL_PASS_THRESHOLD, color=PALETTE["danger"],
               linestyle="--", linewidth=1.5, label=f"Pass ({OVERALL_PASS_THRESHOLD})")
    ax.axvline(AT_RISK_THRESHOLD, color=PALETTE["warning"],
               linestyle="-.", linewidth=1.5, label=f"At Risk ({AT_RISK_THRESHOLD})")
    ax.axvline(df_engineered["avg_marks"].mean(), color="#7EB8F7",
               linestyle=":", linewidth=1.5, label="Class Avg")

    legend_patches = [
        mpatches.Patch(color=PALETTE["success"], label="On Track"),
        mpatches.Patch(color=PALETTE["warning"], label="At Risk"),
        mpatches.Patch(color=PALETTE["danger"], label="Failing"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    ax.set_xlim(0, 105)
    ax.set_xlabel("Average Marks")
    ax.set_ylabel("Student")
    ax.set_title("📊 Student Average Performance Ranking", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _style_fig(fig, ax)


def plot_confusion_matrix(cm: np.ndarray) -> plt.Figure:
    """Visualize model evaluation: how many predictions were correct."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Fail", "Predicted Pass"],
                yticklabels=["Actual Fail", "Actual Pass"],
                linewidths=1, linecolor="#2A2D3A",
                annot_kws={"size": 12, "weight": "bold"},
                ax=ax)
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _style_fig(fig, ax)


def plot_feature_importance(pipeline, feature_cols: list) -> plt.Figure:
    """
    Plot logistic regression coefficients as feature importance.
    Magnitude = importance; sign = direction of effect on FAILURE probability.
    """
    coef = pipeline.named_steps["clf"].coef_[0]

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": coef
    }).sort_values("Coefficient", key=abs, ascending=True)

    colors = [PALETTE["danger"] if c > 0 else PALETTE["success"]
              for c in importance_df["Coefficient"]]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(importance_df["Feature"], importance_df["Coefficient"],
                   color=colors, edgecolor="#2A2D3A", linewidth=0.8)
    ax.axvline(0, color="#C8CDD8", linewidth=0.8)
    ax.set_title("🔍 Feature Importance (Logistic Regression Coefficients)\n"
                 "Positive → increases Fail probability | Negative → decreases it",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Coefficient (effect on Fail probability)")
    fig.tight_layout()
    return _style_fig(fig, ax)


# ─────────────────────────────────────────────
# SECTION 5: PERSONALIZED SUGGESTIONS
# ─────────────────────────────────────────────

def generate_suggestions(student_id: str, df: pd.DataFrame,
                          df_engineered: pd.DataFrame,
                          subject_stats: pd.DataFrame) -> list[str]:
    """
    Generate personalized, actionable improvement suggestions for a student.

    Logic:
        1. Find the student's weakest subjects (below class average)
        2. Quantify the gap from class average
        3. Flag any subjects below the pass threshold
        4. Praise strong subjects to balance feedback
    """
    subject_cols = [c for c in df.columns[1:] if c not in
                    ["avg_marks", "min_marks", "max_marks", "std_marks",
                     "subjects_failed", "pass_rate", "label"]]

    row = df[df["Student_ID"] == student_id]
    if row.empty:
        return ["Student not found."]

    row = row.iloc[0]
    suggestions = []

    # Overall performance context
    avg = df_engineered[df_engineered["Student_ID"] == student_id]["avg_marks"].values[0]
    class_avg = df_engineered["avg_marks"].mean()
    suggestions.append(
        f"Overall average: **{avg:.1f}** vs class average **{class_avg:.1f}** "
        f"({'above' if avg >= class_avg else 'below'} class average by "
        f"{abs(avg - class_avg):.1f} marks)."
    )

    # Subject-level breakdown
    below_pass = []
    below_avg = []
    strong_subjects = []

    for subj in subject_cols:
        mark = row[subj]
        if pd.isna(mark):
            continue
        subj_avg = subject_stats.loc[subj, "Mean"] if subj in subject_stats.index else 50
        gap = mark - subj_avg

        if mark < PASS_THRESHOLD:
            below_pass.append((subj, mark, subj_avg))
        elif gap < -5:
            below_avg.append((subj, mark, subj_avg, gap))
        elif mark >= 70:
            strong_subjects.append((subj, mark))

    # Critical failures
    if below_pass:
        suggestions.append("🚨 **Subjects requiring urgent attention (below pass mark):**")
        for subj, mark, s_avg in sorted(below_pass, key=lambda x: x[1]):
            gap_from_pass = PASS_THRESHOLD - mark
            suggestions.append(
                f"  • **{subj}**: {mark:.0f}/100 — needs **{gap_from_pass:.0f} more marks** "
                f"to pass (class avg: {s_avg:.1f})"
            )

    # Below class average (not failing but lagging)
    if below_avg:
        suggestions.append("⚠️  **Subjects below class average — recommend more practice:**")
        for subj, mark, s_avg, gap in sorted(below_avg, key=lambda x: x[3]):
            pct_below = abs(gap) / s_avg * 100
            suggestions.append(
                f"  • **{subj}**: {mark:.0f}/100 — **{pct_below:.0f}% below** class average "
                f"({s_avg:.1f})"
            )

    # Strengths to acknowledge
    if strong_subjects:
        strong_str = ", ".join([f"{s} ({m:.0f})" for s, m in strong_subjects])
        suggestions.append(f"✅ **Strengths to maintain:** {strong_str}")

    # No issues found
    if not below_pass and not below_avg:
        suggestions.append("🌟 **Great performance!** All subjects are at or above class average. "
                           "Focus on pushing strong subjects to excellence.")

    return suggestions


# ─────────────────────────────────────────────
# SECTION 6: REPORT GENERATION
# ─────────────────────────────────────────────

def generate_csv_report(df_engineered: pd.DataFrame, predictions: pd.DataFrame) -> str:
    """
    Merge engineered features with predictions for a downloadable CSV report.
    """
    report = df_engineered.merge(
        predictions[["Student_ID", "Fail_Probability", "Pass_Probability",
                     "Risk_Level", "Predicted"]],
        on="Student_ID"
    )
    return report.to_csv(index=False)


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    """Convert matplotlib figure to PNG bytes for Streamlit download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# SECTION 7: STREAMLIT UI
# ─────────────────────────────────────────────

def setup_page():
    """Configure Streamlit page and apply custom CSS."""
    st.set_page_config(
        page_title="Smart Student Performance Analyzer",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for a polished, dark-themed academic feel
    st.markdown("""
    <style>
        /* Import premium fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --bg-primary: #0F1117;
            --bg-card: #1A1D27;
            --bg-card2: #1E2132;
            --accent-blue: #2E86AB;
            --accent-green: #28A745;
            --accent-yellow: #FFC107;
            --accent-red: #DC3545;
            --text-primary: #E8ECF0;
            --text-secondary: #9BA3AF;
            --border: #2A2D3A;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Main background */
        .main { background-color: var(--bg-primary); }
        .stApp { background-color: var(--bg-primary); }

        /* Header gradient */
        .header-container {
            background: linear-gradient(135deg, #1a1d27 0%, #16213e 50%, #0f1117 100%);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }
        .header-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -20%;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(46,134,171,0.12) 0%, transparent 70%);
            pointer-events: none;
        }
        .header-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: #E8ECF0;
            margin: 0;
        }
        .header-subtitle {
            color: #9BA3AF;
            font-size: 0.95rem;
            margin: 0.4rem 0 0;
        }

        /* Metric cards */
        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            text-align: center;
            transition: border-color 0.2s;
        }
        .metric-card:hover { border-color: var(--accent-blue); }
        .metric-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        .metric-label {
            font-size: 0.78rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.25rem;
        }
        .metric-sub {
            font-size: 0.82rem;
            color: var(--text-secondary);
            margin-top: 0.2rem;
        }

        /* Section headers */
        .section-header {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--text-primary);
            border-left: 3px solid var(--accent-blue);
            padding-left: 0.8rem;
            margin: 1.5rem 0 1rem;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--bg-card);
            border-radius: 10px;
            padding: 4px;
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: var(--text-secondary);
            border-radius: 7px;
            font-weight: 500;
            font-size: 0.88rem;
            padding: 0.5rem 1.2rem;
        }
        .stTabs [aria-selected="true"] {
            background: var(--accent-blue) !important;
            color: white !important;
        }

        /* Suggestion box */
        .suggestion-box {
            background: var(--bg-card2);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent-blue);
            border-radius: 10px;
            padding: 1.2rem 1.4rem;
            margin: 0.8rem 0;
        }

        /* Risk badges */
        .risk-high   { color: #FF6B6B; font-weight: 600; }
        .risk-medium { color: #FFD93D; font-weight: 600; }
        .risk-low    { color: #6BCB77; font-weight: 600; }

        /* Dataframe styling */
        .stDataFrame { border-radius: 10px; overflow: hidden; }

        /* Upload area */
        .uploadedFile {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: var(--bg-card);
            border-right: 1px solid var(--border);
        }

        /* Buttons */
        .stDownloadButton > button {
            background: var(--accent-blue);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            transition: opacity 0.2s;
        }
        .stDownloadButton > button:hover { opacity: 0.85; }

        /* Divider */
        hr { border-color: var(--border); }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the application header banner."""
    st.markdown("""
    <div class="header-container">
        <div style="display:flex; align-items:center; gap:1rem;">
            <div style="font-size:2.5rem;">🎓</div>
            <div>
                <p class="header-title">Smart Student Performance Analyzer</p>
                <p class="header-subtitle">
                    End-to-end educational analytics · Statistical Analysis · Logistic Regression · Actionable Insights
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, sub: str = "", icon: str = ""):
    """Render a single KPI card using HTML for precision control."""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
        {'<div class="metric-sub">' + sub + '</div>' if sub else ''}
    </div>
    """


def main():
    """
    Application entry point — orchestrates the full Streamlit UI.

    Layout:
        Sidebar  → Configuration & file upload
        Tab 1    → Overview & class statistics
        Tab 2    → Visual analysis (charts)
        Tab 3    → ML predictions (risk table)
        Tab 4    → Individual student deep-dive & suggestions
        Tab 5    → Model diagnostics
    """
    setup_page()
    render_header()

    # ── SIDEBAR ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        pass_thresh = st.slider("Pass Mark Threshold", 30, 60, PASS_THRESHOLD, 5,
                                help="Minimum marks to count a subject as passed")
        at_risk_thresh = st.slider("At-Risk Threshold", 40, 70, AT_RISK_THRESHOLD, 5,
                                   help="Students below this overall average are flagged")

        st.markdown("---")
        st.markdown("### 📂 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Student CSV",
            type=["csv"],
            help="CSV with Student ID as first column, subjects as remaining columns."
        )

        st.markdown("---")
        st.markdown("""
        **Expected CSV Format:**
        ```
        Student_ID, Math, Science, English
        S001, 78, 65, 82
        S002, 45, 50, 38
        ```
        """)

        st.markdown("---")
        st.caption("🔬 ML: Logistic Regression | 🐍 Python · Pandas · Seaborn · Sklearn")

    # ── MAIN AREA ─────────────────────────────────────────
    if uploaded_file is None:
        # Landing state — show instructions
        st.markdown('<div class="section-header">How to get started</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        steps = [
            ("1️⃣", "Upload CSV", "Upload your student marks CSV via the sidebar"),
            ("2️⃣", "Analyze", "Instantly see class statistics and visualizations"),
            ("3️⃣", "Predict", "ML model predicts pass/fail risk for each student"),
            ("4️⃣", "Act", "Get personalized recommendations per student"),
        ]
        for col, (icon, title, desc) in zip([col1, col2, col3, col4], steps):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="text-align:left; padding:1.4rem;">
                    <div style="font-size:1.8rem; margin-bottom:0.6rem;">{icon}</div>
                    <div style="font-weight:600; color:#E8ECF0; margin-bottom:0.3rem;">{title}</div>
                    <div style="font-size:0.83rem; color:#9BA3AF;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.info("📎 Upload a CSV file in the sidebar to begin. A sample CSV is included in the project.")
        return

    # ── LOAD DATA ─────────────────────────────────────────
    df, error = load_and_validate_csv(uploaded_file)
    if error:
        st.error(f"❌ {error}")
        return

    df_eng = engineer_features(df)
    subject_cols = [c for c in df.columns[1:].tolist() if c not in
                    ["avg_marks", "min_marks", "max_marks", "std_marks",
                     "subjects_failed", "pass_rate", "label"]]
    class_stats = compute_class_statistics(df_eng)
    subject_stats = compute_subject_statistics(df_eng)

    # Train model (best-effort; may fail for tiny or homogeneous datasets)
    pipeline, feature_cols, model_result = train_model(df_eng)
    model_ok = pipeline is not None

    if not model_ok:
        st.warning(f"⚠️  ML model skipped: {model_result}")

    predictions = predict_students(pipeline, df_eng, feature_cols) if model_ok else None

    # ── TABS ──────────────────────────────────────────────
    tabs = st.tabs(["📋 Overview", "📊 Visualizations",
                    "🤖 Predictions", "💡 Suggestions", "🔬 Model Info"])

    # ────────────────────────────────────────────────────────
    # TAB 1 — OVERVIEW
    # ────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="section-header">Class-Level KPIs</div>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        cards = [
            (c1, "Total Students", str(class_stats["total"]), "", "👥"),
            (c2, "Class Average", f"{class_stats['class_avg']:.1f}", "out of 100", "📈"),
            (c3, "Passing", str(class_stats["pass_count"]),
             f"{100*class_stats['pass_count']/class_stats['total']:.0f}%", "✅"),
            (c4, "Failing", str(class_stats["fail_count"]),
             f"{100*class_stats['fail_count']/class_stats['total']:.0f}%", "❌"),
            (c5, "Top Scorer", class_stats["topper_name"],
             f"Avg: {class_stats['topper_avg']:.1f}", "🏆"),
            (c6, "Needs Support", str(class_stats["at_risk_count"]), "at-risk students", "⚠️"),
        ]
        for col, label, value, sub, icon in cards:
            with col:
                st.markdown(render_metric_card(label, value, sub, icon), unsafe_allow_html=True)

        st.markdown("---")

        col_left, col_right = st.columns([1.4, 1])

        with col_left:
            st.markdown('<div class="section-header">Subject Statistics Table</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                subject_stats.style
                    .background_gradient(subset=["Mean"], cmap="RdYlGn", vmin=0, vmax=100)
                    .background_gradient(subset=["Difficulty Index"], cmap="RdYlGn_r", vmin=0, vmax=1)
                    .format({"Pass Rate (%)": "{:.1f}%", "Difficulty Index": "{:.3f}"}),
                use_container_width=True
            )

        with col_right:
            st.markdown('<div class="section-header">Student Data Preview</div>',
                        unsafe_allow_html=True)
            preview = df_eng[["Student_ID"] + subject_cols + ["avg_marks", "label"]].copy()
            preview["avg_marks"] = preview["avg_marks"].round(1)
            preview["label"] = preview["label"].map({1: "✅ Pass", 0: "❌ Fail"})
            preview = preview.rename(columns={"avg_marks": "Average", "label": "Status"})
            st.dataframe(preview, use_container_width=True, height=340)

        # Download section
        st.markdown("---")
        st.markdown('<div class="section-header">📥 Downloads</div>', unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            if model_ok:
                report_csv = generate_csv_report(df_eng, predictions)
                st.download_button("⬇️  Download Full Report (CSV)", report_csv,
                                   file_name="student_analysis_report.csv",
                                   mime="text/csv")
        with dl2:
            stats_csv = subject_stats.to_csv()
            st.download_button("⬇️  Download Subject Stats (CSV)", stats_csv,
                               file_name="subject_statistics.csv",
                               mime="text/csv")

    # ────────────────────────────────────────────────────────
    # TAB 2 — VISUALIZATIONS
    # ────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown('<div class="section-header">Student Average Ranking</div>',
                    unsafe_allow_html=True)
        fig_bar = plot_student_average_bar(df_eng)
        st.pyplot(fig_bar, use_container_width=True)
        st.download_button("⬇️ Download Chart", fig_to_png_bytes(fig_bar),
                           "student_ranking.png", "image/png", key="dl_bar")
        plt.close(fig_bar)

        st.markdown("---")
        st.markdown('<div class="section-header">Marks Distribution per Subject</div>',
                    unsafe_allow_html=True)
        fig_dist = plot_marks_distribution(df_eng)
        st.pyplot(fig_dist, use_container_width=True)
        st.download_button("⬇️ Download Chart", fig_to_png_bytes(fig_dist),
                           "distribution.png", "image/png", key="dl_dist")
        plt.close(fig_dist)

        st.markdown("---")
        col_v1, col_v2 = st.columns([1, 1])
        with col_v1:
            st.markdown('<div class="section-header">Subject Difficulty (Box Plot)</div>',
                        unsafe_allow_html=True)
            fig_box = plot_subject_boxplot(df_eng)
            st.pyplot(fig_box, use_container_width=True)
            st.download_button("⬇️ Download", fig_to_png_bytes(fig_box),
                               "boxplot.png", "image/png", key="dl_box")
            plt.close(fig_box)

        with col_v2:
            st.markdown('<div class="section-header">Performance Heatmap</div>',
                        unsafe_allow_html=True)
            fig_heat = plot_performance_heatmap(df_eng)
            st.pyplot(fig_heat, use_container_width=True)
            st.download_button("⬇️ Download", fig_to_png_bytes(fig_heat),
                               "heatmap.png", "image/png", key="dl_heat")
            plt.close(fig_heat)

    # ────────────────────────────────────────────────────────
    # TAB 3 — PREDICTIONS
    # ────────────────────────────────────────────────────────
    with tabs[2]:
        if not model_ok:
            st.warning("Model could not be trained on this dataset.")
        else:
            st.markdown('<div class="section-header">Risk Classification — All Students</div>',
                        unsafe_allow_html=True)

            # Summary row
            p1, p2, p3 = st.columns(3)
            high_risk = (predictions["Risk_Level"].str.contains("High")).sum()
            mod_risk = (predictions["Risk_Level"].str.contains("Moderate")).sum()
            on_track = (predictions["Risk_Level"].str.contains("On Track")).sum()
            with p1:
                st.markdown(render_metric_card("High Risk", str(high_risk),
                            "Immediate intervention needed", "🔴"), unsafe_allow_html=True)
            with p2:
                st.markdown(render_metric_card("Moderate Risk", str(mod_risk),
                            "Monitor closely", "🟡"), unsafe_allow_html=True)
            with p3:
                st.markdown(render_metric_card("On Track", str(on_track),
                            "Performing well", "🟢"), unsafe_allow_html=True)

            st.markdown("---")

            # Filterable predictions table
            filter_risk = st.selectbox("Filter by Risk Level",
                                        ["All", "🔴 High Risk", "🟡 Moderate Risk", "🟢 On Track"])
            display_pred = predictions.copy()
            if filter_risk != "All":
                display_pred = display_pred[display_pred["Risk_Level"] == filter_risk]

            display_pred = display_pred.sort_values("Fail_Probability", ascending=False)
            display_pred["avg_marks"] = display_pred["avg_marks"].round(1)
            display_pred = display_pred.rename(columns={
                "avg_marks": "Average Marks",
                "Fail_Probability": "Fail Prob (%)",
                "Pass_Probability": "Pass Prob (%)",
            })

            st.dataframe(
                display_pred.style
                    .background_gradient(subset=["Fail Prob (%)"],
                                         cmap="RdYlGn_r", vmin=0, vmax=100),
                use_container_width=True, height=420
            )

    # ────────────────────────────────────────────────────────
    # TAB 4 — SUGGESTIONS
    # ────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown('<div class="section-header">Personalized Student Improvement Plan</div>',
                    unsafe_allow_html=True)

        student_list = df_eng["Student_ID"].tolist()
        selected_student = st.selectbox("Select a student", student_list)

        if selected_student:
            student_row = df_eng[df_eng["Student_ID"] == selected_student].iloc[0]
            avg = student_row["avg_marks"]
            subjects_failed = int(student_row["subjects_failed"])
            std = student_row["std_marks"]

            # Quick stats row
            qs1, qs2, qs3, qs4 = st.columns(4)
            with qs1:
                st.markdown(render_metric_card("Average Marks", f"{avg:.1f}", "out of 100", "📊"),
                            unsafe_allow_html=True)
            with qs2:
                st.markdown(render_metric_card("Subjects Failed", str(subjects_failed),
                            f"of {len(subject_cols)} subjects", "❌"), unsafe_allow_html=True)
            with qs3:
                rank = (df_eng["avg_marks"] > avg).sum() + 1
                st.markdown(render_metric_card("Class Rank", f"#{rank}",
                            f"out of {len(df_eng)}", "🏅"), unsafe_allow_html=True)
            with qs4:
                if model_ok:
                    pred_row = predictions[predictions["Student_ID"] == selected_student].iloc[0]
                    risk = pred_row["Risk_Level"]
                    fail_p = pred_row["Fail_Probability"]
                    st.markdown(render_metric_card("Risk Level", risk,
                                f"Fail probability: {fail_p}%", ""), unsafe_allow_html=True)

            st.markdown("---")

            # Subject radar / bar comparison
            st.markdown("**Subject-Level Breakdown vs Class Average:**")
            fig_comp, ax_comp = plt.subplots(figsize=(9, 3.5))
            subj_marks = [student_row[s] for s in subject_cols]
            subj_avgs = [subject_stats.loc[s, "Mean"] if s in subject_stats.index else 50
                         for s in subject_cols]

            x = np.arange(len(subject_cols))
            width = 0.38
            ax_comp.bar(x - width / 2, subj_marks, width, label=f"{selected_student}",
                        color=PALETTE["primary"], alpha=0.85, edgecolor="#1A1D27")
            ax_comp.bar(x + width / 2, subj_avgs, width, label="Class Avg",
                        color=PALETTE["neutral"], alpha=0.65, edgecolor="#1A1D27")
            ax_comp.axhline(PASS_THRESHOLD, color=PALETTE["danger"], linestyle="--",
                            linewidth=1.2, label="Pass Mark")
            ax_comp.set_xticks(x)
            ax_comp.set_xticklabels(subject_cols, rotation=20)
            ax_comp.set_ylabel("Marks")
            ax_comp.set_title(f"Performance Breakdown: {selected_student} vs Class Average")
            ax_comp.legend(fontsize=8)
            ax_comp.set_ylim(0, 105)
            _style_fig(fig_comp, ax_comp)
            st.pyplot(fig_comp, use_container_width=True)
            plt.close(fig_comp)

            st.markdown("---")
            st.markdown("**📝 Personalized Recommendations:**")
            suggestions = generate_suggestions(
                selected_student, df_eng, df_eng, subject_stats
            )
            for s in suggestions:
                st.markdown(f'<div class="suggestion-box">{s}</div>', unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────
    # TAB 5 — MODEL INFO
    # ────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown('<div class="section-header">Model Architecture & Diagnostics</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="suggestion-box">
        <b>Why Logistic Regression?</b><br>
        Logistic Regression was chosen for three reasons specific to this educational context:<br>
        1. <b>Interpretability</b>: Coefficients map directly to feature importance — educators can understand
           <i>why</i> a student is flagged at-risk without a black-box explanation.<br>
        2. <b>Probabilistic output</b>: Unlike SVM or Decision Trees, LR outputs a calibrated probability
           (e.g., "72% chance of failing") — enabling nuanced risk tiering rather than binary labels.<br>
        3. <b>Small data</b>: With typical class sizes (20–200 students), complex models like Neural Networks
           overfit. LR with L2 regularization generalizes well with few samples.<br><br>
        <b>Trade-off</b>: A Random Forest would likely achieve higher accuracy but at the cost of
        interpretability. For a dashboard used by educators, explainability is paramount.
        </div>
        """, unsafe_allow_html=True)

        if not model_ok:
            st.info("Model was not trained — diagnostics unavailable.")
        else:
            metrics = model_result

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(render_metric_card("Accuracy",
                            f"{metrics['accuracy']*100:.1f}%", "on test set", "🎯"),
                            unsafe_allow_html=True)
            with m2:
                auc = metrics.get("auc", None)
                st.markdown(render_metric_card("ROC-AUC",
                            f"{auc:.3f}" if auc else "N/A",
                            "1.0 = perfect", "📐"), unsafe_allow_html=True)
            with m3:
                n_features = len(feature_cols)
                st.markdown(render_metric_card("Features Used", str(n_features),
                            "engineered features", "🔧"), unsafe_allow_html=True)

            st.markdown("---")
            col_cm, col_fi = st.columns([1, 1.4])

            with col_cm:
                st.markdown("**Confusion Matrix (Test Set):**")
                fig_cm = plot_confusion_matrix(metrics["confusion_matrix"])
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)

            with col_fi:
                st.markdown("**Feature Importance (Logistic Coefficients):**")
                fig_fi = plot_feature_importance(pipeline, feature_cols)
                st.pyplot(fig_fi, use_container_width=True)
                plt.close(fig_fi)

            st.markdown("---")
            st.markdown("**Classification Report (Test Set):**")
            cr = metrics["classification_report"]
            cr_df = pd.DataFrame({
                k: {m: f"{v:.2f}" if isinstance(v, float) else str(v)
                    for m, v in cr[k].items()}
                for k in cr if isinstance(cr[k], dict)
            }).T
            st.dataframe(cr_df, use_container_width=True)

            st.markdown("---")
            st.markdown("**Engineered Features:**")
            feat_desc = {
                "avg_marks": "Mean marks across all subjects — primary predictor",
                "min_marks": "Worst subject score — captures subject-level failure risk",
                "std_marks": "Score variability — high std may indicate inconsistency",
                "subjects_failed": "Count of subjects below pass threshold",
                "pass_rate": "Fraction of subjects passed (1 - fail_rate)",
            }
            feat_df = pd.DataFrame([
                {"Feature": k, "Description": v} for k, v in feat_desc.items()
            ])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

            st.markdown("""
            ---
            **How this could scale:**
            - 🗄️  Replace CSV with PostgreSQL / MongoDB for real-time data
            - 🔄  Add semester history for time-series trend modeling
            - 🧠  Upgrade to XGBoost or LightGBM when N > 1,000 students
            - 📡  REST API wrapper (FastAPI) for integration with LMS platforms
            - 📧  Automated alert emails for high-risk students
            """)


if __name__ == "__main__":
    main()
