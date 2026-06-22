import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import io
import os
import glob

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DDoS Detection — SMOTE Evaluation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-banner {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1f3c 50%, #0a2540 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,168,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem; font-weight: 600;
    color: #e8f4fd; margin: 0 0 0.4rem 0; letter-spacing: -0.5px;
}
.hero-subtitle { font-size: 0.95rem; color: #7eb8d4; margin: 0; font-weight: 400; }
.hero-tag {
    display: inline-block;
    background: rgba(0,168,255,0.15);
    border: 1px solid rgba(0,168,255,0.3);
    color: #00a8ff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; padding: 2px 8px; border-radius: 4px;
    margin-bottom: 0.8rem; letter-spacing: 1px; text-transform: uppercase;
}
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px;
    color: #00a8ff; border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0;
}
.winner-card {
    background: linear-gradient(135deg, #0a2540 0%, #0d1f3c 100%);
    border: 2px solid #00e676; border-radius: 12px;
    padding: 1.5rem; text-align: center;
}
.winner-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 2px;
    color: #00e676; margin-bottom: 0.5rem;
}
.winner-name { font-size: 1.6rem; font-weight: 700; color: #e8f4fd; margin: 0; }
.info-box {
    background: rgba(0,168,255,0.06);
    border-left: 3px solid #00a8ff;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem; margin: 1rem 0;
    font-size: 0.88rem; color: #c8dde8; line-height: 1.6;
}
.status-box {
    background: rgba(0,230,118,0.06);
    border-left: 3px solid #00e676;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem; margin: 1rem 0;
    font-size: 0.88rem; color: #c8dde8; line-height: 1.6;
}
.error-box {
    background: rgba(255,82,82,0.06);
    border-left: 3px solid #ff5252;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem; margin: 1rem 0;
    font-size: 0.88rem; color: #c8dde8; line-height: 1.6;
}
section[data-testid="stSidebar"] {
    background: #060e1a; border-right: 1px solid #1e3a5f;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
for key in ['trained', 'results']:
    if key not in st.session_state:
        st.session_state[key] = False if key == 'trained' else None

# ─── Dataset Discovery ────────────────────────────────────────────────────────
def find_dataset():
    """
    Cari dataset CSV di folder yang sama dengan script ini.
    Prioritas: file yang namanya mengandung 'ddos', 'DDoS', 'ids', 'Friday', dst.
    Fallback: CSV pertama yang ditemukan.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_csv = glob.glob(os.path.join(base_dir, "*.csv"))

    if not all_csv:
        return None, []

    # Prioritaskan file dengan nama yang relevan
    priority_keywords = ['ddos', 'DDoS', 'ids', 'IDS', 'friday', 'Friday', 'cic', 'CIC', 'attack']
    for kw in priority_keywords:
        for f in all_csv:
            if kw in os.path.basename(f):
                return f, all_csv

    return all_csv[0], all_csv

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ DDoS Detector")
    st.markdown("---")

    dataset_path, all_csv = find_dataset()

    if all_csv:
        csv_names = [os.path.basename(f) for f in all_csv]
        selected_name = st.selectbox(
            "Dataset Terdeteksi",
            csv_names,
            index=csv_names.index(os.path.basename(dataset_path)) if dataset_path else 0,
            help="File CSV yang ditemukan di folder yang sama dengan app.py"
        )
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), selected_name)
    else:
        st.markdown("""
<div class="error-box">
⚠️ Tidak ada file CSV ditemukan di folder ini.<br>
Letakkan file dataset (.csv) di folder yang sama dengan <code>app.py</code>.
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Hyperparameter")
    test_size    = st.slider("Test Size (%)", 10, 40, 20) / 100

    st.markdown("**Decision Tree & Random Forest**")
    max_depth    = st.slider("Max Depth", 3, 20, 10)
    min_split    = st.slider("Min Samples Split", 2, 50, 10)
    min_leaf     = st.slider("Min Samples Leaf", 1, 20, 5)
    n_est        = st.slider("RF n_estimators", 50, 300, 100, step=50)

    st.markdown("**SMOTE**")
    k_neighbors  = st.slider("k_neighbors", 3, 10, 5)

    st.markdown("**Cross Validation**")
    cv_folds     = st.slider("K-Fold", 3, 10, 5)
    cv_samples   = st.number_input("CV Sample Size", 10000, 100000, 50000, step=5000)
    run_cv       = st.checkbox("Jalankan Cross Validation", value=True)

    st.markdown("---")
    run_btn = st.button("▶  Jalankan Analisis", use_container_width=True, type="primary",
                        disabled=(dataset_path is None))

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-tag">Research Tool · CIC-IDS2017</div>
  <div class="hero-title">🛡️ DDoS Detection — SMOTE Evaluation</div>
  <div class="hero-subtitle">
    Evaluation of SMOTE Implementation in Improving Random Forest-Based DDoS Attack Detection on SDN Networks
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Helper Functions ─────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name):
    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t0
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        'Model'              : model_name,
        'Accuracy (%)'       : round(accuracy_score(y_test, y_pred) * 100, 4),
        'Precision (%)'      : round(precision_score(y_test, y_pred) * 100, 4),
        'Recall (%)'         : round(recall_score(y_test, y_pred) * 100, 4),
        'F1-Score (%)'       : round(f1_score(y_test, y_pred) * 100, 4),
        'ROC-AUC (%)'        : round(roc_auc_score(y_test, y_prob) * 100, 4),
        'Waktu Prediksi (s)' : round(pred_time, 4)
    }
    return metrics, y_pred, y_prob

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def make_cm_fig(predictions, titles):
    class_names = ['BENIGN', 'DDoS']
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.patch.set_facecolor('#060e1a')
    colors = ['#3498db', '#27ae60', '#e67e22', '#8e44ad']
    for (y_pred, y_test), title, color, ax in zip(predictions, titles, colors, axes):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5, cbar=False)
        ax.set_facecolor('#0d1f3c')
        ax.set_title(title, fontsize=10, fontweight='bold', color='#e8f4fd', pad=8)
        ax.set_xlabel('Predicted', fontsize=9, color='#7eb8d4')
        ax.set_ylabel('True', fontsize=9, color='#7eb8d4')
        ax.tick_params(colors='#7eb8d4')
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.75, f'({cm[i,j]/total*100:.1f}%)',
                        ha='center', fontsize=8, color='#aaa')
    plt.tight_layout()
    return fig

def make_roc_fig(roc_data):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#060e1a')
    ax.set_facecolor('#0d1f3c')
    colors  = ['#3498db', '#27ae60', '#e67e22', '#8e44ad']
    lstyles = ['--', '-', '--', '-']
    for (y_prob, y_test, name), color, ls in zip(roc_data, colors, lstyles):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2,
                label=f'{name} (AUC={roc_auc_val:.4f})')
    ax.plot([0,1],[0,1],'w--', linewidth=1, alpha=0.4, label='Random (AUC=0.5)')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel('False Positive Rate', color='#7eb8d4', fontsize=11)
    ax.set_ylabel('True Positive Rate', color='#7eb8d4', fontsize=11)
    ax.set_title('ROC Curve — 4 Model', color='#e8f4fd', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8.5, facecolor='#060e1a', labelcolor='#c8dde8', edgecolor='#1e3a5f')
    ax.tick_params(colors='#7eb8d4')
    ax.grid(True, alpha=0.15, color='#1e3a5f')
    for spine in ax.spines.values(): spine.set_edgecolor('#1e3a5f')
    plt.tight_layout()
    return fig

def make_bar_fig(metrics_keys, all_vals_dict):
    x = np.arange(len(metrics_keys))
    width = 0.20
    labels_order = ['DT', 'RF', 'DT+SMOTE', 'RF+SMOTE']
    colors = ['#3498db', '#27ae60', '#e67e22', '#8e44ad']
    offsets = [-1.5, -0.5, 0.5, 1.5]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor('#060e1a')
    ax.set_facecolor('#0d1f3c')
    for label, color, offset in zip(labels_order, colors, offsets):
        vals = [all_vals_dict[label][m] for m in metrics_keys]
        bars = ax.bar(x + offset*width, vals, width,
                      label=label, color=color, edgecolor='#060e1a', alpha=0.9)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.05,
                    f'{bar.get_height():.2f}', ha='center', va='bottom',
                    fontsize=7, color=color, rotation=90)
    ax.set_xlabel('Metrik Evaluasi', color='#7eb8d4', fontsize=11)
    ax.set_ylabel('Nilai (%)', color='#7eb8d4', fontsize=11)
    ax.set_title('Perbandingan Metrik Evaluasi — 4 Model', color='#e8f4fd', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' (%)','') for m in metrics_keys], fontsize=10, color='#c8dde8')
    all_v = [v for d in all_vals_dict.values() for v in d.values()]
    ax.set_ylim([max(0, min(all_v)-2), 101.5])
    ax.legend(fontsize=9, facecolor='#060e1a', labelcolor='#c8dde8', edgecolor='#1e3a5f')
    ax.tick_params(colors='#7eb8d4')
    ax.grid(True, axis='y', alpha=0.15, color='#1e3a5f')
    for spine in ax.spines.values(): spine.set_edgecolor('#1e3a5f')
    plt.tight_layout()
    return fig

def make_fi_fig(models_fi, X_cols):
    top_n = 15
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#060e1a')
    fi_configs = [
        ('DT (Tanpa SMOTE)',  '#3498db', axes[0][0]),
        ('RF (Tanpa SMOTE)',  '#27ae60', axes[0][1]),
        ('DT + SMOTE',        '#e67e22', axes[1][0]),
        ('RF + SMOTE',        '#8e44ad', axes[1][1]),
    ]
    for (_, importance), (name, color, ax) in zip(models_fi, fi_configs):
        imp = pd.Series(importance, index=X_cols).nlargest(top_n)
        bars = ax.barh(imp.index[::-1], imp.values[::-1], color=color, edgecolor='#060e1a', alpha=0.85)
        ax.set_facecolor('#0d1f3c')
        ax.set_title(f'Top {top_n} Features — {name}', fontsize=10, fontweight='bold', color='#e8f4fd')
        ax.set_xlabel('Importance Score', color='#7eb8d4', fontsize=9)
        ax.tick_params(colors='#7eb8d4', labelsize=8)
        ax.grid(True, axis='x', alpha=0.15, color='#1e3a5f')
        for spine in ax.spines.values(): spine.set_edgecolor('#1e3a5f')
        for bar in bars:
            ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2.,
                    f'{bar.get_width():.4f}', va='center', fontsize=7, color='#c8dde8')
    plt.suptitle('Feature Importance — 4 Model', color='#e8f4fd', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig

def make_cv_fig(cv_results):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('#060e1a')
    ax.set_facecolor('#0d1f3c')
    folds = [f'Fold {i+1}' for i in range(len(list(cv_results.values())[0]))]
    x = np.arange(len(folds))
    width = 0.35
    colors = ['#3498db', '#27ae60']
    for i, (name, scores, color) in enumerate(zip(cv_results.keys(), cv_results.values(), colors)):
        bars = ax.bar(x + (i-0.5)*width, scores*100, width,
                      label=name, color=color, edgecolor='#060e1a', alpha=0.9)
        ax.axhline(scores.mean()*100, color=color, linestyle='--', linewidth=1.5,
                   label=f'{name} Mean: {scores.mean()*100:.2f}%', alpha=0.7)
    ax.set_xlabel('Fold', color='#7eb8d4', fontsize=11)
    ax.set_ylabel('Accuracy (%)', color='#7eb8d4', fontsize=11)
    ax.set_title(f'{len(folds)}-Fold Cross-Validation Accuracy', color='#e8f4fd', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(folds, color='#c8dde8')
    ax.tick_params(colors='#7eb8d4')
    ax.legend(fontsize=8.5, facecolor='#060e1a', labelcolor='#c8dde8', edgecolor='#1e3a5f')
    ax.grid(True, axis='y', alpha=0.15, color='#1e3a5f')
    for spine in ax.spines.values(): spine.set_edgecolor('#1e3a5f')
    plt.tight_layout()
    return fig

# ─── No dataset found ─────────────────────────────────────────────────────────
if dataset_path is None:
    st.markdown("""
<div class="error-box">
❌ <strong>Tidak ada dataset ditemukan.</strong><br><br>
Letakkan file CSV (format CIC-IDS2017) di folder yang sama dengan <code>app.py</code>, 
lalu refresh halaman ini. Kolom wajib: <code>Label</code> berisi nilai <code>BENIGN</code> / <code>DDoS</code>.
</div>
""", unsafe_allow_html=True)
    st.stop()

# ─── Show dataset info before running ─────────────────────────────────────────
if not st.session_state.trained:
    st.markdown(f"""
<div class="status-box">
📂 <strong>Dataset ditemukan:</strong> <code>{os.path.basename(dataset_path)}</code><br>
Atur hyperparameter di sidebar lalu klik <strong>▶ Jalankan Analisis</strong> untuk memulai.
</div>
""", unsafe_allow_html=True)

# ─── Run Analysis ─────────────────────────────────────────────────────────────
if run_btn or st.session_state.trained:
    if run_btn:
        st.session_state.trained = False
        st.session_state.results = None

    if not st.session_state.trained:
        progress = st.progress(0, text="Memuat dataset...")

        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            st.error(f"❌ Gagal membaca dataset: {e}")
            st.stop()

        df.columns = df.columns.str.strip()

        if 'Label' not in df.columns:
            st.error("❌ Kolom 'Label' tidak ditemukan. Pastikan dataset sesuai format CIC-IDS2017.")
            st.stop()

        progress.progress(10, text="Preprocessing data...")
        n_raw = len(df)
        df.drop_duplicates(inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        le = LabelEncoder()
        df['Label_encoded'] = le.fit_transform(df['Label'])
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        X = df.drop(columns=['Label', 'Label_encoded'])
        y = df['Label_encoded']
        X = X.select_dtypes(include=[np.number])

        progress.progress(20, text="Membagi data train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        progress.progress(30, text="Menerapkan SMOTE...")
        t_smote = time.time()
        smote = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        smote_time = time.time() - t_smote

        progress.progress(40, text="Training Decision Tree...")
        t0 = time.time()
        dt_model = DecisionTreeClassifier(
            criterion='gini', max_depth=max_depth,
            min_samples_split=min_split, min_samples_leaf=min_leaf, random_state=42
        )
        dt_model.fit(X_train, y_train)
        dt_train_time = time.time() - t0

        progress.progress(55, text="Training Random Forest...")
        t0 = time.time()
        rf_model = RandomForestClassifier(
            n_estimators=n_est, criterion='gini', max_depth=max_depth,
            min_samples_split=min_split, min_samples_leaf=min_leaf,
            n_jobs=-1, random_state=42
        )
        rf_model.fit(X_train, y_train)
        rf_train_time = time.time() - t0

        progress.progress(68, text="Training Decision Tree + SMOTE...")
        t0 = time.time()
        dt_smote_model = DecisionTreeClassifier(
            criterion='gini', max_depth=max_depth,
            min_samples_split=min_split, min_samples_leaf=min_leaf, random_state=42
        )
        dt_smote_model.fit(X_train_smote, y_train_smote)
        dt_smote_time = time.time() - t0

        progress.progress(80, text="Training Random Forest + SMOTE...")
        t0 = time.time()
        rf_smote_model = RandomForestClassifier(
            n_estimators=n_est, criterion='gini', max_depth=max_depth,
            min_samples_split=min_split, min_samples_leaf=min_leaf,
            n_jobs=-1, random_state=42
        )
        rf_smote_model.fit(X_train_smote, y_train_smote)
        rf_smote_time = time.time() - t0

        progress.progress(88, text="Evaluasi model...")
        dt_metrics,       dt_pred,       dt_prob       = evaluate_model(dt_model,       X_test, y_test, 'Decision Tree')
        rf_metrics,       rf_pred,       rf_prob       = evaluate_model(rf_model,       X_test, y_test, 'Random Forest')
        dt_smote_metrics, dt_smote_pred, dt_smote_prob = evaluate_model(dt_smote_model, X_test, y_test, 'DT + SMOTE')
        rf_smote_metrics, rf_smote_pred, rf_smote_prob = evaluate_model(rf_smote_model, X_test, y_test, 'RF + SMOTE')

        cv_results = {}
        if run_cv:
            progress.progress(92, text="Cross-validation...")
            kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            n_cv = min(cv_samples, len(X))
            X_cv, y_cv = resample(X, y, n_samples=n_cv, random_state=42, stratify=y)
            for model, name in [(dt_model, 'Decision Tree'), (rf_model, 'Random Forest')]:
                scores = cross_val_score(model, X_cv, y_cv, cv=kfold, scoring='accuracy', n_jobs=-1)
                cv_results[name] = scores

        progress.progress(100, text="Selesai!")
        time.sleep(0.3)
        progress.empty()

        st.session_state.trained = True
        st.session_state.results = {
            'df_shape'        : (n_raw, df.shape[0], df.shape[1]),
            'le_classes'      : list(le.classes_),
            'dist_before'     : np.bincount(y_train).tolist(),
            'dist_after'      : np.bincount(y_train_smote).tolist(),
            'smote_time'      : smote_time,
            'train_sizes'     : (len(X_train), len(X_test)),
            'train_dist'      : np.bincount(y_train).tolist(),
            'test_dist'       : np.bincount(y_test).tolist(),
            'dt_metrics'      : dt_metrics,       'rf_metrics'       : rf_metrics,
            'dt_smote_metrics': dt_smote_metrics, 'rf_smote_metrics' : rf_smote_metrics,
            'dt_pred'         : dt_pred,           'rf_pred'          : rf_pred,
            'dt_smote_pred'   : dt_smote_pred,     'rf_smote_pred'    : rf_smote_pred,
            'dt_prob'         : dt_prob,            'rf_prob'          : rf_prob,
            'dt_smote_prob'   : dt_smote_prob,      'rf_smote_prob'    : rf_smote_prob,
            'y_test'          : y_test,
            'dt_train_time'   : dt_train_time,     'rf_train_time'    : rf_train_time,
            'dt_smote_time'   : dt_smote_time,     'rf_smote_time'    : rf_smote_time,
            'cv_results'      : cv_results,
            'feature_importances': [
                (dt_model,       dt_model.feature_importances_),
                (rf_model,       rf_model.feature_importances_),
                (dt_smote_model, dt_smote_model.feature_importances_),
                (rf_smote_model, rf_smote_model.feature_importances_),
            ],
            'X_cols'           : list(X.columns),
            'class_report_dt'  : classification_report(y_test, dt_pred,       target_names=['BENIGN','DDoS']),
            'class_report_rf'  : classification_report(y_test, rf_pred,       target_names=['BENIGN','DDoS']),
            'class_report_dts' : classification_report(y_test, dt_smote_pred, target_names=['BENIGN','DDoS']),
            'class_report_rfs' : classification_report(y_test, rf_smote_pred, target_names=['BENIGN','DDoS']),
            'dataset_name'     : os.path.basename(dataset_path),
        }

    # ─── Display Results ──────────────────────────────────────────────────────
    R = st.session_state.results
    METRICS_KEYS = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'ROC-AUC (%)']

    tabs = st.tabs([
        "📊 Overview",
        "⚖️ SMOTE Effect",
        "🔢 Confusion Matrix",
        "📈 ROC Curve",
        "🌿 Feature Importance",
        "🔁 Cross Validation",
        "📋 Classification Report",
        "💾 Export"
    ])

    # ── Tab 0: Overview ────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown(f"""
<div class="status-box">
📂 <strong>Dataset:</strong> <code>{R['dataset_name']}</code>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Ringkasan Dataset</div>', unsafe_allow_html=True)
        n_raw, n_clean, n_feat = R['df_shape']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Rows", f"{n_raw:,}")
        c2.metric("After Clean", f"{n_clean:,}", delta=f"-{n_raw-n_clean:,}")
        c3.metric("Features", f"{n_feat-2:,}")
        c4.metric("Train / Test", f"{int((1-test_size)*100)}% / {int(test_size*100)}%")

        st.markdown('<div class="section-header">Distribusi Kelas</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Training Set — Sebelum SMOTE**")
            dist_df = pd.DataFrame({
                'Kelas'     : R['le_classes'],
                'Jumlah'    : R['train_dist'],
                'Persen (%)': [round(v/sum(R['train_dist'])*100,2) for v in R['train_dist']]
            })
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**Training Set — Sesudah SMOTE**")
            smote_df = pd.DataFrame({
                'Kelas'     : R['le_classes'],
                'Jumlah'    : R['dist_after'],
                'Persen (%)': [round(v/sum(R['dist_after'])*100,2) for v in R['dist_after']]
            })
            st.dataframe(smote_df, use_container_width=True, hide_index=True)

        st.markdown(f"""
<div class="info-box">
⏱️ SMOTE selesai dalam <strong>{R['smote_time']:.2f} detik</strong>.
Sampel sintetis ditambahkan: <strong>{sum(R['dist_after']) - sum(R['train_dist']):,}</strong>.
Data testing (<strong>{R['train_sizes'][1]:,} baris</strong>) tidak dimodifikasi.
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Performa 4 Model</div>', unsafe_allow_html=True)
        all_m = [R['dt_metrics'], R['rf_metrics'], R['dt_smote_metrics'], R['rf_smote_metrics']]
        results_df = pd.DataFrame(all_m).set_index('Model')
        st.dataframe(results_df, use_container_width=True)

        best_model = max(all_m, key=lambda m: m['F1-Score (%)'])
        st.markdown(f"""
<div class="winner-card">
  <div class="winner-label">🏆 Model Terbaik (F1-Score)</div>
  <div class="winner-name">{best_model['Model']}</div>
  <div style="color:#7eb8d4;margin-top:0.5rem;font-size:0.9rem;">
    F1-Score: <strong style="color:#00e676">{best_model['F1-Score (%)']:.4f}%</strong> &nbsp;|&nbsp;
    Accuracy: <strong style="color:#00e676">{best_model['Accuracy (%)']:.4f}%</strong> &nbsp;|&nbsp;
    ROC-AUC: <strong style="color:#00e676">{best_model['ROC-AUC (%)']:.4f}%</strong>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Tab 1: SMOTE Effect ────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown('<div class="section-header">Pengaruh SMOTE — Decision Tree</div>', unsafe_allow_html=True)
        rows_dt = []
        for m in METRICS_KEYS:
            delta = R['dt_smote_metrics'][m] - R['dt_metrics'][m]
            rows_dt.append({
                'Metrik'          : m,
                'DT (Tanpa SMOTE)': f"{R['dt_metrics'][m]:.4f}%",
                'DT + SMOTE'      : f"{R['dt_smote_metrics'][m]:.4f}%",
                'Δ'               : f"{'▲' if delta>0 else '▼' if delta<0 else '='} {abs(delta):.4f}%"
            })
        st.dataframe(pd.DataFrame(rows_dt), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Pengaruh SMOTE — Random Forest</div>', unsafe_allow_html=True)
        rows_rf = []
        for m in METRICS_KEYS:
            delta = R['rf_smote_metrics'][m] - R['rf_metrics'][m]
            rows_rf.append({
                'Metrik'          : m,
                'RF (Tanpa SMOTE)': f"{R['rf_metrics'][m]:.4f}%",
                'RF + SMOTE'      : f"{R['rf_smote_metrics'][m]:.4f}%",
                'Δ'               : f"{'▲' if delta>0 else '▼' if delta<0 else '='} {abs(delta):.4f}%"
            })
        st.dataframe(pd.DataFrame(rows_rf), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Grafik Perbandingan 4 Model</div>', unsafe_allow_html=True)
        all_vals_dict = {
            'DT'      : {m: R['dt_metrics'][m]       for m in METRICS_KEYS},
            'RF'      : {m: R['rf_metrics'][m]       for m in METRICS_KEYS},
            'DT+SMOTE': {m: R['dt_smote_metrics'][m] for m in METRICS_KEYS},
            'RF+SMOTE': {m: R['rf_smote_metrics'][m] for m in METRICS_KEYS},
        }
        fig_bar = make_bar_fig(METRICS_KEYS, all_vals_dict)
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        st.markdown('<div class="section-header">Waktu Training</div>', unsafe_allow_html=True)
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Decision Tree",   f"{R['dt_train_time']:.2f}s")
        tc2.metric("Random Forest",   f"{R['rf_train_time']:.2f}s")
        tc3.metric("DT + SMOTE",      f"{R['dt_smote_time']:.2f}s")
        tc4.metric("RF + SMOTE",      f"{R['rf_smote_time']:.2f}s")

    # ── Tab 2: Confusion Matrix ────────────────────────────────────────────────
    with tabs[2]:
        st.markdown('<div class="section-header">Confusion Matrix — 4 Model</div>', unsafe_allow_html=True)
        yt = R['y_test']
        predictions = [
            (R['dt_pred'],       yt),
            (R['rf_pred'],       yt),
            (R['dt_smote_pred'], yt),
            (R['rf_smote_pred'], yt),
        ]
        titles_cm = ['DT (Tanpa SMOTE)', 'RF (Tanpa SMOTE)', 'DT + SMOTE', 'RF + SMOTE']
        fig_cm = make_cm_fig(predictions, titles_cm)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    # ── Tab 3: ROC Curve ───────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown('<div class="section-header">ROC Curve — 4 Model</div>', unsafe_allow_html=True)
        yt = R['y_test']
        roc_data = [
            (R['dt_prob'],       yt, 'DT (Tanpa SMOTE)'),
            (R['rf_prob'],       yt, 'RF (Tanpa SMOTE)'),
            (R['dt_smote_prob'], yt, 'DT + SMOTE'),
            (R['rf_smote_prob'], yt, 'RF + SMOTE'),
        ]
        c1, _ = st.columns([2, 1])
        with c1:
            fig_roc = make_roc_fig(roc_data)
            st.pyplot(fig_roc)
            plt.close(fig_roc)

        auc_rows = []
        for y_prob, y_t, name in roc_data:
            fpr, tpr, _ = roc_curve(y_t, y_prob)
            auc_rows.append({'Model': name, 'AUC': round(auc(fpr, tpr)*100, 4)})
        auc_df = pd.DataFrame(auc_rows).sort_values('AUC', ascending=False)
        st.dataframe(auc_df, use_container_width=True, hide_index=True)

    # ── Tab 4: Feature Importance ──────────────────────────────────────────────
    with tabs[4]:
        st.markdown('<div class="section-header">Feature Importance — Top 15</div>', unsafe_allow_html=True)
        models_fi = R['feature_importances']
        fig_fi = make_fi_fig(models_fi, R['X_cols'])
        st.pyplot(fig_fi)
        plt.close(fig_fi)

        st.markdown('<div class="section-header">Top 10 Fitur per Model</div>', unsafe_allow_html=True)
        fi_cols = st.columns(4)
        fi_labels = ['DT', 'RF', 'DT+SMOTE', 'RF+SMOTE']
        for col, (_, imp), label in zip(fi_cols, models_fi, fi_labels):
            with col:
                st.markdown(f"**{label}**")
                fi_series = pd.Series(imp, index=R['X_cols']).nlargest(10).reset_index()
                fi_series.columns = ['Feature', 'Score']
                fi_series['Score'] = fi_series['Score'].round(4)
                st.dataframe(fi_series, use_container_width=True, hide_index=True)

    # ── Tab 5: Cross Validation ────────────────────────────────────────────────
    with tabs[5]:
        if R['cv_results']:
            st.markdown('<div class="section-header">Cross-Validation Results</div>', unsafe_allow_html=True)
            cv_r = R['cv_results']
            col1, col2 = st.columns(2)
            for col, (name, scores) in zip([col1, col2], cv_r.items()):
                with col:
                    st.markdown(f"**{name}**")
                    cv_df = pd.DataFrame({
                        'Fold'         : [f'Fold {i+1}' for i in range(len(scores))],
                        'Accuracy (%)' : (scores * 100).round(4)
                    })
                    cv_df.loc[len(cv_df)] = ['**Mean**',    round(scores.mean()*100, 4)]
                    cv_df.loc[len(cv_df)] = ['**Std Dev**', round(scores.std()*100,  4)]
                    st.dataframe(cv_df, use_container_width=True, hide_index=True)
            fig_cv = make_cv_fig(cv_r)
            st.pyplot(fig_cv)
            plt.close(fig_cv)
        else:
            st.info("Cross-validation tidak dijalankan. Aktifkan di sidebar untuk melihat hasil.")

    # ── Tab 6: Classification Report ──────────────────────────────────────────
    with tabs[6]:
        st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
        cr_tabs = st.tabs(["Decision Tree", "Random Forest", "DT + SMOTE", "RF + SMOTE"])
        for tab, report in zip(cr_tabs, [R['class_report_dt'], R['class_report_rf'],
                                          R['class_report_dts'], R['class_report_rfs']]):
            with tab:
                st.code(report, language='text')

    # ── Tab 7: Export ──────────────────────────────────────────────────────────
    with tabs[7]:
        st.markdown('<div class="section-header">Export Hasil</div>', unsafe_allow_html=True)

        all_m = [R['dt_metrics'], R['rf_metrics'], R['dt_smote_metrics'], R['rf_smote_metrics']]
        export_df = pd.DataFrame(all_m)
        csv_bytes = export_df.to_csv(index=False).encode('utf-8')

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download Hasil Perbandingan (CSV)",
                data=csv_bytes,
                file_name="hasil_perbandingan_4model.csv",
                mime="text/csv",
                use_container_width=True
            )

        yt = R['y_test']
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            predictions = [(R['dt_pred'],yt),(R['rf_pred'],yt),(R['dt_smote_pred'],yt),(R['rf_smote_pred'],yt)]
            fig_cm = make_cm_fig(predictions, ['DT','RF','DT+SMOTE','RF+SMOTE'])
            st.download_button("⬇️ Confusion Matrix", fig_to_bytes(fig_cm), "confusion_matrix.png", "image/png", use_container_width=True)
            plt.close(fig_cm)
        with ex2:
            roc_exp = [(R['dt_prob'],yt,'DT'),(R['rf_prob'],yt,'RF'),(R['dt_smote_prob'],yt,'DT+SMOTE'),(R['rf_smote_prob'],yt,'RF+SMOTE')]
            fig_roc = make_roc_fig(roc_exp)
            st.download_button("⬇️ ROC Curve", fig_to_bytes(fig_roc), "roc_curve.png", "image/png", use_container_width=True)
            plt.close(fig_roc)
        with ex3:
            all_vals_dict = {
                'DT'      : {m: R['dt_metrics'][m]       for m in METRICS_KEYS},
                'RF'      : {m: R['rf_metrics'][m]       for m in METRICS_KEYS},
                'DT+SMOTE': {m: R['dt_smote_metrics'][m] for m in METRICS_KEYS},
                'RF+SMOTE': {m: R['rf_smote_metrics'][m] for m in METRICS_KEYS},
            }
            fig_bar = make_bar_fig(METRICS_KEYS, all_vals_dict)
            st.download_button("⬇️ Bar Chart Metrik", fig_to_bytes(fig_bar), "metrik_4model.png", "image/png", use_container_width=True)
            plt.close(fig_bar)

        if R['cv_results']:
            fig_cv = make_cv_fig(R['cv_results'])
            st.download_button("⬇️ Cross-Validation Chart", fig_to_bytes(fig_cv), "cross_validation.png", "image/png")
            plt.close(fig_cv)

        st.markdown("""
<div class="info-box">
💡 Semua grafik tersedia dalam format PNG resolusi tinggi (150 dpi).
CSV berisi semua metrik evaluasi keempat model.
</div>
""", unsafe_allow_html=True)