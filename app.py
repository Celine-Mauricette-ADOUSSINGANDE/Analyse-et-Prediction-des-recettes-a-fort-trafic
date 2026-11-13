"""
Recipe traffic dashboard (FR / EN)
Run with:  streamlit run dashboard.py

No data is embedded or simulated in this code: every figure is computed once a CSV is loaded
(upload it from the sidebar, or put recipe_site_traffic_2212.csv with ";" separator next to this file).
Optional: put logo-efrei-blanc.png next to this file to show the logo in the header.
"""
import base64
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

st.set_page_config(page_title="Recipe traffic dashboard", page_icon="🍲", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Palette and constants
# ----------------------------------------------------------------------------
NAVY, BLUE, SKY, ORANGE, INK, GREY = "#173868", "#1F5FA8", "#8DB6E3", "#F26B21", "#0F2240", "#5B6F8A"
NUTRI_COLS = ["calories", "carbohydrate", "sugar", "protein"]
DEFAULT_PATH = "recipe_site_traffic_2212.csv"
LOGO_PATH = "logo-efrei-blanc.png"

# ----------------------------------------------------------------------------
# Translations: key -> (French, English)
# ----------------------------------------------------------------------------
TX = {
    "title": ("Analyse et prédiction des recettes à fort trafic", "Analysis and Prediction of High-Traffic Recipes"),
    "subtitle": ("🎯 Problématique : quelles recettes mettre en avant pour maximiser le trafic du site et donc les ventes d'abonnements ?",
                 "🎯 Business question: which recipes should we feature to maximize site traffic and therefore subscription sales?"),
    "filters": ("Filtres", "Filters"),
    "upload": ("Charger le CSV (séparateur « ; »)", "Upload the CSV (\";\" separator)"),
    "nofile": ("Fichier `{p}` introuvable. Place-le à côté de `dashboard.py` ou charge-le ci-dessous.",
               "File `{p}` not found. Put it next to `dashboard.py` or upload it below."),
    "categories": ("Catégories", "Categories"),
    "servings": ("Portions", "Servings"),
    "traffic": ("Trafic", "Traffic"),
    "all": ("Tous", "All"), "high": ("Fort", "High"), "low": ("Faible", "Low"),
    "nutri_ranges": ("Plages nutritionnelles", "Nutrition ranges"),
    "hide_imputed": ("Exclure la nutrition imputée", "Exclude imputed nutrition"),
    "reset_note": ("Les filtres s'appliquent aux onglets Vue d'ensemble, Explorer et Données. Les tests et le modèle utilisent toutes les données.",
                   "Filters apply to the Overview, Explore and Data tabs. Tests and the model use the full dataset."),
    "empty": ("Aucune recette ne correspond aux filtres. Élargis la sélection.",
              "No recipe matches the filters. Widen your selection."),
    "chip_recipes": ("recettes", "recipes"), "chip_cats": ("catégories", "categories"),
    "chip_high": ("de fort trafic", "high traffic"),
    "tab_over": ("🏠 Vue d'ensemble", "🏠 Overview"), "tab_explore": ("🔍 Explorer", "🔍 Explore"),
    "tab_stats": ("🧪 Tests statistiques", "🧪 Statistical tests"), "tab_model": ("🤖 Modèles et prédiction", "🤖 Models & prediction"),
    "tab_data": ("🗂️ Données", "🗂️ Data"),
    "k_sel": ("Recettes", "Recipes"), "k_sel_sub": ("{p:.0%} du total", "{p:.0%} of total"),
    "k_rate": ("Fort trafic", "High traffic"), "k_rate_sub": ("{d:+.1f} pts vs global", "{d:+.1f} pts vs overall"),
    "k_cal": ("Calories (médiane)", "Calories (median)"), "k_pro": ("Protéines (médiane)", "Protein (median)"),
    "k_cat": ("Catégories", "Categories"),
    "rate_cat": ("Taux de fort trafic par catégorie", "High-traffic rate by category"),
    "split": ("Répartition du trafic", "Traffic split"),
    "heat": ("Taux de fort trafic : catégorie × portions", "High-traffic rate: category × servings"),
    "rate": ("Taux", "Rate"), "recipes": ("Recettes", "Recipes"), "category": ("Catégorie", "Category"),
    "insight_over": ("Dans la sélection, **{top}** est la catégorie la plus performante ({tr:.0%}) et **{bot}** la moins performante ({br:.0%}).",
                     "In the selection, **{top}** is the best-performing category ({tr:.0%}) and **{bot}** the weakest ({br:.0%})."),
    "variable": ("Variable", "Variable"), "bins": ("Classes", "Bins"), "split_traffic": ("Séparer par trafic", "Split by traffic"),
    "hist": ("Distribution", "Distribution"), "box": ("Boîtes à moustaches", "Box plots"),
    "stats": ("Statistiques", "Statistics"),
    "count": ("Effectif", "Count"), "mean": ("Moyenne", "Mean"), "median": ("Médiane", "Median"),
    "std": ("Écart-type", "Std dev"),
    "skew_note": ("Asymétrie de la sélection : **{s:.2f}** (> 1 : étalée vers la droite, un log aide les modèles linéaires).",
                  "Skewness of the selection: **{s:.2f}** (> 1: stretched to the right; a log helps linear models)."),
    "relation": ("Relation entre deux variables", "Relationship between two variables"),
    "x_axis": ("Axe X", "X axis"), "y_axis": ("Axe Y", "Y axis"), "log": ("Échelle log", "Log scale"),
    "scatter_hint": ("Molette pour zoomer, glisser pour se déplacer, double-clic pour réinitialiser.",
                     "Scroll to zoom, drag to pan, double-click to reset."),
    "corr": ("Corrélations", "Correlations"),
    "mw": ("Nutrition vs trafic : test de Mann-Whitney", "Nutrition vs traffic: Mann-Whitney test"),
    "mw_note": ("p < 0,01 : lien net · 0,01 ≤ p < 0,05 : à la limite · p ≥ 0,05 : pas de lien. Quatre tests sont faits sans correction : à lire prudemment.",
                "p < 0.01: clear link · 0.01 ≤ p < 0.05: borderline · p ≥ 0.05: no link. Four tests are run without correction: read cautiously."),
    "clear": ("net", "clear"), "borderline": ("limite", "borderline"), "none": ("aucun", "none"),
    "chi_cat": ("Catégorie vs trafic", "Category vs traffic"), "chi_serv": ("Portions vs trafic", "Servings vs traffic"),
    "cramer": ("V de Cramér", "Cramér's V"), "pval": ("p-value", "p-value"),
    "strength": ("lien {s}", "{s} link"),
    "weak": ("faible", "weak"), "moderate": ("modéré", "moderate"), "strong": ("fort", "strong"),
    "missing": ("Valeurs manquantes", "Missing values"),
    "missing_txt": ("{n} recettes ({p:.1%}) n'ont aucune donnée nutritionnelle (imputées par la médiane de leur catégorie). Taux de fort trafic : **{rm:.0%}** pour elles contre **{ro:.0%}** pour les autres. L'écart vient surtout de la composition par catégorie.",
                    "{n} recipes ({p:.1%}) have no nutrition data (imputed with their category median). High-traffic rate: **{rm:.0%}** for them vs **{ro:.0%}** for the others. The gap mostly comes from the category mix."),
    "model_title": ("Comparaison des modèles (jeu de test 20 %)", "Model comparison (20% test set)"),
    "model_note": ("Modèles entraînés sur 80 % des données (stratifié, graine 42) avec les hyperparamètres retenus dans le notebook.",
                   "Models trained on 80% of the data (stratified, seed 42) with the hyperparameters selected in the notebook."),
    "model": ("Modèle", "Model"), "precision": ("Précision", "Precision"), "recall": ("Rappel", "Recall"),
    "roc": ("Courbes ROC", "ROC curves"), "fpr": ("Taux de faux positifs", "False positive rate"),
    "tpr": ("Taux de vrais positifs", "True positive rate"),
    "confusion": ("Matrice de confusion", "Confusion matrix"), "actual": ("Réel", "Actual"), "predicted": ("Prédit", "Predicted"),
    "coef": ("Variables les plus influentes (régression logistique)", "Most influential variables (logistic regression)"),
    "coef_note": ("Coefficient > 0 : augmente la probabilité de fort trafic ; < 0 : la diminue.",
                  "Coefficient > 0: increases the probability of high traffic; < 0: decreases it."),
    "predict": ("Prédire une nouvelle recette", "Predict a new recipe"),
    "proba": ("Probabilité de fort trafic", "Probability of high traffic"),
    "verdict_hi": ("Fort trafic probable", "High traffic likely"), "verdict_lo": ("Faible trafic probable", "Low traffic likely"),
    "verdict_unc": ("Incertain", "Uncertain"),
    "predict_note": ("La catégorie pèse bien plus que les valeurs nutritionnelles : change-la pour voir.",
                     "Category weighs far more than the nutrition values: change it to see."),
    "carbs": ("Glucides (g)", "Carbohydrates (g)"), "sugar": ("Sucre (g)", "Sugar (g)"), "protein": ("Protéines (g)", "Protein (g)"),
    "calories": ("Calories", "Calories"),
    "data": ("recettes sélectionnées", "selected recipes"), "download": ("⬇️ Télécharger la sélection (CSV)", "⬇️ Download the selection (CSV)"),
    "recipe": ("Recette", "Recipe"),
    "data_src": ("Source des données", "Data source"),
    "welcome_t": ("Charge tes données pour démarrer", "Load your data to get started"),
    "welcome": ("Aucune donnée n'est intégrée à l'application : tous les calculs se font une fois le fichier chargé. Utilise le bouton **Charger le CSV** de la barre latérale (colonnes attendues : {cols}).",
                "No data is embedded in the app: every computation runs once the file is loaded. Use the **Upload the CSV** button in the sidebar (expected columns: {cols})."),
    "bad_cols": ("Colonnes manquantes dans le fichier : {c}", "Missing columns in the file: {c}"),
    "outliers": ("Valeurs aberrantes", "Outliers"),
    "o_keep": ("Conserver", "Keep"), "o_cap": ("Plafonner (P1–P99)", "Cap (P1–P99)"), "o_drop": ("Exclure (> 3×IQR)", "Exclude (> 3×IQR)"),
    "o_help": ("Traitement appliqué aux 4 variables nutritionnelles pour tous les onglets et le modèle.",
               "Treatment applied to the 4 nutrition variables in every tab and in the model."),
    "tab_quality": ("🧹 Qualité des données", "🧹 Data quality"), "tab_conc": ("✅ Conclusions", "✅ Conclusions"),
    "q_rows": ("Lignes", "Rows"), "q_dups": ("Doublons", "Duplicates"), "q_miss": ("Nutrition manquante", "Nutrition missing"),
    "q_out": ("Lignes aberrantes (1,5×IQR)", "Outlier rows (1.5×IQR)"), "q_after": ("Lignes après traitement", "Rows after treatment"),
    "q_miss_t": ("Valeurs manquantes par colonne (%)", "Missing values by column (%)"),
    "q_out_t": ("Détection des valeurs aberrantes (méthode IQR)", "Outlier detection (IQR method)"),
    "q_var": ("Variable", "Variable"), "q_fence": ("Seuil haut (1,5×IQR)", "Upper fence (1.5×IQR)"),
    "q_n15": ("Au-dessus 1,5×IQR", "Above 1.5×IQR"), "q_n3": ("Au-dessus 3×IQR", "Above 3×IQR"), "q_max": ("Maximum", "Maximum"),
    "q_ba": ("Avant / après traitement", "Before / after treatment"), "q_before": ("Avant", "Before"), "q_afterl": ("Après", "After"),
    "q_ba_note": ("Méthode choisie : **{m}**. Les valeurs extrêmes sont plausibles (grosses recettes) mais très rares : les plafonner évite qu'elles dominent les moyennes et les modèles linéaires.",
                  "Chosen method: **{m}**. Extreme values are plausible (large recipes) but very rare: capping stops them from dominating means and linear models."),
    "q_crit": ("Analyse critique du jeu de données", "Critical review of the dataset"),
    "c1": ("**Cible binaire** : `high_traffic` indique seulement fort / non fort, sans volume de visites ni date : impossible de chiffrer l'impact réel.",
           "**Binary target**: `high_traffic` only says high / not high, with no visit volume or date: the real impact cannot be quantified."),
    "c2": ("**Hypothèse sur la cible** : {n:.0%} des lignes n'ont pas de valeur et sont lues comme « Low ». Si ce sont en réalité des données absentes, les taux sont biaisés.",
           "**Target assumption**: {n:.0%} of rows have no value and are read as “Low”. If they are actually missing data, the rates are biased."),
    "c3": ("**Données nutritionnelles manquantes** : {n} recettes ({p:.1%}), toutes variables à la fois ; aucune dans {z}. Imputation par la médiane de la catégorie.",
           "**Missing nutrition data**: {n} recipes ({p:.1%}), all variables at once; none in {z}. Imputed with the category median."),
    "c4": ("**Valeurs extrêmes** : {n} lignes dépassent 1,5×IQR sur au moins une variable ; distributions très asymétriques (asymétrie max {s:.1f}).",
           "**Extreme values**: {n} rows exceed 1.5×IQR on at least one variable; strongly skewed distributions (max skewness {s:.1f})."),
    "c5": ("**Petits effectifs** : la plus petite catégorie compte {m} recettes ; par catégorie × portions, certaines cases ont très peu de recettes, donc des taux instables.",
           "**Small counts**: the smallest category has {m} recipes; by category × servings some cells hold very few recipes, so rates are unstable."),
    "c6": ("**Variables absentes** : ni ingrédients, ni temps de préparation, ni saison, ni auteur ; données observationnelles, on mesure des associations, pas des causes.",
           "**Missing variables**: no ingredients, preparation time, season or author; observational data, so we measure associations, not causes."),
    "c7": ("**Nettoyage nécessaire** : {a} valeurs de portions du type « as a snack » et {b} catégories au lieu de 10 (« Chicken Breast » fusionné).",
           "**Cleaning required**: {a} servings values like \"as a snack\" and {b} categories instead of 10 (\"Chicken Breast\" merged)."),
    "conc_t": ("Conclusions et recommandations", "Conclusions and recommendations"),
    "conc_top": ("Catégories à mettre en avant", "Categories to feature"),
    "conc_n": ("{n} recettes", "{n} recipes"),
    "conc_1": ("**Prioriser {c1}, {c2} et {c3}** : plus de {r:.0%} de leurs recettes atteignent un fort trafic, contre {o:.0%} en moyenne sur le site.",
               "**Prioritize {c1}, {c2} and {c3}**: over {r:.0%} of their recipes reach high traffic, vs {o:.0%} on average across the site."),
    "conc_2": ("**Ne pas se fier à la nutrition ni aux portions** : le lien avec le trafic est faible (V de Cramér de la catégorie {v:.2f} ; portions : p = {p:.2f}).",
               "**Do not rely on nutrition or servings**: their link with traffic is weak (category Cramér's V {v:.2f}; servings: p = {p:.2f})."),
    "conc_3": ("**Utiliser le modèle comme outil de tri** : {m} obtient un F1 de {f:.2f} et un ROC AUC de {a:.2f} ; quand il annonce un fort trafic, il a raison {pr:.0%} du temps.",
               "**Use the model as a screening tool**: {m} reaches an F1 of {f:.2f} and a ROC AUC of {a:.2f}; when it predicts high traffic, it is right {pr:.0%} of the time."),
    "conc_4": ("**Pistes** : collecter le volume de visites, les ingrédients et la saison ; mesurer l'effet réel d'une mise en avant par un test A/B.",
               "**Next steps**: collect visit volumes, ingredients and season; measure the real effect of featuring a recipe with an A/B test."),
}


def tr(key, **kw):
    s = TX[key][0 if LANG == "fr" else 1]
    return s.format(**kw) if kw else s


LANG_CHOICE = st.sidebar.radio("🌐", ["Français", "English"], horizontal=True, label_visibility="collapsed")
LANG = "fr" if LANG_CHOICE == "Français" else "en"
LABELS = {"calories": tr("calories"), "carbohydrate": tr("carbs"), "sugar": tr("sugar"),
          "protein": tr("protein"), "servings": tr("servings")}

# ----------------------------------------------------------------------------
# Style (CSS)
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
#MainMenu, footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{background: transparent;}}
.block-container {{padding-top: 1.2rem; max-width: 1350px;}}
section[data-testid="stSidebar"] {{background: #EAF1FA;}}
.hero {{background: linear-gradient(120deg, {NAVY} 0%, {BLUE} 65%, #2F7BD0 100%); border-radius: 22px;
  padding: 26px 34px; color: white; display: flex; justify-content: space-between; align-items: center;
  box-shadow: 0 10px 26px rgba(23,56,104,.28); margin-bottom: 18px;}}
.hero h1 {{color: white; margin: 0; font-size: 2.1rem; font-weight: 800; padding: 0;}}
.hero p {{margin: 6px 0 0 0; color: #C9DBF0; font-size: 1.02rem;}}
.chip {{display: inline-block; background: rgba(255,255,255,.16); border-radius: 999px; padding: 4px 14px;
  margin: 12px 8px 0 0; font-size: .86rem; color: white;}}
.chip b {{color: #FFB84D;}}
.kpi {{background: white; border-radius: 18px; padding: 16px 20px; box-shadow: 0 3px 14px rgba(23,56,104,.09);}}
.kpi .label {{color: {GREY}; font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600;}}
.kpi .value {{color: {NAVY}; font-size: 2rem; font-weight: 800; line-height: 1.25;}}
.kpi .sub {{font-size: .82rem; color: {GREY};}}
.kpi .up {{color: #1B8A5A; font-weight: 600;}} .kpi .down {{color: #C0392B; font-weight: 600;}}
[class*="st-key-card_"] {{background: white !important; border-radius: 18px !important; border: none !important;
  box-shadow: 0 3px 14px rgba(23,56,104,.08); padding: 6px 10px;}}
.stTabs [data-baseweb="tab-list"] {{gap: 8px;}}
.stTabs [data-baseweb="tab"] {{background: white; border-radius: 12px; padding: 8px 18px; height: auto;
  box-shadow: 0 1px 6px rgba(23,56,104,.07);}}
.stTabs [aria-selected="true"] {{background: {NAVY};}}
.stTabs [aria-selected="true"] p {{color: white !important; font-weight: 700;}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{display: none;}}
.insight {{background: #EAF1FA; border-radius: 12px; padding: 10px 16px; color: {INK}; font-size: .93rem; margin-top: 6px;}}
.badge {{display:inline-block; border-radius: 999px; padding: 2px 12px; font-weight: 700; font-size: .8rem;}}
.b-clear {{background:#D9F2E6; color:#12704A;}} .b-border {{background:#FFE9CC; color:#A25A00;}} .b-none {{background:#E6EDF7; color:{GREY};}}
.gauge {{background:#E6EDF7; border-radius: 999px; height: 18px; overflow: hidden;}}
.gauge > div {{height: 100%; border-radius: 999px;}}
.bigproba {{font-size: 3rem; font-weight: 800; color: {NAVY}; line-height: 1.1;}}
h2, h3 {{color: {NAVY};}}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data (real file only)
# ----------------------------------------------------------------------------
REQUIRED = ["recipe", "calories", "carbohydrate", "sugar", "protein", "category", "servings", "high_traffic"]


@st.cache_data(show_spinner="…")
def load_data(path_or_file):
    """Read and clean the CSV. Everything (counts, rates, tests) is computed from this data only."""
    raw = pd.read_csv(path_or_file, sep=";")
    missing_cols = [c for c in REQUIRED if c not in raw.columns]
    if missing_cols:
        return None, {"missing_cols": missing_cols}
    d = raw.copy()
    info = {"n_rows": len(d), "n_cols": d.shape[1], "duplicates": int(d.duplicated().sum()),
            "na_pct": (d.isna().mean() * 100).round(2).to_dict(),
            "na_target": float(d["high_traffic"].isna().mean()),
            "n_cat_before": int(d["category"].nunique()),
            "odd_servings": int(d["servings"].astype(str).str.contains(r"[A-Za-z]").sum())}
    d["high_traffic"] = d["high_traffic"].fillna("Low").map({"High": 1, "Low": 0})
    d["category"] = d["category"].replace({"Chicken Breast": "Chicken"})
    d["servings"] = d["servings"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    d["nutrition_missing"] = d[NUTRI_COLS].isna().any(axis=1)
    for col in NUTRI_COLS:
        d[col] = d[col].fillna(d.groupby("category")[col].transform("median"))
    return d, info


@st.cache_data(show_spinner="…")
def outlier_report(d: pd.DataFrame):
    rows, flag = [], pd.Series(False, index=d.index)
    for c in NUTRI_COLS:
        q1, q3 = d[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        hi15, hi3 = q3 + 1.5 * iqr, q3 + 3 * iqr
        flag |= (d[c] > hi15) | (d[c] < q1 - 1.5 * iqr)
        rows.append({"var": c, "fence": hi15, "n15": int((d[c] > hi15).sum()), "n3": int((d[c] > hi3).sum()),
                     "max": d[c].max(), "skew": d[c].skew()})
    return pd.DataFrame(rows), int(flag.sum())


@st.cache_data(show_spinner="…")
def treat_outliers(d: pd.DataFrame, method: str):
    out = d.copy()
    if method == "cap":
        for c in NUTRI_COLS:
            lo, hi = out[c].quantile([0.01, 0.99])
            out[c] = out[c].clip(lo, hi)
    elif method == "drop":
        keep = pd.Series(True, index=out.index)
        for c in NUTRI_COLS:
            q1, q3 = out[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            keep &= out[c].between(q1 - 3 * iqr, q3 + 3 * iqr)
        out = out[keep]
    return out


st.sidebar.markdown(f"### 📁 {tr('data_src')}")
uploaded = st.sidebar.file_uploader(tr("upload"), type="csv", label_visibility="collapsed")
source = uploaded if uploaded is not None else (DEFAULT_PATH if os.path.exists(DEFAULT_PATH) else None)
if source is None:
    st.markdown(f"## 🍲 {tr('welcome_t')}")
    st.info(tr("welcome", cols=", ".join(f"`{c}`" for c in REQUIRED)))
    st.stop()
df_clean, INFO = load_data(source)
if df_clean is None:
    st.error(tr("bad_cols", c=", ".join(INFO["missing_cols"])))
    st.stop()

OUT_LABELS = {"keep": tr("o_keep"), "cap": tr("o_cap"), "drop": tr("o_drop")}
out_method = st.sidebar.selectbox(f"🧹 {tr('outliers')}", list(OUT_LABELS), index=1,
                                  format_func=OUT_LABELS.get, help=tr("o_help"))
df = treat_outliers(df_clean, out_method)
for _d in (df_clean, df):
    _d["traffic"] = np.where(_d["high_traffic"] == 1, tr("high"), tr("low"))
COLOR_TRAFFIC = alt.Scale(domain=[tr("low"), tr("high")], range=[SKY, ORANGE])


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def style(chart, height=300):
    return (chart.properties(height=height)
            .configure(background="#FFFFFF")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=INK, titleColor=GREY, gridColor="#E6EDF7", domainColor="#C9D6E8",
                            tickColor="#C9D6E8", labelFontSize=12, titleFontSize=12)
            .configure_legend(labelColor=INK, titleColor=GREY, orient="bottom", labelFontSize=12))


def show(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:                      # older Streamlit versions
        st.altair_chart(chart, use_container_width=True)


_card_n = [0]


def card():
    """Bordered container styled as a white card (key-based CSS; falls back on older Streamlit)."""
    _card_n[0] += 1
    try:
        return st.container(border=True, key=f"card_{_card_n[0]}")
    except TypeError:
        return st.container(border=True)


def kpi(label, value, sub="", tone=""):
    st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                f'<div class="sub {tone}">{sub}</div></div>', unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def card_title(text):
    st.markdown(f"##### {text}")


# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
st.sidebar.markdown(f"### 🍲 {tr('filters')}")
cats = sorted(df["category"].unique())
sel_cats = st.sidebar.multiselect(tr("categories"), cats, default=cats)
servs = sorted(df["servings"].unique())
sel_serv = st.sidebar.multiselect(tr("servings"), servs, default=servs)
sel_traffic = st.sidebar.radio(tr("traffic"), [tr("all"), tr("high"), tr("low")], horizontal=True)
st.sidebar.markdown(f"**{tr('nutri_ranges')}**")
ranges = {c: st.sidebar.slider(LABELS[c], float(df[c].min()), float(df[c].max()),
                               (float(df[c].min()), float(df[c].max()))) for c in NUTRI_COLS}
hide_imputed = st.sidebar.checkbox(tr("hide_imputed"))
st.sidebar.caption(tr("reset_note"))

mask = df["category"].isin(sel_cats) & df["servings"].isin(sel_serv)
for c, (lo, hi) in ranges.items():
    mask &= df[c].between(lo, hi)
if sel_traffic != tr("all"):
    mask &= df["traffic"] == sel_traffic
if hide_imputed:
    mask &= ~df["nutrition_missing"]
f = df[mask]

# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
logo_html = ""
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as fh:
        logo_html = f'<img src="data:image/png;base64,{base64.b64encode(fh.read()).decode()}" style="height:54px;">'
overall = df["high_traffic"].mean()
st.markdown(f"""
<div class="hero">
  <div>
    <h1>🍲 {tr('title')}</h1>
    <p>{tr('subtitle')}</p>
    <span class="chip"><b>{len(df)}</b> {tr('chip_recipes')}</span>
    <span class="chip"><b>{df['category'].nunique()}</b> {tr('chip_cats')}</span>
    <span class="chip"><b>{overall:.0%}</b> {tr('chip_high')}</span>
  </div>
  <div>{logo_html}</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([tr("tab_over"), tr("tab_explore"), tr("tab_stats"), tr("tab_model"), tr("tab_quality"),
                tr("tab_conc"), tr("tab_data")])


# ----------------------------------------------------------------------------
# Models (trained once on the full dataset, as in the notebook)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="…")
def train_models(data: pd.DataFrame):
    cat_cols, num_cols = ["category"], NUTRI_COLS + ["servings"]
    X, y = data[num_cols + cat_cols], data["high_traffic"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    prep_lin = ColumnTransformer([
        ("log_scale", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                ("scale", StandardScaler())]), NUTRI_COLS),
        ("scale", StandardScaler(), ["servings"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    prep_tree = ColumnTransformer([("pass", "passthrough", num_cols),
                                   ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    models = {
        "Logistic Regression": Pipeline([("prep", prep_lin), ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
        "Random Forest": Pipeline([("prep", prep_tree), ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=5, random_state=42, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("prep", prep_tree), ("clf", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42))]),
    }
    out = {"fitted": {}, "metrics": [], "roc": {}, "cm": {}}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred, proba = m.predict(X_te), m.predict_proba(X_te)[:, 1]
        out["fitted"][name] = m
        out["metrics"].append({"model": name, "accuracy": accuracy_score(y_te, pred),
                               "precision": precision_score(y_te, pred), "recall": recall_score(y_te, pred),
                               "f1": f1_score(y_te, pred), "auc": roc_auc_score(y_te, proba)})
        fpr, tpr, _ = roc_curve(y_te, proba)
        out["roc"][name] = pd.DataFrame({"fpr": fpr, "tpr": tpr, "model": name})
        out["cm"][name] = confusion_matrix(y_te, pred)
    out["n_test"] = len(X_te)
    return out


# ============================================================================
# TAB 1 : Overview
# ============================================================================
with tabs[0]:
    if f.empty:
        st.warning(tr("empty"))
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            kpi(tr("k_sel"), f"{len(f):,}", tr("k_sel_sub", p=len(f) / len(df)))
        with k2:
            d = (f["high_traffic"].mean() - overall) * 100
            kpi(tr("k_rate"), f"{f['high_traffic'].mean():.1%}", tr("k_rate_sub", d=d), "up" if d >= 0 else "down")
        with k3:
            kpi(tr("k_cal"), f"{f['calories'].median():.0f}", "kcal")
        with k4:
            kpi(tr("k_pro"), f"{f['protein'].median():.1f}", "g")
        with k5:
            kpi(tr("k_cat"), f"{f['category'].nunique()}", f"/ {df['category'].nunique()}")
        st.write("")

        left, right = st.columns([3, 2])
        by_cat = f.groupby("category").agg(rate=("high_traffic", "mean"), recipes=("recipe", "count")).reset_index()
        with left:
            with card():
                card_title(tr("rate_cat"))
                bars = (alt.Chart(by_cat).mark_bar(cornerRadiusEnd=6, color=BLUE)
                        .encode(x=alt.X("rate:Q", axis=alt.Axis(format="%"), title=None, scale=alt.Scale(domain=[0, 1])),
                                y=alt.Y("category:N", sort="-x", title=None),
                                color=alt.Color("rate:Q", scale=alt.Scale(range=[SKY, NAVY]), legend=None),
                                tooltip=[alt.Tooltip("category:N", title=tr("category")),
                                         alt.Tooltip("rate:Q", format=".1%", title=tr("rate")),
                                         alt.Tooltip("recipes:Q", title=tr("recipes"))]))
                labels = bars.mark_text(align="left", dx=4, color=INK, fontSize=12).encode(
                    text=alt.Text("rate:Q", format=".0%"), color=alt.value(INK))
                show(style(bars + labels, 330))
        with right:
            with card():
                card_title(tr("split"))
                sp = f["traffic"].value_counts().reset_index()
                sp.columns = ["traffic", "recipes"]
                donut = (alt.Chart(sp).mark_arc(innerRadius=80, cornerRadius=6)
                         .encode(theta="recipes:Q", color=alt.Color("traffic:N", scale=COLOR_TRAFFIC, title=None),
                                 tooltip=["traffic:N", "recipes:Q"]))
                show(style(donut, 330))

        with card():
            card_title(tr("heat"))
            heat = f.groupby(["category", "servings"]).agg(rate=("high_traffic", "mean"), recipes=("recipe", "count")).reset_index()
            base = alt.Chart(heat).encode(x=alt.X("servings:O", title=tr("servings")), y=alt.Y("category:N", title=None))
            rect = base.mark_rect(cornerRadius=4).encode(
                color=alt.Color("rate:Q", scale=alt.Scale(range=["#EAF1FA", NAVY], domain=[0, 1]), title=tr("rate"),
                                legend=alt.Legend(format="%")),
                tooltip=[alt.Tooltip("category:N", title=tr("category")), alt.Tooltip("servings:O", title=tr("servings")),
                         alt.Tooltip("rate:Q", format=".0%", title=tr("rate")), alt.Tooltip("recipes:Q", title=tr("recipes"))])
            txt = base.mark_text(fontSize=11).encode(text=alt.Text("rate:Q", format=".0%"),
                                                    color=alt.condition("datum.rate > 0.55", alt.value("white"), alt.value(INK)))
            show(style(rect + txt, 340))
        best, worst = by_cat.sort_values("rate").iloc[-1], by_cat.sort_values("rate").iloc[0]
        insight(tr("insight_over", top=best["category"], tr=best["rate"], bot=worst["category"], br=worst["rate"]))

# ============================================================================
# TAB 2 : Explore
# ============================================================================
with tabs[1]:
    if f.empty:
        st.warning(tr("empty"))
    else:
        with card():
            card_title(tr("hist"))
            c1, c2, c3 = st.columns([2, 1, 1])
            var = c1.selectbox(tr("variable"), NUTRI_COLS + ["servings"], format_func=LABELS.get)
            nb = c2.slider(tr("bins"), 10, 60, 30)
            split_t = c3.checkbox(tr("split_traffic"), value=True)
            enc = dict(x=alt.X(f"{var}:Q", bin=alt.Bin(maxbins=nb), title=LABELS[var]), y=alt.Y("count():Q", title=tr("recipes")))
            if split_t:
                enc["color"] = alt.Color("traffic:N", scale=COLOR_TRAFFIC, title=None)
                enc["tooltip"] = ["traffic:N", "count():Q"]
            hist = alt.Chart(f).mark_bar(color=BLUE, opacity=0.9, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(**enc)
            show(style(hist.interactive(), 300))
            st.markdown(f'<div class="insight">💡 {tr("skew_note", s=f[var].skew())}</div>', unsafe_allow_html=True)
        st.write("")
        l, r = st.columns(2)
        with l:
            with card():
                card_title(tr("box"))
                box = alt.Chart(f).mark_boxplot(extent="min-max", size=46).encode(
                    x=alt.X("traffic:N", title=None, sort=[tr("low"), tr("high")]),
                    y=alt.Y(f"{var}:Q", title=LABELS[var]),
                    color=alt.Color("traffic:N", scale=COLOR_TRAFFIC, legend=None))
                show(style(box, 300))
        with r:
            with card():
                card_title(tr("stats"))
                s = f.groupby("traffic")[var].describe()[["count", "mean", "50%", "std", "min", "max"]]
                s.columns = [tr("count"), tr("mean"), tr("median"), tr("std"), "Min", "Max"]
                st.dataframe(s.round(1))
        st.write("")
        with card():
            card_title(tr("relation"))
            c1, c2, c3 = st.columns(3)
            xv = c1.selectbox(tr("x_axis"), NUTRI_COLS, index=0, format_func=LABELS.get)
            yv = c2.selectbox(tr("y_axis"), NUTRI_COLS, index=3, format_func=LABELS.get)
            lg = c3.checkbox(tr("log"))
            stype = "symlog" if lg else "linear"
            smp = f if len(f) <= 2000 else f.sample(2000, random_state=0)
            sc = alt.Chart(smp).mark_circle(size=70, opacity=0.65).encode(
                x=alt.X(f"{xv}:Q", scale=alt.Scale(type=stype), title=LABELS[xv]),
                y=alt.Y(f"{yv}:Q", scale=alt.Scale(type=stype), title=LABELS[yv]),
                color=alt.Color("traffic:N", scale=COLOR_TRAFFIC, title=None),
                tooltip=["recipe:Q", "category:N", "servings:Q", "calories:Q", "carbohydrate:Q", "sugar:Q", "protein:Q", "traffic:N"])
            show(style(sc.interactive(), 400))
            st.caption(tr("scatter_hint"))

# ============================================================================
# TAB 3 : Statistical tests (full dataset, as in the notebook EDA)
# ============================================================================
with tabs[2]:
    l, r = st.columns([3, 2])
    with l:
        with card():
            card_title(tr("mw"))
            rows = []
            for c in NUTRI_COLS:
                u, p = mannwhitneyu(df.loc[df.high_traffic == 1, c], df.loc[df.high_traffic == 0, c])
                rows.append((c, df[c].skew(), p))
            html = ["<table style='width:100%;border-collapse:collapse;font-size:.95rem'>"
                    f"<tr style='color:{GREY};text-align:left'><th>{tr('variable')}</th><th>Skew</th><th>{tr('pval')}</th><th></th></tr>"]
            for c, sk, p in rows:
                cls, lab = (("b-clear", tr("clear")) if p < 0.01 else ("b-border", tr("borderline")) if p < 0.05 else ("b-none", tr("none")))
                html.append(f"<tr style='border-top:1px solid #E6EDF7'><td style='padding:9px 0'><b>{LABELS[c]}</b></td>"
                            f"<td>{sk:.2f}</td><td>{p:.4f}</td><td><span class='badge {cls}'>{lab}</span></td></tr>")
            html.append("</table>")
            st.markdown("".join(html), unsafe_allow_html=True)
            st.caption(tr("mw_note"))
    with r:
        with card():
            card_title(tr("chi_cat"))
            ct = pd.crosstab(df["category"], df["high_traffic"])
            chi2, p, _, _ = chi2_contingency(ct)
            v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
            strength = tr("strong") if v >= 0.3 else tr("moderate") if v >= 0.1 else tr("weak")
            st.markdown(f"<div class='bigproba'>{v:.2f}</div><div style='color:{GREY}'>{tr('cramer')} · {tr('strength', s=strength)}<br>χ² = {chi2:.1f} · {tr('pval')} = {p:.2e}</div>", unsafe_allow_html=True)
        st.write("")
        with card():
            card_title(tr("chi_serv"))
            chi2s, ps, _, _ = chi2_contingency(pd.crosstab(df["servings"], df["high_traffic"]))
            st.markdown(f"<div class='bigproba'>p = {ps:.2f}</div><div style='color:{GREY}'>χ² = {chi2s:.1f}</div>", unsafe_allow_html=True)

    l, r = st.columns(2)
    with l:
        with card():
            card_title(tr("corr"))
            corr = df[NUTRI_COLS + ["servings", "high_traffic"]].corr().round(2)
            cl = corr.reset_index().melt("index", var_name="variable", value_name="r")
            b = alt.Chart(cl).encode(x=alt.X("index:N", title=None), y=alt.Y("variable:N", title=None))
            hm = b.mark_rect(cornerRadius=4).encode(color=alt.Color("r:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1]), title="r"),
                                                    tooltip=["index:N", "variable:N", "r:Q"])
            tx = b.mark_text(fontSize=12).encode(text=alt.Text("r:Q", format=".2f"),
                                                 color=alt.condition("abs(datum.r) > 0.6", alt.value("white"), alt.value(INK)))
            show(style(hm + tx, 340))
    with r:
        with card():
            card_title(tr("missing"))
            nm = df["nutrition_missing"]
            st.markdown(tr("missing_txt", n=int(nm.sum()), p=nm.mean(), rm=df.loc[nm, "high_traffic"].mean(), ro=df.loc[~nm, "high_traffic"].mean()))
            mt = df.groupby(["category", "nutrition_missing"]).agg(rate=("high_traffic", "mean"), n=("recipe", "count")).reset_index()
            mt["nutrition_missing"] = mt["nutrition_missing"].map({True: "NaN", False: "OK"})
            ch = alt.Chart(mt).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y("category:N", title=None), x=alt.X("rate:Q", axis=alt.Axis(format="%"), title=None),
                yOffset="nutrition_missing:N", color=alt.Color("nutrition_missing:N", scale=alt.Scale(domain=["OK", "NaN"], range=[BLUE, ORANGE]), title=None),
                tooltip=["category:N", "nutrition_missing:N", alt.Tooltip("rate:Q", format=".0%"), "n:Q"])
            show(style(ch, 300))

# ============================================================================
# TAB 4 : Models and prediction
# ============================================================================
with tabs[3]:
    M = train_models(df)
    with card():
        card_title(tr("model_title"))
        mt = pd.DataFrame(M["metrics"]).set_index("model")
        mt.columns = ["Accuracy", tr("precision"), tr("recall"), "F1", "ROC AUC"]
        mt.index.name = tr("model")
        st.dataframe(mt.style.format("{:.3f}").highlight_max(color="#C9DBF0"))
        st.caption(tr("model_note"))
    st.write("")
    l, r = st.columns(2)
    with l:
        with card():
            card_title(tr("roc"))
            rc = pd.concat(M["roc"].values())
            line = alt.Chart(rc).mark_line(strokeWidth=3).encode(
                x=alt.X("fpr:Q", title=tr("fpr")), y=alt.Y("tpr:Q", title=tr("tpr")),
                color=alt.Color("model:N", scale=alt.Scale(range=[ORANGE, BLUE, SKY]), title=None))
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[5, 5], color=GREY).encode(x="x:Q", y="y:Q")
            show(style(line + diag, 360))
    with r:
        with card():
            card_title(tr("confusion"))
            name = st.selectbox(tr("model"), list(M["cm"]), label_visibility="collapsed")
            cm = M["cm"][name]
            lab = [tr("low"), tr("high")]
            cdf = pd.DataFrame([{tr("actual"): lab[i], tr("predicted"): lab[j], "n": int(cm[i, j])} for i in range(2) for j in range(2)])
            bb = alt.Chart(cdf).encode(x=alt.X(f"{tr('predicted')}:N", sort=lab), y=alt.Y(f"{tr('actual')}:N", sort=lab))
            rr = bb.mark_rect(cornerRadius=6).encode(color=alt.Color("n:Q", scale=alt.Scale(range=["#EAF1FA", NAVY]), legend=None))
            tt = bb.mark_text(fontSize=26, fontWeight="bold").encode(text="n:Q", color=alt.condition("datum.n > 40", alt.value("white"), alt.value(INK)))
            show(style(rr + tt, 310))
    st.write("")
    l, r = st.columns([3, 2])
    lr = M["fitted"]["Logistic Regression"]
    with l:
        with card():
            card_title(tr("coef"))
            names = [n.split("__")[-1] for n in lr.named_steps["prep"].get_feature_names_out()]
            coefs = pd.DataFrame({"feature": names, "coef": lr.named_steps["clf"].coef_[0]})
            coefs = coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index).head(12)
            cb = alt.Chart(coefs).mark_bar(cornerRadiusEnd=5).encode(
                x=alt.X("coef:Q", title=None), y=alt.Y("feature:N", sort=alt.EncodingSortField("coef", order="descending"), title=None),
                color=alt.condition("datum.coef > 0", alt.value(ORANGE), alt.value(BLUE)), tooltip=["feature:N", alt.Tooltip("coef:Q", format=".2f")])
            show(style(cb, 340))
            st.caption(tr("coef_note"))
    with r:
        with card():
            card_title("🔮 " + tr("predict"))
            pm = st.selectbox(tr("model"), list(M["fitted"]), key="pred_model")
            cat = st.selectbox(tr("category"), sorted(df["category"].unique()))
            sv = st.select_slider(tr("servings"), options=sorted(df["servings"].unique()), value=4)
            a, b2 = st.columns(2)
            cal = a.number_input(tr("calories"), 0.0, 5000.0, float(df["calories"].median()))
            ca = b2.number_input(tr("carbs"), 0.0, 600.0, float(df["carbohydrate"].median()))
            su = a.number_input(tr("sugar"), 0.0, 200.0, float(df["sugar"].median()))
            pr = b2.number_input(tr("protein"), 0.0, 400.0, float(df["protein"].median()))
            new = pd.DataFrame({"calories": [cal], "carbohydrate": [ca], "sugar": [su], "protein": [pr], "servings": [sv], "category": [cat]})
            pb = float(M["fitted"][pm].predict_proba(new)[0, 1])
            verdict, col = ((tr("verdict_hi"), ORANGE) if pb >= 0.6 else (tr("verdict_lo"), BLUE) if pb <= 0.4 else (tr("verdict_unc"), GREY))
            st.markdown(f"<div style='color:{GREY}'>{tr('proba')}</div><div class='bigproba'>{pb:.0%}</div>"
                        f"<div class='gauge'><div style='width:{pb*100:.0f}%;background:{col}'></div></div>"
                        f"<div style='margin-top:8px;font-weight:700;color:{col}'>{verdict}</div>", unsafe_allow_html=True)
            st.caption(tr("predict_note"))

# ============================================================================
# TAB 5 : Data quality and critical review
# ============================================================================
with tabs[4]:
    rep_df, n_out_rows = outlier_report(df_clean)
    nm = df_clean["nutrition_missing"]
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi(tr("q_rows"), f"{INFO['n_rows']:,}", f"{INFO['n_cols']} " + ("colonnes" if LANG == "fr" else "columns"))
    with k2:
        kpi(tr("q_dups"), f"{INFO['duplicates']}")
    with k3:
        kpi(tr("q_miss"), f"{int(nm.sum())}", f"{nm.mean():.1%}")
    with k4:
        kpi(tr("q_out"), f"{n_out_rows}", f"{n_out_rows / len(df_clean):.1%}")
    with k5:
        kpi(tr("q_after"), f"{len(df):,}", OUT_LABELS[out_method])
    st.write("")
    l, r = st.columns(2)
    with l:
        with card():
            card_title(tr("q_miss_t"))
            na = pd.DataFrame({"col": list(INFO["na_pct"]), "pct": list(INFO["na_pct"].values())})
            nb_ = alt.Chart(na).mark_bar(cornerRadiusEnd=5, color=BLUE).encode(
                x=alt.X("pct:Q", title=None), y=alt.Y("col:N", sort="-x", title=None),
                tooltip=["col:N", alt.Tooltip("pct:Q", format=".1f")])
            show(style(nb_ + nb_.mark_text(align="left", dx=4, color=INK).encode(text=alt.Text("pct:Q", format=".1f")), 300))
    with r:
        with card():
            card_title(tr("q_out_t"))
            tbl = rep_df.copy()
            tbl["var"] = tbl["var"].map(LABELS)
            tbl = tbl.drop(columns="skew")
            tbl.columns = [tr("q_var"), tr("q_fence"), tr("q_n15"), tr("q_n3"), tr("q_max")]
            st.dataframe(tbl.round(1), hide_index=True)
    with card():
        card_title(tr("q_ba"))
        vq = st.selectbox(tr("variable"), NUTRI_COLS, format_func=LABELS.get, key="q_var")
        ba = pd.concat([pd.DataFrame({"v": df_clean[vq], "stage": tr("q_before")}),
                        pd.DataFrame({"v": df[vq], "stage": tr("q_afterl")})])
        bx = alt.Chart(ba).mark_boxplot(extent="min-max", size=60).encode(
            x=alt.X("stage:N", title=None, sort=[tr("q_before"), tr("q_afterl")]), y=alt.Y("v:Q", title=LABELS[vq]),
            color=alt.Color("stage:N", scale=alt.Scale(range=[SKY, BLUE]), legend=None))
        show(style(bx, 300))
        st.markdown(tr("q_ba_note", m=OUT_LABELS[out_method]))
    st.write("")
    with card():
        card_title(tr("q_crit"))
        zero_cats = sorted(set(df_clean["category"]) - set(df_clean.loc[nm, "category"]))
        smallest = int(df_clean["category"].value_counts().min())
        crit = [tr("c1"), tr("c2", n=INFO["na_target"]),
                tr("c3", n=int(nm.sum()), p=nm.mean(), z=", ".join(zero_cats) if zero_cats else "—"),
                tr("c4", n=n_out_rows, s=rep_df["skew"].max()), tr("c5", m=smallest), tr("c6"),
                tr("c7", a=INFO["odd_servings"], b=INFO["n_cat_before"])]
        st.markdown("\n".join(f"- {c}" for c in crit))

# ============================================================================
# TAB 6 : Conclusions and recommendations (computed from the loaded data)
# ============================================================================
with tabs[5]:
    rates = df.groupby("category").agg(rate=("high_traffic", "mean"), n=("recipe", "count")).sort_values("rate", ascending=False)
    top3 = rates.head(3)
    base_rate = df["high_traffic"].mean()
    ct_c = pd.crosstab(df["category"], df["high_traffic"])
    chi2_c = chi2_contingency(ct_c)[0]
    v_c = float(np.sqrt(chi2_c / (ct_c.values.sum() * (min(ct_c.shape) - 1))))
    p_s = chi2_contingency(pd.crosstab(df["servings"], df["high_traffic"]))[1]
    mt_ = pd.DataFrame(M["metrics"]).set_index("model")
    bm = mt_["f1"].idxmax()
    st.markdown(f"### {tr('conc_t')}")
    cols = st.columns(3)
    for col, (cat_name, row) in zip(cols, top3.iterrows()):
        with col:
            kpi(cat_name, f"{row['rate']:.0%}", tr("conc_n", n=int(row["n"])), "up")
    st.write("")
    with card():
        card_title(tr("conc_top"))
        st.markdown("\n".join([
            "- " + tr("conc_1", c1=top3.index[0], c2=top3.index[1], c3=top3.index[2], r=top3["rate"].min(), o=base_rate),
            "- " + tr("conc_2", v=v_c, p=p_s),
            "- " + tr("conc_3", m=bm, f=mt_.loc[bm, "f1"], a=mt_.loc[bm, "auc"], pr=mt_.loc[bm, "precision"]),
            "- " + tr("conc_4")]))

# ============================================================================
# TAB 7 : Data
# ============================================================================
with tabs[6]:
    if f.empty:
        st.warning(tr("empty"))
    else:
        with card():
            card_title(f"{len(f)} {tr('data')}")
            table = f.drop(columns=["high_traffic", "nutrition_missing"]).rename(columns=LABELS | {"recipe": tr("recipe"), "category": tr("category"), "traffic": tr("traffic")})
            st.dataframe(table, hide_index=True)
            st.download_button(tr("download"), data=table.to_csv(index=False, sep=";").encode("utf-8"),
                               file_name="recipes_selection.csv", mime="text/csv")