"""
Streamlit app: Predicting high-traffic recipes
Run with:  streamlit run app.py
The file recipe_site_traffic_2212.csv (";" separator) must be in the same folder,
or be uploaded through the sidebar.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import chi2_contingency, mannwhitneyu

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)

st.set_page_config(page_title="Recipes: traffic prediction", page_icon="🍲", layout="wide")
sns.set_theme(style="whitegrid")
FOREST, TOMATO, SAGE = "#2D4A3E", "#E4572E", "#8FB8A0"

NUTRI_COLS = ["calories", "carbohydrate", "sugar", "protein"]
NUM_COLS = NUTRI_COLS + ["servings"]
CAT_COLS = ["category"]
DEFAULT_PATH = "recipe_site_traffic_2212.csv"


# ----------------------------------------------------------------------------
# 1. Loading and cleaning (same logic as the notebook)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data…")
def load_raw(path_or_file):
    return pd.read_csv(path_or_file, sep=";")


@st.cache_data(show_spinner="Cleaning data…")
def clean(raw: pd.DataFrame):
    df = raw.copy()
    info = {
        "n_rows": len(df),
        "duplicates": int(df.duplicated().sum()),
        "recipe_unique": bool(df["recipe"].is_unique),
        "n_categories_before": int(df["category"].nunique()),
        "odd_servings": int(df["servings"].astype(str).str.contains("snack").sum()),
    }
    # Target: NaN -> Low -> 0 / 1
    df["high_traffic"] = df["high_traffic"].fillna("Low").map({"High": 1, "Low": 0})
    # Categories
    df["category"] = df["category"].replace({"Chicken Breast": "Chicken"})
    # Servings
    df["servings"] = df["servings"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    # Missing nutrition values
    info["missing"] = int(df[NUTRI_COLS].isna().all(axis=1).sum())
    df["nutrition_missing"] = df[NUTRI_COLS].isna().any(axis=1).astype(int)
    for col in NUTRI_COLS:
        df[col] = df[col].fillna(df.groupby("category")[col].transform("median"))
    return df, info


# ----------------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------------
def build_models(params=None):
    p = params or {}
    prep_linear = ColumnTransformer([
        ("log_scale", Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), NUTRI_COLS),
        ("scale", StandardScaler(), ["servings"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    prep_tree = ColumnTransformer([
        ("pass", "passthrough", NUTRI_COLS + ["servings"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return {
        "Logistic Regression": Pipeline([
            ("prep", prep_linear),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=p.get("lr_C", 1))),
        ]),
        "Random Forest": Pipeline([
            ("prep", prep_tree),
            ("clf", RandomForestClassifier(random_state=42, n_jobs=-1,
                                           n_estimators=p.get("rf_n", 400),
                                           max_depth=p.get("rf_depth", 5),
                                           min_samples_leaf=p.get("rf_leaf", 1))),
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", prep_tree),
            ("clf", GradientBoostingClassifier(random_state=42,
                                               n_estimators=p.get("gb_n", 100),
                                               learning_rate=p.get("gb_lr", 0.05),
                                               max_depth=p.get("gb_depth", 2))),
        ]),
    }


PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
    "Random Forest": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 5, 10],
                      "clf__min_samples_leaf": [1, 3, 5]},
    "Gradient Boosting": {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                          "clf__max_depth": [2, 3, 4]},
}


@st.cache_resource(show_spinner="Training models…")
def train_all(df: pd.DataFrame, run_gridsearch: bool, test_size: float, seed: int):
    X = df[NUM_COLS + CAT_COLS]
    y = df["high_traffic"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = build_models()

    cv_rows, fitted, best_params = {}, {}, {}
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv,
                                scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
        cv_rows[name] = {m: scores[f"test_{m}"].mean() for m in
                         ["accuracy", "precision", "recall", "f1", "roc_auc"]}
        if run_gridsearch:
            gs = GridSearchCV(model, PARAM_GRIDS[name], cv=cv, scoring="f1", n_jobs=-1)
            gs.fit(X_train, y_train)
            fitted[name] = gs.best_estimator_
            best_params[name] = {**gs.best_params_, "f1_cv": round(gs.best_score_, 3)}
        else:
            fitted[name] = model.fit(X_train, y_train)
            best_params[name] = "notebook parameters (no GridSearch)"

    rows, curves, cms, reports = [], {}, {}, {}
    for name, model in fitted.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred),
                     "Recall": recall_score(y_test, pred),
                     "F1": f1_score(y_test, pred),
                     "ROC AUC": roc_auc_score(y_test, proba)})
        curves[name] = roc_curve(y_test, proba)
        cms[name] = confusion_matrix(y_test, pred)
        reports[name] = classification_report(y_test, pred)
    results = pd.DataFrame(rows).set_index("Model")
    cv_df = pd.DataFrame(cv_rows).T
    cv_df.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    return dict(fitted=fitted, results=results, cv=cv_df, curves=curves, cms=cms,
                reports=reports, best_params=best_params,
                shapes=(X_train.shape, X_test.shape), y_rates=(y_train.mean(), y_test.mean()))


# ----------------------------------------------------------------------------
# Sidebar: data source and navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🍲 Recipes")
page = st.sidebar.radio("Navigation", [
    "1. Context", "2. Data and cleaning", "3. Exploratory analysis",
    "4. Modeling", "5. Prediction", "6. Conclusion"])

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload the CSV (\";\" separator)", type="csv")
try:
    raw = load_raw(uploaded if uploaded else DEFAULT_PATH)
except FileNotFoundError:
    st.error(f"File `{DEFAULT_PATH}` not found. Put it next to `app.py` "
             "or upload it in the sidebar.")
    st.stop()
df, info = clean(raw)


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def interp(text):
    """Interpretation block shown under each chart / table."""
    st.info("💡 **Interpretation.** " + text)


def strength_v(v):
    return "weak" if v < 0.1 else "moderate" if v < 0.3 else "strong"


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
if page == "1. Context":
    st.title("Predicting high-traffic recipes")
    st.markdown(
        "The recipe website wants to sell subscriptions. The more visits a recipe attracts, "
        "the higher the chance of a sale. **Objective: predict which recipes "
        "will generate high traffic**, in order to choose the ones to feature.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recipes", len(df))
    c2.metric("Variables", raw.shape[1])
    c3.metric("Categories", df["category"].nunique())
    c4.metric("High traffic", f"{df['high_traffic'].mean():.1%}")
    st.subheader("Data dictionary")
    st.table(pd.DataFrame({
        "Variable": ["recipe", "calories", "carbohydrate", "sugar", "protein",
                     "category", "servings", "high_traffic"],
        "Description": ["Unique recipe identifier", "Number of calories",
                        "Carbohydrates (g)", "Sugar (g)", "Protein (g)",
                        "Recipe type (10 groups)", "Number of servings",
                        "\"High\" if traffic was high (target)"]}))

elif page == "2. Data and cleaning":
    st.title("Data and cleaning")
    st.subheader("Raw data preview")
    st.dataframe(raw.head(10))
    c1, c2, c3 = st.columns(3)
    c1.metric("Duplicates", info["duplicates"])
    c2.metric("Unique identifier", "Yes" if info["recipe_unique"] else "No")
    c3.metric("Rows without nutrition", f"{info['missing']} ({info['missing']/info['n_rows']:.1%})")

    st.subheader("Cleaning steps")
    st.markdown(f"""
1. **Target**: empty values of `high_traffic` = "Low", then coded 0 / 1.
2. **Categories**: {info['n_categories_before']} values in the data; "Chicken Breast" merged with "Chicken" → **10** categories.
3. **Servings**: {info['odd_servings']} values like "4 as a snack" converted to the integer; column cast to `int`.
4. **Missing values**: the 4 nutrition variables are missing together on {info['missing']} recipes; imputed with the **category median**.
""")
    st.subheader("Distribution of servings after cleaning")
    vc = df["servings"].value_counts().sort_index()
    st.bar_chart(vc)
    interp(f"The most frequent value is **{vc.idxmax()} servings** ({vc.max()} recipes, "
           f"{vc.max()/len(df):.0%} of the total). The \"as a snack\" values were absorbed: only "
           f"{len(vc)} values remain ({', '.join(map(str, vc.index))}), so the variable can be used as a number.")
    with st.expander("Cleaned data"):
        st.dataframe(df.head(20))
        st.write(df.drop(columns="recipe").describe().T)
        asym = (df[NUTRI_COLS].mean() > df[NUTRI_COLS].median()).all()
        interp("For every nutrition variable, the **mean is above the median** and the maximum is far "
               "from the 75th percentile: a few extreme recipes pull the means upward "
               "(right-skewed distributions)." if asym else
               "Means and medians are close: little visible skewness.")

elif page == "3. Exploratory analysis":
    st.title("Exploratory analysis")
    t1, t2, t3, t4, t5, t6 = st.tabs(["Target", "Categories", "Nutrition",
                                      "Servings", "Missing values", "Summary"])

    with t1:
        rate = df["high_traffic"].mean()
        c1, c2 = st.columns(2)
        c1.metric("High traffic", f"{rate:.1%}")
        c2.metric("Low traffic", f"{1-rate:.1%}")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(data=df, x="high_traffic", hue="high_traffic", palette=[SAGE, TOMATO], legend=False, ax=ax)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Low (0)", "High (1)"]); ax.set_xlabel("")
        show(fig)
        base = max(rate, 1 - rate)
        interp(f"**{rate:.1%}** of recipes generate high traffic: the classes are moderately imbalanced. "
               f"A model that always predicted \"high traffic\" would already be right **{base:.1%}** of the time: this is the "
               "minimum bar to beat. That is why we track **F1** and **ROC AUC** rather than accuracy alone.")

    with t2:
        order = df["category"].value_counts().index
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, y="category", order=order, color=FOREST, ax=ax)
            ax.set_title("Recipes per category"); ax.set_ylabel("")
            show(fig)
            vcat = df["category"].value_counts()
            interp(f"Categories range from **{vcat.min()}** ({vcat.idxmin()}) to **{vcat.max()}** recipes ({vcat.idxmax()}). "
                   "Each category is represented well enough for its high-traffic rate to be reliable.")
        with c2:
            rates = df.groupby("category")["high_traffic"].mean().sort_values(ascending=False) * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=rates.values, y=rates.index, color=FOREST, ax=ax)
            ax.set_title("High-traffic rate (%)"); ax.set_ylabel("")
            for i, v in enumerate(rates.values):
                ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=9)
            show(fig)
            top3 = ", ".join(f"{c} ({r:.0f}%)" for c, r in rates.head(3).items())
            bot = rates.index[-1]
            interp(f"The most successful recipes are **{top3}**. At the other end, **{bot}** reaches only {rates.iloc[-1]:.0f}%. "
                   f"The gap between extremes is **{rates.iloc[0]-rates.iloc[-1]:.0f} points**: "
                   "knowing the category already tells you a lot about traffic.")
        ct = pd.crosstab(df["category"], df["high_traffic"])
        chi2, p, dof, _ = chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        c1, c2, c3 = st.columns(3)
        c1.metric("Chi²", f"{chi2:.1f}"); c2.metric("p-value", f"{p:.2e}"); c3.metric("Cramér's V", f"{v:.2f}")
        interp(f"The chi² test gives p = {p:.2e}: category and traffic are **not independent** (association "
               f"{'significant' if p < 0.05 else 'not significant'}). Cramér's V is **{v:.2f}**: a **{strength_v(v)}** link "
               "(guide: < 0.1 weak, 0.1 to 0.3 moderate, > 0.3 strong). It is the most discriminating variable in the dataset.")

    with t3:
        var = st.selectbox("Nutrition variable", NUTRI_COLS)
        c1, c2 = st.columns(2)
        with c1:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            sns.histplot(df[var], bins=30, color=FOREST, ax=axes[0]); axes[0].set_title(var)
            sns.histplot(np.log1p(df[var]), bins=30, color=SAGE, ax=axes[1]); axes[1].set_title(f"log1p({var})")
            plt.tight_layout(); show(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.boxplot(data=df, x="high_traffic", y=var, hue="high_traffic", palette=[SAGE, TOMATO], legend=False, ax=ax)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Low", "High"]); ax.set_xlabel("")
            show(fig)
        mh = df.loc[df.high_traffic == 1, var].median(); ml = df.loc[df.high_traffic == 0, var].median()
        interp(f"**Histograms**: skewness of {df[var].skew():.2f} (mean {df[var].mean():.1f} vs median "
               f"{df[var].median():.1f}); after log1p the distribution is much more symmetric, which suits "
               f"logistic regression better. **Boxplots**: median of {mh:.1f} for high-traffic recipes vs "
               f"{ml:.1f} for the others; the boxes **overlap heavily**, so this variable alone separates the two groups poorly.")
        st.subheader("Skewness and Mann-Whitney test")
        rows = []
        for col in NUTRI_COLS:
            u, p = mannwhitneyu(df.loc[df.high_traffic == 1, col], df.loc[df.high_traffic == 0, col])
            rows.append({"Variable": col, "Skewness": df[col].skew(), "U statistic": u, "p-value": p})
        st.dataframe(pd.DataFrame(rows).set_index("Variable").style.format(
            {"Skewness": "{:.2f}", "U statistic": "{:,.0f}", "p-value": "{:.4f}"}))
        tab = pd.DataFrame(rows).set_index("Variable")
        clear = tab.index[tab["p-value"] < 0.01].tolist(); borderline = tab.index[(tab["p-value"] >= 0.01) & (tab["p-value"] < 0.05)].tolist()
        interp(f"The Mann-Whitney test compares the two groups without assuming normality. "
               f"**Clear** link (p < 0.01): {', '.join(clear) if clear else 'no variable'}. "
               f"**Borderline** link (0.01 ≤ p < 0.05): {', '.join(borderline) if borderline else 'no variable'}, "
               "to be read cautiously because four tests are run without correction. "
               "The skewness (> 2 everywhere) confirms the value of the log transform.")
        corr = df[NUM_COLS + ["high_traffic"]].corr()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlations (numeric variables)")
        show(fig)
        ct_cible = corr["high_traffic"].drop("high_traffic")
        cf = corr.loc[NUM_COLS, NUM_COLS]
        pairs = cf.where(np.triu(np.ones(cf.shape), 1).astype(bool)).stack()
        best_pair = pairs.abs().idxmax()
        interp(f"No numeric variable is strongly correlated with the target (strongest: **{ct_cible.abs().idxmax()}**, "
               f"r = {ct_cible[ct_cible.abs().idxmax()]:.2f}), which confirms that nutrition explains little of the traffic. "
               f"Among the explanatory variables, the strongest correlation is **{best_pair[0]} / {best_pair[1]}** "
               f"(r = {pairs[best_pair]:.2f}): no worrying redundancy for logistic regression.")

    with t4:
        rate_s = df.groupby("servings")["high_traffic"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x=rate_s.index, y=rate_s.values, color=FOREST, ax=ax)
        ax.set_ylabel("High-traffic rate (%)"); ax.set_xlabel("Servings")
        for i, v in enumerate(rate_s.values):
            ax.text(i, v + 1, f"{v:.1f}", ha="center")
        show(fig)
        chi2, p, _, _ = chi2_contingency(pd.crosstab(df["servings"], df["high_traffic"]))
        st.metric("Chi² servings vs target: p-value", f"{p:.3f}")
        interp(f"Rates vary from {rate_s.min():.1f}% to {rate_s.max():.1f}% depending on the number of servings, only "
               f"{rate_s.max()-rate_s.min():.1f} points. The chi² test (p = {p:.2f}) "
               f"{'does not allow us to conclude there is a link' if p >= 0.05 else 'indicates a significant link'} between servings and traffic. "
               "This variable is therefore not very informative.")

    with t5:
        grp = df.groupby("nutrition_missing")["high_traffic"].agg(["mean", "count"])
        grp.index = ["Nutrition provided", "Nutrition missing"]
        grp["mean"] = (grp["mean"] * 100).round(1)
        st.dataframe(grp.rename(columns={"mean": "High-traffic rate (%)", "count": "Count"}))
        st.markdown("**By category** (missing values are concentrated in high-traffic categories):")
        by_cat = df.groupby(["category", "nutrition_missing"])["high_traffic"].agg(["mean", "count"]).round(2)
        st.dataframe(by_cat)
        r_m = df.loc[df.nutrition_missing == 1, "high_traffic"].mean(); r_o = df.loc[df.nutrition_missing == 0, "high_traffic"].mean()
        n_zero = (pd.crosstab(df["category"], df["nutrition_missing"]).get(1, pd.Series(dtype=int)) == 0).sum()
        interp(f"Recipes without nutrition data have a high-traffic rate of **{r_m:.0%}** vs **{r_o:.0%}**. But the by-category table shows "
               f"that {n_zero} categories have no missing values (the least popular ones) and that the direction of the gap changes from one category to another, "
               "with small counts: the overall gap comes from the **category mix**, not from the missingness itself. "
               "Tested in cross-validation, the `nutrition_missing` variable did not improve the AUC; it was dropped.")

    with t6:
        st.markdown("""
- **Target**: ~61% high traffic, moderately imbalanced classes → F1 and ROC AUC.
- **Category**: dominant variable (Cramér's V ≈ 0.58), one-hot encoded.
- **Nutrition**: skewed, weak effect → `log1p` for the linear model.
- **Servings** and **missing values**: no effect of their own.
""")

elif page == "4. Modeling":
    st.title("Modeling")
    st.markdown("Three models are compared: logistic regression, Random Forest and Gradient Boosting. "
                "All preprocessing lives inside scikit-learn *pipelines* (no data leakage).")
    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Test set share", 0.1, 0.4, 0.2, 0.05)
    seed = c2.number_input("Random seed", value=42, step=1)
    use_gs = c3.checkbox("Re-run GridSearch (slower)", value=False,
                         help="Otherwise, the best hyperparameters found in the notebook are used.")

    if st.button("Start training", type="primary") or "trained" in st.session_state:
        st.session_state["trained"] = True
        out = train_all(df, use_gs, test_size, int(seed))
        (n_tr, _), (n_te, _) = out["shapes"]
        st.caption(f"Training: {n_tr} recipes · Test: {n_te} recipes · "
                   f"high-traffic share: {out['y_rates'][0]:.3f} / {out['y_rates'][1]:.3f}")

        st.subheader("Cross-validation (5 folds, training set)")
        st.dataframe(out["cv"].style.format("{:.3f}").highlight_max(color="#C9D6CE"))
        bcv = out["cv"]["F1"].idxmax()
        interp(f"Cross-validation evaluates each model on 5 splits of the training set: it is a more stable estimate than a single test. "
               f"Best F1: **{bcv}** ({out['cv'].loc[bcv, 'F1']:.3f}). The gap between models is "
               f"{out['cv']['F1'].max() - out['cv']['F1'].min():.3f} F1 points: very small.")

        st.subheader("Evaluation on the test set")
        res = out["results"]
        st.dataframe(res.style.format("{:.3f}").highlight_max(color="#C9D6CE"))
        best_name = res["F1"].idxmax()
        gap = (res["F1"] - out["cv"]["F1"]).abs().max()
        interp(f"Best test F1: **{best_name}** ({res.loc[best_name, 'F1']:.3f}), best ROC AUC: **{res['ROC AUC'].idxmax()}** "
               f"({res['ROC AUC'].max():.3f}). Test and cross-validation scores are consistent (max F1 gap: {gap:.3f}): "
               "no sign of overfitting. The three models are very close; on ~190 test recipes, a 0.01 gap is not significant. "
               "We therefore favor the simplest and most interpretable model (logistic regression).")

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4.5))
            for name, (fpr, tpr, _) in out["curves"].items():
                ax.plot(fpr, tpr, label=f"{name} (AUC {res.loc[name, 'ROC AUC']:.3f})")
            ax.plot([0, 1], [0, 1], "k--", label="Chance")
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.set_title("ROC curves"); ax.legend(fontsize=8)
            show(fig)
            interp(f"The closer a curve is to the top-left corner, the better the model; the diagonal corresponds to chance (AUC = 0.5). "
                   f"Here the AUCs range from {res['ROC AUC'].min():.2f} to {res['ROC AUC'].max():.2f}: a high-traffic recipe gets a higher score "
                   f"than a low-traffic recipe in about {res['ROC AUC'].max():.0%} of cases. The curves almost overlap.")
        with c2:
            choice = st.selectbox("Confusion matrix for:", list(out["cms"]),
                                  index=list(out["cms"]).index(best_name))
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            sns.heatmap(out["cms"][choice], annot=True, fmt="d", cmap="Greens", cbar=False,
                        xticklabels=["Low", "High"], yticklabels=["Low", "High"], ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            show(fig)
            st.code(out["reports"][choice])
            tn, fp, fn, tp = out["cms"][choice].ravel()
            interp(f"Out of {tn+fp+fn+tp} test recipes: **{tp}** high-traffic recipes correctly spotted, **{tn}** low-traffic recipes correctly set aside. "
                   f"**{fp} false positives** (promoted recipes that will not get high traffic) and **{fn} false negatives** (good recipes missed). "
                   f"Precision = {tp/(tp+fp):.2f}: when the model predicts high traffic, it is right {tp/(tp+fp):.0%} of the time. "
                   f"Recall = {tp/(tp+fn):.2f}: it finds {tp/(tp+fn):.0%} of the high-traffic recipes.")

        st.subheader("Feature importance")
        model_imp = st.selectbox("Model", list(out["fitted"]), key="imp_model")
        pipe = out["fitted"][model_imp]
        names = pipe.named_steps["prep"].get_feature_names_out()
        clf = pipe.named_steps["clf"]
        imp = pd.Series(clf.feature_importances_ if hasattr(clf, "feature_importances_") else clf.coef_[0],
                        index=names)
        imp = imp.reindex(imp.abs().sort_values(ascending=False).index).head(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x=imp.values, y=[n.split("__")[-1] for n in imp.index], color=FOREST, ax=ax)
        ax.set_title("Coefficients" if not hasattr(clf, "feature_importances_") else "Importances")
        show(fig)
        top = [n.split("__")[-1] for n in imp.index[:3]]
        if hasattr(clf, "feature_importances_"):
            interp(f"Importance = share of impurity reduction contributed by each variable. Most influential: **{', '.join(top)}**. "
                   "The \"category_…\" columns are the one-hot encoded categories; their cumulative weight dominates, which matches the exploratory analysis.")
        else:
            pos = [n.split("__")[-1] for n, v in imp.items() if v > 0][:2]; neg = [n.split("__")[-1] for n, v in imp.items() if v < 0][:2]
            interp(f"Coefficient > 0: the variable **increases** the probability of high traffic; < 0: it **decreases** it (other variables held fixed). "
                   f"Strongest positive effects: {', '.join(pos) or '-'}; negative: {', '.join(neg) or '-'}. "
                   "Nutrition variables are standardized after log1p: their coefficients are comparable with each other. "
                   "Categories clearly dominate.")
        with st.expander("Selected hyperparameters"):
            st.json(out["best_params"])
    else:
        st.info("Click \"Start training\" to train and compare the models.")

elif page == "5. Prediction":
    st.title("Predict the traffic of a new recipe")
    out = train_all(df, False, 0.2, 42)
    model_name = st.selectbox("Model", list(out["fitted"]), index=0)
    c1, c2 = st.columns(2)
    with c1:
        category = st.selectbox("Category", sorted(df["category"].unique()))
        servings = st.select_slider("Servings", options=sorted(df["servings"].unique()), value=4)
    with c2:
        calories = st.number_input("Calories", 0.0, 5000.0, 300.0)
        carbs = st.number_input("Carbohydrates (g)", 0.0, 600.0, 20.0)
        sugar = st.number_input("Sugar (g)", 0.0, 200.0, 5.0)
        protein = st.number_input("Protein (g)", 0.0, 400.0, 15.0)

    new = pd.DataFrame({"calories": [calories], "carbohydrate": [carbs], "sugar": [sugar],
                        "protein": [protein], "servings": [servings], "category": [category]})
    proba = out["fitted"][model_name].predict_proba(new)[0, 1]
    st.metric("Probability of high traffic", f"{proba:.1%}")
    st.progress(float(proba))
    if proba >= 0.5:
        st.success("Prediction: **high traffic**")
    else:
        st.error("Prediction: **low traffic**")
    level = "very high" if proba >= 0.8 else "high" if proba >= 0.6 else "uncertain" if proba >= 0.4 else "low" if proba >= 0.2 else "very low"
    interp(f"The model estimates the probability of high traffic at **{proba:.0%}**: **{level}** confidence. "
           "A probability close to 50% means the model is hesitating. To test sensitivity, change the category: "
           "it moves the prediction far more than the nutrition values.")
    st.caption("Decision threshold at 0.5. The probability reflects the history of this dataset, not a guarantee.")

elif page == "6. Conclusion":
    st.title("Conclusion")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Key takeaways")
        st.markdown("""
- **Category** is the dominant driver of traffic.
- Nutrition and servings add little.
- Logistic regression: F1 ≈ 0.81 and ROC AUC ≈ 0.87 on the test set.
- The three models are equivalent: the simplest is selected.
""")
    with c2:
        st.subheader("Limitations and next steps")
        st.markdown("""
- Small sample: 947 recipes.
- No data on ingredients or preparation time.
- Imputation computed before the train / test split: should move into the pipeline.
- Adjust the threshold to the cost of a wrongly promoted recipe.
""")