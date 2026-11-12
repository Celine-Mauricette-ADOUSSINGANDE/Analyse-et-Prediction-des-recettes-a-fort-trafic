"""
Application Streamlit : Prédire les recettes à fort trafic
Lancer avec :  streamlit run app.py
Le fichier recipe_site_traffic_2212.csv (séparateur « ; ») doit être dans le même dossier,
ou être chargé via la barre latérale.
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

st.set_page_config(page_title="Recettes : prédiction du trafic", page_icon="🍲", layout="wide")
sns.set_theme(style="whitegrid")
FOREST, TOMATO, SAGE = "#2D4A3E", "#E4572E", "#8FB8A0"

NUTRI_COLS = ["calories", "carbohydrate", "sugar", "protein"]
NUM_COLS = NUTRI_COLS + ["servings"]
CAT_COLS = ["category"]
DEFAULT_PATH = "recipe_site_traffic_2212.csv"


# ----------------------------------------------------------------------------
# 1. Chargement et nettoyage (même logique que le notebook)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement des données…")
def load_raw(path_or_file):
    return pd.read_csv(path_or_file, sep=";")


@st.cache_data(show_spinner="Nettoyage des données…")
def clean(raw: pd.DataFrame):
    df = raw.copy()
    info = {
        "n_lignes": len(df),
        "doublons": int(df.duplicated().sum()),
        "recipe_unique": bool(df["recipe"].is_unique),
        "n_categories_avant": int(df["category"].nunique()),
        "servings_atypiques": int(df["servings"].astype(str).str.contains("snack").sum()),
    }
    # Cible : NaN -> Low -> 0 / 1
    df["high_traffic"] = df["high_traffic"].fillna("Low").map({"High": 1, "Low": 0})
    # Catégories
    df["category"] = df["category"].replace({"Chicken Breast": "Chicken"})
    # Portions
    df["servings"] = df["servings"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    # Valeurs manquantes nutritionnelles
    info["manquants"] = int(df[NUTRI_COLS].isna().all(axis=1).sum())
    df["nutrition_missing"] = df[NUTRI_COLS].isna().any(axis=1).astype(int)
    for col in NUTRI_COLS:
        df[col] = df[col].fillna(df.groupby("category")[col].transform("median"))
    return df, info


# ----------------------------------------------------------------------------
# 2. Modèles
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
        "Régression logistique": Pipeline([
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
    "Régression logistique": {"clf__C": [0.01, 0.1, 1, 10]},
    "Random Forest": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 5, 10],
                      "clf__min_samples_leaf": [1, 3, 5]},
    "Gradient Boosting": {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                          "clf__max_depth": [2, 3, 4]},
}


@st.cache_resource(show_spinner="Entraînement des modèles…")
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
            best_params[name] = "paramètres du notebook (sans GridSearch)"

    rows, curves, cms, reports = [], {}, {}, {}
    for name, model in fitted.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({"Modèle": name,
                     "Accuracy": accuracy_score(y_test, pred),
                     "Précision": precision_score(y_test, pred),
                     "Rappel": recall_score(y_test, pred),
                     "F1": f1_score(y_test, pred),
                     "ROC AUC": roc_auc_score(y_test, proba)})
        curves[name] = roc_curve(y_test, proba)
        cms[name] = confusion_matrix(y_test, pred)
        reports[name] = classification_report(y_test, pred)
    results = pd.DataFrame(rows).set_index("Modèle")
    cv_df = pd.DataFrame(cv_rows).T
    cv_df.columns = ["Accuracy", "Précision", "Rappel", "F1", "ROC AUC"]
    return dict(fitted=fitted, results=results, cv=cv_df, curves=curves, cms=cms,
                reports=reports, best_params=best_params,
                shapes=(X_train.shape, X_test.shape), y_rates=(y_train.mean(), y_test.mean()))


# ----------------------------------------------------------------------------
# Barre latérale : source de données et navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🍲 Recettes")
page = st.sidebar.radio("Navigation", [
    "1. Contexte", "2. Données et nettoyage", "3. Analyse exploratoire",
    "4. Modélisation", "5. Prédiction", "6. Conclusion"])

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Charger le CSV (séparateur « ; »)", type="csv")
try:
    raw = load_raw(uploaded if uploaded else DEFAULT_PATH)
except FileNotFoundError:
    st.error(f"Fichier `{DEFAULT_PATH}` introuvable. Place-le à côté de `app.py` "
             "ou charge-le dans la barre latérale.")
    st.stop()
df, info = clean(raw)


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def interp(texte):
    """Bloc d'interprétation affiché sous chaque graphique / tableau."""
    st.info("💡 **Interprétation.** " + texte)


def force_v(v):
    return "faible" if v < 0.1 else "modéré" if v < 0.3 else "fort"


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
if page == "1. Contexte":
    st.title("Prédire les recettes à fort trafic")
    st.markdown(
        "Le site de recettes veut vendre des abonnements. Plus une recette attire de visites, "
        "plus la probabilité d'une vente augmente. **Objectif : prédire quelles recettes "
        "généreront un fort trafic**, pour choisir celles à mettre en avant.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recettes", len(df))
    c2.metric("Variables", raw.shape[1])
    c3.metric("Catégories", df["category"].nunique())
    c4.metric("Fort trafic", f"{df['high_traffic'].mean():.1%}")
    st.subheader("Dictionnaire des variables")
    st.table(pd.DataFrame({
        "Variable": ["recipe", "calories", "carbohydrate", "sugar", "protein",
                     "category", "servings", "high_traffic"],
        "Description": ["Identifiant unique de la recette", "Nombre de calories",
                        "Glucides (g)", "Sucre (g)", "Protéines (g)",
                        "Type de recette (10 groupes)", "Nombre de portions",
                        "« High » si le trafic était élevé (cible)"]}))

elif page == "2. Données et nettoyage":
    st.title("Données et nettoyage")
    st.subheader("Aperçu des données brutes")
    st.dataframe(raw.head(10))
    c1, c2, c3 = st.columns(3)
    c1.metric("Doublons", info["doublons"])
    c2.metric("Identifiant unique", "Oui" if info["recipe_unique"] else "Non")
    c3.metric("Lignes sans nutrition", f"{info['manquants']} ({info['manquants']/info['n_lignes']:.1%})")

    st.subheader("Étapes de nettoyage")
    st.markdown(f"""
1. **Cible** : valeurs vides de `high_traffic` = « Low », puis codage 0 / 1.
2. **Catégories** : {info['n_categories_avant']} valeurs dans les données ; « Chicken Breast » fusionné avec « Chicken » → **10** catégories.
3. **Portions** : {info['servings_atypiques']} valeurs du type « 4 as a snack » ramenées à l'entier ; colonne convertie en `int`.
4. **Valeurs manquantes** : les 4 variables nutritionnelles manquent ensemble sur {info['manquants']} recettes ; imputation par la **médiane de la catégorie**.
""")
    st.subheader("Répartition des portions après nettoyage")
    vc = df["servings"].value_counts().sort_index()
    st.bar_chart(vc)
    interp(f"La modalité la plus fréquente est **{vc.idxmax()} portions** ({vc.max()} recettes, "
           f"{vc.max()/len(df):.0%} du total). Les valeurs « as a snack » ont été absorbées : il ne reste que "
           f"{len(vc)} modalités ({', '.join(map(str, vc.index))}), ce qui rend la variable exploitable comme nombre.")
    with st.expander("Données nettoyées"):
        st.dataframe(df.head(20))
        st.write(df.drop(columns="recipe").describe().T)
        asym = (df[NUTRI_COLS].mean() > df[NUTRI_COLS].median()).all()
        interp("Pour chaque variable nutritionnelle, la **moyenne est supérieure à la médiane** et le maximum est très "
               "éloigné du 75e percentile : quelques recettes extrêmes tirent les moyennes vers le haut "
               "(distributions asymétriques à droite)." if asym else
               "Les moyennes et médianes sont proches : peu d'asymétrie visible.")

elif page == "3. Analyse exploratoire":
    st.title("Analyse exploratoire")
    t1, t2, t3, t4, t5, t6 = st.tabs(["Cible", "Catégories", "Nutrition",
                                      "Portions", "Valeurs manquantes", "Synthèse"])

    with t1:
        rate = df["high_traffic"].mean()
        c1, c2 = st.columns(2)
        c1.metric("Fort trafic", f"{rate:.1%}")
        c2.metric("Faible trafic", f"{1-rate:.1%}")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(data=df, x="high_traffic", hue="high_traffic", palette=[SAGE, TOMATO], legend=False, ax=ax)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Faible (0)", "Fort (1)"]); ax.set_xlabel("")
        show(fig)
        base = max(rate, 1 - rate)
        interp(f"**{rate:.1%}** des recettes génèrent un fort trafic : les classes sont modérément déséquilibrées. "
               f"Un modèle qui prédirait toujours « fort trafic » aurait déjà **{base:.1%}** de bonnes réponses : c'est le "
               "seuil minimal à battre. C'est pourquoi on suit le **F1** et le **ROC AUC** plutôt que la seule accuracy.")

    with t2:
        order = df["category"].value_counts().index
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, y="category", order=order, color=FOREST, ax=ax)
            ax.set_title("Recettes par catégorie"); ax.set_ylabel("")
            show(fig)
            vcat = df["category"].value_counts()
            interp(f"Les catégories comptent de **{vcat.min()}** ({vcat.idxmin()}) à **{vcat.max()}** recettes ({vcat.idxmax()}). "
                   "Chaque catégorie est assez représentée pour que son taux de fort trafic soit fiable.")
        with c2:
            rates = df.groupby("category")["high_traffic"].mean().sort_values(ascending=False) * 100
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=rates.values, y=rates.index, color=FOREST, ax=ax)
            ax.set_title("Taux de fort trafic (%)"); ax.set_ylabel("")
            for i, v in enumerate(rates.values):
                ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=9)
            show(fig)
            top3 = ", ".join(f"{c} ({r:.0f} %)" for c, r in rates.head(3).items())
            bot = rates.index[-1]
            interp(f"Les recettes les plus à succès sont **{top3}**. À l'opposé, **{bot}** n'atteint que {rates.iloc[-1]:.0f} %. "
                   f"L'écart entre extrêmes est de **{rates.iloc[0]-rates.iloc[-1]:.0f} points** : "
                   "connaître la catégorie renseigne déjà beaucoup sur le trafic.")
        ct = pd.crosstab(df["category"], df["high_traffic"])
        chi2, p, dof, _ = chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        c1, c2, c3 = st.columns(3)
        c1.metric("Chi²", f"{chi2:.1f}"); c2.metric("p-value", f"{p:.2e}"); c3.metric("V de Cramér", f"{v:.2f}")
        interp(f"Le test du chi² donne p = {p:.2e} : la catégorie et le trafic ne sont **pas indépendants** (association "
               f"{'significative' if p < 0.05 else 'non significative'}). Le V de Cramér vaut **{v:.2f}** : lien **{force_v(v)}** "
               "(repères : < 0,1 faible, 0,1 à 0,3 modéré, > 0,3 fort). C'est la variable la plus discriminante du jeu de données.")

    with t3:
        var = st.selectbox("Variable nutritionnelle", NUTRI_COLS)
        c1, c2 = st.columns(2)
        with c1:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            sns.histplot(df[var], bins=30, color=FOREST, ax=axes[0]); axes[0].set_title(var)
            sns.histplot(np.log1p(df[var]), bins=30, color=SAGE, ax=axes[1]); axes[1].set_title(f"log1p({var})")
            plt.tight_layout(); show(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.boxplot(data=df, x="high_traffic", y=var, hue="high_traffic", palette=[SAGE, TOMATO], legend=False, ax=ax)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Faible", "Fort"]); ax.set_xlabel("")
            show(fig)
        mh = df.loc[df.high_traffic == 1, var].median(); ml = df.loc[df.high_traffic == 0, var].median()
        interp(f"**Histogrammes** : asymétrie de {df[var].skew():.2f} (moyenne {df[var].mean():.1f} contre médiane "
               f"{df[var].median():.1f}) ; après log1p la distribution est bien plus symétrique, ce qui convient mieux "
               f"à la régression logistique. **Boxplots** : médiane de {mh:.1f} pour les recettes à fort trafic contre "
               f"{ml:.1f} pour les autres ; les boîtes se **chevauchent largement**, donc cette variable seule sépare mal les deux groupes.")
        st.subheader("Asymétrie et test de Mann-Whitney")
        rows = []
        for col in NUTRI_COLS:
            u, p = mannwhitneyu(df.loc[df.high_traffic == 1, col], df.loc[df.high_traffic == 0, col])
            rows.append({"Variable": col, "Asymétrie": df[col].skew(), "Statistique U": u, "p-value": p})
        st.dataframe(pd.DataFrame(rows).set_index("Variable").style.format(
            {"Asymétrie": "{:.2f}", "Statistique U": "{:,.0f}", "p-value": "{:.4f}"}))
        tab = pd.DataFrame(rows).set_index("Variable")
        nets = tab.index[tab["p-value"] < 0.01].tolist(); limite = tab.index[(tab["p-value"] >= 0.01) & (tab["p-value"] < 0.05)].tolist()
        interp(f"Le test de Mann-Whitney compare les deux groupes sans supposer de loi normale. "
               f"Lien **net** (p < 0,01) : {', '.join(nets) if nets else 'aucune variable'}. "
               f"Lien **à la limite** (0,01 ≤ p < 0,05) : {', '.join(limite) if limite else 'aucune variable'}, "
               "à interpréter prudemment car quatre tests sont réalisés sans correction. "
               "L'asymétrie (> 2 partout) confirme l'intérêt du log.")
        corr = df[NUM_COLS + ["high_traffic"]].corr()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Corrélations (variables numériques)")
        show(fig)
        ct_cible = corr["high_traffic"].drop("high_traffic")
        cf = corr.loc[NUM_COLS, NUM_COLS]
        pairs = cf.where(np.triu(np.ones(cf.shape), 1).astype(bool)).stack()
        best_pair = pairs.abs().idxmax()
        interp(f"Aucune variable numérique n'est fortement corrélée à la cible (la plus forte : **{ct_cible.abs().idxmax()}**, "
               f"r = {ct_cible[ct_cible.abs().idxmax()]:.2f}), ce qui confirme que la nutrition explique peu le trafic. "
               f"Entre variables explicatives, la corrélation la plus forte est **{best_pair[0]} / {best_pair[1]}** "
               f"(r = {pairs[best_pair]:.2f}) : pas de redondance inquiétante pour la régression logistique.")

    with t4:
        rate_s = df.groupby("servings")["high_traffic"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x=rate_s.index, y=rate_s.values, color=FOREST, ax=ax)
        ax.set_ylabel("Taux de fort trafic (%)"); ax.set_xlabel("Portions")
        for i, v in enumerate(rate_s.values):
            ax.text(i, v + 1, f"{v:.1f}", ha="center")
        show(fig)
        chi2, p, _, _ = chi2_contingency(pd.crosstab(df["servings"], df["high_traffic"]))
        st.metric("Chi² portions vs cible : p-value", f"{p:.3f}")
        interp(f"Les taux varient de {rate_s.min():.1f} % à {rate_s.max():.1f} % selon le nombre de portions, soit seulement "
               f"{rate_s.max()-rate_s.min():.1f} points. Le test du chi² (p = {p:.2f}) "
               f"{'ne permet pas de conclure à un lien' if p >= 0.05 else 'indique un lien significatif'} entre portions et trafic. "
               "Cette variable est donc peu informative.")

    with t5:
        grp = df.groupby("nutrition_missing")["high_traffic"].agg(["mean", "count"])
        grp.index = ["Nutrition renseignée", "Nutrition manquante"]
        grp["mean"] = (grp["mean"] * 100).round(1)
        st.dataframe(grp.rename(columns={"mean": "Taux fort trafic (%)", "count": "Effectif"}))
        st.markdown("**Par catégorie** (les manquants sont concentrés dans des catégories à fort trafic) :")
        by_cat = df.groupby(["category", "nutrition_missing"])["high_traffic"].agg(["mean", "count"]).round(2)
        st.dataframe(by_cat)
        r_m = df.loc[df.nutrition_missing == 1, "high_traffic"].mean(); r_o = df.loc[df.nutrition_missing == 0, "high_traffic"].mean()
        n_zero = (pd.crosstab(df["category"], df["nutrition_missing"]).get(1, pd.Series(dtype=int)) == 0).sum()
        interp(f"Les recettes sans nutrition ont un taux de fort trafic de **{r_m:.0%}** contre **{r_o:.0%}**. Mais le tableau par catégorie montre "
               f"que {n_zero} catégories n'ont aucun manquant (les moins populaires) et que le sens de l'écart change d'une catégorie à l'autre, "
               "avec de petits effectifs : l'écart global vient de la **composition par catégorie**, pas du manque lui-même. "
               "Testée en validation croisée, la variable `nutrition_missing` n'améliorait pas l'AUC ; elle a été retirée.")

    with t6:
        st.markdown("""
- **Cible** : ~61 % de fort trafic, classes modérément déséquilibrées → F1 et ROC AUC.
- **Catégorie** : variable dominante (V de Cramér ≈ 0,58), encodée en one-hot.
- **Nutrition** : asymétrique, effet faible → `log1p` pour le modèle linéaire.
- **Portions** et **valeurs manquantes** : pas d'effet propre.
""")

elif page == "4. Modélisation":
    st.title("Modélisation")
    st.markdown("Trois modèles sont comparés : régression logistique, Random Forest et Gradient Boosting. "
                "Tout le prétraitement est dans des *pipelines* scikit-learn (pas de fuite de données).")
    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Part du jeu de test", 0.1, 0.4, 0.2, 0.05)
    seed = c2.number_input("Graine aléatoire", value=42, step=1)
    use_gs = c3.checkbox("Relancer le GridSearch (plus long)", value=False,
                         help="Sinon, les meilleurs hyperparamètres trouvés dans le notebook sont utilisés.")

    if st.button("Lancer l'entraînement", type="primary") or "trained" in st.session_state:
        st.session_state["trained"] = True
        out = train_all(df, use_gs, test_size, int(seed))
        (n_tr, _), (n_te, _) = out["shapes"]
        st.caption(f"Entraînement : {n_tr} recettes · Test : {n_te} recettes · "
                   f"part de fort trafic : {out['y_rates'][0]:.3f} / {out['y_rates'][1]:.3f}")

        st.subheader("Validation croisée (5 plis, jeu d'entraînement)")
        st.dataframe(out["cv"].style.format("{:.3f}").highlight_max(color="#C9D6CE"))
        bcv = out["cv"]["F1"].idxmax()
        interp(f"La validation croisée évalue chaque modèle sur 5 découpages du jeu d'entraînement : c'est une estimation plus stable qu'un seul test. "
               f"Meilleur F1 : **{bcv}** ({out['cv'].loc[bcv, 'F1']:.3f}). L'écart entre modèles est de "
               f"{out['cv']['F1'].max() - out['cv']['F1'].min():.3f} point de F1 : très faible.")

        st.subheader("Évaluation sur le jeu de test")
        res = out["results"]
        st.dataframe(res.style.format("{:.3f}").highlight_max(color="#C9D6CE"))
        best_name = res["F1"].idxmax()
        gap = (res["F1"] - out["cv"]["F1"]).abs().max()
        interp(f"Meilleur F1 sur le test : **{best_name}** ({res.loc[best_name, 'F1']:.3f}), meilleur ROC AUC : **{res['ROC AUC'].idxmax()}** "
               f"({res['ROC AUC'].max():.3f}). Les scores test et validation croisée sont cohérents (écart max de F1 : {gap:.3f}) : "
               "pas de signe de surapprentissage. Les trois modèles sont très proches ; sur ~190 recettes de test, un écart de 0,01 n'est pas significatif. "
               "On privilégie donc le modèle le plus simple et interprétable (régression logistique).")

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4.5))
            for name, (fpr, tpr, _) in out["curves"].items():
                ax.plot(fpr, tpr, label=f"{name} (AUC {res.loc[name, 'ROC AUC']:.3f})")
            ax.plot([0, 1], [0, 1], "k--", label="Hasard")
            ax.set_xlabel("Taux de faux positifs"); ax.set_ylabel("Taux de vrais positifs")
            ax.set_title("Courbes ROC"); ax.legend(fontsize=8)
            show(fig)
            interp(f"Plus une courbe est proche du coin supérieur gauche, meilleur est le modèle ; la diagonale correspond au hasard (AUC = 0,5). "
                   f"Ici les AUC vont de {res['ROC AUC'].min():.2f} à {res['ROC AUC'].max():.2f} : une recette à fort trafic reçoit un score plus élevé "
                   f"qu'une recette à faible trafic dans environ {res['ROC AUC'].max():.0%} des cas. Les courbes sont quasiment superposées.")
        with c2:
            choice = st.selectbox("Matrice de confusion pour :", list(out["cms"]),
                                  index=list(out["cms"]).index(best_name))
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            sns.heatmap(out["cms"][choice], annot=True, fmt="d", cmap="Greens", cbar=False,
                        xticklabels=["Faible", "Fort"], yticklabels=["Faible", "Fort"], ax=ax)
            ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
            show(fig)
            st.code(out["reports"][choice])
            tn, fp, fn, tp = out["cms"][choice].ravel()
            interp(f"Sur {tn+fp+fn+tp} recettes de test : **{tp}** fortes correctement repérées, **{tn}** faibles correctement écartées. "
                   f"**{fp} faux positifs** (recettes promues qui n'auront pas de fort trafic) et **{fn} faux négatifs** (bonnes recettes manquées). "
                   f"Précision = {tp/(tp+fp):.2f} : quand le modèle annonce un fort trafic, il a raison {tp/(tp+fp):.0%} du temps. "
                   f"Rappel = {tp/(tp+fn):.2f} : il retrouve {tp/(tp+fn):.0%} des recettes à fort trafic.")

        st.subheader("Importance des variables")
        model_imp = st.selectbox("Modèle", list(out["fitted"]), key="imp_model")
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
            interp(f"Importance = part de réduction d'impureté apportée par chaque variable. Les plus influentes : **{', '.join(top)}**. "
                   "Les colonnes « category_… » sont les catégories encodées en one-hot ; leur poids cumulé domine, ce qui rejoint l'analyse exploratoire.")
        else:
            pos = [n.split("__")[-1] for n, v in imp.items() if v > 0][:2]; neg = [n.split("__")[-1] for n, v in imp.items() if v < 0][:2]
            interp(f"Coefficient > 0 : la variable **augmente** la probabilité de fort trafic ; < 0 : elle la **diminue** (à autres variables fixées). "
                   f"Plus forts effets positifs : {', '.join(pos) or '-'} ; négatifs : {', '.join(neg) or '-'}. "
                   "Les variables nutritionnelles sont standardisées après log1p : leurs coefficients sont comparables entre eux. "
                   "Les catégories dominent nettement.")
        with st.expander("Hyperparamètres retenus"):
            st.json(out["best_params"])
    else:
        st.info("Clique sur « Lancer l'entraînement » pour entraîner et comparer les modèles.")

elif page == "5. Prédiction":
    st.title("Prédire le trafic d'une nouvelle recette")
    out = train_all(df, False, 0.2, 42)
    model_name = st.selectbox("Modèle", list(out["fitted"]), index=0)
    c1, c2 = st.columns(2)
    with c1:
        category = st.selectbox("Catégorie", sorted(df["category"].unique()))
        servings = st.select_slider("Portions", options=sorted(df["servings"].unique()), value=4)
    with c2:
        calories = st.number_input("Calories", 0.0, 5000.0, 300.0)
        carbs = st.number_input("Glucides (g)", 0.0, 600.0, 20.0)
        sugar = st.number_input("Sucre (g)", 0.0, 200.0, 5.0)
        protein = st.number_input("Protéines (g)", 0.0, 400.0, 15.0)

    new = pd.DataFrame({"calories": [calories], "carbohydrate": [carbs], "sugar": [sugar],
                        "protein": [protein], "servings": [servings], "category": [category]})
    proba = out["fitted"][model_name].predict_proba(new)[0, 1]
    st.metric("Probabilité de fort trafic", f"{proba:.1%}")
    st.progress(float(proba))
    if proba >= 0.5:
        st.success("Prédiction : **fort trafic**")
    else:
        st.error("Prédiction : **faible trafic**")
    niveau = "très élevée" if proba >= 0.8 else "élevée" if proba >= 0.6 else "incertaine" if proba >= 0.4 else "faible" if proba >= 0.2 else "très faible"
    interp(f"Le modèle estime à **{proba:.0%}** la probabilité de fort trafic : confiance **{niveau}**. "
           "Une probabilité proche de 50 % signifie que le modèle hésite. Pour tester la sensibilité, change la catégorie : "
           "elle fait varier la prédiction bien plus que les valeurs nutritionnelles.")
    st.caption("Seuil de décision à 0,5. La probabilité reflète l'historique de ce jeu de données, pas une garantie.")

elif page == "6. Conclusion":
    st.title("Conclusion")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("À retenir")
        st.markdown("""
- La **catégorie** est le facteur dominant du trafic.
- La nutrition et les portions apportent peu.
- Régression logistique : F1 ≈ 0,81 et ROC AUC ≈ 0,87 sur le test.
- Les trois modèles sont équivalents : le plus simple est retenu.
""")
    with c2:
        st.subheader("Limites et pistes")
        st.markdown("""
- Petit échantillon : 947 recettes.
- Pas de données sur les ingrédients ni le temps de préparation.
- Imputation calculée avant la séparation train / test : à intégrer au pipeline.
- Ajuster le seuil selon le coût d'une fausse promotion.
""")