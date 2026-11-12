"""
Mini tableau de bord interactif : exploration des recettes
Lancer avec :  streamlit run dashboard.py
Le fichier recipe_site_traffic_2212.csv (séparateur « ; ») doit être dans le même dossier,
ou être chargé via la barre latérale.
"""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Tableau de bord des recettes", page_icon="🍲", layout="wide")

BLUE, ORANGE, SKY, NAVY = "#1F5FA8", "#F26B21", "#8DB6E3", "#173868"
NUTRI_COLS = ["calories", "carbohydrate", "sugar", "protein"]
LABELS = {"calories": "Calories", "carbohydrate": "Glucides (g)", "sugar": "Sucre (g)",
          "protein": "Protéines (g)", "servings": "Portions"}
DEFAULT_PATH = "recipe_site_traffic_2212.csv"
COLOR_TRAFFIC = alt.Scale(domain=["Faible", "Fort"], range=[SKY, ORANGE])


def show_chart(chart):
    """Affiche un graphique Altair pleine largeur (compatible anciennes et nouvelles versions)."""
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


# ----------------------------------------------------------------------------
# Chargement et nettoyage (même logique que le notebook)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement des données…")
def load_data(path_or_file):
    df = pd.read_csv(path_or_file, sep=";")
    df["high_traffic"] = df["high_traffic"].fillna("Low").map({"High": 1, "Low": 0})
    df["category"] = df["category"].replace({"Chicken Breast": "Chicken"})
    df["servings"] = df["servings"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    df["nutrition_missing"] = df[NUTRI_COLS].isna().any(axis=1)
    for col in NUTRI_COLS:
        df[col] = df[col].fillna(df.groupby("category")[col].transform("median"))
    df["trafic"] = np.where(df["high_traffic"] == 1, "Fort", "Faible")
    return df


# ----------------------------------------------------------------------------
# Barre latérale : source et filtres
# ----------------------------------------------------------------------------
st.sidebar.title("🍲 Filtres")
uploaded = st.sidebar.file_uploader("Charger le CSV (séparateur « ; »)", type="csv")
try:
    df = load_data(uploaded if uploaded else DEFAULT_PATH)
except FileNotFoundError:
    st.error(f"Fichier `{DEFAULT_PATH}` introuvable. Place-le à côté de `dashboard.py` "
             "ou charge-le dans la barre latérale.")
    st.stop()

cats = sorted(df["category"].unique())
sel_cats = st.sidebar.multiselect("Catégories", cats, default=cats)
sel_serv = st.sidebar.multiselect("Portions", sorted(df["servings"].unique()),
                                  default=sorted(df["servings"].unique()))
sel_traffic = st.sidebar.radio("Trafic", ["Tous", "Fort", "Faible"], horizontal=True)

st.sidebar.markdown("**Plages nutritionnelles**")
ranges = {}
for col in NUTRI_COLS:
    lo, hi = float(df[col].min()), float(df[col].max())
    ranges[col] = st.sidebar.slider(LABELS[col], lo, hi, (lo, hi))
hide_imputed = st.sidebar.checkbox("Exclure les recettes à nutrition imputée", value=False)

mask = df["category"].isin(sel_cats) & df["servings"].isin(sel_serv)
for col, (lo, hi) in ranges.items():
    mask &= df[col].between(lo, hi)
if sel_traffic != "Tous":
    mask &= df["trafic"] == sel_traffic
if hide_imputed:
    mask &= ~df["nutrition_missing"]
f = df[mask]

# ----------------------------------------------------------------------------
# En-tête et indicateurs
# ----------------------------------------------------------------------------
st.title("Tableau de bord des recettes")
st.caption("Explore les recettes avec les filtres de la barre latérale : tous les graphiques se mettent à jour.")

if f.empty:
    st.warning("Aucune recette ne correspond aux filtres. Élargis la sélection.")
    st.stop()

overall = df["high_traffic"].mean()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Recettes sélectionnées", f"{len(f)}", f"{len(f)/len(df):.0%} du total", delta_color="off")
c2.metric("Taux de fort trafic", f"{f['high_traffic'].mean():.1%}",
          f"{(f['high_traffic'].mean() - overall)*100:+.1f} pts vs global")
c3.metric("Calories (médiane)", f"{f['calories'].median():.0f}")
c4.metric("Protéines (médiane)", f"{f['protein'].median():.1f} g")
c5.metric("Catégories", f["category"].nunique())

tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'ensemble", "📈 Distributions",
                                  "🔵 Relations", "🗂️ Données"])

# ----------------------------------------------------------------------------
# Onglet 1 : vue d'ensemble
# ----------------------------------------------------------------------------
with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("Taux de fort trafic par catégorie")
        by_cat = (f.groupby("category")
                    .agg(taux=("high_traffic", "mean"), recettes=("recipe", "count"))
                    .reset_index())
        chart = (alt.Chart(by_cat).mark_bar(color=BLUE)
                 .encode(x=alt.X("taux:Q", axis=alt.Axis(format="%"), title="Taux de fort trafic"),
                         y=alt.Y("category:N", sort="-x", title=None),
                         tooltip=[alt.Tooltip("category:N", title="Catégorie"),
                                  alt.Tooltip("taux:Q", format=".1%", title="Taux"),
                                  alt.Tooltip("recettes:Q", title="Recettes")])
                 .properties(height=330))
        show_chart(chart)
    with right:
        st.subheader("Répartition du trafic")
        split = f["trafic"].value_counts().reset_index()
        split.columns = ["trafic", "recettes"]
        donut = (alt.Chart(split).mark_arc(innerRadius=70)
                 .encode(theta="recettes:Q", color=alt.Color("trafic:N", scale=COLOR_TRAFFIC, title="Trafic"),
                         tooltip=["trafic:N", "recettes:Q"])
                 .properties(height=330))
        show_chart(donut)

    st.subheader("Taux de fort trafic : catégorie × portions")
    heat = (f.groupby(["category", "servings"])
              .agg(taux=("high_traffic", "mean"), recettes=("recipe", "count")).reset_index())
    heatmap = (alt.Chart(heat).mark_rect()
               .encode(x=alt.X("servings:O", title="Portions"), y=alt.Y("category:N", title=None),
                       color=alt.Color("taux:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                                       title="Taux", legend=alt.Legend(format="%")),
                       tooltip=[alt.Tooltip("category:N", title="Catégorie"),
                                alt.Tooltip("servings:O", title="Portions"),
                                alt.Tooltip("taux:Q", format=".0%", title="Taux"),
                                alt.Tooltip("recettes:Q", title="Recettes")])
               .properties(height=330))
    show_chart(heatmap)
    st.caption("Les cases vides correspondent à des combinaisons absentes de la sélection. "
               "Survole une case pour voir l'effectif.")

# ----------------------------------------------------------------------------
# Onglet 2 : distributions
# ----------------------------------------------------------------------------
with tab2:
    cc1, cc2, cc3 = st.columns([2, 1, 1])
    var = cc1.selectbox("Variable", NUTRI_COLS + ["servings"], format_func=LABELS.get)
    nbins = cc2.slider("Nombre de classes", 10, 60, 30)
    split_traffic = cc3.checkbox("Séparer par trafic", value=True)

    enc = dict(x=alt.X(f"{var}:Q", bin=alt.Bin(maxbins=nbins), title=LABELS[var]),
               y=alt.Y("count():Q", title="Recettes"))
    if split_traffic:
        enc["color"] = alt.Color("trafic:N", scale=COLOR_TRAFFIC, title="Trafic")
        enc["tooltip"] = ["trafic:N", "count():Q"]
    hist = alt.Chart(f).mark_bar(color=BLUE, opacity=0.85).encode(**enc).properties(height=320)
    show_chart(hist.interactive())

    left, right = st.columns(2)
    with left:
        st.subheader("Boîtes à moustaches par trafic")
        box = (alt.Chart(f).mark_boxplot(extent="min-max", size=40)
               .encode(x=alt.X("trafic:N", title=None, sort=["Faible", "Fort"]),
                       y=alt.Y(f"{var}:Q", title=LABELS[var]),
                       color=alt.Color("trafic:N", scale=COLOR_TRAFFIC, legend=None))
               .properties(height=300))
        show_chart(box)
    with right:
        st.subheader("Statistiques")
        stats = f.groupby("trafic")[var].describe()[["count", "mean", "50%", "std", "min", "max"]]
        stats.columns = ["Effectif", "Moyenne", "Médiane", "Écart-type", "Min", "Max"]
        st.dataframe(stats.round(1))
        st.caption(f"Asymétrie de la sélection : {f[var].skew():.2f} "
                   "(> 1 : distribution étalée vers la droite).")

# ----------------------------------------------------------------------------
# Onglet 3 : relations
# ----------------------------------------------------------------------------
with tab3:
    cc1, cc2, cc3 = st.columns(3)
    x_var = cc1.selectbox("Axe X", NUTRI_COLS, index=0, format_func=LABELS.get)
    y_var = cc2.selectbox("Axe Y", NUTRI_COLS, index=3, format_func=LABELS.get)
    log_scale = cc3.checkbox("Échelle logarithmique", value=False)
    scale_type = "symlog" if log_scale else "linear"

    sample = f if len(f) <= 2000 else f.sample(2000, random_state=0)
    scatter = (alt.Chart(sample).mark_circle(size=60, opacity=0.6)
               .encode(x=alt.X(f"{x_var}:Q", scale=alt.Scale(type=scale_type), title=LABELS[x_var]),
                       y=alt.Y(f"{y_var}:Q", scale=alt.Scale(type=scale_type), title=LABELS[y_var]),
                       color=alt.Color("trafic:N", scale=COLOR_TRAFFIC, title="Trafic"),
                       tooltip=["recipe:Q", "category:N", "servings:Q", "calories:Q", "carbohydrate:Q",
                                "sugar:Q", "protein:Q", "trafic:N"])
               .properties(height=420).interactive())
    show_chart(scatter)
    st.caption("Molette pour zoomer, glisser pour se déplacer, double-clic pour réinitialiser. "
               "Survole un point pour voir la recette.")

    st.subheader("Corrélations")
    corr = f[NUTRI_COLS + ["servings", "high_traffic"]].corr().round(2)
    corr_long = corr.reset_index().melt("index", var_name="variable", value_name="r")
    base = alt.Chart(corr_long).encode(x=alt.X("index:N", title=None), y=alt.Y("variable:N", title=None))
    heat_corr = base.mark_rect().encode(
        color=alt.Color("r:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1]), title="r"),
        tooltip=["index:N", "variable:N", "r:Q"])
    text = base.mark_text(fontSize=12).encode(text=alt.Text("r:Q", format=".2f"),
                                             color=alt.condition("abs(datum.r) > 0.6",
                                                                 alt.value("white"), alt.value("black")))
    show_chart((heat_corr + text).properties(height=320))

# ----------------------------------------------------------------------------
# Onglet 4 : données
# ----------------------------------------------------------------------------
with tab4:
    st.subheader(f"{len(f)} recettes")
    table = f.drop(columns=["high_traffic", "nutrition_missing"]).rename(columns=LABELS | {
        "recipe": "Recette", "category": "Catégorie", "trafic": "Trafic"})
    st.dataframe(table, hide_index=True)
    st.download_button("⬇️ Télécharger la sélection (CSV)",
                       data=table.to_csv(index=False, sep=";").encode("utf-8"),
                       file_name="recettes_selection.csv", mime="text/csv")