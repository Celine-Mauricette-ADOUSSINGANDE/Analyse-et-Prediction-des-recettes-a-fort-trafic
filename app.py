"""
Mini interactive dashboard: exploring the recipes
Run with:  streamlit run dashboard.py
The file recipe_site_traffic_2212.csv (";" separator) must be in the same folder,
or be uploaded through the sidebar.
"""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Recipe dashboard", page_icon="🍲", layout="wide")

BLUE, ORANGE, SKY, NAVY = "#1F5FA8", "#F26B21", "#8DB6E3", "#173868"
NUTRI_COLS = ["calories", "carbohydrate", "sugar", "protein"]
LABELS = {"calories": "Calories", "carbohydrate": "Carbohydrates (g)", "sugar": "Sugar (g)",
          "protein": "Protein (g)", "servings": "Servings"}
DEFAULT_PATH = "recipe_site_traffic_2212.csv"
COLOR_TRAFFIC = alt.Scale(domain=["Low", "High"], range=[SKY, ORANGE])


def show_chart(chart):
    """Show a full-width Altair chart (works with old and new Streamlit versions)."""
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


# ----------------------------------------------------------------------------
# Loading and cleaning (same logic as the notebook)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data…")
def load_data(path_or_file):
    df = pd.read_csv(path_or_file, sep=";")
    df["high_traffic"] = df["high_traffic"].fillna("Low").map({"High": 1, "Low": 0})
    df["category"] = df["category"].replace({"Chicken Breast": "Chicken"})
    df["servings"] = df["servings"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    df["nutrition_missing"] = df[NUTRI_COLS].isna().any(axis=1)
    for col in NUTRI_COLS:
        df[col] = df[col].fillna(df.groupby("category")[col].transform("median"))
    df["traffic"] = np.where(df["high_traffic"] == 1, "High", "Low")
    return df


# ----------------------------------------------------------------------------
# Sidebar: source and filters
# ----------------------------------------------------------------------------
st.sidebar.title("🍲 Filters")
uploaded = st.sidebar.file_uploader("Upload the CSV (\";\" separator)", type="csv")
try:
    df = load_data(uploaded if uploaded else DEFAULT_PATH)
except FileNotFoundError:
    st.error(f"File `{DEFAULT_PATH}` not found. Put it next to `dashboard.py` "
             "or upload it in the sidebar.")
    st.stop()

cats = sorted(df["category"].unique())
sel_cats = st.sidebar.multiselect("Categories", cats, default=cats)
sel_serv = st.sidebar.multiselect("Servings", sorted(df["servings"].unique()),
                                  default=sorted(df["servings"].unique()))
sel_traffic = st.sidebar.radio("Traffic", ["All", "High", "Low"], horizontal=True)

st.sidebar.markdown("**Nutrition ranges**")
ranges = {}
for col in NUTRI_COLS:
    lo, hi = float(df[col].min()), float(df[col].max())
    ranges[col] = st.sidebar.slider(LABELS[col], lo, hi, (lo, hi))
hide_imputed = st.sidebar.checkbox("Exclude recipes with imputed nutrition", value=False)

mask = df["category"].isin(sel_cats) & df["servings"].isin(sel_serv)
for col, (lo, hi) in ranges.items():
    mask &= df[col].between(lo, hi)
if sel_traffic != "All":
    mask &= df["traffic"] == sel_traffic
if hide_imputed:
    mask &= ~df["nutrition_missing"]
f = df[mask]

# ----------------------------------------------------------------------------
# Header and KPIs
# ----------------------------------------------------------------------------
st.title("Recipe dashboard")
st.caption("Explore the recipes with the sidebar filters: every chart updates.")

if f.empty:
    st.warning("No recipe matches the filters. Widen your selection.")
    st.stop()

overall = df["high_traffic"].mean()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Selected recipes", f"{len(f)}", f"{len(f)/len(df):.0%} of total", delta_color="off")
c2.metric("High-traffic rate", f"{f['high_traffic'].mean():.1%}",
          f"{(f['high_traffic'].mean() - overall)*100:+.1f} pts vs overall")
c3.metric("Calories (median)", f"{f['calories'].median():.0f}")
c4.metric("Protein (median)", f"{f['protein'].median():.1f} g")
c5.metric("Categories", f["category"].nunique())

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Distributions",
                                  "🔵 Relationships", "🗂️ Data"])

# ----------------------------------------------------------------------------
# Tab 1: overview
# ----------------------------------------------------------------------------
with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("High-traffic rate by category")
        by_cat = (f.groupby("category")
                    .agg(rate=("high_traffic", "mean"), recipes=("recipe", "count"))
                    .reset_index())
        chart = (alt.Chart(by_cat).mark_bar(color=BLUE)
                 .encode(x=alt.X("rate:Q", axis=alt.Axis(format="%"), title="High-traffic rate"),
                         y=alt.Y("category:N", sort="-x", title=None),
                         tooltip=[alt.Tooltip("category:N", title="Category"),
                                  alt.Tooltip("rate:Q", format=".1%", title="Rate"),
                                  alt.Tooltip("recipes:Q", title="Recipes")])
                 .properties(height=330))
        show_chart(chart)
    with right:
        st.subheader("Traffic split")
        split = f["traffic"].value_counts().reset_index()
        split.columns = ["traffic", "recipes"]
        donut = (alt.Chart(split).mark_arc(innerRadius=70)
                 .encode(theta="recipes:Q", color=alt.Color("traffic:N", scale=COLOR_TRAFFIC, title="Traffic"),
                         tooltip=["traffic:N", "recipes:Q"])
                 .properties(height=330))
        show_chart(donut)

    st.subheader("High-traffic rate: category × servings")
    heat = (f.groupby(["category", "servings"])
              .agg(rate=("high_traffic", "mean"), recipes=("recipe", "count")).reset_index())
    heatmap = (alt.Chart(heat).mark_rect()
               .encode(x=alt.X("servings:O", title="Servings"), y=alt.Y("category:N", title=None),
                       color=alt.Color("rate:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                                       title="Rate", legend=alt.Legend(format="%")),
                       tooltip=[alt.Tooltip("category:N", title="Category"),
                                alt.Tooltip("servings:O", title="Servings"),
                                alt.Tooltip("rate:Q", format=".0%", title="Rate"),
                                alt.Tooltip("recipes:Q", title="Recipes")])
               .properties(height=330))
    show_chart(heatmap)
    st.caption("Empty cells are combinations absent from the selection. "
               "Hover a cell to see the count.")

# ----------------------------------------------------------------------------
# Tab 2: distributions
# ----------------------------------------------------------------------------
with tab2:
    cc1, cc2, cc3 = st.columns([2, 1, 1])
    var = cc1.selectbox("Variable", NUTRI_COLS + ["servings"], format_func=LABELS.get)
    nbins = cc2.slider("Number of bins", 10, 60, 30)
    split_traffic = cc3.checkbox("Split by traffic", value=True)

    enc = dict(x=alt.X(f"{var}:Q", bin=alt.Bin(maxbins=nbins), title=LABELS[var]),
               y=alt.Y("count():Q", title="Recipes"))
    if split_traffic:
        enc["color"] = alt.Color("traffic:N", scale=COLOR_TRAFFIC, title="Traffic")
        enc["tooltip"] = ["traffic:N", "count():Q"]
    hist = alt.Chart(f).mark_bar(color=BLUE, opacity=0.85).encode(**enc).properties(height=320)
    show_chart(hist.interactive())

    left, right = st.columns(2)
    with left:
        st.subheader("Box plots by traffic")
        box = (alt.Chart(f).mark_boxplot(extent="min-max", size=40)
               .encode(x=alt.X("traffic:N", title=None, sort=["Low", "High"]),
                       y=alt.Y(f"{var}:Q", title=LABELS[var]),
                       color=alt.Color("traffic:N", scale=COLOR_TRAFFIC, legend=None))
               .properties(height=300))
        show_chart(box)
    with right:
        st.subheader("Statistics")
        stats = f.groupby("traffic")[var].describe()[["count", "mean", "50%", "std", "min", "max"]]
        stats.columns = ["Count", "Mean", "Median", "Std dev", "Min", "Max"]
        st.dataframe(stats.round(1))
        st.caption(f"Skewness of the selection: {f[var].skew():.2f} "
                   "(> 1: distribution stretched to the right).")

# ----------------------------------------------------------------------------
# Tab 3: relationships
# ----------------------------------------------------------------------------
with tab3:
    cc1, cc2, cc3 = st.columns(3)
    x_var = cc1.selectbox("X axis", NUTRI_COLS, index=0, format_func=LABELS.get)
    y_var = cc2.selectbox("Y axis", NUTRI_COLS, index=3, format_func=LABELS.get)
    log_scale = cc3.checkbox("Logarithmic scale", value=False)
    scale_type = "symlog" if log_scale else "linear"

    sample = f if len(f) <= 2000 else f.sample(2000, random_state=0)
    scatter = (alt.Chart(sample).mark_circle(size=60, opacity=0.6)
               .encode(x=alt.X(f"{x_var}:Q", scale=alt.Scale(type=scale_type), title=LABELS[x_var]),
                       y=alt.Y(f"{y_var}:Q", scale=alt.Scale(type=scale_type), title=LABELS[y_var]),
                       color=alt.Color("traffic:N", scale=COLOR_TRAFFIC, title="Traffic"),
                       tooltip=["recipe:Q", "category:N", "servings:Q", "calories:Q", "carbohydrate:Q",
                                "sugar:Q", "protein:Q", "traffic:N"])
               .properties(height=420).interactive())
    show_chart(scatter)
    st.caption("Scroll to zoom, drag to pan, double-click to reset. "
               "Hover a point to see the recipe.")

    st.subheader("Correlations")
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
# Tab 4: data
# ----------------------------------------------------------------------------
with tab4:
    st.subheader(f"{len(f)} recipes")
    table = f.drop(columns=["high_traffic", "nutrition_missing"]).rename(columns=LABELS | {
        "recipe": "Recipe", "category": "Category", "traffic": "Traffic"})
    st.dataframe(table, hide_index=True)
    st.download_button("⬇️ Download the selection (CSV)",
                       data=table.to_csv(index=False, sep=";").encode("utf-8"),
                       file_name="recipes_selection.csv", mime="text/csv")