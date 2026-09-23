from pathlib import Path

import streamlit as st

from project import (
    DATA_PATH,
    OUTPUT_DIR,
    category_prediction_model,
    clean_dataset,
    correlation_heatmap,
    discount_by_category_chart,
    fk_advantage_chart,
    generate_business_insights,
    load_data,
    price_boxplot_by_category,
    price_distribution_chart,
    rating_distribution_chart,
    top_brands_chart,
    top_categories_chart,
    top_keywords_chart,
)


st.set_page_config(
    page_title="Flipkart Pricing Analytics",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: #151923;
        border: 1px solid #2b3240;
        border-radius: 10px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: #aab3c2; }
    [data-testid="stMetricValue"] { color: #f5f7fb; }
    .model-kicker {
        color: #ff4b4b;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .model-copy { color: #aab3c2; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def run_analysis(data_path):
    raw_df = load_data(data_path)
    df = clean_dataset(raw_df)

    top_categories = top_categories_chart(df)
    price_distribution_chart(df)
    average_discount = discount_by_category_chart(df)
    top_brands = top_brands_chart(df)
    correlation_heatmap(df)
    rating_distribution_chart(df)
    fk_grouped = fk_advantage_chart(df)
    top_keywords = top_keywords_chart(df)
    price_boxplot_by_category(df)
    accuracy, report = category_prediction_model(df)
    generate_business_insights(
        df,
        top_categories,
        average_discount,
        top_brands,
        fk_grouped,
        top_keywords,
        accuracy,
    )
    return df, accuracy, report


def show_chart(filename, caption):
    chart_path = BASE_DIR / OUTPUT_DIR / filename
    if chart_path.exists():
        st.image(str(chart_path), caption=f"{filename} | {caption}", use_container_width=True)
    else:
        st.warning(f"Chart output is missing: {filename}")


st.title("Flipkart Product & Pricing Analytics")
st.caption("Explore catalog concentration, pricing strategy, brands, ratings, and category prediction.")

data_path = BASE_DIR / DATA_PATH
if not data_path.exists():
    st.error(f"Dataset not found: {data_path.name}")
    st.stop()

with st.spinner("Cleaning data and generating analysis..."):
    try:
        df, accuracy, report = run_analysis(str(data_path))
    except Exception as error:
        st.error(f"The analysis could not be completed: {error}")
        st.stop()

metric_columns = st.columns(4)
metric_columns[0].metric("Products analyzed", f"{len(df):,}")
metric_columns[1].metric("Average retail price", f"INR {df['retail_price'].mean():,.0f}")
metric_columns[2].metric("Average discount", f"{df['discount_percent'].mean():.1f}%")
metric_columns[3].metric("Category model accuracy", f"{accuracy:.1%}")

overview_tab, pricing_tab, catalog_tab, model_tab = st.tabs(
    ["Overview", "Pricing", "Catalog quality", "Prediction model"]
)

with overview_tab:
    left, right = st.columns(2)
    with left:
        show_chart("top_categories.png", "Top categories by listing count")
    with right:
        show_chart("top_brands.png", "Top brands by listing count")

with pricing_tab:
    left, right = st.columns(2)
    with left:
        show_chart("price_distribution.png", "Retail and discounted price distributions")
        show_chart("discount_by_category.png", "Average discount by category")
    with right:
        show_chart("fk_advantage_discount.png", "FK Advantage versus regular products")
        show_chart("price_boxplot_category.png", "Discounted price spread by category")

with catalog_tab:
    left, right = st.columns(2)
    with left:
        show_chart("rating_distribution.png", "Ratings for rated products")
        show_chart("top_keywords.png", "Most frequent product-name keywords")
    with right:
        show_chart("correlation_heatmap.png", "Correlations among numeric features")
    st.write(f"Products with a rating: {int(df['has_rating'].sum()):,} ({df['has_rating'].mean():.1%})")

with model_tab:
    st.markdown('<div class="model-kicker">Machine learning evaluation</div>', unsafe_allow_html=True)
    st.header("Product category prediction")
    st.markdown(
        '<div class="model-copy">A TF-IDF text pipeline and logistic regression classifier predict the main category from each product name.</div>',
        unsafe_allow_html=True,
    )

    report_lines = report.strip().splitlines()
    report_rows = []
    summary = {}
    for line in report_lines:
        parts = line.split()
        if len(parts) == 5 and parts[0] not in {"precision", "recall", "f1-score", "support"}:
            try:
                report_rows.append(
                    {
                        "Category": " ".join(parts[:-4]),
                        "Precision": float(parts[-4]),
                        "Recall": float(parts[-3]),
                        "F1 score": float(parts[-2]),
                        "Support": int(parts[-1]),
                    }
                )
            except ValueError:
                continue
        elif parts and parts[0] in {"accuracy", "macro", "weighted"}:
            if parts[0] == "accuracy" and len(parts) >= 3:
                summary["Test samples"] = int(parts[-1])
            elif parts[0] in {"macro", "weighted"} and len(parts) >= 5:
                summary[f"{parts[0].title()} F1"] = float(parts[-2])

    model_metrics = st.columns(4)
    model_metrics[0].metric("Test accuracy", f"{accuracy:.1%}")
    model_metrics[1].metric("Test samples", f"{summary.get('Test samples', 0):,}")
    model_metrics[2].metric("Macro F1", f"{summary.get('Macro F1', 0):.2f}")
    model_metrics[3].metric("Weighted F1", f"{summary.get('Weighted F1', 0):.2f}")

    st.divider()
    report_col, image_col = st.columns([1.1, 1], gap="large")
    with report_col:
        st.subheader("Per-category performance")
        if report_rows:
            st.dataframe(
                report_rows,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Precision": st.column_config.NumberColumn(format="%.2f"),
                    "Recall": st.column_config.NumberColumn(format="%.2f"),
                    "F1 score": st.column_config.NumberColumn(format="%.2f"),
                },
            )
        else:
            st.code(report, language="text")
    with image_col:
        show_chart(
            "category_prediction_confusion_matrix.png",
            "Actual versus predicted categories",
        )
