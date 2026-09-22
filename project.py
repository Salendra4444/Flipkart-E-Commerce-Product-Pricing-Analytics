import os
import re
import ast
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_PATH = "flipkart_com-ecommerce_sample.csv"
OUTPUT_DIR = "screenshots"
INSIGHTS_FILE = "business_insights.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    return df


def extract_categories(tree_value):
    if pd.isna(tree_value):
        return "Unknown", "Unknown"
    try:
        parsed = ast.literal_eval(tree_value)
        text = parsed[0] if isinstance(parsed, list) and len(parsed) > 0 else str(tree_value)
    except (ValueError, SyntaxError):
        text = str(tree_value)
    text = text.strip().strip('"')
    parts = [p.strip() for p in text.split(">>") if p.strip()]
    main_cat = parts[0] if len(parts) > 0 else "Unknown"
    sub_cat = parts[1] if len(parts) > 1 else main_cat
    return main_cat, sub_cat


def clean_rating(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def clean_dataset(df):
    df = df.copy()
    df.drop_duplicates(subset="uniq_id", inplace=True)

    categories = df["product_category_tree"].apply(extract_categories)
    df["main_category"] = categories.apply(lambda x: x[0])
    df["sub_category"] = categories.apply(lambda x: x[1])

    df["retail_price"] = pd.to_numeric(df["retail_price"], errors="coerce")
    df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors="coerce")
    df = df[(df["retail_price"] > 0) & (df["discounted_price"] > 0)]
    df = df[df["discounted_price"] <= df["retail_price"]]

    df["discount_amount"] = df["retail_price"] - df["discounted_price"]
    df["discount_percent"] = (df["discount_amount"] / df["retail_price"]) * 100

    df["product_rating"] = df["product_rating"].apply(clean_rating)
    df["overall_rating"] = df["overall_rating"].apply(clean_rating)
    df["has_rating"] = df["product_rating"].notna()

    df["brand"] = df["brand"].fillna("Unknown")
    df["brand"] = df["brand"].replace("", "Unknown")

    df["is_FK_Advantage_product"] = df["is_FK_Advantage_product"].fillna(False)
    df["is_FK_Advantage_product"] = df["is_FK_Advantage_product"].astype(bool)

    df["description"] = df["description"].fillna("")
    df["description_length"] = df["description"].apply(lambda x: len(str(x).split()))

    df["product_name"] = df["product_name"].fillna("Unknown Product")
    df["product_name_length"] = df["product_name"].apply(lambda x: len(str(x).split()))

    return df


def top_categories_chart(df, n=15):
    counts = df["main_category"].value_counts().head(n)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=counts.values, y=counts.index, palette="viridis")
    plt.title(f"Top {n} Product Categories by Listing Count")
    plt.xlabel("Number of Products")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_categories.png"), dpi=150)
    plt.close()
    return counts


def price_distribution_chart(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df["retail_price"] <= df["retail_price"].quantile(0.95)]["retail_price"],
                 bins=50, color="steelblue", label="Retail Price", kde=True)
    sns.histplot(df[df["discounted_price"] <= df["discounted_price"].quantile(0.95)]["discounted_price"],
                 bins=50, color="orange", label="Discounted Price", kde=True, alpha=0.6)
    plt.title("Distribution of Retail vs Discounted Prices (95th percentile capped)")
    plt.xlabel("Price (INR)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "price_distribution.png"), dpi=150)
    plt.close()


def discount_by_category_chart(df, n=10):
    top_cats = df["main_category"].value_counts().head(n).index
    subset = df[df["main_category"].isin(top_cats)]
    avg_discount = subset.groupby("main_category")["discount_percent"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_discount.values, y=avg_discount.index, palette="magma")
    plt.title(f"Average Discount % by Category (Top {n} Categories)")
    plt.xlabel("Average Discount (%)")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "discount_by_category.png"), dpi=150)
    plt.close()
    return avg_discount


def top_brands_chart(df, n=15):
    subset = df[df["brand"] != "Unknown"]
    counts = subset["brand"].value_counts().head(n)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=counts.values, y=counts.index, palette="crest")
    plt.title(f"Top {n} Brands by Number of Listings")
    plt.xlabel("Number of Products")
    plt.ylabel("Brand")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_brands.png"), dpi=150)
    plt.close()
    return counts


def correlation_heatmap(df):
    cols = ["retail_price", "discounted_price", "discount_percent", "description_length", "product_name_length"]
    corr = df[cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()


def rating_distribution_chart(df):
    rated = df[df["has_rating"]]
    plt.figure(figsize=(8, 6))
    sns.countplot(x=rated["product_rating"].astype(int), palette="Set2")
    plt.title("Distribution of Product Ratings (Rated Products Only)")
    plt.xlabel("Rating")
    plt.ylabel("Number of Products")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rating_distribution.png"), dpi=150)
    plt.close()
    return rated["product_rating"].value_counts().sort_index()


def fk_advantage_chart(df):
    grouped = df.groupby("is_FK_Advantage_product")["discount_percent"].mean()
    plt.figure(figsize=(6, 6))
    sns.barplot(x=grouped.index.map({True: "FK Advantage", False: "Regular"}), y=grouped.values, palette="pastel")
    plt.title("Average Discount %: FK Advantage vs Regular Products")
    plt.xlabel("Product Type")
    plt.ylabel("Average Discount (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fk_advantage_discount.png"), dpi=150)
    plt.close()
    return grouped


STOPWORDS = set("""a an the for and or with in of to from women men womens mens s
girl girls boy boys kids by pack set combo pcs piece pieces new x cm ml""".split())


def top_keywords_chart(df, n=20):
    words = []
    for name in df["product_name"].dropna():
        tokens = re.findall(r"[a-zA-Z]+", str(name).lower())
        words.extend([t for t in tokens if t not in STOPWORDS and len(t) > 2])
    counter = Counter(words)
    common = counter.most_common(n)
    labels, values = zip(*common)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=list(values), y=list(labels), palette="flare")
    plt.title(f"Top {n} Keywords in Product Names")
    plt.xlabel("Frequency")
    plt.ylabel("Keyword")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_keywords.png"), dpi=150)
    plt.close()
    return common


def price_boxplot_by_category(df, n=8):
    top_cats = df["main_category"].value_counts().head(n).index
    subset = df[df["main_category"].isin(top_cats)]
    capped = subset[subset["discounted_price"] <= subset["discounted_price"].quantile(0.95)]
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=capped, x="discounted_price", y="main_category", palette="coolwarm")
    plt.title(f"Discounted Price Spread Across Top {n} Categories")
    plt.xlabel("Discounted Price (INR)")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "price_boxplot_category.png"), dpi=150)
    plt.close()


def category_prediction_model(df, n_classes=8):
    top_cats = df["main_category"].value_counts().head(n_classes).index
    subset = df[df["main_category"].isin(top_cats)].copy()
    X_text = subset["product_name"].astype(str)
    y = subset["main_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=top_cats)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=top_cats, yticklabels=top_cats)
    plt.title(f"Category Prediction Confusion Matrix (Accuracy: {accuracy:.2%})")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "category_prediction_confusion_matrix.png"), dpi=150)
    plt.close()

    return accuracy, report


def generate_business_insights(df, top_cats, avg_discount, top_brands, fk_grouped, top_keywords, accuracy):
    lines = []
    lines.append("FLIPKART E-COMMERCE PRODUCT & PRICING ANALYTICS - BUSINESS INSIGHTS")
    lines.append("=" * 70)
    lines.append(f"Total products analyzed after cleaning: {len(df)}")
    lines.append(f"Average retail price: INR {df['retail_price'].mean():.2f}")
    lines.append(f"Average discounted price: INR {df['discounted_price'].mean():.2f}")
    lines.append(f"Average discount offered: {df['discount_percent'].mean():.2f}%")
    lines.append("")
    lines.append("Top 5 categories by listing volume:")
    for cat, count in top_cats.head(5).items():
        lines.append(f"  - {cat}: {count} listings")
    lines.append("")
    lines.append("Categories with highest average discount:")
    for cat, pct in avg_discount.head(3).items():
        lines.append(f"  - {cat}: {pct:.2f}% average discount")
    lines.append("")
    lines.append("Top 5 most listed brands:")
    for brand, count in top_brands.head(5).items():
        lines.append(f"  - {brand}: {count} listings")
    lines.append("")
    lines.append("FK Advantage vs Regular product discount comparison:")
    for key, val in fk_grouped.items():
        label = "FK Advantage" if key else "Regular"
        lines.append(f"  - {label}: {val:.2f}% average discount")
    lines.append("")
    lines.append("Most frequent keywords in product names:")
    for word, freq in top_keywords[:10]:
        lines.append(f"  - {word}: {freq} occurrences")
    lines.append("")
    lines.append(f"Bonus ML model - product category prediction from product name")
    lines.append(f"Logistic Regression + TF-IDF test accuracy: {accuracy:.2%}")
    lines.append("")
    lines.append("KEY TAKEAWAYS:")
    lines.append("1. A small number of categories dominate the catalog, suggesting")
    lines.append("   inventory concentration risk and opportunity for category expansion.")
    lines.append("2. Discount strategy varies significantly by category, indicating")
    lines.append("   category-specific pricing and promotion strategies are already in play.")
    lines.append("3. FK Advantage products show a different discount pattern than regular")
    lines.append("   listings, useful for evaluating the program's pricing impact.")
    lines.append("4. Product names carry strong category signal, meaning lightweight text")
    lines.append("   models can automate category tagging and catalog quality checks.")
    lines.append("5. A large share of products have no customer rating, highlighting a")
    lines.append("   gap in review collection that could be addressed with post-purchase")
    lines.append("   engagement campaigns.")

    with open(INSIGHTS_FILE, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


def main():
    df_raw = load_data(DATA_PATH)
    df = clean_dataset(df_raw)

    top_cats = top_categories_chart(df)
    price_distribution_chart(df)
    avg_discount = discount_by_category_chart(df)
    top_brands = top_brands_chart(df)
    correlation_heatmap(df)
    rating_distribution_chart(df)
    fk_grouped = fk_advantage_chart(df)
    top_keywords = top_keywords_chart(df)
    price_boxplot_by_category(df)
    accuracy, report = category_prediction_model(df)

    generate_business_insights(df, top_cats, avg_discount, top_brands, fk_grouped, top_keywords, accuracy)

    df.to_csv("cleaned_flipkart_data.csv", index=False)


if __name__ == "__main__":
    main()
