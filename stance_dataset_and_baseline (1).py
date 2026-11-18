# Indic Climate Stance Challenge - Milestone 2
# Dataset prep + statistics + TF-IDF baseline. Run cell by cell in Google Colab.
# Each "# %%" marks a cell.

# %% 1. Install
# !pip install -q datasets scikit-learn pandas matplotlib seaborn

# %% 2. Load Dhoroni (Bengali climate news, stance among other labels)
import pandas as pd
from datasets import load_dataset

ds = load_dataset("ciol-research/Dhoroni")
print(ds)  # shows splits and column names
df = pd.concat([ds[s].to_pandas().assign(split_orig=s) for s in ds.keys()],
               ignore_index=True)
print(df.columns.tolist())
print(df.head(3).T)

# %% 3. CONFIG - edit after looking at the printed columns above
# Auto-guess columns; override manually if the guess is wrong.
TEXT_COL = next((c for c in df.columns if c.lower() in
                 ["text", "content", "news", "article", "body", "headline"]), None)
STANCE_COL = next((c for c in df.columns if "stance" in c.lower()), None)
print("TEXT_COL =", TEXT_COL, "| STANCE_COL =", STANCE_COL)
print(df[STANCE_COL].value_counts(dropna=False))  # see the real label names

# Map the dataset's stance labels to the challenge labels.
# Edit the keys to match the value_counts printed above.
LABEL_MAP = {
    # "supportive": "Favour",
    # "against":    "Against",
    # "neutral":    "None",
}

# %% 4. Build the Bengali part of the dataset
bn = pd.DataFrame({
    "text": df[TEXT_COL].astype(str),
    "label": df[STANCE_COL].astype(str).str.strip().str.lower().map(
        {k.lower(): v for k, v in LABEL_MAP.items()}),
    "language": "bn",
    "source": "Dhoroni (ciol-research/Dhoroni)",
})
bn = bn.dropna(subset=["label"])
print(len(bn), "Bengali rows kept")

# %% 5. Hindi part (translated + self-written). Optional but recommended.
# Create hindi.csv with columns: text,label  (labels: Favour/Against/None)
# and upload it to Colab. Then run this cell.
import os
if os.path.exists("hindi.csv"):
    hi = pd.read_csv("hindi.csv")
    hi["language"] = "hi"
    hi["source"] = "translated/self-written"
    data = pd.concat([bn, hi[["text", "label", "language", "source"]]],
                     ignore_index=True)
else:
    data = bn.copy()
    print("hindi.csv not found - using Bengali only for now")

# %% 6. Dataset statistics (for the report and slides)
data["text"] = data["text"].str.strip()
print("Instances:", len(data))
print("\nBy language:\n", data["language"].value_counts())
print("\nClass distribution:\n", data["label"].value_counts())
print("\nClass x language:\n", pd.crosstab(data["language"], data["label"]))
print("Duplicates:", data.duplicated("text").sum())
print("Missing/empty text:", (data["text"].isna() | (data["text"] == "")).sum())
data["n_words"] = data["text"].str.split().str.len()
print("\nWords per text:\n", data["n_words"].describe())

# Script mix: share of Latin letters in each text (code-mixing / Roman script proxy)
import re
def latin_ratio(t):
    letters = re.findall(r"[^\W\d_]", t)
    return sum(c.isascii() for c in letters) / max(len(letters), 1)
data["latin_ratio"] = data["text"].apply(latin_ratio)
print("\nTexts with >20% Latin letters (code-mix proxy):",
      round((data["latin_ratio"] > 0.2).mean() * 100, 1), "%")

# %% 7. Charts
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
data["label"].value_counts().plot.bar(ax=ax[0], title="Class distribution")
data["n_words"].clip(upper=data["n_words"].quantile(0.99)).plot.hist(
    bins=40, ax=ax[1], title="Words per text")
plt.tight_layout(); plt.savefig("dataset_stats.png", dpi=150); plt.show()

# %% 8. Save the final dataset
data.drop_duplicates("text").to_csv("climate_stance_dataset.csv", index=False)

# %% 9. Train/val/test split (stratified, fixed seed)
from sklearn.model_selection import train_test_split
d = data.drop_duplicates("text").reset_index(drop=True)
d["strat"] = d["language"] + "_" + d["label"]
vc = d["strat"].value_counts()
d = d[d["strat"].isin(vc[vc >= 3].index)]   # avoid classes too tiny to split
train, temp = train_test_split(d, test_size=0.3, random_state=42, stratify=d["strat"])
val, test = train_test_split(temp, test_size=0.5, random_state=42,
                             stratify=temp["strat"].where(temp["strat"].map(
                                 temp["strat"].value_counts()) >= 2, "other"))
print(len(train), len(val), len(test))

# %% 10. Baseline: TF-IDF + Logistic Regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay

# word + character n-grams help with spelling noise and Indic morphology
vec = make_union(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
)
clf = make_pipeline(vec, LogisticRegression(max_iter=2000, class_weight="balanced"))
clf.fit(train["text"], train["label"])

pred = clf.predict(test["text"])
print("Macro F1:", round(f1_score(test["label"], pred, average="macro"), 4))
print(classification_report(test["label"], pred, digits=3))
ConfusionMatrixDisplay.from_predictions(test["label"], pred, xticks_rotation=45)
plt.tight_layout(); plt.savefig("baseline_confusion.png", dpi=150); plt.show()

# %% 11. Per-language score (shows the two-language challenge)
for lang, g in test.groupby("language"):
    p = clf.predict(g["text"])
    print(lang, "Macro F1:", round(f1_score(g["label"], p, average="macro"), 4))

# %% 12. Save splits for the transformer notebook
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)
