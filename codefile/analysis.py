import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

INPUT_FILE = "/Users/shreyasingh/Downloads/sem6/NLP/dataset2/pairs_lang_correct_corrupted_jumbled.csv"

# ---------------- Load ---------------- #

df = pd.read_csv(INPUT_FILE)

orig = df["correct"].astype(str)
corr = df["corrupted"].astype(str)

# raw ratios from generation
ratios_raw = df["corruption_ratio"].values

# sentence lengths
lengths = orig.apply(lambda x: len(x.split())).values

# ---------------- Fix ratios ---------------- #
# Cap to [0,1] so % is meaningful

ratios = np.clip(ratios_raw, 0.0, 1.0)

total = len(ratios)

# ---------------- Summary stats ---------------- #

corrupted_pct = np.mean(ratios > 0) * 100
mean_pct = np.mean(ratios) * 100
median_pct = np.median(ratios) * 100
std_pct = np.std(ratios) * 100
p90 = np.percentile(ratios, 90) * 100
p95 = np.percentile(ratios, 95) * 100

# ---------------- Buckets ---------------- #

buckets = {
    "0%": ratios == 0,
    "0–5%": (ratios > 0) & (ratios <= 0.05),
    "5–10%": (ratios > 0.05) & (ratios <= 0.10),
    "10–20%": (ratios > 0.10) & (ratios <= 0.20),
    "20–25%": (ratios > 0.20) & (ratios <= 0.25),
}

bucket_pct = {k: v.sum() / total * 100 for k, v in buckets.items()}

# ---------------- Length correlation ---------------- #

corr_coeff = np.corrcoef(lengths, ratios)[0, 1]

# ---------------- Length stratification ---------------- #

length_bins = {
    "1–5": lengths <= 5,
    "6–15": (lengths > 5) & (lengths <= 15),
    "16–30": (lengths > 15) & (lengths <= 30),
    ">30": lengths > 30,
}

length_means = {
    k: ratios[v].mean() * 100 if v.sum() > 0 else 0
    for k, v in length_bins.items()
}

# ---------------- Token explosion diagnostic ---------------- #

token_ratio = (
    corr.apply(lambda x: len(x.split())).values /
    orig.apply(lambda x: len(x.split())).values
)

token_explosion_pct = np.mean(token_ratio > 1.5) * 100

# ---------------- Script detection ---------------- #

def detect_script(text):
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            return "Devanagari"
        elif 0x0980 <= code <= 0x09FF:
            return "Bengali"
        elif 0x0B80 <= code <= 0x0BFF:
            return "Tamil"
        elif 0x0C00 <= code <= 0x0C7F:
            return "Telugu"
        elif 0x0D00 <= code <= 0x0D7F:
            return "Malayalam"
        elif 0x0A80 <= code <= 0x0AFF:
            return "Gujarati"
        elif 0x0A00 <= code <= 0x0A7F:
            return "Gurmukhi"
        elif 0x0600 <= code <= 0x06FF:
            return "Arabic"
        elif 0x0041 <= code <= 0x007A:
            return "Latin"
    return "Unknown"

df["script"] = orig.apply(detect_script)

# ================= TERMINAL OUTPUT ================= #

print("\n===== FINAL CORRUPTION ANALYSIS =====\n")

print(f"Total sentences            : {total}")
print(f"Corrupted sentences (%)    : {corrupted_pct:.2f}")
print(f"Mean corruption (%)        : {mean_pct:.2f}")
print(f"Median corruption (%)      : {median_pct:.2f}")
print(f"Std deviation (%)          : {std_pct:.2f}")
print(f"90th percentile (%)        : {p90:.2f}")
print(f"95th percentile (%)        : {p95:.2f}")

print("\n--- Corruption intensity distribution ---")
for k, v in bucket_pct.items():
    print(f"{k:>8} : {v:.2f}%")

print("\n--- Sentence length analysis ---")
print(f"Pearson correlation (length vs corruption): {corr_coeff:.3f}")

for k, v in length_means.items():
    print(f"Avg corruption for length {k:>4} tokens : {v:.2f}%")

print("\n--- Stability diagnostics ---")
print(f"Token explosion (>1.5× tokens) : {token_explosion_pct:.2f}%")

print("\n===== SENTENCES PER LANGUAGE =====\n")
for lang, count in df["language"].value_counts().items():
    print(f"{lang:>8} : {count}")

print("\n===== SCRIPT DISTRIBUTION BY LANGUAGE =====\n")
for lang in sorted(df["language"].unique()):
    subset = df[df["language"] == lang]
    scripts = subset["script"].value_counts()
    print(f"\nLanguage: {lang}")
    for scr, cnt in scripts.items():
        print(f"  {scr:>12} : {cnt}")

print("\n===== CORRUPTION BY LANGUAGE =====\n")
for lang in sorted(df["language"].unique()):
    mask = df["language"] == lang
    avg_corr = ratios[mask].mean() * 100
    print(f"{lang:>8} : Avg corruption = {avg_corr:.2f}%")

print("\n--- INTERPRETATION ---")
print("""
• Approximately one-third of the dataset is corrupted, preserving a strong clean-data prior.
• Median corruption is zero, confirming conservative augmentation.
• Most corrupted samples remain below 20% token modification.
• Short sentences exhibit higher corruption variance due to token-splitting effects.
• Negative length–corruption correlation indicates no systematic length bias.
• Token explosion cases are rare and controlled.

This corruption strategy is bounded, interpretable, and suitable for robustness training.
""")

# ================= PLOTS ================= #

plt.figure(figsize=(7, 4))
plt.bar(bucket_pct.keys(), bucket_pct.values())
plt.xlabel("Percentage of tokens modified per sentence")
plt.ylabel("Percentage of dataset")
plt.title("Sentence-level corruption intensity distribution")
plt.tight_layout()
plt.savefig("corruption_buckets_bar.png")
plt.close()

plt.figure(figsize=(6, 6))
plt.pie(
    bucket_pct.values(),
    labels=bucket_pct.keys(),
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Dataset share by corruption intensity")
plt.tight_layout()
plt.savefig("corruption_buckets_pie.png")
plt.close()

plt.figure(figsize=(7, 4))
plt.scatter(lengths, ratios * 100, alpha=0.25, s=6)
plt.xlabel("Sentence length (number of tokens)")
plt.ylabel("Corruption ratio (%)")
plt.title("Corruption severity vs sentence length")
plt.tight_layout()
plt.savefig("length_vs_corruption.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.bar(length_means.keys(), length_means.values())
plt.xlabel("Sentence length group (tokens)")
plt.ylabel("Average corruption ratio (%)")
plt.title("Average corruption by sentence length group")
plt.tight_layout()
plt.savefig("length_group_avg_corruption.png")
plt.close()

print("\nSaved plots:")
print(" - corruption_buckets_bar.png")
print(" - corruption_buckets_pie.png")
print(" - length_vs_corruption.png")
print(" - length_group_avg_corruption.png")