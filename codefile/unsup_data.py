import pandas as pd
from datasets import Dataset, DatasetDict

files = {
    "asm_Beng.tsv": "asm_Beng",
    "ben_Beng.tsv": "ben_Beng",
    "guj_Gujr.tsv": "guj_Gujr",
    "mar_Deva.tsv": "mar_Deva",
    "ory_Orya.tsv": "ory_Orya",
    "tam_Taml.tsv": "tam_Taml",
    "mal_Mlym.tsv": "mal_Mlym",
}

all_rows = []

for fname, lang_code in files.items():
    print(f"Loading {fname} ...")
    df = pd.read_csv(fname, sep="\t")

    # Expect: src_lang, tgt_lang, src, tgt
    if not {"tgt_lang", "tgt"}.issubset(df.columns):
        raise ValueError(f"{fname} has unexpected columns: {df.columns}")

    for text in df["tgt"].dropna():
        all_rows.append({
            "lang": lang_code,
            "text": str(text).strip(),
        })

print(f"Total sentences across 5 languages: {len(all_rows)}")

# Create full dataset
full = Dataset.from_list(all_rows)
df_all = full.to_pandas()
# Save the full dataset (no split)
df_all.to_csv("unsup_5lang_full.csv", index=False)

print("Saved unsup_5lang_full.csv")