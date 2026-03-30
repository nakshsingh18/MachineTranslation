import random
import pandas as pd
import augly.text as txtaugs

INPUT_FILE = "unsup_5lang_full.csv"
OUTPUT_FILE = "pairs_lang_correct_corrupted_jumbled.csv"

random.seed(42)

# ---------------- word jumbling ---------------- #

def swap_adjacent_words(words):
    if len(words) < 2:
        return words
    i = random.randint(0, len(words) - 2)
    words = words.copy()
    words[i], words[i + 1] = words[i + 1], words[i]
    return words


# ---------------- SAFE AugLy augmenters ---------------- #
# NOTE: whitespace corruption intentionally low

AUGLY_AUGS = [
    ("ReplaceSimilarChars", txtaugs.ReplaceSimilarChars(p=0.20)),
    ("ReplaceSimilarUnicodeChars", txtaugs.ReplaceSimilarUnicodeChars(p=0.20)),
    ("SplitWords", txtaugs.SplitWords(p=0.08)),
    ("MergeWords", txtaugs.MergeWords(p=0.08)),
    ("ChangeCase", txtaugs.ChangeCase(p=0.08)),
    ("InsertWhitespaceChars", txtaugs.InsertWhitespaceChars(p=0.02)),  # 🔴 reduced
    ("InsertPunctuationChars", txtaugs.InsertPunctuationChars(p=0.05)),
]


# ---------------- safety filter ---------------- #

def is_over_fragmented(text: str) -> bool:
    """
    Reject outputs that are basically character soup.
    """
    if not text:
        return True

    tokens = text.split()
    if not tokens:
        return True

    single_char_ratio = sum(len(t) == 1 for t in tokens) / len(tokens)
    return single_char_ratio > 0.35


# ---------------- AugLy application ---------------- #

def apply_augly_noise(text: str):
    # 40% chance of applying TWO augmentations
    num_ops = random.choices([1, 2], weights=[0.6, 0.4])[0]
    chosen = random.sample(AUGLY_AUGS, num_ops)

    out = text
    used = []

    for aug_name, aug in chosen:
        out = aug(out)
        used.append(aug_name)

        # normalize AugLy output
        if isinstance(out, list):
            out = " ".join(t for t in out if isinstance(t, str))
        elif not isinstance(out, str):
            out = str(out)

    # 🔴 critical filter
    if is_over_fragmented(out):
        return text, "REJECTED_OVER_FRAGMENTED", False

    return out, "+".join(used), out != text


# ---------------- corruption function ---------------- #

def make_wrong_sentence(text: str):
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if not text:
        return text, None, False

    # Step 1: word order corruption
    words = text.split()
    words = swap_adjacent_words(words)
    corrupted = " ".join(words)

    # Step 2: AugLy corruption
    corrupted, aug_name, augly_used = apply_augly_noise(corrupted)

    return corrupted, aug_name, augly_used


# ---------------- main ---------------- #

def main():
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)

    if "lang" not in df.columns or "text" not in df.columns:
        raise ValueError("Expected columns 'lang' and 'text'")

    print(f"Total rows: {len(df)}")
    print("Creating (language, correct, corrupted) pairs...")

    MAX_PRINT = 5
    printed = 0

    corrupted_texts = []

    for lang, text in zip(df["lang"].astype(str), df["text"].astype(str)):
        corrupted, aug_name, augly_used = make_wrong_sentence(text)
        corrupted_texts.append(corrupted)

        if augly_used and printed < MAX_PRINT:
            printed += 1
            print(f"\n===== AUGLY SAMPLE {printed} =====")
            print(f"AUGMENTER USED:\n{aug_name}")
            print(f"\nLANGUAGE:\n{lang}")
            print(f"\nCORRECT:\n{text}")
            print(f"\nCORRUPTED:\n{corrupted}")
            print("===============================")

    out_df = pd.DataFrame({
        "language": df["lang"].astype(str),
        "correct": df["text"].astype(str),
        "corrupted": corrupted_texts,
    })

    print(f"\nSaving to {OUTPUT_FILE} ...")
    out_df.to_csv(OUTPUT_FILE, index=False)

    print("Done.")
    print(f"Saved file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
