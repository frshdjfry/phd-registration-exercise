import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = "."

TEXT_COLUMN = "text_clean"

def load_brysbaert(path):
    df = pd.read_csv(path)
    df["Word"] = df["Word"].str.lower()
    df = df[["Word", "Conc.M"]]
    df = df.drop_duplicates("Word")
    df = df.rename(columns={"Word": "word", "Conc.M": "concreteness_rating"})
    df["concreteness_rating"] = pd.to_numeric(df["concreteness_rating"], errors="coerce")
    return df

def load_glasgow(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(
        columns={
            "words_unnamed: 0_level_1": "word",
            "length_unnamed: 1_level_1": "length",
        }
    )
    df["word"] = df["word"].str.lower()
    df = df[
        [
            "word",
            "imageability_mean",
            "valence_mean",
            "arousal_mean",
            "dominance_mean",
        ]
    ]
    df = df.drop_duplicates("word")
    df = df.rename(
        columns={
            "imageability_mean": "imageability_rating",
            "valence_mean": "valence_rating",
            "arousal_mean": "arousal_rating",
            "dominance_mean": "dominance_rating",
        }
    )
    for c in ["imageability_rating", "valence_rating", "arousal_rating", "dominance_rating"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_lancaster(path):
    df = pd.read_csv(path)
    df["Word"] = df["Word"].str.lower()
    df = df[
        [
            "Word",
            "Visual.mean",
            "Auditory.mean",
            "Haptic.mean",
            "Gustatory.mean",
            "Olfactory.mean",
            "Interoceptive.mean",
        ]
    ]
    df = df.drop_duplicates("Word")
    df = df.rename(
        columns={
            "Word": "word",
            "Visual.mean": "sensory_visual_strength",
            "Auditory.mean": "sensory_auditory_strength",
            "Haptic.mean": "sensory_haptic_strength",
            "Gustatory.mean": "sensory_gustatory_strength",
            "Olfactory.mean": "sensory_olfactory_strength",
            "Interoceptive.mean": "sensory_interoceptive_strength",
        }
    )
    numeric_cols = [
        "sensory_visual_strength",
        "sensory_auditory_strength",
        "sensory_haptic_strength",
        "sensory_gustatory_strength",
        "sensory_olfactory_strength",
        "sensory_interoceptive_strength",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_elp(path):
    df = pd.read_csv(path)
    df["Word"] = df["Word"].str.lower()
    df = df[["Word", "I_Mean_RT"]]
    df = df.drop_duplicates("Word")
    df = df.rename(
        columns={"Word": "word", "I_Mean_RT": "lexical_decision_reaction_time"}
    )
    df["lexical_decision_reaction_time"] = pd.to_numeric(
        df["lexical_decision_reaction_time"], errors="coerce"
    )
    return df

def load_subtlex(path):
    df = pd.read_csv(path)
    df["Word"] = df["Word"].str.lower()
    df = df[["Word", "Lg10WF"]]
    df = df.drop_duplicates("Word")
    df = df.rename(columns={"Word": "word", "Lg10WF": "word_frequency_log10"})
    df["word_frequency_log10"] = pd.to_numeric(
        df["word_frequency_log10"], errors="coerce"
    )
    return df

def build_master_lexicon():
    brys = load_brysbaert(os.path.join(DATA_DIR, "brysbaert.csv"))
    glas = load_glasgow(os.path.join(DATA_DIR, "glasgow_fixed.csv"))
    lanc = load_lancaster(os.path.join(DATA_DIR, "lancaster.csv"))
    elp = load_elp(os.path.join(DATA_DIR, "elp.csv"))
    subt = load_subtlex(os.path.join(DATA_DIR, "subtlex.csv"))
    lex = brys.merge(glas, on="word", how="outer")
    lex = lex.merge(lanc, on="word", how="outer")
    lex = lex.merge(elp, on="word", how="outer")
    lex = lex.merge(subt, on="word", how="outer")
    lex = lex.set_index("word")
    for c in lex.columns:
        if lex[c].dtype == object:
            lex[c] = pd.to_numeric(lex[c], errors="coerce")
    return lex

def tokenize(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"\b\w+\b", text.lower())

def compute_text_features(text, lexicon, numeric_features):
    tokens = tokenize(text)
    covered_tokens = [t for t in tokens if t in lexicon.index]
    result = {}
    result["sentence_token_count"] = len(tokens)
    result["sentence_token_count_with_lexicon_entry"] = len(covered_tokens)
    if not covered_tokens:
        for feature in numeric_features:
            result[f"{feature}_mean"] = np.nan
        return result
    for feature in numeric_features:
        values = []
        for token in covered_tokens:
            value = lexicon.at[token, feature]
            if pd.notna(value):
                values.append(float(value))
        if values:
            result[f"{feature}_mean"] = sum(values) / float(len(values))
        else:
            result[f"{feature}_mean"] = np.nan
    return result

def main():
    movies = pd.read_parquet(os.path.join(DATA_DIR, "movies.parquet"))
    dialogues = pd.read_parquet(os.path.join(DATA_DIR, "dialogues_clean.parquet"))
    target_genres = ["Action", "Comedy", "Drama", "Crime", "Horror"]
    if "first_genre" in movies.columns:
        movies_filtered = movies[movies["first_genre"].isin(target_genres)]
    else:
        movies_filtered = movies.copy()
    movies_filtered = movies_filtered.copy()
    movies_filtered["decade"] = (movies_filtered["year"] // 10) * 10
    valid_decades = [2000, 2010, 2020]
    movies_filtered = movies_filtered[movies_filtered["decade"].isin(valid_decades)]
    movies_filtered = movies_filtered[(movies_filtered["runtime"].notna()) & (movies_filtered["runtime"] > 0)]
    dialogues_filtered = dialogues.copy()
    dialogues_filtered["dialogue_duration"] = dialogues_filtered["end_time"] - dialogues_filtered["start_time"]
    dialogues_filtered = dialogues_filtered.merge(
        movies_filtered[["movie_id", "first_genre", "first_country", "decade", "runtime"]],
        on="movie_id",
        how="inner",
    )
    dialogues_filtered = dialogues_filtered[
        (dialogues_filtered["dialogue_duration"] > 0)
        & (dialogues_filtered["dialogue_duration"] <= 20)
    ]
    dialogues_filtered = dialogues_filtered[
        dialogues_filtered[["text_emotion", "lexical_density"]].notna().all(axis=1)
    ].copy()
    lexicon = build_master_lexicon()
    numeric_features = lexicon.select_dtypes(include="number").columns
    feature_rows = []
    for text in tqdm(dialogues_filtered[TEXT_COLUMN].tolist()):
        feature_rows.append(compute_text_features(text, lexicon, numeric_features))
    features_df = pd.DataFrame(feature_rows, index=dialogues_filtered.index)
    dialogues_with_features = pd.concat([dialogues_filtered, features_df], axis=1)
    output_path = os.path.join(DATA_DIR, "dialogues_with_text_features.csv")
    dialogues_with_features.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
