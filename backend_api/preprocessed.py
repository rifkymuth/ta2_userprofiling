import os
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from concurrent.futures import ProcessPoolExecutor

PATH = ""


def preprocessing(df):
    # Proses preprocessing lengkap
    # still not working
    text_cf = case_folding(df)
    text_cl = cleaning(text_cf)
    text_sw = stopwords_removal(text_cl)
    text_sm = stemming(text_sw)

    df["tweet"] = text_sm
    df = df.rename(columns={"tweet": "Tweet"})
    df["Tweet"] = df["Tweet"].fillna("")

    df.to_csv(PATH + "hasil_preprocessing.csv", index=False)  # Where PATH

    list_word = text_sm.apply(lambda x: x.split()).tolist()
    df_list_words = (
        pd.DataFrame(list_word)
        .stack()  # mengubah list 2D menjadi format 1D.
        .reset_index(drop=True)  # mereset indeks setelah proses stacking.
        .to_frame(
            name="Word"
        )  # mengonversi hasil stacking menjadi DataFrame dengan kolom "Word".
        .groupby("Word")
        .size()  #  menghitung frekuensi kemunculan setiap kata.
        .reset_index(
            name="Total"
        )  # mengonversi hasil groupby ke DataFrame dengan kolom 'Word' dan 'Total'.
        .sort_values(
            by=["Total", "Word"], ascending=[False, True]
        )  # mengurutkan hasil berdasarkan total dan kata.
        .reset_index(drop=True)
    )
    df_list_words.to_csv(PATH + "list_words.csv", index=False)


# --- Word2Vec Vectorization ---
def get_sentence_vector(tokens, model):
    word_vecs = [model.wv[word] for word in tokens if word in model.wv]
    if not word_vecs:
        return np.zeros(model.vector_size)
    return np.mean(word_vecs, axis=0)


def case_folding(text):
    text = text.casefold()
    return text


def cleaning(text):

    # Menghapus karakter baris atau tab (\n, \r, \t..)
    text = re.sub(r"\n|\r|\t", " ", text)

    # Menghapus URL
    text = re.sub(r"http[s]?\://\S+", "", text)

    # Menghapus hashtag
    text = re.sub(r"#\S+", "", text)

    # Menghapus mentions (@)
    text = re.sub(r"@\S+", "", text)

    # Menghapus teks dalam tanda kurung
    text = re.sub(r"(\(.*\))|(\[.*\])", "", text)

    # Menghapus tanda baca
    text = re.sub(r"[^\w\s]", "", text)

    # Menghapus semua karakter yang bukan huruf alfabet
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Hapus spasi ganda yang mungkin tersisa
    text = re.sub(r"\s+", " ", text).strip()

    # Hapus kata yang terdiri dari karakter berulang
    text = re.sub(r"\b(a+|z+)\w*\b", "", text, flags=re.IGNORECASE)

    # Hapus kata dengan panjang kurang dari 3 atau lebih dari 7 karakter
    text = re.sub(r"\b\w{1,2}\b", "", text)  # kata sangat pendek (1-2 karakter)
    text = re.sub(r"\b\w{10,}\b", "", text)  # kata sangat panjang (10 karakter ke atas)

    return text


# Create a slang dictionary
def load_slang_dict(slang_path_file):
    with open(slang_path_file, "r") as file:
        slang_data = file.read().splitlines()
    return dict(slang.split(":") for slang in slang_data)


# Create a stopwords list
def load_stopwords(stopword_path_file):
    with open(stopword_path_file, "r") as file:
        custom_stopwords = file.read().splitlines()
    # Combine custom stopwords with NLTK stopwords
    nltk_stopwords = set(stopwords.words("indonesian"))
    new_stopwords = set(custom_stopwords).union(nltk_stopwords)
    return new_stopwords


# Function to replace slang words and remove stopwords
def replace_slangs_remove_stopwords(text, slangs, stopwords):
    # Tokenizing text into tokens
    tokens = text.split(" ")
    filtered_tokens = []

    for token in tokens:
        # Replace slang words
        if token in slangs:
            token = slangs[token]
        # Add token to result if not in stopwords
        if token not in stopwords:
            filtered_tokens.append(token)
    # print(filtered_tokens)

    # Combine filtered tokens back into a string
    return " ".join(filtered_tokens)


def stopwords_removal(text):
    # # Paths to files
    # slang_path_file = "slang_id.txt"
    # stopword_path_file = "stopwords_id.txt"

    # # Load resources
    # slangs = load_slang_dict(slang_path_file)
    # stopwords_combined = load_stopwords(stopword_path_file)

    cleaned_text = replace_slangs_remove_stopwords(text, slangs, stopwords_combined)
    return cleaned_text


def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    words = text.split()

    text_lm = []
    for w in words:
        lm = stemmer.stem(w)
        text_lm.append(lm)

    return " ".join(text_lm)


def preprocess_pipeline(text):
    try:
        text = case_folding(text)
        text = cleaning(text)
        text = stopwords_removal(text)
        text = stemming(text)
        return text
    except Exception as e:
        print(f"Error processing text: {e}")
        return None


def parallel_preprocess(series, func, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, series))
    return pd.Series(results, index=series.index)


# Paths to files
slang_path_file = "slang_id.txt"
stopword_path_file = "stopwords_id.txt"

# Load resources
slangs = load_slang_dict(slang_path_file)
stopwords_combined = load_stopwords(stopword_path_file)

if __name__ == "__main__":
    # data_indonlu_train = pd.read_csv(
    #     "./data/indonlu_dataset_smsa_doc_sentiment/train_preprocess.tsv",
    #     delimiter="\t",
    #     header=None,
    #     names=["Tweet", "sentimen"],
    # )
    # data_indonlu_valid = pd.read_csv(
    #     "./data/indonlu_dataset_smsa_doc_sentiment/valid_preprocess.tsv",
    #     delimiter="\t",
    #     header=None,
    #     names=["Tweet", "sentimen"],
    # )
    # data_indonlu_test = pd.read_csv(
    #     "./data/indonlu_dataset_smsa_doc_sentiment/test_preprocess.tsv",
    #     delimiter="\t",
    #     header=None,
    #     names=["Tweet", "sentimen"],
    # )

    # data = pd.concat([data_indonlu_train, data_indonlu_valid, data_indonlu_test])
    data = pd.read_csv(
        "./data/topic classification data/topic_classification_balanced.csv",
        delimiter=";",
    )

    # print(data["isi"].value_counts())

    data.loc[26659:, "isi_preprocessed"] = parallel_preprocess(
        data.loc[26659:, "isi_preprocessed"], preprocess_pipeline
    )

    # data.to_csv("data_news_updated_filtered_preprocessing.csv", index=False)
    data.to_csv("data_news_updated_filtered_preprocessing.csv", index=False, sep=";")
