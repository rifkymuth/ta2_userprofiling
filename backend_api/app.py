import os
from flask import Flask, request, jsonify, send_file
import json
import re
import nltk
import pandas as pd
import numpy as np
import emoji
from machine_learning import (
    sentiment_predict_indobert_model,
    sentiment_predict_indodistil_model_new,
    sentiment_predict_distilbert_model_new,
    topic_classification_indobert_model,
    IndoBERTClass,
)
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics.pairwise import cosine_similarity
from torch import cuda
from concurrent.futures import ProcessPoolExecutor
from bertopic_model import predict_topic
from langdetect import detect, detect_langs, LangDetectException
from argostranslate import package, translate
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

device = "cuda" if cuda.is_available() else "cpu"
print("device is " + device)

PATH = "./user_profiling_ta2/models/"
# PATH = ""

# Load word2vec
model = Word2Vec.load(PATH + "model_word2vec.model")


@app.route("/topic_classification", methods=["GET"])
def topic_classification():
    # # Input file dari hasil sentimen
    # df = pd.read_csv(PATH + "hasil_sentimen.csv", delimiter=",")

    # Input json dari pengguna
    with open(PATH + "hasil_sentimen.json", "r", encoding="utf-8") as json_file:
        json_content = json.load(json_file)
    df = pd.DataFrame(json_content)

    print("\n=== Predicting topic...")
    try:
        df = topic_classification_indobert_model(df)

        print("\n=== Prediction complete!")

        #  Save to csv
        df.to_csv(
            PATH + "hasil_sentimen_topic_classification.csv",
            index=False,
        )

        # Save as json
        df.to_json(
            PATH + "hasil_sentimen_topic_classification.json",
            index=False,
        )
        # df.drop(columns=['vector']).to_json(PATH + "temp.json", index=False)

        print("\n=== Sending file!")
    except Exception as e:
        return returnAPI(500, "Error", f"{e}")

    # return send_file(
    #     PATH + "hasil_sentimen_topic_classification.csv", as_attachment=True
    # )

    return send_file(
        PATH + "hasil_sentimen_topic_classification.json", as_attachment=True
    )

    # return send_file(PATH + "temp.json", as_attachment=True)


@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    # # Input file dari pengguna
    # input_file = request.files["file"]
    # df = pd.read_csv(input_file, delimiter=";")

    # Input json dari pengguna
    input_json = request.json
    json_tweets = input_json["tweets"]
    df = pd.DataFrame(json_tweets)

    print("\n=== Input file recieved!")

    # Preprocessing
    df = preprocess_task(df)

    # # Prediction
    # try:
    #     # Load indobert sentiment model from local
    #     indobert_model = load_model(PATH + "model_indobert_sentiment_analysis.pkl")

    #     print("\n=== Predicting sentiments...")
    #     df = sentiment_predict_indobert_model(df, indobert_model)

    #     df = df.rename(columns={"text": "tweet", "post": "text"})  # fix column names

    #     df.to_csv(PATH + "hasil_sentimen.csv", index=False)
    # except Exception as e:
    #     return returnAPI(500, "Error", f"{e}")

    try:
        df = df.rename(columns={"text": "tweet", "post": "text"})  # fix column names

        print("\n=== Predicting sentiments...")
        # df = sentiment_predict_indodistil_model_new(df)
        df = sentiment_predict_distilbert_model_new(df)

        # # Save to csv file
        # df.to_csv(PATH + "hasil_sentimen.csv", index=False)

        # Save as json
        df.to_json(PATH + "hasil_sentimen.json", index=False)
        # df.drop(columns=['vector']).to_json(PATH + "temp.json", index=False)
    except Exception as e:
        return returnAPI(500, "Error", f"{e}")

    print("\n=== Prediction complete!")
    print("\n=== Sending file!")

    return send_file(PATH + "hasil_sentimen.json", as_attachment=True)
    # return send_file(PATH + "temp.json", as_attachment=True)


@app.route("/topic_modeling", methods=["GET"])
def topic_modelling():

    # # Read preprocessed input file
    # df = pd.read_csv(PATH + "hasil_sentimen.csv", delimiter=",")

    # Read preprocessed json
    with open(PATH + "hasil_sentimen.json", "r", encoding="utf-8") as json_file:
        json_content = json.load(json_file)
    df = pd.DataFrame(json_content)

    df["text"] = df["text"].astype(str)

    _, topic_model = predict_topic(df["text"], stopwords_combined)

    topics = topic_model.get_topics()

    topics_json = {
        topic_id: [
            {"word": word, "probability": float(prob)} for word, prob in words_probs
        ]
        for topic_id, words_probs in zip(topics, topics.values())
    }

    data = {
        "topics model": topics_json,
    }

    return returnAPI(200, "Success", data)


@app.route("/similarity", methods=["POST"])
def similarity():
    # # Input file dari hasil sentimen
    # df = pd.read_csv(PATH + "hasil_sentimen.csv", delimiter=",")

    # Input json dari pengguna
    with open(PATH + "hasil_sentimen.json", "r", encoding="utf-8") as json_file:
        json_content = json.load(json_file)
    df = pd.DataFrame(json_content)

    # Input teks dari pengguna
    input_text = request.form["text"]

    # Proses preprocessing lengkap
    text_cf = case_folding(input_text)
    text_cl = cleaning(text_cf)
    text_sw = stopwords_removal(text_cl)
    text_sm = stemming(text_sw)

    print("\n=== Hasil Preprocessing ===")
    print("Case Folding     :", text_cf)
    print("Cleaning         :", text_cl)
    print("Stopword Removal :", text_sw)
    print("Stemming         :", text_sm)

    # --- Hitung Cosine Similarity ---
    input_vector = get_sentence_vector(text_sm.split(), model)
    # input_vector = get_sentence_vector(text_cf.split(), model)

    # Ubah kolom vector dari string menjadi array NumPy
    # df["vector"] = df["vector"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df["vector"] = df["post"].apply(
        lambda text: get_sentence_vector(text.split(), model)
    )
    df["vector"] = df["vector"].apply(lambda x: np.array(x))

    similarities = []
    for i, tweet_vector in enumerate(df["vector"]):
        sim = cosine_similarity([input_vector], [tweet_vector])[0][0]
        similarities.append(sim)

    # # --- Tambahkan kolom similarity ke DataFrame ---
    df["similarity"] = similarities

    # --- Filter tweet yang similarity-nya > 0.6 ---
    df_filtered = df[df["similarity"] > 0.6].copy()
    percentage = df_filtered["similarity"].count() / df["similarity"].count()
    print(df_filtered["similarity"].count())
    print(df["similarity"].count())
    print(percentage)

    # # --- Urutkan berdasarkan similarity ---
    df_filtered.sort_values(by="similarity", ascending=False, inplace=True)

    # Hitung persentase sentimen
    sentiment_percent = df_filtered["sentimen"].value_counts(normalize=True) * 100

    print(
        f"\n=== Persentase sentimen dari post yang mirip dengan teks yang dimasukkan: {percentage:.5f}% ==="
    )
    print(f"Positif : {sentiment_percent.get('Positif', 0):.2f}%")
    print(f"Netral  : {sentiment_percent.get('Netral', 0):.2f}%")
    print(f"Negatif : {sentiment_percent.get('Negatif', 0):.2f}%")

    data = {
        "Similar posts": f"{round(percentage, 5)}%",
        "Positif": f"{round(sentiment_percent.get('Positif', 0), 2)}%",
        "Netral": f"{round(sentiment_percent.get('Netral', 0), 2)}%",
        "Negatif": f"{round(sentiment_percent.get('Negatif', 0), 2)}%",
    }

    return returnAPI(200, "Success", data)


def returnAPI(code=200, message="", data=[]):
    status = "success"
    if code != 200:
        status = "failed"
    returnArray = {"code": code, "status": status, "message": message, "data": data}
    return jsonify(returnArray)


# --- Word2Vec Vectorization ---
def get_sentence_vector(tokens, model):
    word_vecs = [model.wv[word] for word in tokens if word in model.wv]
    if not word_vecs:
        return np.zeros(model.vector_size)
    return np.mean(word_vecs, axis=0)


def case_folding(text):
    text = text.casefold()
    return text

def transform_emoji(text):
    # Transform :emoji_name: to emoji winking face
    return re.sub(r":([a-z0-9_]+):", lambda m: "emoji " + m.group(1).replace("_", " "), text)


def cleaning(text, lang="id"):

    # Menghapus karakter baris atau tab (\n, \r, \t..)
    text = re.sub(r"\n|\r|\t", " ", text)

    # Menghapus URL
    text = re.sub(r"http[s]?\://\S+", "", text)

    # Menghapus hashtag
    text = re.sub(r"#\S+", "", text)

    # Menghapus mentions (@)
    text = re.sub(r"@\S+", "", text)

    # Menghapus quotes (")
    text = re.sub(r"\"\S+", "", text)

    # Menghapus teks dalam tanda kurung
    text = re.sub(r"(\(.*\))|(\[.*\])", "", text)

    # Menghapus tanda baca
    text = re.sub(r"[^\w\s]", "", text)

    # Mengubah emoji
    text = transform_emoji(emoji.demojize(text, language=lang))

    # Menghapus semua karakter yang bukan huruf alfabet
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Hapus spasi ganda yang mungkin tersisa
    text = re.sub(r"\s+", " ", text).strip()

    # Hapus kata yang terdiri dari karakter berulang
    text = re.sub(r"\b(a+|z+)\w*\b", "", text, flags=re.IGNORECASE)

    # Hapus kata dengan panjang kurang dari 2 atau lebih dari 7 karakter
    # text = re.sub(r"\b\w{,1}\b", "", text)  # kata sangat pendek (1 karakter)
    # text = re.sub(r"\b\w{10,}\b", "", text)  # kata sangat panjang (10 karakter ke atas)

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
    return stemmer.stem(text)


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
        max_workers = os.cpu_count()-1 or 4
        # max_workers = 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, series))
    results_df = pd.Series(results, index=series.index)
    return results_df.dropna().reset_index(drop=True)

def multithreading_preprocess(series, func, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count()-1 or 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, series))
    return pd.Series(results, index=series.index).dropna().reset_index(drop=True)

def preprocess_task(df):
    print("\n=== Preprocessing...")
    df["post"] = df["post"].astype(str)

    # making a copy
    df["text"] = df["post"]

    # detecting language
    df['lang'] = df['post'].apply(safe_detect)

    # translating to indonesian
    print("\n== Translating...")
    # df["post"] = argos_batch_translate(df["post"].tolist(), df['lang'].tolist())
    df["post"] = df.apply(
        lambda row: argos_single_translate(row["post"], row["lang"]),
        axis=1
    )

    # preprocessing
    print("\n== Preprocessing pipeline...")
    df["post"] = multithreading_preprocess(series=df["post"], func=preprocess_pipeline)

    df["post"] = df["post"].astype(str)


    print("\n=== Preprocessing done!")
    print("data     :", df["post"].sample(5))

    return df

def safe_detect(text):
    try:
        return detect(text)
    except LangDetectException:
        return "un"
    
def install_argos_translate():
    package.update_package_index()
    available_packages = package.get_available_packages()
    model_installed = any(
        pkg.from_code == "en" and pkg.to_code == "id"
        for pkg in package.get_installed_packages()
    )
    if not model_installed:
        print("Downloading Argos Translate model: English → Indonesian...")
        package_to_install = next(
            p for p in available_packages if p.from_code == "en" and p.to_code == "id"
        )
        package.install_from_path('translate-en_id-1_9.argosmodel')
        print("Model installed.")
    else:
        print("Argos Translate model already installed.")

def argos_single_translate(text, lang, from_lang="en", to_lang="id"):
    if lang == from_lang:
        return translate.translate(text, from_lang, to_lang)
    else:
        return text

def argos_batch_translate(texts, langs):
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        results = list(executor.map(argos_single_translate, texts, langs))
    return results

# Paths to files
slang_path_file = "slang_id.txt"
stopword_path_file = "stopwords_id.txt"

# Load resources
install_argos_translate()
nltk.download("stopwords")
slangs = load_slang_dict(slang_path_file)
stopwords_combined = load_stopwords(stopword_path_file)
factory = StemmerFactory()
stemmer = factory.create_stemmer()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5295, debug=True)
