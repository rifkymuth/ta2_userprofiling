import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    BertTokenizer,
    AutoModelForSequenceClassification,
    DistilBertTokenizer,
)
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from datasets import Dataset

device = "cuda" if cuda.is_available() else "cpu"

topic_classification_model_name = "cahya/bert-base-indonesian-522M"
topic_classification_model_name2 = "cahya/bert-base-indonesian-1.5G"
tokenizer = BertTokenizer.from_pretrained(topic_classification_model_name)

sentiment_predict_model_name = "cahya/distilbert-base-indonesian"
sentiment_predict_model_name2 = "cahya/bert-base-indonesian-1.5G"
sentiment_predict_model_name3 = "distilbert-base-uncased"
tokenizer_sentiment = DistilBertTokenizer.from_pretrained(sentiment_predict_model_name3)

# CONSTANTS
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 2e-05
N_CLASSES = 3

ROOT_PATH = "./user_profiling_ta2/models/"
# ROOT_PATH = ".."


class IndoBERTClass(torch.nn.Module):
    def __init__(self):
        super(IndoBERTClass, self).__init__()
        self.indoBERT = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.indoBERT.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.indoBERT(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        output = self.drop(pooled_output)
        return self.out(output)


class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.tweets = dataframe["tweet"].to_numpy()
        self.sentiment = dataframe["sentimen"].to_numpy()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        tweets = str(self.tweets[item])
        sentiment = self.sentiment[item]
        encoding = self.tokenizer.encode_plus(
            tweets,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "tweet_text": tweets,
            "ids": encoding["input_ids"].flatten(),
            "mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(sentiment, dtype=torch.long),
        }

    def __len__(self):
        return self.len


def load_model_dict(model_path):
    model = IndoBERTClass()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_model(model_path):
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.indoBERT.attn_implementation = "eager"
    model.eval()

    return model


def input_dataloader(batch_input):
    tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
    input_params = {"batch_size": VALID_BATCH_SIZE, "shuffle": False, "num_workers": 2}

    input_set = Triage(batch_input, tokenizer, MAX_LEN)

    return DataLoader(input_set, **input_params)


def input_loader(batch_input):
    input_set = []

    tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
    for review_text in batch_input:
        encoded_review = tokenizer.encode_plus(
            review_text,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        input_set.append(encoded_review)
    return input_set


def inference(model, input_loader):
    predicted_list = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(input_loader, 0):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)

            outputs = model(ids, mask).squeeze()
            _, prediction = torch.max(outputs.data, dim=1)
            predicted_list.extend(prediction.tolist())
    return predicted_list


def inference_seq(model, input_set):
    predicted_list = []
    for idx, data in enumerate(input_set):
        ids = data["input_ids"].to(device, dtype=torch.long)
        mask = data["attention_mask"].to(device, dtype=torch.long)

        output = model(ids, mask)
        _, prediction = torch.max(output, dim=1)
        predicted_list.append(int(prediction))
    return predicted_list


def sentiment_predict(data):
    # Load indobert sentiment model from local
    SENTIMENT_PREDICT_MODEL_PATH = ROOT_PATH + "/model_indobert_sentiment_analysis.pkl"

    model = load_model(SENTIMENT_PREDICT_MODEL_PATH)

    # Add new column for prediction
    data["sentimen"] = 0

    # Preparing input
    input_set = input_loader(data["Tweet"])

    # Predicting input using model
    predicted = inference_seq(model, input_set)

    # Finalizing predicted result into dataframe
    predicted_data = pd.Series(data=predicted)
    data["sentimen"] = predicted_data.sub(1)
    data["sentimen"] = data["hasil_sentimen"].map(
        {0: "Negatif", 1: "Netral", 2: "Positif"}
    )

    return data


def sentiment_predict_indobert_model(data, model):

    # Add new column for prediction
    data["sentimen"] = ""

    # Preparing input
    input_set = input_loader(data["Tweet"])

    # Predicting input using model
    predicted = inference_seq(model, input_set)

    # Finalizing predicted result into dataframe
    predicted_data = pd.Series(data=predicted)
    data["sentimen"] = predicted_data
    data["sentimen"] = data["sentimen"].map({0: "Negatif", 1: "Netral", 2: "Positif"})

    return data


def sentiment_predict_indodistil_model_new(data):
    # constant
    SENTIMENT_PREDICT_MODEL_PATH = ROOT_PATH + "/my_indodistilbert_sentimen"

    data["text"] = data["text"].astype(str)

    # tokenized input with DistilBertTokenizer
    tokenized_input = tokenizer_sentiment(
        data["text"].tolist(), return_tensors="pt", truncation=True, padding=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_PREDICT_MODEL_PATH
    )

    # Forward pass
    model.eval()

    # Forward pass
    with torch.no_grad():
        outputs = model(**tokenized_input)

        # The logits (raw model output)
        logits = outputs.logits

        predicted_class = torch.argmax(logits, dim=-1)

    predicted_data = pd.Series(data=predicted_class)
    data["sentimen"] = predicted_data
    data["sentimen"] = data["sentimen"].map({0: "Negatif", 1: "Netral", 2: "Positif"})

    return data


def sentiment_predict_distilbert_model_new(data):
    # constant
    SENTIMENT_PREDICT_MODEL_PATH = ROOT_PATH + "/my_distilbert_sentimen"

    data["text"] = data["text"].astype(str)

    # tokenized input with DistilBertTokenizer
    tokenized_input = tokenizer_sentiment(
        data["text"].tolist(), return_tensors="pt", truncation=True, padding=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_PREDICT_MODEL_PATH
    )

    # Forward pass
    model.eval()

    # Forward pass
    with torch.no_grad():
        outputs = model(**tokenized_input)

        # The logits (raw model output)
        logits = outputs.logits

        predicted_class = torch.argmax(logits, dim=-1)

    predicted_data = pd.Series(data=predicted_class)
    data["sentimen"] = predicted_data
    data["sentimen"] = data["sentimen"].map({0: "Negatif", 1: "Netral", 2: "Positif"})

    return data


def tokenize(batch):
    return tokenizer(batch["text"], max_length=MAX_LEN, truncation=True)


def topic_classification_indobert_model(data):
    # constant
    TOPIC_CLASSIFICATION_MODEL_PATH = ROOT_PATH + "/my_indobert_topic_classification"

    # # initialize topik column
    # data["topik"] = ""

    data["text"] = data["text"].astype(str)

    # tokenized input with BertTokenizer
    tokenized_input = tokenizer(
        data["text"].tolist(), return_tensors="pt", truncation=True, padding=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        TOPIC_CLASSIFICATION_MODEL_PATH
    )

    # Forward pass
    model.eval()

    with torch.no_grad():
        outputs = model(**tokenized_input)

        # The logits (raw model output)
        logits = outputs.logits

        predicted_class = torch.argmax(logits, dim=-1)
        predicted_class_proba = torch.softmax(logits, dim=-1)

        for pred_class, pred_proba, i in zip(
            predicted_class, predicted_class_proba, range(len(predicted_class))
        ):
            if pred_proba[pred_class] < 0.7:
                predicted_class[i] = -1

    # mapping prediction to class
    data["topik"] = pd.Series(data=predicted_class).map(
        {
            -1: "lain-lain",
            0: "berita dan politik",
            1: "ekonomi",
            2: "hot",
            3: "kesehatan",
            4: "olahraga",
            5: "teknologi",
        }
    ).tolist()

    return data
