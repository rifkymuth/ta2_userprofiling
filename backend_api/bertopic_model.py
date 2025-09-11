from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from fastopic import FASTopic
from topmost import Preprocess


# Initialize BERTopic components
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
representation_model = KeyBERTInspired()
vectorizer_model = CountVectorizer(ngram_range=(1, 2))


def predict_topic(data, stopwords_combined):
    # topic_model = BERTopic(
    #     representation_model=representation_model,
    #     ctfidf_model=ctfidf_model,
    #     vectorizer_model=vectorizer_model,
    #     language="multilingual",
    #     top_n_words=5,
    #     nr_topics="auto",
    # )

    # # topic_model = BERTopic( nr_topics="auto")
    # print(data)
    # topics, _ = topic_model.fit_transform(data)

    # # View the top topics
    # print(topic_model.get_topics())
    preprocess = Preprocess()
    topic_model = FASTopic(6, device='cpu')
    topics, _ = topic_model.fit_transform(data)

    return topics, topic_model
