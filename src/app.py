# import libraries
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from loguru import logger

import flair
from flair.data import Sentence
from flair.embeddings import WordEmbeddings

glove_embedding = WordEmbeddings("glove")

import nltk.tokenize
from nltk.tokenize import word_tokenize

nltk.download("punkt")

import contractions
import string

punc = string.punctuation
from nltk.corpus import stopwords

nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import spacy

try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    import spacy.cli

    print("Model not found. Downloading.")
    spacy.cli.download("en_core_web_md")
    import en_core_web_md

    nlp = en_core_web_md.load()

# define processing functions
def tokenisation(s):
    return word_tokenize(s)


def contract(s):
    return list(map(lambda w: contractions.fix(w), s))


def puntuations(s):
    return [w for w in s if w not in punc]


def stopwords_rm(s):
    return [w for w in s if w not in STOP_WORDS]


def lemm(s):
    return [token.lemma_ for token in nlp(" ".join(s))]


def pipe(s):
    return " ".join(lemm(stopwords_rm(puntuations(contract(tokenisation(s))))))

# define function to tras
def word_embedding(s):
    sentence = Sentence(s)
    glove_embedding.embed(sentence)
    sentence_matrix = sum([np.matrix(token.embedding) for token in sentence]) / len(
        sentence
    )
    return np.array(sentence_matrix).ravel()


df_raw = pd.read_csv("data/final_data.csv", index_col=0)
cols = ["content_ps", "content_re"]

for col in cols:
    df_raw[col] = df_raw[col].apply(lambda row: pipe(row))
    df_raw[col] = df_raw[col].apply(word_embedding)

ps = pd.DataFrame(df_raw["content_ps"].to_list())
re = pd.DataFrame(df_raw["content_re"].to_list())
X = pd.concat([ps, re], axis=1)
y = df_raw["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, train_size=0.8
)
model = RandomForestClassifier(
    n_estimators=1200,
    max_depth=40,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features="auto",
    bootstrap=False,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = round(accuracy_score(y_pred, y_test), 2)
logger.info(f"Accuracy score of trained model is: {score}")

df_selected_re = pd.read_csv("data/selected_resoruces .csv", index_col=0)
resources_list = df_selected_re.iloc[:10, 2].tolist()
content_re_list = df_selected_re.iloc[:10, 3].tolist()
df_re1 = df_raw[df_raw["title_re"] == resources_list[0]]
df_re2 = df_raw[df_raw["title_re"] == resources_list[1]]
df_re3 = df_raw[df_raw["title_re"] == resources_list[2]]
df_re4 = df_raw[df_raw["title_re"] == resources_list[3]]
df_re5 = df_raw[df_raw["title_re"] == resources_list[4]]
df_re6 = df_raw[df_raw["title_re"] == resources_list[5]]
df_re7 = df_raw[df_raw["title_re"] == resources_list[6]]
df_re8 = df_raw[df_raw["title_re"] == resources_list[7]]
df_re9 = df_raw[df_raw["title_re"] == resources_list[8]]
df_re10 = df_raw[df_raw["title_re"] == resources_list[9]]
dfs = [df_re1, df_re2, df_re3, df_re4, df_re5, df_re6, df_re7, df_re8, df_re9, df_re10]

# For new PS
content_re_embedded = list(map(word_embedding, map(pipe, content_re_list)))


def process_new_ps(ps):
    return word_embedding(pipe(ps))


def predict_new_ps(ps, re, model):
    recom = []

    for i in range(10):
        new_x = np.append(process_new_ps(ps), re[i]).reshape(1, -1)
        # print(model.predict_proba(new_x))
        de = True if model.predict(new_x) > 0.7 else False
        recom.append(de)
    return recom

# initiate the flask application
app = Flask(__name__, template_folder="templates")


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        ps = request.form["message"]
        predict_new_ps(ps, content_re_embedded, model)
        recom_re = list(
            df_selected_re.iloc[predict_new_ps(ps, content_re_embedded, model)]
            .T.to_dict()
            .values()
        )

    return render_template("result.html", recom_re=recom_re)


if __name__ == "__main__":
    app.run()
