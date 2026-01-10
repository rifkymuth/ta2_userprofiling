import json
import os
import pytest
from flask import jsonify, Response
import pandas as pd

# IMPORTANT: replace 'myapp' with the actual module that defines `app` and the route.
import app as app_module
from app import app as flask_app
from app import returnAPI


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Mock PATH to a temporary directory to avoid touching real filesystem
    # mock_path = str(tmp_path) + os.sep
    mock_path = ""
    monkeypatch.setattr(app_module, "PATH", mock_path, raising=False)

    # Mock preprocess_task to be a no-op
    def fake_preprocess_task(df):
        return df
    monkeypatch.setattr(app_module, "preprocess_task", fake_preprocess_task, raising=False)

    # Mock sentiment predictor to a lightweight deterministic function
    def fake_sentiment_predict(df):
        with open(mock_path + "hasil_sentimen_ipank.json", "r", encoding="utf-8") as json_file:
            json_content = json.load(json_file)
        new_df = pd.DataFrame(json_content)
        return new_df
    monkeypatch.setattr(
        app_module, "sentiment_predict_distilbert_model_new", fake_sentiment_predict, raising=False
    )

    def fake_topic_classification_indobert_model(df):
        with open(mock_path + "hasil_sentimen_topic_classification_ipank.json", "r", encoding="utf-8") as json_file:
            json_content = json.load(json_file)
        new_df = pd.DataFrame(json_content)
        return new_df
    monkeypatch.setattr(
        app_module, "topic_classification_indobert_model", fake_topic_classification_indobert_model, raising=False
    )

    def fake_predict_topic(df, stopwords_combined):
        return {
            "code": 200,
            "data": {
                "topics model": {
                    "0": [
                        {
                            "probability": 0.011669946834445,
                            "word": "dah"
                        },
                        {
                            "probability": 0.011668877676129341,
                            "word": "moment"
                        },
                        {
                            "probability": 0.011651130393147469,
                            "word": "sctv"
                        },
                        {
                            "probability": 0.01164942979812622,
                            "word": "landooo"
                        },
                        {
                            "probability": 0.011643322184681892,
                            "word": "emma"
                        }
                    ],
                    "1": [
                        {
                            "probability": 0.01216336153447628,
                            "word": "balap"
                        },
                        {
                            "probability": 0.012159440666437149,
                            "word": "pole"
                        },
                        {
                            "probability": 0.011734075844287872,
                            "word": "tegang"
                        },
                        {
                            "probability": 0.011681676842272282,
                            "word": "rekor"
                        },
                        {
                            "probability": 0.011681275442242622,
                            "word": "lengan"
                        }
                    ],
                    "2": [
                        {
                            "probability": 0.012616325169801712,
                            "word": "pilih"
                        },
                        {
                            "probability": 0.012527203187346458,
                            "word": "nonton"
                        },
                        {
                            "probability": 0.012149923481047153,
                            "word": "life"
                        },
                        {
                            "probability": 0.01214636117219925,
                            "word": "malas"
                        },
                        {
                            "probability": 0.012083981186151505,
                            "word": "voters"
                        }
                    ],
                    "3": [
                        {
                            "probability": 0.012507762759923935,
                            "word": "terbahakbahak"
                        },
                        {
                            "probability": 0.012506915256381035,
                            "word": "cinta"
                        },
                        {
                            "probability": 0.012484833598136902,
                            "word": "tertawa"
                        },
                        {
                            "probability": 0.01248275674879551,
                            "word": "kacau"
                        },
                        {
                            "probability": 0.012473657727241516,
                            "word": "tiktok"
                        }
                    ],
                    "4": [
                        {
                            "probability": 0.012698529288172722,
                            "word": "gua"
                        },
                        {
                            "probability": 0.01263517513871193,
                            "word": "wajah"
                        },
                        {
                            "probability": 0.012548181228339672,
                            "word": "banget"
                        },
                        {
                            "probability": 0.01249393355101347,
                            "word": "orang"
                        },
                        {
                            "probability": 0.012440264225006104,
                            "word": "mata"
                        }
                    ],
                    "5": [
                        {
                            "probability": 0.012453433126211166,
                            "word": "hati"
                        },
                        {
                            "probability": 0.011937795206904411,
                            "word": "rusak"
                        },
                        {
                            "probability": 0.011911015957593918,
                            "word": "merah"
                        },
                        {
                            "probability": 0.011910446919500828,
                            "word": "rakyat"
                        },
                        {
                            "probability": 0.011836234480142593,
                            "word": "pas"
                        }
                    ]
                }
            },
            "message": "Success",
            "status": "success"
        }
    monkeypatch.setattr(
        app_module, "predict_topic", fake_predict_topic, raising=False
    )

    with flask_app.test_client() as c:
        yield c


def test_predict_sentiment_success(client):
    # Input uses 'post' key so that the route's rename (post -> text) takes effect
    payload = {
        "name": "Ipankkk",
        "verified": "false",
        "followers": 195,
        "following": 265,
        "image_url": "/pic/pbs.twimg.com%2Fprofile_images%2F1963007334709657608%2FmCZox9DH_400x400.jpg",
        "tweets": [
            {
                "post": "date",
                "date": "2025-09-10 15:30:00",
                "link": "https://nitter.net/PostsOfCats/status/1965800174778753309#m",
                "comment": "358",
                "retweet": "34216",
                "quote": "895",
                "likes": "248719"
            },
            {
                "post": "Tom Lembong shot by Tony Hartawan for TEMPO.",
                "date": "2025-09-12 12:22:00",
                "link": "https://nitter.net/IndoPopBase/status/1966477550303027311#m",
                "comment": "1052",
                "retweet": "10388",
                "quote": "3547",
                "likes": "60917"
            },
            {
                "post": "scene fight anime + lady gaga - judas  🔥🔥🔥",
                "date": "2025-09-12 13:51:00",
                "link": "https://nitter.net/pankparampank/status/1966500019550163411#m",
                "comment": "0",
                "retweet": "0",
                "quote": "0",
                "likes": "0"
            },
        ]
    }

    resp = client.post("http://127.0.0.1:5295/predict_sentiment", json=payload)
    assert resp.status == "200 OK"

    # Response is a file attachment; verify header
    cd = resp.headers.get("Content-Disposition", "")

    # The endpoint writes DataFrame to JSON using default orient='columns'
    # So the JSON is a dict of column -> list-of-values
    resp_file_text = resp.data.decode("utf-8")
    data = json.loads(resp_file_text)
    assert "text" in data
    assert "sentimen" in data
    # Ensure the number of rows matches input
    assert len(data["text"]) == 3
    # Ensure our mocked predictor produced expected values
    assert data["sentimen"]["0"] == "Negatif"
    assert data["sentimen"]["1"] == "Netral"
    assert data["sentimen"]["2"] == "Positif"


def test_predict_sentiment_model_error(client, monkeypatch):
    # Re-patch the model function to raise an error and assert 500 handling
    def boom(_df):
        raise RuntimeError("model failed")
    monkeypatch.setattr(
        app_module, "sentiment_predict_distilbert_model_new", boom, raising=True
    )

    payload = {"tweets": [{"post": "anything"}]}
    resp = client.post("/predict_sentiment", json=payload)

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["message"] == "Error"


def test_predict_sentiment_ig_success(client, monkeypatch):
    def fake_sentiment_predict_ig(df):
        with open("" + "hasil_sentimen copy.json", "r", encoding="utf-8") as json_file:
            json_content = json.load(json_file)
        new_df = pd.DataFrame(json_content)
        return new_df
    monkeypatch.setattr(
        app_module, "sentiment_predict_distilbert_model_new", fake_sentiment_predict_ig, raising=False
    )
    # Input uses 'post' key so that the route's rename (post -> text) takes effect
    payload = {
        "image": "https://instagram.fada2-2.fna.fbcdn.net/v/t51.2885-19/541336050_18505124941065704_5677980891095465827_n.jpg?stp=dst-jpg_s150x150_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6InByb2ZpbGVfcGljLmRqYW5nby4xMDgwLmMyIn0&_nc_ht=instagram.fada2-2.fna.fbcdn.net&_nc_cat=108&_nc_oc=Q6cZ2QEg4xaDRREI043JqRgbovhClh8oDd_cq8bWLKd4nxmYtuKL6ru8M_7NVQaFQpuMh0c&_nc_ohc=WkcTIH5_-SEQ7kNvwFWKXYF&_nc_gid=4Le9OKLjyQUhI0WNh2sDNA&edm=AEF8tYYBAAAA&ccb=7-5&oh=00_AfY2U3zxRqfbNF_f8hrk2GQ4R80q4LkDKLp7crv8m33mDA&oe=68C87B2B&_nc_sid=1e20d2",
        "username": "@dnsrmdhan",
        "name": "urbae",
        "description": "Streamer kecil \ud83e\udd0f\ud83c\udffb",
        "totalPost": "10",
        "totalFollowers": "571",
        "totalFollowing": "495",
        "posts": [
            {
            "caption": "they say life is a book, well, i'm turning the page i\u2019m not looking back at it at all...",
            "images": [
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/521594230_18498162538065704_3939557267520276205_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY4Mjc4Ny5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=7nfO-kB5pB4Q7kNvwFw-K8n&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfbgBUYA2x6-D5fH5EXOCKU9r3mMU4f3UZapspNLZYx4YA&oe=68C85D2B&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/521500347_18498162547065704_6859152553093574156_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY4Mjc4Ny5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=IqalmL8DsI4Q7kNvwEcpNo7&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfajkQLBNPA5n2bRgH6gbDsy8k43TuDN-B_o9F37QJl9WQ&oe=68C869E8&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/523418515_18498162556065704_4937775529271524804_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY4Mjc4Ny5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=Xs0bv6Abu-UQ7kNvwHp0hLv&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfZ34iuh-nQWUkePHKehgQo2yICVnePRv9rH4_G4eNdmrw&oe=68C86F0D&_nc_sid=bc0c2c"
            ]
            },
            {
            "caption": "im too important ask me if i got competition i said no they disappoint...",
            "images": [
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/491898713_18483477907065704_3844018028740474215_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=QXxnnyJviqsQ7kNvwFuehAC&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_Afa9WghoaTZy5hCOeUn_NRG4irnjDKVfhnWnqOxiyCSSoQ&oe=68C874F3&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/495169928_18483477931065704_475906611746683639_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=76NT-R_YNZAQ7kNvwGtwTuI&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfYpV01WfCl5n2ns7pXwjee_kS3ZZyBBRKnJbTcwh9bBZw&oe=68C862AA&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/491897847_18483477922065704_6367198648916244072_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=aOCQk2to-VkQ7kNvwG0IlHC&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfZij_c0XJDC8Oc2A_IEscGgU38kTa-0a2pDWZzTIwp5Kg&oe=68C87A8B&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/491898995_18483477940065704_7779745862894426900_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=c0w3VLRceJMQ7kNvwEojBbC&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfbFY_ZGAKOofqb2Kx4YE2UvHpXS8UkxvE0f0VMBgzRzaA&oe=68C8788D&_nc_sid=bc0c2c"
            ]
            },
            {
            "caption": "i think that's what life is about truly findin' yourself and then closin' your eyes, and dyin' in yo...",
            "images": [
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/485494006_18475209238065704_7089701127129618286_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=9DiT-67mN98Q7kNvwFfcAAO&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfaDAJYNzQHLg5HKvaCc_78JnNmJWdK3KMqC-sYXYDlLvA&oe=68C87FC1&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/485641022_18475209247065704_6526732903330331503_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=bIcjDiQdFRYQ7kNvwGuyF4G&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfYwrnnfEHnlZ0lOQTOODF3UzJtkenJo8eUSwSu7z3fpcA&oe=68C84EB1&_nc_sid=bc0c2c"
            ]
            },
            {
            "caption": "they say my life deserve a camera, I'm a GoPro...",
            "images": [
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/481877001_18471152314065704_6988853708928231596_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=KQNwnJGwOKoQ7kNvwEsen-p&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfaeSROlGJdiPu9G-nwopYLRmaxbemGhtfobR79s3cpbfQ&oe=68C84DA8&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/481602581_18471152326065704_8602335989452227219_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=GbmuHQeltKoQ7kNvwGsIolJ&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_Afai6XiJ2eJEeKre0Addb2A4uys53AeHDV69mayw5-gExw&oe=68C87C3A&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/481830789_18471152335065704_9209329941472955319_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=GmpSzMNOt5QQ7kNvwFKVqEE&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfY6i8UL6Hh_9F1GZ_Xi0C3fGT0e3NDP-XQs9lGiiSl2Jg&oe=68C85BDD&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/482171993_18471152344065704_6504443874986146963_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=gBk6bgRcROUQ7kNvwGblYha&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfZUQzgZhizODHZ-qTH9P-ERZaO9JSbmrv_eu_DoMYZebw&oe=68C86E26&_nc_sid=bc0c2c"
            ]
            },
            {
            "caption": "\u03ad\u03ba\u03b8\u03b5\u03c3\u03b7...",
            "images": [
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/475029318_18465905212065704_1839998253662452240_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=gbHcnXYefyMQ7kNvwHnOlD0&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfYxImoU0PsVcaRIG6dsXXiu0eD8QEd1p_W5NrvLD-OHQw&oe=68C86467&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/475191086_18465905230065704_8463795306840718998_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=obmZjQaLrYAQ7kNvwE6tnOs&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfbZCvYNRoNWrqe-cgLRkJYpKxEoMJsjxqc5gRuzVsUCPw&oe=68C8686A&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/475150438_18465905239065704_3182385956236460874_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=8JfHHzXQyncQ7kNvwFWU6id&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfYTu1zGJuvKkF4iuQoMRtAlSqZCVZ53N2t6Y38EB7--uQ&oe=68C84B12&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/475304169_18465905257065704_1906304174647544331_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=fEzZcksDhCgQ7kNvwHhPV8n&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfakAJBDumEyzSj1rRa3qqmHZ5iFbZ3FPIhe05Mbx-P8Kg&oe=68C86B1F&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.2885-15/475049623_18465905272065704_2315981248242574882_n.webp?stp=dst-webp_s640x640_sh0.08&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmY3NTc2MS5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=is9oyM6uC0oQ7kNvwEGhifm&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfaE-eTSJoIx8XHgfex_jg9mDMxhugk40RQwkZhcxlgmpw&oe=68C87446&_nc_sid=bc0c2c"
            ]
            },
            {
            "caption": "my world revolves around a black hole, the same black hole that's in place of my soul...",
            "images": [
                "https://scontent-lga3-2.cdninstagram.com/v/t51.29350-15/469746277_611085724590293_4366409743721097620_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-2.cdninstagram.com&_nc_cat=105&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=1b2YstuHl9UQ7kNvwHSPT-1&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfbJtSDzCcrx5FaUViAbLrPMmgb5rTHcJvlQQXX8Pngj5w&oe=68C86F50&_nc_sid=bc0c2c",
                "https://scontent-lga3-1.cdninstagram.com/v/t51.29350-15/470172476_591285456731030_505624954810931797_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-1.cdninstagram.com&_nc_cat=102&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=AGCUscv886gQ7kNvwGSBoH7&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfbKu0Yrt5w64mxe2_2qXWyHS5EIgrWf538GyMmLd-hopg&oe=68C870BA&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.29350-15/470165159_1617867625481538_6264587946506044361_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=106&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=tOosXzDv_r8Q7kNvwEOB6fe&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_Afa2YjfDvOlP7EK8FI4um6WeRDQrvWazHHxS53wEGhxRVg&oe=68C86F98&_nc_sid=bc0c2c"
            ]
            },
            {
            "caption": "where stories come alive...",
            "images": [
                "https://scontent-lga3-1.cdninstagram.com/v/t51.29350-15/468792969_1323874462305567_108293523930345244_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-1.cdninstagram.com&_nc_cat=103&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=pL8Oue-m9P4Q7kNvwGgf9Lv&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfYntvrEyj9p-yTjqPNT7oP5Nbo422JMCND6ceezV0GgPQ&oe=68C86A8B&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.29350-15/468792561_948739513355302_3715884861392078991_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=110&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=SGCSPJiNaW0Q7kNvwEvD2LS&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfZMgEcVfn7MAbdld146XSSg72ofPs-hKirZ45A-U-G96g&oe=68C85641&_nc_sid=bc0c2c",
                "https://scontent-lga3-1.cdninstagram.com/v/t51.29350-15/468815838_599738122505175_3028326317574502379_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-1.cdninstagram.com&_nc_cat=103&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=vgFrgOSv2NwQ7kNvwHzBMgQ&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfYaRlJmESWNIbngxXQ-S5Vut0cgdxJeUxcLf2dH76HeRg&oe=68C873D3&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.29350-15/468797605_920851183336050_6810137238275052479_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=106&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=E6Ub8XYm7NgQ7kNvwFsBsHS&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_Afa9XYcScozKgJmrSKAY1ww-cHivw5h0Kij8zoisHdpAvg&oe=68C854D2&_nc_sid=bc0c2c",
                "https://scontent-lga3-1.cdninstagram.com/v/t51.29350-15/468741790_989638379570351_3267854120910034781_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-1.cdninstagram.com&_nc_cat=111&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=oQVy6s8X4CkQ7kNvwEEx1J-&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_Afb3F0AezALCWxNjdOCxCwb58O-1hVwZT8MM70pm2Ei7hw&oe=68C85659&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.29350-15/468799750_957353902879012_5782150449679712907_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=104&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=m6eQoEsSZycQ7kNvwHloves&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfbUwfUEqwcm5dmFRnmWQ8DIOrimODkKmROk2IaB3ft7cg&oe=68C85E02&_nc_sid=bc0c2c",
                "https://scontent-lga3-3.cdninstagram.com/v/t51.29350-15/468985446_1303250887500608_8321786201812126331_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_cat=108&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=aKZihgzfDMAQ7kNvwE589lB&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfZ0A5Y1hsnr1mabN5WUln14UCcz2z5PTLO3ucNAphYkFg&oe=68C85843&_nc_sid=bc0c2c"
            ]
            },
            {
            "caption": "you bring the sun out when its cloudy \ud83c\udf3b...",
            "images": [
                "https://scontent-lga3-2.cdninstagram.com/v/t51.29350-15/466982650_1045670950691453_5436079563261968406_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-2.cdninstagram.com&_nc_cat=105&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=1t7g1DwezngQ7kNvwGXJ7zP&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfayHC_AG9HBI4JiqT8vtkb9g9e7BdLhYtJv4cLtoAFd9Q&oe=68C85C73&_nc_sid=bc0c2c",
                "https://scontent-lga3-1.cdninstagram.com/v/t51.29350-15/467037741_1340688980632166_5628646754367690965_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-1.cdninstagram.com&_nc_cat=102&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=6BT99waAF8gQ7kNvwFX98OV&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfZaRQOyflUKFcOfFcO7ynEaUg7Ofxfn7xv-RLx2nXu5Kw&oe=68C8557F&_nc_sid=bc0c2c",
                "https://scontent-lga3-1.cdninstagram.com/v/t51.29350-15/466784288_473428439189495_6726069318721665375_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-1.cdninstagram.com&_nc_cat=111&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=RQNZcaBQTycQ7kNvwF-d6HA&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfanenB2Bf6-jd8qYzZQj1hkSWgTAzbctZwCJazwY4vFLg&oe=68C87377&_nc_sid=bc0c2c"
            ]
            },
            {
            "caption": "I cry out for help, do they listen?...",
            "images": [
                "https://scontent-lga3-2.cdninstagram.com/v/t51.29350-15/456607131_853523926729749_2048710482812060428_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-2.cdninstagram.com&_nc_cat=109&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=Bn_15aUuI-8Q7kNvwGjYsPW&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfaYsXJjoh2CBbD5GCPqcVw3nF6Bdug7nTw1cQPT7urGbg&oe=68C84FF6&_nc_sid=bc0c2c",
                "https://scontent-lga3-2.cdninstagram.com/v/t51.29350-15/456816122_1622395325004537_6582291773298618965_n.webp?stp=dst-jpg_e35_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi42MzF4NjMxLnNkci5mMjkzNTAuZGVmYXVsdF9pbWFnZS5jMiJ9&_nc_ht=scontent-lga3-2.cdninstagram.com&_nc_cat=109&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=zS7DH-BvBd4Q7kNvwFuzFBs&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfZsyvtOai4U-L9QSEv-BKuAEJb5-QDslRrd6YBM6hjZhg&oe=68C8649A&_nc_sid=bc0c2c"
            ]
            },
            {
            "caption": "if he wanna fight .40 bite, like tyson...",
            "images": [
                "https://scontent-lga3-1.cdninstagram.com/v/t51.29350-15/455154951_493971463584335_4687850638061478697_n.webp?stp=dst-jpg_e35_s640x640_sh0.08_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6ImltYWdlX3VybGdlbi4xMDgweDEwODAuc2RyLmYyOTM1MC5kZWZhdWx0X2ltYWdlLmMyIn0&_nc_ht=scontent-lga3-1.cdninstagram.com&_nc_cat=111&_nc_oc=Q6cZ2QGsZWspwaGs3ULvr-Hf-NSMhBHiPaeXawn3i0kGDh8yVxzEf0AZhrPxEiKkUwDBPL9nhbQFY5i3pIkfP-GzNsfs&_nc_ohc=2YvROTCV1a0Q7kNvwHqSRu2&_nc_gid=pf66b7veeozrZ9kpsxHQZQ&edm=APU89FABAAAA&ccb=7-5&oh=00_AfZUGT9FqX8Ve7wljGN5mEXo6b2pSl7hvNhzO8jTZaMelw&oe=68C86138&_nc_sid=bc0c2c"
            ]
            }
        ]
    }

    resp = client.post("http://127.0.0.1:5295/predict_sentiment_ig", json=payload)
    assert resp.status == "200 OK"

    # Response is a file attachment; verify header
    cd = resp.headers.get("Content-Disposition", "")

    # The endpoint writes DataFrame to JSON using default orient='columns'
    # So the JSON is a dict of column -> list-of-values
    resp_file_text = resp.data.decode("utf-8")
    data = json.loads(resp_file_text)
    assert "text" in data
    assert "sentimen" in data
    # Ensure the number of rows matches input
    assert len(data["text"]) == 12
    # Ensure our mocked predictor produced expected values
    assert data["sentimen"]["0"] == "Negatif"
    assert data["sentimen"]["2"] == "Negatif"
    assert data["sentimen"]["4"] == "Netral"


def test_predict_sentiment_ig_model_error(client, monkeypatch):
    # Re-patch the model function to raise an error and assert 500 handling
    def boom(_df):
        raise RuntimeError("model failed")
    monkeypatch.setattr(
        app_module, "sentiment_predict_distilbert_model_new", boom, raising=True
    )

    payload = {"posts": [{"caption": "anything", "images": ["", ""]}]}
    resp = client.post("/predict_sentiment_ig", json=payload)

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["message"] == "Error"

hasil_sentimen_json = {
    "text": {
        "0": "date",
        "1": "tom lembong tembak tony hartawan tempo",
        "2": "tarung lady gaga judbas recitas"
    },
    "date": {
        "0": "2025-09-10 15:30:00",
        "1": "2025-09-12 12:22:00",
        "2": "2025-09-12 13:51:00"
    },
    "link": {
        "0": "https://nitter.net/PostsOfCats/status/1965800174778753309#m",
        "1": "https://nitter.net/IndoPopBase/status/1966477550303027311#m",
        "2": "https://nitter.net/pankparampank/status/1966500019550163411#m"
    },
    "comment": {
        "0": "358",
        "1": "1052",
        "2": "0"
    },
    "retweet": {
        "0": "34216",
        "1": "10388",
        "2": "0"
    },
    "quote": {
        "0": "895",
        "1": "3547",
        "2": "0"
    },
    "likes": {
        "0": "248719",
        "1": "60917",
        "2": "0"
    },
    "tweet": {
        "0": "date",
        "1": "Tom Lembong shot by Tony Hartawan for TEMPO.",
        "2": "scene fight anime + lady gaga - judas  🔥🔥🔥"
    },
    "lang": {
        "0": "id",
        "1": "en",
        "2": "en"
    },
    "sentimen": {
        "0": "Negatif",
        "1": "Netral",
        "2": "Positif"
    }
}

def test_topic_classification_success(client):
    # Input uses 'post' key so that the route's rename (post -> text) takes effect
    payload = hasil_sentimen_json

    resp = client.post("http://127.0.0.1:5295/topic_classification", json=payload)
    assert resp.status == "200 OK"

    # Response is a file attachment; verify header
    cd = resp.headers.get("Content-Disposition", "")

    # The endpoint writes DataFrame to JSON using default orient='columns'
    # So the JSON is a dict of column -> list-of-values
    resp_file_text = resp.data.decode("utf-8")
    data = json.loads(resp_file_text)
    assert "text" in data
    assert "topik" in data
    # Ensure the number of rows matches input
    assert len(data["topik"]) == 3
    # Ensure our mocked predictor produced expected values
    assert data["topik"]["0"] == "lain-lain"
    assert data["topik"]["1"] == "gosip"
    assert data["topik"]["2"] == "gosip"


def test_topic_classification_error(client, monkeypatch):
    # Re-patch the model function to raise an error and assert 500 handling
    def boom(_df):
        raise RuntimeError("model failed")
    monkeypatch.setattr(
        app_module, "topic_classification_indobert_model", boom, raising=True
    )

    payload = {"text": {"0": "anything"}}
    resp = client.post("http://127.0.0.1:5295/topic_classification", json=payload)

    assert resp.status_code == 200
    data = json.loads(resp.data.decode("utf-8"))
    assert data["code"] == 500
    assert data["message"] == "Error"

def test_topic_modeling_success(client):
    # Input uses 'post' key so that the route's rename (post -> text) takes effect
    payload = hasil_sentimen_json

    resp = client.post("http://127.0.0.1:5295/topic_modeling", json=payload)
    assert resp.status == "200 OK"

    # Response is a file attachment; verify header
    cd = resp.headers.get("Content-Disposition", "")

    # The endpoint writes DataFrame to JSON using default orient='columns'
    # So the JSON is a dict of column -> list-of-values
    resp_file_text = resp.data.decode("utf-8")
    data = json.loads(resp_file_text)
    assert data["code"] == 200
    topic_models_data = data["data"]["data"]
    assert len(topic_models_data["topics model"]) == 6
    assert len(topic_models_data["topics model"]["0"]) == 5


def test_topic_modeling_error(client, monkeypatch):
    # Re-patch the model function to raise an error and assert 500 handling
    def boom(_df):
        raise RuntimeError("model failed")
    monkeypatch.setattr(
        app_module, "predict_topic", boom, raising=True
    )

    payload = {"text": {"0": "anything"}}
    resp = client.post("http://127.0.0.1:5295/topic_modeling", json=payload)

    assert resp.status_code == 500