from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
import re
import pickle
import nltk
from nltk import WordNetLemmatizer
from fastapi.middleware.cors import CORSMiddleware
import motor.motor_asyncio
from datetime import date


app = FastAPI()
origins = ["*"]
prediction_model = pickle.load(open('./model.pkl', 'rb'))
tfidf_v = pickle.load(open('./tfidf.pkl', 'rb'))
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = "mongodb+srv://ananthu:ananthu@cluster0.pzvdb.mongodb.net/newsset?retryWrites=true&w=majority"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
database = client.newsset
news_collection = database.get_collection("news")

class Output:
	def __init__(self, news='Please provide a headline', prediction='None', status='500 Internal Server Error'):
		self.news = news
		self.prediction = prediction
		self.status = status

@app.get("/status")
async def status():
	return "200 OK"

@app.get("/api/predict")
async def root(request: Request):
	try:
		news = request.query_params['news']
		return fake_news_det(news)
	except Exception:
		return Output()

@app.get("/api/predictednews")
async def root(request: Request):
	allNews = []
	async for news in news_collection.find():
		allNews.append(newsHelper(news))
	return allNews


@app.post("/api/sendnews")
async def root(request: Request):
	jsonBody = await request.json()
	prediction = fake_news_det(jsonBody["title"])
	today = date.today()
	datePosted = today.strftime("%B %d, %Y")
	data = {
		"title": jsonBody["title"],
		"source": jsonBody["source"],
		"date_posted": datePosted,
		"prediction": prediction.prediction
	}
	await news_collection.insert_one(jsonable_encoder(data))
	return data

def newsHelper(news) -> dict:
    return {
        "id": str(news["_id"]),
        "title": news["title"],
        "source": news["source"],
        "date_posted": news["date_posted"],
        "prediction": news["prediction"],
    }

def fake_news_det(news):
	review = news
	corpus = []
	review = re.sub(r'[^a-zA-Z\s]', '', review)
	review = review.lower()
	review = nltk.word_tokenize(review)
	for y in review :
		if y not in stopwords:
			corpus.append(lemmatizer.lemmatize(y))     
	input_data = [' '.join(corpus)]
	vectorized_input_data = tfidf_v.transform(input_data)
	prediction = prediction_model.predict(vectorized_input_data)
	if prediction[0] == 1:
		return Output(news, "Fake", "200 OK")
	else:
		return Output(news, "Real", "200 OK")