from fastapi import FastAPI, Request
import re
import pickle
import nltk
from nltk import WordNetLemmatizer
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
origins = ["*"]
loaded_model = pickle.load(open('./model.pkl', 'rb'))
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


class Output:
	def __init__(self, news='Please provide a headline', prediction='None', status='500 Internal Server Error'):
		self.news = news
		self.prediction = prediction
		self.status = status


@app.get("/predict")
async def root(request: Request):
	try:
		news = request.query_params['news']
		return fake_news_det(news)
	except Exception:
		return Output()


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
	prediction = loaded_model.predict(vectorized_input_data)
	if prediction[0] == 1:
		return Output(news, "Fake", "200 OK")
	else:
		return Output(news, "Real", "200 OK")