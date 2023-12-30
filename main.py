import numpy as np
import pandas as pd
from functools import wraps
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests
from flask import Flask, render_template, request, redirect, url_for, session, flash
from passlib.hash import sha256_crypt
from werkzeug.security import generate_password_hash, check_password_hash
from tmdbv3api import TMDb
import pymongo
tmdb = TMDb()
tmdb.api_key = '29a13b0b1d0e910dd62dfc79c6d354a2'

from tmdbv3api import Movie

# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

def create_sim():
    data = pd.read_csv('main_data.csv')
    
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    return data,sim


def rcmd(m):
    m = m.lower()
    try:
        data.head()
        sim.shape
    except:
        data, sim = create_sim()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie your searched is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(sim[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11]
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l

def ListOfGenres(genre_json):
    if genre_json:
        genres = []
        genre_str = ", " 
        for i in range(0,len(genre_json)):
            genres.append(genre_json[i]['name'])
        return genre_str.join(genres)

def date_convert(s):
    MONTHS = ['January', 'February', 'Match', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December']
    y = s[:4]
    m = int(s[5:-3])
    d = s[8:]
    month_name = MONTHS[m-1]

    result= month_name + ' ' + d + ' '  + y
    return result

def MinsToHours(duration):
    if duration%60==0:
        return "{:.0f} hours".format(duration/60)
    else:
        return "{:.0f} hours {} minutes".format(duration/60,duration%60)

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())


client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["TMDB"]
usersCollection = db["users"]

app = Flask(__name__)
@app.route("/")
def home():
    a = request.cookies.get('stars')
    suggestions = get_suggestions()
    return render_template('layout.html')

def login_required(f):
	@wraps(f)
	def wrap(*args, **kwargs):
		if 'logged_in' in session:
			return f(*args, **kwargs)
		else:
			return redirect(url_for('login_page'))
	return wrap

@app.route("/logout/")
@login_required
def logout():
	session.clear()
	return render_template("layout.html")

@app.route('/login/', methods=['GET','POST'])
def login_page():
	error = ''
	if request.method == "POST":
		attempted_email = request.form['email']
		attempted_password = request.form['password']
		if usersCollection.find_one({'email':attempted_email}) is None:
			error = "Email doesn't exists, register first!"
		elif sha256_crypt.verify(attempted_password, usersCollection.find_one({'email':attempted_email})['password']):
			session['logged_in'] = True
			session['email'] = attempted_email
			return render_template('home.html')
		else:
			error = "Invalid credentials, try again!"
	return render_template("login.html", error=error, title='Login')

@app.route('/register/', methods=['GET','POST'])
def register_page():
	error = ''
	if request.method == "POST":
		email = request.form['email']
		password = sha256_crypt.encrypt(str(request.form['password']))
		if usersCollection.find_one({'email':email}) is None:
			usersCollection.insert_one({'email':email, 'password':password})
			session['logged_in'] = True
			session['email'] = email
			return render_template('login.html')
		else:
			error="Email id already exists!"
	return render_template("register.html", error=error, title='Register')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie') # get movie name from the URL
    r = rcmd(movie)
    movie = movie.upper()
    if type(r)==type('string'): # no such movie found in the database
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        tmdb_movie = Movie()
        result = tmdb_movie.search(movie)

        # get movie id and movie title
        movie_id = result[0].id
        movie_name = result[0].title
        
        # making API call
        response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
        data_json = response.json()
        imdb_id = data_json['imdb_id']
        poster = data_json['poster_path']
        img_path = 'https://image.tmdb.org/t/p/original{}'.format(poster)

        # getting list of genres form json
        genre = ListOfGenres(data_json['genres'])

        # web scraping to get user reviews from IMDB site
        sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
        soup = bs.BeautifulSoup(sauce,'lxml')
        soup_result = soup.find_all("div",{"class":"text show-more__control"})

        reviews_list = [] # list of reviews
        reviews_status = [] # list of comments (good or bad)
        for reviews in soup_result:
            if reviews.string:
                reviews_list.append(reviews.string)
                # passing the review to our model
                movie_review_list = np.array([reviews.string])
                movie_vector = vectorizer.transform(movie_review_list)
                pred = clf.predict(movie_vector)
                reviews_status.append('Good' if pred else 'Bad')

        # combining reviews and comments into dictionary
        movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))} 

        # getting votes with comma as thousands separators
        vote_count = "{:,}".format(result[0].vote_count)
        
        # convert date to readable format (eg. 10-06-2019 to June 10 2019)
        rd = date_convert(result[0].release_date)

        # getting the status of the movie (released or not)
        status = data_json['status']

        # convert minutes to hours minutes (eg. 148 minutes to 2 hours 28 mins)
        runtime = MinsToHours(data_json['runtime'])

        # getting the posters for the recommended movies
        poster = []
        movie_title_list = []
        for movie_title in r:
            list_result = tmdb_movie.search(movie_title)
            movie_id = list_result[0].id
            response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
            data_json = response.json()
            poster.append('https://image.tmdb.org/t/p/original{}'.format(data_json['poster_path']))
        movie_cards = {poster[i]: r[i] for i in range(len(r))}

        # get movie names for auto completion
        suggestions = get_suggestions()
        
        return render_template('recommend.html',movie=movie,mtitle=r,t='l',cards=movie_cards,
            result=result[0],reviews=movie_reviews,img_path=img_path,genres=genre,vote_count=vote_count,
            release_date=rd,status=status,runtime=runtime)


if __name__ == '__main__':
    app.secret_key = '123123123'
    app.run(debug=True)
