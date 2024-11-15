
from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

# Load data
books = pd.read_csv('books.csv')
ratings = pd.read_csv('ratings.csv')

# Preprocess and merge data
ratings_with_name = ratings.merge(books, on='ISBN')
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]

@app.route('/')
def recommend():
    books = popular_df.to_dict(orient='records')
    return render_template('recommend.html', books=books)

if __name__ == '__main__':
    app.run(debug=True)
