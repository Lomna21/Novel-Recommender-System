from flask import Flask, render_template, request
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the data and models (assuming you have these prepared as CSVs or pickled files)
books = pd.read_csv('books.csv')
pt = pd.read_csv('pt.csv', index_col=0)  # Pivot table of book titles and some user interaction data
similarity_scores = np.load('similarity_scores.npy')  # Precomputed similarity matrix

# Define the recommend function
def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data

# Route for the home page with form submission
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        book_name = request.form['book_name']
        recommendations = recommend(book_name)
        return render_template('recommend_book.html', recommendations=recommendations)
    return render_template('recommend_book.html')

if __name__ == '__main__':
    app.run(debug=True)
