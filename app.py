from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

# Load the data (assuming popular.pkl is available)
popular_df = pd.read_pickle('popular.pkl')

@app.route('/')
def home():
    # Convert DataFrame to HTML table
    table_html = popular_df.to_html(classes='table table-striped', index=False)
    return render_template('index.html', table_html=table_html)

if __name__ == '__main__':
    app.run(debug=True)
