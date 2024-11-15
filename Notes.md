# Requirements of the Projects

pickle
flask
numpy
pandas
gunicorn

Breif me about your project.
--> The primary objective of my proect was to build a predictive model which recommends novel to users based on genre. So the workflow and the architecture of my projest goes like firstly i imported the data set from kaggle(else -- uci-ml- reprogetory). 
Overfitting concept.
- supervised or unsupervised or reinforcement 
- regression and classification
- why this algorithm.
- how you will train for new books launched.
- how will you make your model efficient -- bagging and boosting


------- PICKLE --------

Pickle is a Python module that provides a way to serialize and deserialize Python objects. Serialization is the process of converting an object into a format that can be stored (such as a file or a database) or transmitted (such as over a network) and later reconstructed. Deserialization is the reverse process: converting the serialized data back into an object.

-- Common Use Cases of pickle --
Saving Model States - Save machine learning models to disk after training so they can be reloaded and used without retraining.
Data Persistence - Save complex data structures (like dictionaries, lists, or custom objects) to disk.
Caching - Store intermediate results to disk to avoid recomputing them.

-- Important Considerations for pickle --
Security - Never unpickle data from an untrusted or unauthenticated source. The pickle module is not secure against erroneous or maliciously constructed data.
Compatibility - Pickled data may not be compatible across different Python versions.
Efficiency - Pickle is not the most efficient serialization format in terms of speed or space. Alternatives like JSON (for simple data structures)



------- FLASK --------

--> About model starts from here <--
--> First we will import the libraries pandas and numpy --> refer import librearies using

--> then we will read all the data available with us from the .csv files
--> eg. books = pd.read_csv('books.csv') --> Refere uploading new file

--> data in books frame -- ISBN (unique number of each book), Book-Title, Book-Author, Year-Of-Publication, Publisher, Image
--> data in users frame -- User-ID, Location, Age (there are lot of missing values of age but we are not using age in our model)
--> data in ratings frame -- User-ID, ISBN, Book-Rating

Exploratory Data Analysis (EDA) is a critical step in the process of building machine learning models. It involves analyzing and visualizing the data to understand its structure, identify patterns, detect anomalies, and uncover relationships between variables. Here's a breakdown of what EDA typically involves:

1. Data cleaning --> Handling missing values , Removing duplicates and correcting errors (identifying and fixing incorrect data.
2. Descriptive Statistics --> measures include mean, meadian, mode standard deviation, variance, skewness.
3. Data visualization --> various plots and charts to visualize data -- histogram, bar graph, scatter plot, correlation matrices
4. Identifying patterns and Relationships
5. Feature Engineering --> Creating new features from existing ones to better capture underlying patterns in the data.
   Techniques include binning, polynomial features, interactions, and transformations (e.g., log, square root).
6. Outlier Detection --> Deciding whether to remove, transform, or keep outliers based on their impact.
7. Dimensionality Reduction --> Reducing the number of features to simplify the model and improve performance. Techniques include Principal Component Analysis (PCA) and t-SNE.
8. Data Transformation --> Normalizing or standardizing data to ensure features are on a similar scale. (e.g., log transformation for skewed data).



------- NUMPY -------

NumPy (Numerical Python) is a powerful library in Python that plays a crucial role in scientific computing and data manipulation. Here are some of its primary roles and uses:

- Array Creation and Manipulation
- Efficiency and Performance
- Integration with Other Libraries
- Linear Algebra and Random Number Generation



------- PANDAS -------

Pandas is a powerful open-source data manipulation and analysis library for Python. It is particularly well-suited for handling structured data, making it a crucial tool in data science, data analysis, and machine learning workflows. Here are the primary roles and uses of the Pandas library:

1. Data Structures:
   Series: A one-dimensional labeled array that can hold data of any type (integers, strings, floating-point numbers, etc.). It is similar to a column in a spreadsheet or a database.

2. Data Cleaning and Preparation:
   Role: Pandas provides a wide range of tools to clean and prepare data for analysis. This includes handling missing data, removing duplicates, transforming data types, and more.

3. Data Transformation:
   Role: Pandas allows for the reshaping and transformation of data. This includes filtering, sorting, grouping, and merging datasets.

4. Data Analysis:
   Role: The library includes powerful tools for exploratory data analysis (EDA), such as descriptive statistics, correlation analysis, and data visualization (in conjunction with libraries like Matplotlib and Seaborn).

5. Data Input and Output:
   Role: Pandas can read and write data from various formats, making it easy to import and export data. Supported formats include CSV, Excel, SQL databases, JSON, HTML, and more.

6. Time Series Analysis:
   Role: Pandas provides robust support for time series data, allowing for date-based indexing, resampling, time shifting, and rolling window calculations.

7. Integration with Other Libraries:
   Role: Pandas seamlessly integrates with other Python libraries, such as NumPy for numerical operations, Matplotlib and Seaborn for plotting, and SciPy and scikit-learn for scientific computing and machine learning.


--> What is collaborative filtering, and how did you use it in your project?
•	Answer: Collaborative filtering is a technique used to make recommendations based on the preferences and behaviours of similar users. In my project, I used collaborative filtering to analyse user interactions and preferences to provide personalized novel recommendations based on the behaviour of users with similar tastes.

--> Can you explain the popularity filtering algorithm and how it was implemented in your project?
•	Answer: The popularity filtering algorithm ranks items based on their popularity across all users. I implemented this by aggregating user ratings and interactions to identify the top 50 most popular novels. This was done using a combination of data aggregation techniques with Pandas to compute rankings and display the most popular novels.

--> How did you handle missing or incomplete data in your recommender system?
•	Answer: I handled missing or incomplete data by using data imputation techniques and by designing the system to work with incomplete datasets. For instance, I used default values or average ratings where necessary and ensured that the system could still provide recommendations even with partial data.

--> What performance metrics did you use to evaluate the effectiveness of your recommender system?
•	Answer: I used metrics such as precision, recall, and mean squared error (MSE) to evaluate the effectiveness of the recommendations. Precision and recall help in assessing the relevance of recommendations, while MSE helps measure the accuracy of predicted ratings compared to actual ratings.

--> How did you ensure the scalability of your recommender system?
•	Answer: To ensure scalability, I focused on efficient data processing and optimized algorithms. I used Flask for the web application to handle requests efficiently and Gunicorn as the WSGI server to manage multiple requests simultaneously. I also ensured that data handling and computation were efficient, enabling the system to scale as the number of users and data grows.

--> What challenges did you face while implementing the recommender system and how did you overcome them?
•	Answer: One challenge was ensuring the accuracy and relevance of recommendations. I addressed this by fine-tuning the collaborative filtering algorithm and popularity metrics, and by continuously testing and refining the system based on user feedback and interaction data.

--> How did you integrate the front-end with the back-end of the recommender system?
•	Answer: I used Flask to build the back-end API that handles data processing and recommendation logic. The front-end, built with HTML and CSS, communicated with the back-end via RESTful API endpoints. Flask served the necessary data to the front-end, which then displayed the recommendations to users.

--> What improvements or additional features would you add to this recommender system if given more time?
•	Answer: If given more time, I would consider integrating additional features such as content-based filtering to complement collaborative filtering, adding user feedback mechanisms to continuously improve recommendations, and implementing advanced machine learning techniques to enhance recommendation accuracy.


# Popularity Based Recommender System

--> Tells top 50 books with highest average rating by people
--> Consider only those books on which minimum of 250 people have given the rating
--> Group by book title and taken count of book rating. Then after finding the count we have calculated the mean (average rating) on each book. Then from that we will select top 50 books in which there is atleast 250 readers rating on each book. Otherwise there can be the case on which only 2 readers has given the rating/review and it is comming on top but originally it is not popular book.

# Colabrotive Filtering Based Recommender System

--> we have chosen only those users who have given atleast 200 ratings on book (means regular readers/genuine readers) and we will consider only those books on which there are atleast 50 ratings has been given.

--> Why 200 and 50 only why not other numbers
-- Cause at these numbers I have got best result after going through various number through hit and trial method.

--> pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
--> pt.fillna(0,inplace=True) --> This line of code replaces all NaN values in the DataFrame pt with 0.
inplace=True modifies the DataFrame pt in place, so it does not return a new DataFrame but rather updates the existing one.

We will make pivot table to transform data into dimensional form.
A pivot table is a powerful data summarization tool commonly used in spreadsheet applications like Microsoft Excel, Google Sheets, and data analysis software like Pandas in Python. Pivot tables allow you to Data reorganization, Data Summarization, filtering and sorting and analyze large datasets quickly and efficiently.

--> from sklearn.metrics.pairwise import cosine_similarity
from cosine_similarity we will find the euclidian distance between all the books from each other.

--> We will make recommend function in which we will share the name of the book and it will recommend similar books to us.

--> similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]

The enumerate function adds a counter to the iterable. So, enumerate(similarity_scores[index]) will produce a sequence of tuples where each tuple contains an index and the corresponding similarity score.
For example, if similarity_scores[index] is [1.0, 0.2, 0.8, 0.5], enumerate(similarity_scores[index]) will produce [(0, 1.0), (1, 0.2), (2, 0.8), (3, 0.5)].

The sorted function sorts the list of tuples.
The key=lambda x: x[1] part tells the sorted function to sort the tuples based on the second element (the similarity score) of each tuple.
reverse=True sorts the list in descending order (from highest similarity to lowest).

import pickle
pickle.dump(popular_df,open('popular.pkl','wb'))

pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))

# import libraries using

--> # import (name of library)

# uploading new file

--> name which we want = pd(for pandas).read_csv('file_name')
eg. --> books = pd.read_csv('Books.csv')

# head()

The head() method returns a specified number of rows, string from the top. The head() method returns the first 5 rows if a number is not specified. Note: The column names will also be returned, in addition to the specified rows.

# filename.shape()

It tells about the dimension of the file. i.e. number of rows and columns
eg. print(books.shape) --> output will be -- (271360,8)

# filename.isnull().sum()

eg. --> books.isnull().sum() --> output will be --> ISBN - 0, Book-title - 0, Book-Author - 1,.....
Tells about total number of null values in the file coloum wise.

# filename.duplicated().sum()

eg. --> books.duplicated().sum()
Tells about total duplicated values in the file.

# Merge two table or data set

eg. --> ratings_with_name = ratings.merge(books,on='ISBN')
In this example ratings_with_name is name given to new merged table/dataset.
syntex --> new_name = first_table_name.merge(second_table_name,on='on which column we want to merge, like in DBMS primary key column has to be present in both tables')

# groupby()

# num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()

Group by is used to group things with same name or thing in a column.
syntex --> name_of_table.groupby('column_name')
.count() is used to count the frequency in a column. (used to count the number of occurrences of a particular element appearing in the string.)
syntex --> .count()['column name']
.reset_index() is used to reset the indexes.

# rename()

# num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)

rename() is used to rename the column.
syntex --> table_name.rename(columns={'old_name':'new_name'})

When inplace = True , the data is modified in place, which means it will return nothing and the dataframe is now updated.
When inplace = False , which is the default, then the operation is performed and it returns a copy of the object.

# avg_rating_df = ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()

.mean() is used to take mean of data

# .sort_values('avg_rating',ascending=False)

.sort_values() is used to sort in cloumns. ascending = false means in decending order

# x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200

# padhe_likhe_users = x[x].index

x = return boolean cause at the end a condition is given
padhe_likhe_users = x[x].index, padhe_likhe_users me saare true values in x ka index store karega.

# filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]

table_1('column_name').isin(table_2) tells wheater a value present in that table or not.
The isin() method checks if the Dataframe contains the specified values. It returns a DataFrame similar to the original DataFrame, but the original values have been replaced with True if the value was one of the specified values, otherwise False .

# pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')

.pivot_table() --> makes a new table from old table in which it groups unique values within one or more descrete values.
pivot() and pivot_table(): Group unique values within one or more discrete categories.
index = rows kon si hongi , columns = column kon si hongi , values = what will be values assigned for a perticular box of table.

stack() and unstack(): Pivot a column or row level to the opposite axis respectively.

# from sklearn.metrics.pairwise import cosine_similarity

We are importing this file i.e. cosine_similarity from sklearn.metrics.pairwise because this helps us to find the euclidian distance between two vectors.
similarity_scores = cosine_similarity(pt)
In this project we will find euclidian distance between all books from final table so that we can get data of similar books and higher rated books.

# def recommend(book_name):

    # index fetch(first we fetch index of the book)
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)
    return data

Its a function to return suggestion/recommendation
index = np.where(pt.index==book_name)[0][0] --> tells about the index of the particular book in the table.

# How to unpack pkl file

Your pkl file is, in fact, a serialized pickle file, which means it has been dumped using Python's pickle module.
To un-pickle the data you can:

import pickle
with open('serialized.pkl', 'rb') as f:
data = pickle.load(f)
