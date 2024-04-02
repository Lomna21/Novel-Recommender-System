# Requirements of the Projects
flask
numpy
pandas
gunicorn

# import libraries using 
--> # import (name of library)

# uploading new file 
--> name which we want = pd(for pandas).read_csv('file_name')
eg. --> books = pd.read_csv('Books.csv')

# head()
The head() method returns a specified number of rows, string from the top. The head() method returns the first 5 rows if a number is not specified. Note: The column names will also be returned, in addition to the specified rows.

# filename.shape()
It tells about the dimension of the file. i.e. number of rows and columns

# filename.isnull().sum()
eg. --> books.isnull().sum()
Tells about total number of null values in the file.

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







