import numpy as np
import streamlit as st
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# Load books data
books_list = pickle.load(open('books1.pkl', 'rb'))
books_list = books_list['Book-Title']
books_df = pickle.load(open('books1.pkl', 'rb'))
images = pickle.load(open('books2.pkl', 'rb'))
Popular_books = pickle.load(open("Popular_books.pkl", "rb"))
Data_sample = pickle.load(open('Data_sample.pkl', 'rb'))
Data_sample = Data_sample["Book_Title"].values
Data_sample_pt = pickle.load(open('Data_sample_pt.pkl', 'rb'))
Similarity_score = pickle.load(open('Similarity_score.pkl', 'rb'))
popular_books_Children = pickle.load(open("popular_books_Children.pkl", "rb"))
popular_books_Teenage = pickle.load(open("popular_books_Teenage.pkl", "rb"))
popular_books_Young_adults = pickle.load(open("popular_books_Young_adults.pkl", "rb"))
popular_books_Middle_aged_adults = pickle.load(open("popular_books_Middle_aged_adults.pkl", "rb"))
popular_books_Old_Aged_adults = pickle.load(open("popular_books_Old_Aged_adults.pkl", "rb"))


st.sidebar.title("Navigation")
page_select = st.sidebar.selectbox("Go to", ["Popularity Based Recommendation", "User Based Recommendation",
                                             "Content Based Recommendation"])


if page_select == "Popularity Based Recommendation":
    st.title("Popular Books")
    if st.button("List of 50 Popular Books "):
        st.write(Popular_books)
        for index, row in Popular_books.iterrows():
            title = row[['Book_Title', 'Ratings_Average', 'Ratings_Count']]
            image_link = row['Image_URL_M']
            st.image(image_link, width=100, caption=title, use_column_width=False)
    if st.button("Popular among Children"):
        st.write(popular_books_Children)
        for index, row in popular_books_Children.iterrows():
            title = row[['Book-Title', 'avg_ratings', 'num_ratings']]
            image_link = row['Image-URL-M']
            st.image(image_link, width=150, caption=title, use_column_width=False)
    if st.button("Popular among Teenage"):
        st.write(popular_books_Teenage)
        for index, row in popular_books_Teenage.iterrows():
            title = row[['Book-Title', 'avg_ratings', 'num_ratings']]
            image_link = row['Image-URL-M']
            st.image(image_link, width=150, caption=title, use_column_width=False)
    if st.button("Popular among Young Adult"):
        st.write(popular_books_Young_adults)
        for index, row in popular_books_Young_adults.iterrows():
            title = row[['Book-Title', 'avg_ratings', 'num_ratings']]
            image_link = row['Image-URL-M']
            st.image(image_link, width=150, caption=title, use_column_width=False)
    if st.button("Popular among Adult"):
        st.write(popular_books_Middle_aged_adults)
        for index, row in popular_books_Middle_aged_adults.iterrows():
            title = row[['Book-Title', 'avg_ratings', 'num_ratings']]
            image_link = row['Image-URL-M']
            st.image(image_link, width=150, caption=title, use_column_width=False)
    if st.button("Popular among Senior"):
        st.write(popular_books_Old_Aged_adults)
        for index, row in popular_books_Old_Aged_adults.iterrows():
            title = row[['Book-Title', 'avg_ratings', 'num_ratings']]
            image_link = row['Image-URL-M']
            st.image(image_link, width=150, caption=title, use_column_width=False)


def recommend(book_name):
    index = np.where(Data_sample_pt.index == book_name)[0][0]
    books_list = sorted(list(enumerate(Similarity_score[index])), key=lambda x: x[1], reverse=True)[1:11]

    recommended_books = []
    for i in books_list:
        recommended_books.append(Data_sample_pt.index[i[0]])
    return recommended_books


if page_select == "User Based Recommendation":
    st.title("User Based Recommendation")
    selected_books = st.selectbox("Select a Book and you will get 10 Books based on similar users", Data_sample)
    st.write('You selected:', selected_books)
    if st.button("Recommend"):
        recommendations = recommend(selected_books)
        for i in recommendations:
            st.write(i)


# Create pivot table and user rating matrix
table = books_df.pivot_table(columns='User-ID', index='Book-Title', values='Book-Rating').fillna(0)
user_rating_matrix = csr_matrix(table.values)

# Create KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_rating_matrix)


# Collaborative filtering page
if page_select == "Content Based Recommendation":
    st.title("Content Based Recommendation")
    select_book = st.selectbox('Select books', books_list)
    st.write('You selected:', select_book)
    selected_index = books_df[books_df['Book-Title'] == select_book].index[0]
    if st.button("Recommend"):
        if selected_index < len(table):
            distances, indices = model_knn.kneighbors(table.iloc[selected_index, :].values.reshape(1, -1),
                                                      n_neighbors=6)
            recommendations = []
            for i in range(1, len(distances.flatten())):
                recommendations.append(
                    '{0}: {1}:'.format(int(i), str(table.index.values[indices.flatten()[i]])))

            for i in recommendations:
                st.write(i)
                # Get the image link from the pop_image dataframe
                book_title = i.split(':')[1].strip()
                try:
                    image_link = images[images['Book-Title'] == book_title]['images-L'].values[0]
                    st.image(image_link, caption=book_title, use_column_width=False)
                except:
                    st.write("Image not found for the book " + book_title)
