import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from scipy import sparse

anime_ratings_df = pd.read_csv("rating.csv")
anime_ratings_df.shape
print(anime_ratings_df.head())

anime_ratings = anime_ratings_df.loc[anime_ratings_df.rating != -1].reset_index()[['user_id','anime_id','rating']]
print(anime_ratings.shape)
anime_ratings.head()

train_df, valid_df = train_test_split(anime_ratings, test_size=0.2)

#resetting indices to avoid indexing errors in the future
train_df = train_df.reset_index()[['user_id', 'anime_id', 'rating']]
valid_df = valid_df.reset_index()[['user_id', 'anime_id', 'rating']]

def col_encode(column):
	keys = column.unique()
	key_to_id = {key:idx for idx,key in enumerate(keys)}
	return key_to_id , np.array([key_to_id[x] for x in column]) , len(keys)

def df_encode(anime_df):
    anime_ids, anime_df['anime_id'], num_anime = col_encode(anime_df['anime_id'])
    user_ids, anime_df['user_id'], num_users = col_encode(anime_df['user_id'])
    return anime_df, num_users, num_anime, user_ids, anime_ids	

anime_df, num_users, num_anime, user_ids, anime_ids = df_encode(train_df)
print("Number of users :", num_users)
print("Number of anime :", num_anime)
anime_df.head()

def create_embeddings(n, K):
    """
    Creates a random numpy matrix of shape n, K with uniform values in (0, 11/K)
    n: number of items/users
    K: number of factors in the embedding 
    """
    return 11*np.random.random((n, K)) / K

def create_sparse_matrix(df, rows, cols, column_name="rating"):
    """ Returns a sparse utility matrix""" 
    return sparse.csc_matrix((df[column_name].values,(df['user_id'].values, df['anime_id'].values)),shape=(rows, cols))

anime_df, num_users, num_anime, user_ids, anime_ids = df_encode(train_df)
Y = create_sparse_matrix(anime_df, num_users, num_anime)

def predict(df, emb_user, emb_anime):
    """ This function computes df["prediction"] without doing (U*V^T).
    
    Computes df["prediction"] by using elementwise multiplication of the corresponding embeddings and then 
    sum to get the prediction u_i*v_j. This avoids creating the dense matrix U*V^T.
    """
    df['prediction'] = np.sum(np.multiply(emb_anime[df['anime_id']],emb_user[df['user_id']]), axis=1)
    return df

lmbda = 0.0002

def cost(df, emb_user, emb_anime):
    """ Computes mean square error"""
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_anime.shape[0])
    predicted = create_sparse_matrix(predict(df, emb_user, emb_anime), emb_user.shape[0], emb_anime.shape[0], 'prediction')
    return np.sum((Y-predicted).power(2))/df.shape[0] 

def gradient(df, emb_user, emb_anime):
    """ Computes the gradient for user and anime embeddings"""
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_anime.shape[0])
    predicted = create_sparse_matrix(predict(df, emb_user, emb_anime), emb_user.shape[0], emb_anime.shape[0], 'prediction')
    delta =(Y-predicted)
    grad_user = (-2/df.shape[0])*(delta*emb_anime) + 2*lmbda*emb_user
    grad_anime = (-2/df.shape[0])*(delta.T*emb_user) + 2*lmbda*emb_anime
    return grad_user, grad_anime

def gradient_descent(df, emb_user, emb_anime, iterations=2000, learning_rate=0.01, df_val=None):
    """ 
    Computes gradient descent with momentum (0.9) for given number of iterations.
    emb_user: the trained user embedding
    emb_anime: the trained anime embedding
    """
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_anime.shape[0])
    beta = 0.9
    grad_user, grad_anime = gradient(df, emb_user, emb_anime)
    v_user = grad_user
    v_anime = grad_anime
    for i in range(iterations):
        grad_user, grad_anime = gradient(df, emb_user, emb_anime)
        v_user = beta*v_user + (1-beta)*grad_user
        v_anime = beta*v_anime + (1-beta)*grad_anime
        emb_user = emb_user - learning_rate*v_user
        emb_anime = emb_anime - learning_rate*v_anime
        if(not (i+1)%50):
            print("\niteration", i+1, ":")
            print("train mse:",  cost(df, emb_user, emb_anime))
            if df_val is not None:
                print("validation mse:",  cost(df_val, emb_user, emb_anime))
    return emb_user, emb_anime

emb_user = create_embeddings(num_users, 3)
emb_anime = create_embeddings(num_anime, 3)
emb_user, emb_anime = gradient_descent(anime_df, emb_user, emb_anime, iterations=800, learning_rate=1)

def encode_new_data(valid_df, user_ids, anime_ids):
    """ Encodes valid_df with the same encoding as train_df.
    """
    df_val_chosen = valid_df['anime_id'].isin(anime_ids.keys()) & valid_df['user_id'].isin(user_ids.keys())
    valid_df = valid_df[df_val_chosen]
    valid_df['anime_id'] =  np.array([anime_ids[x] for x in valid_df['anime_id']])
    valid_df['user_id'] = np.array([user_ids[x] for x in valid_df['user_id']])
    return valid_df

print("before encoding:", valid_df.shape)
valid_df = encode_new_data(valid_df, user_ids, anime_ids)
print("after encoding:", valid_df.shape)

train_mse = cost(train_df, emb_user, emb_anime)
val_mse = cost(valid_df, emb_user, emb_anime)
print(train_mse, val_mse)

valid_df[70:80].head()
