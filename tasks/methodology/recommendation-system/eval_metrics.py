import turicreate as tc
from turicreate.toolkits.recommender.util import precision_recall_by_user
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import scipy.sparse as sp
import numpy as np


def personalization_score(recom):
    '''
    Scores the model on how unique recommendations are to individual users.
    A low score indicates recommendations are very similar. High scores are better.
    Returns:  
        Average recommendation similarity for all users (0 to 1).
    '''
    # get all unique items recommended
    uniq_recom = unique_recomendations(recom)
    # create matrix for recommendations
    rec_matrix = make_rec_matrix(recom.to_dataframe(), uniq_recom)
    rec_matrix_sparse = sp.csr_matrix(rec_matrix.values)
    # calculate similarity for every user's recommendation list
    similarity = cosine_similarity(X=rec_matrix_sparse, dense_output=False)
    # get indicies for upper right triangle w/o diagonal
    upper_right = np.triu_indices(similarity.shape[0], k=1)
    # calculate average similarity
    personalization = np.mean(similarity[upper_right])
    return (1-personalization)


def unique_recomendations(recom):
    recom_flattened = [r['item'] for r in recom]
    return list(set(recom_flattened))


def coverage_score(recom, num_items):
    '''
    Scores the model based on the percentage of items 
    that are recommended out of all available items.
    Returns:  
        Percentage out of 100
    '''
    uniq_recom = len(unique_recomendations(recom))
    return round(float(uniq_recom/num_items)*100, 2)


def precision_recall_rmse(testing_data, recom, model):
    precision_recall_by_user(testing_data, recom)
    eval_norm = tc.recommender.util.compare_models(
        testing_data, [model], model_names=['pearson'])
    return eval_norm[0]['precision_recall_overall'].to_dataframe(), eval_norm[0]['rmse_overall']


def make_rec_matrix(recom_df, uniq_recom):
    users = list(set(recom_df.user.values))
    rec_matrix = pd.DataFrame(index=range(
        len(users)), columns=uniq_recom)
    rec_matrix.fillna(0, inplace=True)
    for i in range(len(users)):
        rec_matrix.loc[i, recom_df[recom_df.user ==
                                   users[i]].item.values] = 1
    return rec_matrix
