'''
Method for recommending items to users using collaborative filtering (item to item)
Input:  dataset in tsv format 'user  item    variable'
'''
import sys
import os
import turicreate as tc
from turicreate.toolkits.recommender.util import precision_recall_by_user, random_split_by_user
import numpy as np
import pandas as pd
import zipfile
import warnings
import mlflow
import time
import datetime
from mlflow.utils.logging_utils import eprint
from eval_metrics import unique_recomendations, coverage_score, precision_recall_rmse, make_rec_matrix, personalization_score


def preprocess(dataset):
    # subsample
    # items_included = dataset.item.value_counts().nlargest(50)
    # users_included = dataset.user.value_counts().nlargest(10000)
    # dataset_sampled = dataset[dataset.item.isin(items_included.index.values) &
    #                           dataset.user.isin(users_included.index.values)]
    matrix = pd.pivot_table(dataset, values='variable',
                            index='user', columns='item')
    return matrix


def normalize(matrix):
    matrix_normalized = (matrix-matrix.min())/(matrix.max()-matrix.min())
    d = matrix_normalized.reset_index()
    d.index.names = ['scaled_var']
    dataset_normalized = pd.melt(
        d, id_vars=['user'], value_name='scaled_var').dropna()
    return dataset_normalized


def split_data(dataset):
    train, test = random_split_by_user(
        tc.SFrame(dataset), item_id='item', user_id='user')
    return tc.SFrame(train), tc.SFrame(test)


def create_model(training_data):
    return tc.item_similarity_recommender.create(training_data,
                                                 user_id='user',
                                                 item_id='item',
                                                 target='scaled_var',
                                                 similarity_type='pearson')


def save_recom(recom):
    recom_df = recom.to_dataframe()
    st = datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y-%m-%d-%H:%M:%S')
    if not os.path.exists(os.path.join('output', st)):
        os.makedirs(os.path.join('output', st))
    recom_df.to_csv(os.path.join(
        'output', st, 'recommendations_by_user.csv'), index=False, header=True)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    # # parse args:
    dataset_file_name = sys.argv[2]
    data_path = os.path.join('data', dataset_file_name)
    recom_n = int(sys.argv[3])
    # check and create directories:
    if not os.path.exists(data_path):
        sys.exit("data file does not exist.")
    # load data:
    try:
        eprint('\nloading data...\n')
        with zipfile.ZipFile(data_path, 'r') as zip_ref:
            dataset = pd.read_csv(zip_ref.open(
                dataset_file_name.split('.zip')[0]), sep="\t", header=None, names=['user', 'item', 'variable'])
        zip_ref.close()
    except OSError as err:
        eprint("OS error:", err)
    except:
        eprint("unexpected error:", sys.exc_info()[0])
        raise
    else:
        eprint('\ndataset has', len(dataset), 'entries\n')
    # format dataset
    matrix = preprocess(dataset)
    # normalize data
    dataset_normalized = normalize(matrix)
    users_to_recommend = list(dataset_normalized.user.values)
    # split data into training and testing
    training_data, testing_data = split_data(dataset_normalized)
    num_items = len(list(set(training_data.to_dataframe().item.values)))
    # show raw data stats
    raw_variable_quantiles = np.quantile(
        dataset.variable.values, [0, .25, .5, .75, 1])
    eprint('\nquantiles:', raw_variable_quantiles, '\n')
    with mlflow.start_run():
        try:
            # train and store model
            model = create_model(training_data)
            # create recomendations
            recom = model.recommend(users=users_to_recommend, k=recom_n)
        except:
            eprint('run failed')
            raise
        else:
            eprint('\nsaving recommendations...\n')
            save_recom(recom)
            # calculate metrics
            eprint('\n*** calculating metrics ***\n')

            eprint('\ncalculating precision, recall and rmse...\n')
            precision_recall, rmse = precision_recall_rmse(
                testing_data, recom, model)

            eprint('\ncalculating coverage score...\n')
            coverage = coverage_score(recom, num_items)
            eprint('coverage:', coverage, "%")

            eprint('\ncalculating personalization score...\n')
            personalization = personalization_score(recom)
            eprint('personalization:', personalization, '\n')

            # record run
            mlflow.log_param("num recommendations", recom_n)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("coverage", coverage)
            mlflow.log_metric("personalization", personalization)
