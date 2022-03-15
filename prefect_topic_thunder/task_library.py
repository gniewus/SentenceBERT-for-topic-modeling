from modules.utils import preprocess_articles_for_bert, load_labeled_data, remove_seo_title_marker
from modules.utils import c_tf_idf, extract_topic_sizes, extract_top_n_words_per_topic
from modules.modeling import _preload_umap_reduce, _new_umap_reduce, cluster
import os
import json
import pickle
import torch
import sqlalchemy
import umap
import boto3
import botocore
from sentence_transformers import SentenceTransformer, models
from collections import Counter
from variables import *
from prefect import task, Flow, context, case, Task
from prefect.tasks.mysql.mysql import MySQLExecute
from prefect.run_configs import LocalRun
from prefect.schedules import IntervalSchedule, CronSchedule
from prefect.storage.local import Local
from prefect.engine.executors import LocalExecutor
from prefect.run_configs.docker import DockerRun
from prefect.core.parameter import Parameter
import pandas as pd
import numpy as np


print(">>> Using: '{}' database".format(HUD_DATABASE_NAME))
logger = context.get("logger")

# TODO: Switch from JSON to YAML becasue of comments in the file


def _build_query_to_select_rows_by_article_id(article_ids, table_name: str) -> str:
    """Utility function to build a query to select row by article ids. 
    It takes a list of of article_ids and returns all the rows with matching article_uid.
    Can contain duplicates.
    It allows to pass integer X as article_ids which select all with a limit on X """
    if type(article_ids) == str:
        article_ids = [article_ids]

    query = "SELECT * FROM {} t ".format(table_name)
    if isinstance(article_ids, (np.ndarray, np.generic, list)):
        article_ids = ["'{}'".format(uid) for uid in article_ids]
        query += " WHERE t.article_uid in ({})".format(', '.join(article_ids))

    if "embeddings" in table_name:
        query += 'ORDER BY t.article_created_at DESC '

    if type(article_ids) == int:
        query += "LIMIT {}".format(article_ids)
    query += ";"
    return query


@task(name="Read the {} table".format(EMBEDDINGS_TABLE))
def read_embeddings_table(DB_URL, article_ids=None, embeddings_table=EMBEDDINGS_TABLE):
    db_connection = sqlalchemy.create_engine(DB_URL)

    if article_ids is None:
        df = pd.read_sql_table(embeddings_table, con=db_connection,
                               index_col='article_uid', parse_dates=['article_created_at'])
    elif type(article_ids) == list or type(article_ids) == int:
        query = _build_query_to_select_rows_by_article_id(
            article_ids, embeddings_table)
        df = pd.read_sql(query, con=db_connection, index_col="article_uid", parse_dates=[
                         'article_created_at'])
    else:
        logger.error("Wrong type of article_ids")
        raise Exception("Wrong type of article_ids")

    df.embedding = df.embedding.apply(pickle.loads)
    logger.info("Succesfully read {} article embeddings from '{}'".format(
        df.shape[0], embeddings_table))
    return df


@task(name=" Read the current umap model s3 path from the config file")
def get_current_umap_model_s3_path_from_config(config):
    return config['umap']['main_model_s3_path']


@task(name="Reduce Dim with a preloaded UMAP")
def reduce_dim(array: np.array, umap_params: dict) -> np.array:
    # Path to the pkl file of the UMAP
    if len(array.shape) == 1:
        array = np.vstack(array)
    model_fname = os.path.basename(umap_params['main_model_s3_path'])
    model_local_path = os.path.join(umap_params['location'], model_fname)
    return _preload_umap_reduce(array, model=model_local_path)


@task(name="Apply pre-fitted HDBSCAN clustering.")
def HDBSCAN(embeddings,timestamps=None, hdbscan_parameters={}):
    if len(embeddings.shape) == 1:
        embeddings = np.vstack(embeddings)
    if timestamps is not None:#
        from sklearn.preprocessing import MinMaxScaler
        scaler= MinMaxScaler()
        ts=scaler.fit_transform(np.vstack(timestamps))
        embeddings = np.hstack((embeddings,ts))
    labels = cluster(embeddings, **hdbscan_parameters)
    # ctr = Counter(label)
    # for key,value in ctr.most_common():
    #     if value > 100:
    #         indicies = np.argwhere(racing_labels == 7)

    return labels


@task(name="Reduce Dim with a new UMAP")
def reduce_dim_fresh(array: np.array, params: dict) -> np.array:
    """ Reduce dimensionality using a newly fitted UMAP model and retunr array. """
    return _new_umap_reduce(embeddings=array, args=params)


@task(name="Get Embeddings Column as array")
def get_embeddings_column(dataframe: pd.DataFrame) -> np.array:
    return dataframe.embedding.values


@task(name="Store topic labes in a table")
def store_topic_lables(DB_URL, article_ids, topic_labels):
    columns = ['article_uid', 'topic_label']
    df = pd.DataFrame(zip(article_ids, topic_labels), columns=columns)
    logger.info("Succesfully clustered {} articles".format(len(article_ids)))
    logger.info(df.groupby(topic_labels).count())
    df = df.set_index('article_uid')

    df.to_sql(TOPIC_LABELS_TABLE, if_exists="append", con=sqlalchemy.create_engine(
        DB_URL), dtype={'article_uid': sqlalchemy.types.CHAR(64)})


@task(name='Upload model file to S3')
def upload_model_file(model_path, bucket, dest_path):

    client = boto3.client('s3')
    logger.info(
        'Uploading {} to s3://{}/{}'.format(model_path, bucket, dest_path))
    try:
        response = client.upload_file(
            model_path, "ci-topic-thunder", dest_path)
        logger.info(response)
    except Exception as e:
        logger.error("Error uploading", e)


@task(name="Train the UMAP model")
def train_umap_model(DB_URL, embeddings, params, TRAIN_N):
    " This function train the UMAP model and dumps it to the path specified in the configuration file."
    db_connection = sqlalchemy.create_engine(DB_URL)

    if len(embeddings.shape) == 1:
        emb = np.vstack(embeddings)

    umap_params = params['umap']["parameters"]

    MODEL_NAME = params['model']['model_name']
    if "/" in MODEL_NAME:
        MODEL_NAME = MODEL_NAME.split("/")[1]

    fitted_umap_clustering = umap.UMAP(n_neighbors=umap_params['n_neighbors'],
                                       min_dist=0.01, random_state=0, metric='cosine',
                                       n_components=umap_params['n_components']).fit(emb)
    logger.info('UMAP model was fitted on last {} articles with following params.'.format(
        TRAIN_N), umap_params)
    if TRAIN_N > 100000:
        training_size = str(TRAIN_N)[:3]+"k"
    elif TRAIN_N > 1000:
        training_size = str(TRAIN_N)[:2]+"k"
    else:
        training_size = str(TRAIN_N)

    pkl_filename = "umap_{}_{}-neighbors_{}-comps.pkl".format(training_size,
                                                              umap_params['n_neighbors'],
                                                              umap_params['n_components'])

    if not os.path.exists("{}/{}".format(params['umap']['location'], MODEL_NAME)):
        p = "{}/{}".format(params['umap']['location'], MODEL_NAME)
        os.makedirs(p, exist_ok=True)

    model_local_path = "{}/{}/{}".format(
        params['umap']['location'], MODEL_NAME, pkl_filename)
    with open(model_local_path, 'wb') as file:
        pickle.dump(fitted_umap_clustering, file)

    logger.info("--- Model stored localy in '{}'".format(model_local_path))
    return model_local_path


@task
def get_umap_s3_path_from_config(config, path_to_model_pkl):
    fname = os.path.basename(path_to_model_pkl)
    path = os.path.join(config['umap']['s3_path'], fname)
    logger.info("Returning '{}' as storage path".format(path))
    return path


class GetParameterTask(Task):
    """ Class for accessing the parameters from the dictionary. """

    def __init__(self, task_kwargs={}, **kwargs):
        super().__init__(**task_kwargs)

    @staticmethod
    def nested_get(dic, keys):
        for key in keys:
            dic = dic[key]
        return dic

    def run(self, config, keys_array):
        logger.info("Accessing the'{}' attributes from config file".format(
            ' > '.join(keys_array)))
        self.config = config
        self.keys = keys_array
        try:
            return self.nested_get(self.config, self.keys)
        except KeyError as key:
            logger.error("Could not find key in config dict")


@task(name="Check If table exists")
def check_if_table_exists(DB_URL, table_name):
    db_connection = sqlalchemy.create_engine(DB_URL)
    return db_connection.dialect.has_table(db_connection, table_name)


@task(name="Create table if not exists")
def create_embeddings_table(DB_URL, table_name=TMP_EMBEDDINGS_TABLE_NAME):
    ''' Create a temporary table for embeddings with reduced dimensionality. It takes the table name from the ENV variable'''
    db_connection = sqlalchemy.create_engine(DB_URL)
    sql = """ create table IF NOT EXISTS {}
        (
            article_uid        char(64)                             PRIMARY KEY,
            embedding          blob                                 null,
            article_created_at datetime                             null,
            date               datetime default current_timestamp() not null,
            constraint {}_article_uid_uindex
                unique (article_uid)
        ); """.format(table_name, table_name)

    db_connection.execute(sql)
    logger.info('Created table {}'.format(table_name))


@task
def sync_model_with_s3(model_params):
    """" Takes the paramters and syncronises the model with the main_model_s3_path. 
    It just downloads the model if it doesn't exist."""
    model_path_on_s3 = model_params['main_model_s3_path']
    local_path = model_params['location']

    fname = os.path.basename(model_params['main_model_s3_path'])
    client = boto3.client('s3')
    try:
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        local_path_to_model = os.path.join(local_path, fname)
        if not os.path.isfile(local_path_to_model):
            #There is nothing 
            
            logger.info('Downloading model {} from S3 to {}'.format(
                fname, local_path_to_model))
            client.download_file('ci-topic-thunder',
                                 model_path_on_s3, local_path_to_model)
            logger.info(
                's3://{}/{} downloaded successfuly'.format(S3_BUCKET, model_path_on_s3))
        elif not os.path.getsize(local_path_to_model) == client.head_object(Bucket=S3_BUCKET, Key=model_path_on_s3)["ContentLength"]:
            os.remove(local_path_to_model)
            logger.info('Downloading model {} from S3 to {}'.format(
                fname, local_path_to_model))
            client.download_file('ci-topic-thunder',
                                 model_path_on_s3, local_path_to_model)
            logger.info(
                's3://{}/{} downloaded successfuly'.format(S3_BUCKET, model_path_on_s3))
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise e

    return True


@task
def join_embeddings_with_aritcle_ids(embeddings: np.array, df: pd.DataFrame):
    article_ids = df.index
    created_at = df.created_at.values
    return pd.DataFrame(zip(embeddings, created_at),
                        columns=['embedding', 'article_created_at'],
                        index=article_ids)


@task(name='Clean seo_title and text.')
def clean_text(data: pd.DataFrame, **kwargs):
    data['text'] = data['seo_title'].apply(
        lambda x: remove_seo_title_marker(x, True)) + ". " + data["text"]
    return preprocess_articles_for_bert(data, **kwargs)


class SBERT_Pretrained(Task):
    def __init__(self, model_name='T-Systems-onsite/bert-german-dbmdz-uncased-sentence-stsb', task_kwargs={}, **kwargs):

        word_embedding_model = models.Transformer(
            model_name, max_seq_length=512)

        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        print(kwargs)
        self.model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model])
        logger.info("Loaded pretrained model {}".format(model_name))
        super().__init__(**task_kwargs)

    def run(self, articles: list, batch_size=32):
        if articles:
            return self.model.encode(
                articles,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=False
            )
        else:
            logger.warning("Warning empty list!")
            return articles


@task(name='Normalize vectors')
def normalize(array):
    if not torch.is_tensor(
            array):
        array = torch.tensor(array)
    normalized = array / array.norm(dim=1)[:, None]

    return normalized.cpu().numpy()


@task(name="Read configuration File")
def read_config_file():
    config_path = "./config.json" if os.path.isfile(
        "./config.json") else "./topic_thunder/config.json"
    try:
        with open(config_path, "r") as config:
            config_body = json.load(config)
            logger.info(config_body)
            return config_body
    except FileNotFoundError as err:
        from glob import glob
        logger.warning(glob('./*'))
        raise FileNotFoundError("Could not find config file", err)


@task(name="Check for new artcles to Embedd")
def _get_new_articles(limit=250):
    db_connection = sqlalchemy.create_engine(DB_URL)
    """ Pull latest articles from the database. If limit is None then it pulls all articles.
        param: limit - tells how far in time shall we go back 
    """
    logger.info(">>> Pulling latest articles from the database")
    SRC_SQL_QUERY = 'SELECT * FROM {} t '.format(SRC_TABLE_NAME)
    SRC_SQL_QUERY += 'ORDER BY t.created_at DESC'

    if limit:
        SRC_SQL_QUERY += ' LIMIT {}'.format(limit)

    TARGET_SQL_QUERY = "SELECT t.article_uid FROM {} t".format(
        EMBEDDINGS_TABLE)
    logger.info(TARGET_SQL_QUERY)
    # TODO: Find a more efficient way to check if article is already embedded
    df = pd.read_sql_query(SRC_SQL_QUERY, con=db_connection, parse_dates=[
                           "created_at"]).set_index("article_uid")
    logger.info("Table: {} read sucessfuly".format(SRC_SQL_QUERY))
    target_df = pd.read_sql_query(
        TARGET_SQL_QUERY, con=db_connection, parse_dates=["article_created_at"])

    # TODO: Prep of befeore reading data. Casting?
    df.drop(target_df['article_uid'], inplace=True, errors="ignore")
    logger.info(
        "Found {} new articles - {} are already in the db".format(df.shape[0], target_df.shape[0]))

    return df


@task(name="Store reduced Embeddings ")
def store_reduced_embeddings(DB_URL, reduced_embeddings, embeddings_df):
    """ Store reduced embeddings in a temporary table labeled  used form UMAP"""
    db_connection = sqlalchemy.create_engine(DB_URL)
    tmp_df = embeddings_df.copy()
    tmp_df.embedding = pd.Series([ np.array(x) for  x in reduced_embeddings],index=tmp_df.index)
    tmp_df.embedding = tmp_df.embedding.apply(pickle.dumps)
    tmp_df.to_sql(TMP_EMBEDDINGS_TABLE_NAME, con=db_connection,
                         if_exists="append")
    logger.info("Stored {} embeddings in {}".format(
        tmp_df.shape[0], TMP_EMBEDDINGS_TABLE_NAME))


@task(name="Store embedded articles in the table ")
def _save_embeddings_to_table(DB_URL, results_df):
    # print(results_df)
    db_connection = sqlalchemy.create_engine(DB_URL)
    results_df.embedding = results_df.embedding.apply(pickle.dumps)
    results_df.to_sql(name=EMBEDDINGS_TABLE,
                      if_exists="append", con=db_connection)
    logger.info("Stored {} new embeddings in {}".format(
        results_df.shape[0], EMBEDDINGS_TABLE))
    return True


@task(name="CheckIfDataFrameIsEmpty")
def check_if_df_not_empty(df): return (not df.empty)
