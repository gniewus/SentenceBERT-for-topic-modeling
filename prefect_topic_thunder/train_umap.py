  
  
from variables import *

from modules.modeling import _preload_umap_reduce,_new_umap_reduce, cluster,CTFIDFVectorizer
import os,glob,json,pickle,dotenv,datetime,torch,sqlalchemy,time,umap,boto3,json
from prefect import task, Flow,context,case,Task,unmapped,task,apply_map,flatten
from prefect.tasks.aws.s3 import S3Upload,S3Download
from prefect.schedules import IntervalSchedule
from prefect.tasks.control_flow import ifelse
from prefect.core.parameter import Parameter
#from embedding_flow import read_config_file
import pandas as pd
import numpy as np
from task_library import *
from task_library import _build_query_to_select_rows_by_article_id
from prefect.tasks.mysql.mysql import MySQLExecute
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import sqlalchemy
from modules.utils import c_tf_idf,extract_topic_sizes,extract_top_n_words_per_topic
from prefect.engine.executors import LocalDaskExecutor
from prefect.environments import LocalEnvironment
from prefect.environments.storage import Docker



logger = context.get("logger")
#######################
schedule = IntervalSchedule(
    start_date=datetime.datetime.now() + datetime.timedelta(seconds=1),
    interval=datetime.timedelta(hours=4),
)


with Flow("Train & store UMAP model") as training_flow:
    TRAIN_N= Parameter("NUM_OF_ARTICLES_TO_TRAIN",default=50000)
    DB_URL = 'mysql+pymysql://{}:{}@{}/{}'.format(HUD_LOGIN, HUD_PASSWORD,HUD_HOST, HUD_DATABASE_NAME)
    # Task definitions
    config = read_config_file()

    embeddings_df = read_embeddings_table(DB_URL,TRAIN_N)
    embeddings_col = get_embeddings_column(embeddings_df)
    path_to_model = train_umap_model(DB_URL,embeddings_col,config,TRAIN_N=TRAIN_N)
    path_s3 = get_umap_s3_path_from_config(config,path_to_model)

    #open_file(path_to_model)
    upload_model_file(path_to_model,dest_path=path_s3,bucket="ci-topic-thunder")




if __name__ == "__main__":

    training_flow.executor = LocalExecutor()
    training_flow.storage = Local()
    training_flow.run_config = LocalRun(env={"SOME_VAR": "value"})
    training_flow.register(project_name='tt')
    training_flow.run()