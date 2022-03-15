from variables import *

from modules.modeling import _preload_umap_reduce,_new_umap_reduce, cluster,CTFIDFVectorizer
import os,glob,json,pickle,dotenv,datetime,torch,sqlalchemy,time,mlflow,umap,boto3
from prefect import task, Flow,context,case,Task
from prefect.tasks.aws.s3 import S3Upload,S3Download
from prefect.schedules import IntervalSchedule
from prefect.tasks.control_flow import ifelse
from prefect.core.parameter import Parameter
#from embedding_flow import read_config_file
import pandas as pd
import numpy as np
from task_library import *
from  task_library import sync_model_with_s3

logger = context.get("logger")
#######################
schedule = IntervalSchedule(
    start_date=datetime.datetime.now() + datetime.timedelta(seconds=1),
    interval=datetime.timedelta(hours=4),
)

# with Flow("Reducing the dimensionality of the embeddings stored in the database") as running_flow:
#     #######################
#     # Flow for reading the embedded articles and reducing dimensionality.
#     ################

    
#     DB_URL = 'mysql+pymysql://{}:{}@{}/{}'.format(HUD_LOGIN, HUD_PASSWORD,HUD_HOST, HUD_DATABASE_NAME)

#     # 1.Read the general config file
#     config = read_config_file()
#     # 2.Get embedded articles table
#     embeddings_df = read_embeddings_table(DB_URL)
#     # 3. Get Parameter Task for selection of diffrent parts of configuration JSON
#     get_params  = GetParameterTask(name="Grab Parameters for the UMAP",slug='get_umap_params')
#     # 4. Select the embeddings column from the DF
#     embeddings = get_embeddings_column(embeddings_df)
#     # 5. Use get_params to get UMAP parameters like location of the the UMAP model  
#     umap_parameters = get_params(config=config,keys_array=['umap'])
#     # 6. Use parameters and embeddings colum to reduce the dimenionality 
#     reduced_embeddings  = reduce_dim(embeddings,umap_parameters,upstream_tasks=[sync_model_with_s3(umap_parameters)])
#     # 7. Make sure the table is there for storage of columns
#     create_table  = create_embeddings_table(DB_URL)
#     # 8. Save the reduced stuff to the database table
#     store_reduced_embeddings(DB_URL,reduced_embeddings,embeddings_df, upstream_tasks=[create_table])
    
    #hdbscan_parameters = get_params(config=config,keys_array=['HDBSCAN','parameters'])
    #HDBSCAN(reduced_embeddings,hdbscan_parameters)
    #logger.info(reduced_embeddings)


# with Flow("Train & store UMAP model") as training_flow:
#     TRAIN_N= Parameter("NUM_OF_ARTICLES_TO_TRAIN",default=50000)
#     DB_URL = 'mysql+pymysql://{}:{}@{}/{}'.format(HUD_LOGIN, HUD_PASSWORD,HUD_HOST, HUD_DATABASE_NAME)
#     db_connection = sqlalchemy.create_engine(DB_URL)
#     # Task definitions
#     config = read_config_file()
    
#     embeddings_df = read_embeddings_table(DB_URL,TRAIN_N)
#     embeddings_col = get_embeddings_column(embeddings_df)
#     path_to_model = train_umap_model(db_connection,embeddings_col,config,TRAIN_N=TRAIN_N)
#     path_s3 = get_umap_s3_path_from_config(config,path_to_model)

#     #open_file(path_to_model)
#     upload_model_file(path_to_model,dest_path=path_s3,bucket="ci-topic-thunder")
    



# with Flow("Cluster with HDBSCAN ") as clustering_flow:
#     ######################
#     # Flow for reading the embedded articles and reducing dimensionality.
#     ################

    
#     DB_URL = 'mysql+pymysql://{}:{}@{}/{}'.format(HUD_LOGIN, HUD_PASSWORD,HUD_HOST, HUD_DATABASE_NAME)

#     # 1.Read the general config file
#     config = read_config_file()
#     # 2.Get embedded articles table
#     embeddings_df = read_embeddings_table(DB_URL,embeddings_table=TMP_EMBEDDINGS_TABLE_NAME)
#     # 3. Get Parameter Task for selection of diffrent parts of configuration JSON
#     get_params  = GetParameterTask(name="Load parameters from the file",slug='get_umap_params')
#     hdbscan_parameters = get_params(config=config,keys_array=["clustering",'HDBSCAN'])

#     # 4. We need to make sure that model is trained and avalible.
#     # In case the model is not on S3 it will thrown and error
#     model_synchronised = sync_model_with_s3(hdbscan_parameters)

#     case(model_synchronised,False):
#         # We need to train the model
#         logger.error("Model for HDBSCAN is not avalible on s3 ")

#     case(model_synchronised,True):  
#     # 5. Select the embeddings column from the DF
#         reduced_embeddings_column = get_embeddings_column(embeddings_df)
    

#     # 6. Load the pre-fitted HDBSCAN
    
get_unix_timestamps =task(lambda df: (df.article_created_at.astype(np.int64) // 10**9).to_numpy())



with Flow("Cluster with a fresh HDBSCAN ") as clustering_flow:
    ######################
    # Flow for reading the embedded articles and reducing dimensionality.
    ################

    DatababaseExecutor = MySQLExecute(HUD_DATABASE_NAME, HUD_LOGIN, HUD_PASSWORD,HUD_HOST)

    
    DB_URL = 'mysql+pymysql://{}:{}@{}/{}'.format(HUD_LOGIN, HUD_PASSWORD,HUD_HOST, HUD_DATABASE_NAME)

    # 1.Read the general config file
    config = read_config_file()
    # 2.Get embedded articles table
    embeddings_df = read_embeddings_table(DB_URL, article_ids=None, embeddings_table=TMP_EMBEDDINGS_TABLE_NAME)
    # 3. Get Parameter Task for selection of diffrent parts of configuration JSON
    get_params  = GetParameterTask(name="Load parameters from the file",slug='get_umap_params')    
    hdbscan_parameters = get_params(config=config,keys_array=['clustering','HDBSCAN','parameters'])
    reduced_embeddings = get_embeddings_column(embeddings_df)

    timestamps = get_unix_timestamps(embeddings_df)

    topic_labels = HDBSCAN(reduced_embeddings,timestamps,hdbscan_parameters)
    article_ids= task(lambda df: list(df.index.values))(embeddings_df)

    table_flushed = DatababaseExecutor(query='''truncate table {};'''.format(TOPIC_LABELS_TABLE))
    store_topic_lables(DB_URL,article_ids,topic_labels,upstream_tasks=[table_flushed])
    

if __name__ == "__main__":
    #training_flow.run()
    #running_flow.run()
    clustering_flow.run()