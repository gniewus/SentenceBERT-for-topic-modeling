from variables import SRC_TABLE_NAME, EMBEDDINGS_TABLE,HUD_DATABASE_NAME,HUD_LOGIN, HUD_PASSWORD,HUD_HOST, HUD_DATABASE_NAME
from modules.utils import preprocess_articles_for_bert, load_labeled_data, remove_seo_title_marker
from sentence_transformers import SentenceTransformer, util, models
from prefect import task, Flow, context, case, Task ,Parameter
from prefect.schedules import IntervalSchedule, CronSchedule

from prefect.storage.local import Local


import pandas as pd
from prefect.run_configs.docker import DockerRun
from prefect.engine.executors import LocalExecutor
from prefect.run_configs import LocalRun

from task_library import GetParameterTask, get_embeddings_column,reduce_dim,read_config_file,read_embeddings_table, _get_new_articles,clean_text
from task_library import  check_if_df_not_empty, normalize, join_embeddings_with_aritcle_ids, _save_embeddings_to_table, SBERT_Pretrained
from task_library import  create_embeddings_table, store_reduced_embeddings,sync_model_with_s3,check_if_table_exists,create_embeddings_table
from datetime import timedelta, datetime

logger = context.get("logger")

schedule = IntervalSchedule(
    start_date=datetime.utcnow() + timedelta(seconds=1),
    interval=timedelta(minutes=1),
)


s = CronSchedule("0 */3 * * *")
if __name__ == "__main__":
    with Flow("Article Embedding and dim. reduction ", schedule=schedule ) as flow:
        ################
        # This flow is responsible for embedding and dim. reduction of the articles.
        # You can use this parameter for initial run and set it to None or 200 000 to process all the artiles
        limit = Parameter('window_size_to_check_for_new_articles', default = 1000)

        DB_URL = 'mysql+pymysql://{}:{}@{}/{}'.format(HUD_LOGIN, HUD_PASSWORD,HUD_HOST, HUD_DATABASE_NAME)
        # 1.Read the configuration file
        config = read_config_file()
        embeddings_table_created = create_embeddings_table(DB_URL,EMBEDDINGS_TABLE)

        # 2.Repzthoad the new <limit> articles from the database that are not yet embedded in the embeddings table
        articles = _get_new_articles(limit=limit,upstream_tasks=[embeddings_table_created,config])
        # 3. Check if there are any articles to process
        new_articles_present = check_if_df_not_empty(articles)



        with case(new_articles_present, True):
            # If there are new articles to be processed
            cleaned_articles_pdf = clean_text(articles)
            # TODO: Here comes a parameter conditioned switch to read config and evenetually downloload a fine-tuned model
            encoded = SBERT_Pretrained()(cleaned_articles_pdf)
            normalized = normalize(encoded)
            embeddings_df = join_embeddings_with_aritcle_ids(normalized, articles)
            #embeddings_df = read_embeddings_table(DB_URL,upstream_tasks=[store_original_embeddings] )
            get_params  = GetParameterTask(name="Grab Parameters for the UMAP",slug='get_umap_params')
            #logger.info(umap_parameters)
            #reduced_embeddings = reduce_dim_fresh(embeddings_df,umap_parameters)

            embeddings = get_embeddings_column(embeddings_df)
            umap_parameters = get_params(config=config,keys_array=['umap'])
            reduced_embeddings  = reduce_dim(embeddings,umap_parameters,upstream_tasks=[sync_model_with_s3(umap_parameters)])
            create_reduced_embeddings_table  = create_embeddings_table(DB_URL,upstream_tasks=[reduced_embeddings])


            stored_reduced_embeddings = store_reduced_embeddings(DB_URL,reduced_embeddings,embeddings_df, upstream_tasks=[create_reduced_embeddings_table])
            store_original_embeddings = _save_embeddings_to_table(DB_URL,embeddings_df, upstream_tasks=[stored_reduced_embeddings])

        with case(new_articles_present, False):
            logger.info("No new embeddings found. Tables {} and {} are in sync.".format(
                SRC_TABLE_NAME, EMBEDDINGS_TABLE))

        # db_connection.dispose()
        logger.info("DONE")

    #flow.executor = LocalExecutor()
    #flow.storage = Local()
    #flow.run_config = LocalRun(env={"SOME_VAR": "value"})
    #flow.register(project_name='tt')<
    flow.run(parameters=dict(window_size_to_check_for_new_articles=25000))
