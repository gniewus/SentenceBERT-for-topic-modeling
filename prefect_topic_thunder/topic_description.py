from variables import *

from modules.modeling import _preload_umap_reduce,_new_umap_reduce, cluster,CTFIDFVectorizer
import os,glob,json,pickle,dotenv,datetime,torch,sqlalchemy,time,mlflow,umap,boto3,json
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import sqlalchemy
from modules.utils import c_tf_idf, extract_topic_sizes, extract_top_n_words_per_topic


logger = context.get("logger")
#######################
schedule = IntervalSchedule(
    start_date=datetime.datetime.now() + datetime.timedelta(seconds=1),
    interval=datetime.timedelta(hours=4),
)


@task
def get_topic_ids_table(DB_URL) -> np.array:
    return pd.read_sql_table(TOPIC_LABELS_TABLE, con=sqlalchemy.create_engine(DB_URL))


get_topic_ids_list_from_df = task(
    lambda df: df.topic_label.unique(), name="get_topic_ids_from_df")

select_article_ids_based_on_topic_label = \
    task(lambda df, topic_label:
         df.loc[df['topic_label'] == topic_label].article_uid.values, name='select_article_ids')


@task
def print_values(value):
    logger.info(value)
    return value


@task(name="Get_raw_entites_of_topic_for_article_uids")
def get_named_entities_table_per_article_ids(DB_URL, topic_label, list_of_article_uids):
    """ Returns a list of DFs for all the lists of article_ids for every topic"""
    # if str(topic_label) == '-1':
    #    return pd.DataFrame()
    start = time.time()
    logger.info("{} articles in topic group".format(len(list_of_article_uids)))
    query = _build_query_to_select_rows_by_article_id(
        list_of_article_uids, "raw_article_entities")
    #query = query.replace("*","article_uid,number_of_mentions,type_of_text,name,salience")
    logger.info(" Executing query {} ...".format(query[:256]))
    df = pd.read_sql_query(query, con=sqlalchemy.create_engine(DB_URL),
                           parse_dates=["created_at"]).set_index("article_uid")
    logger.info("Execution is finished in: {:.2f}".format(time.time() - start))
    if df.empty:
        logger.warning("There are no named entities found for these articles: {}".format(
            list_of_article_uids))

    return df


@task(name="ParseNamedEntities")
def parse_named_entities_df(df_with_json):
    res = []
    if df_with_json.empty:
        logger.warning("Empty DF with Named Entites ")
        return []
    a = [parse_google_named_entities(text)
         for text in df_with_json["text"].values]
    b = [parse_google_named_entities(seo_title)
         for seo_title in df_with_json["seo_title"].values]
    c = [parse_google_named_entities(
        kicker_headline) for kicker_headline in df_with_json["kicker_headline"].values]

    res.append(a)
    res.append(b * 3)
    res.append(c * 6)

    logger.info("Total named entities from text,seo and kicker are: {}".format(
        len(flatten_list(res))))
    return flatten_list(res)


def parse_google_named_entities(json_obj, deduplicate=False):
    # Parse objets and deduplicate list of them
    def parse_named_entity(ne):

        if "type" in ne and "text" in ne:
            if type(ne["text"]) == str:
                return [{"text": ne["name"], "type":ne["type"], "salience":ne["salience"]}]
            elif type(ne["text"]) == dict:
                return [{"text": ne["text"]['content'], "type":ne["type"], "salience":ne["salience"]}]
        elif "type" not in ne and "mentions" in ne:
            return [{"text": men["text"]['content'], "type":men['type'], "salience":ne['salience']} for men in ne["mentions"]]

        elif 'name' in ne and 'type' and ne:
            return [{"text": ne["name"], "type":ne["type"], "salience":ne["salience"]}]
        else:
            return [{"text": ne["name"], "type":None, "salience":ne["salience"]}]

    # print(json.loads(json_obj)[0]["entities"])
    obj = json.loads(json_obj)
    list_of_objects = []

    for ne in obj[0]["entities"]:
        tmp = parse_named_entity(ne)
        list_of_objects.extend(tmp)

    if not deduplicate:
        list_of_objects = deduplicate_list(list_of_objects)

    return list_of_objects


@task(name="Filter out NER that are not informative")
def filter_out_named_entities(list_of_NER_dicts):
    list_of_NER_dicts = flatten_list(list_of_NER_dicts)
    # Remove None types
    def func(d): return d['type'] != None and 'bild' not in d['text'].lower(
    ) and 'UNKNOWN' not in d['type'] and 'NUMBER' not in d['type']
    list_of_NER_dicts = list(filter(func, list_of_NER_dicts))

    #list_of_objects = list(filter(lambda d: d['type'] != "OTHER", list_of_NER_dicts))
    #list_of_objects = list(filter(lambda d: "bild" not in ne["name"].lower(), list_of_objects))

    return list_of_NER_dicts


@task(name="Apply Class TFIDF")
def apply_class_tfidf(filtered_named_entities, article_uid_per_class):
    documents_total = sum([len(el) for el in article_uid_per_class])
    vocabulary = flatten_list(filtered_named_entities)

    count_vectorizer = CountVectorizer(lowercase=False, tokenizer=lambda x: [
                                       el.lower().replace(".", "") for el in x])
    term_document_matrix = count_vectorizer.fit_transform(
        filtered_named_entities)
    words = count_vectorizer.get_feature_names()
    ctfidf = CTFIDFVectorizer().fit_transform(
        term_document_matrix, n_samples=documents_total)
    return [[words[index] for index in ctfidf.toarray()[label].argsort()[-10:]] for label, val in enumerate(filtered_named_entities)]


# Flatten list of lists
def flatten_list(t): return [item for sublist in t for item in sublist]

# Deduplicated list of objects
def deduplicate_list(list_of_objects): return [i for n, i in enumerate(
    list_of_objects) if i not in list_of_objects[n + 1:]]  


stringify_NER_dict = task(lambda list_of_objects: [
                          ne['text'] for ne in list_of_objects])


@task(name="Zip and save to the database")
def zip_and_save_topic_descriptions_to_db(DB_URL, topic_descriptions, topic_labels, articles_per_topic_id):
    topic_sizes = [len(el) for el in articles_per_topic_id]
    topic_descriptions = [", ".join(liste) for liste in topic_descriptions]
    data = list(zip(topic_labels, topic_descriptions, topic_sizes))
    df = pd.DataFrame(data, columns=[
                      'topic_label', 'topic_description', 'topic_size']).set_index('topic_label')
    df.topic_description.astype(str)
    # ,dtype={"topic_description":sqlalchemy.types.String(256)})
    return df.to_sql(TOPIC_DESC_TABLE, con=sqlalchemy.create_engine(DB_URL), if_exists="replace")


with Flow("Build topic descriptions") as topic_desc_flow:
    DB_URL = 'mysql+pymysql://{}:{}@{}/{}'.format(
        HUD_LOGIN, HUD_PASSWORD, HUD_HOST, HUD_DATABASE_NAME)
    DatababaseExecutor = MySQLExecute(
        HUD_DATABASE_NAME, HUD_LOGIN, HUD_PASSWORD, HUD_HOST)

    ##########################
    # Flow to build up topic descriptions from the topic labels
    # 1. Read topic_labels table - <article_uid, topic_label>
    topic_labels_table = get_topic_ids_table(DB_URL)
    #topic_labels_table = task(lambda df: df.head(2500))(topic_labels_table)

    # 2. Get all unique topic labels
    topic_labels_list = get_topic_ids_list_from_df(topic_labels_table)
    # 3. For every topic label get a DataFrame with named entites response JSON from the raw_article_entities
    list_of_article_ids_per_topic_label = select_article_ids_based_on_topic_label.map(
        topic_label=topic_labels_list, df=unmapped(topic_labels_table))
    # 4. Map every DF with a function that return a list of parameters
    list_of_entities_df = get_named_entities_table_per_article_ids.map(
        DB_URL=unmapped(DB_URL),
        topic_label=topic_labels_list,
        list_of_article_uids=list_of_article_ids_per_topic_label)

    config = read_config_file()
    # print_values(list_of_entities_df)
    # 5.Parse every DF to a list of words df
    parse_named_entities_json_to_dict_list = parse_named_entities_df.map(
        df_with_json=list_of_entities_df)
    # print_values(parse_named_entities_json_to_dict_list)

    # 6. Filter out the NUMBER and COMMON types out of Named Entities
    filtered_named_entities = filter_out_named_entities.map(
        parse_named_entities_json_to_dict_list)
    # print_values([filtered_named_entities,topic_labels_list])

    # 7. NER Dict to string representation
    stringified_named_entities = stringify_NER_dict.map(
        filtered_named_entities)

    # 8. Class TFIDF on named entities and return to 10 words
    top_k_words_per_topic = apply_class_tfidf(
        stringified_named_entities, list_of_article_ids_per_topic_label)
    # print_values(top_k_words_per_topic)

    # 9. Make sure that the table is avalible and ready
    table_created = DatababaseExecutor(query=""" create table IF NOT EXISTS {}
        (
            topic_label        int(6)                             PRIMARY KEY,
            topic_desc         text                                 null,
            topic_size int                             null,
            date               datetime default current_timestamp() not null,
            constraint {}_topic_label_uindex
                unique (topic_label)
        ); """.format(TOPIC_DESC_TABLE, TOPIC_DESC_TABLE), upstream_tasks=[top_k_words_per_topic])
    table_flushed = DatababaseExecutor(query='''DELETE FROM {};'''.format(TOPIC_DESC_TABLE))

    # 10. Store the stuff in the database
    zip_and_save_topic_descriptions_to_db(DB_URL, top_k_words_per_topic, topic_labels_list,
                                          list_of_article_ids_per_topic_label, upstream_tasks=[table_created,table_flushed])


if __name__ == "__main__":
    # training_flow.run()
    topic_desc_flow.run()
    # flow.run()
