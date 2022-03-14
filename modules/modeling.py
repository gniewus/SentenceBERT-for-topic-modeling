import time
import umap

import numpy as np
import plotly.express as px
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from modules import utils
import pandas as pd
import pickle
from sklearn.preprocessing import normalize


# tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")


def get_sentence_embeddings(array, sbert_worde_embedding_model, pooling_mode_max_tokens=False, output_model=False, **kwargs):
    """
        This function takes array of (preprocessed) sentence embeddings, a model and one paramter and returns the embeddings
    :param array:
    :param sbert_worde_embedding_model:
    :param pooling_mode_max_tokens:
    :return:
    """
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(sbert_worde_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=pooling_mode_max_tokens)

    # join BERT model and pooling to get the sentence transformer
    model = SentenceTransformer(modules=[sbert_worde_embedding_model, pooling_model])

    start_time = time.time()
    embeddings = model.encode(array, show_progress_bar=True, **kwargs)
    print("--- Embedding dimension {}".format(embeddings.shape[1]))
    print("--- %d Documnets encoded %s seconds ---" % (len(array), (time.time() - start_time)))
    if not output_model:
        return embeddings
    else:
        return embeddings, model


def umap_for_viz(embeddings, df, n_neighbors, min_dist):
    """
    This function performs dimensionality reduction for the visualization.
    :param embeddings: array of embedded vectors
    :param df:  Articles DF with dates and headlines
    :param n_neighbors:
    :param min_dist:
    :return:
    """
    umap_data = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist, metric='cosine', random_state=0) \
        .fit_transform(embeddings)
    res = pd.DataFrame(umap_data, columns=['x', "y"], index=df.index)
    res["headline"] = df["headline"].values
    res["created_at"] = df["created_at"].values

    return res


import os


def _splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def load_umap_viz_and_reduce(embeddings,
                             viz_model="bert-german-dbmdz-uncased-sentence-stsb/umap_viz_100_19-neighbors_0.01-min-dist.pkl"):
    # Load the Model back from file
    return _preload_umap_reduce(embeddings, viz_model)


def _preload_umap_reduce(embeddings, model):
    start_time = time.time()

    viz_model_path = "../models/" + model
    with open(viz_model_path, 'rb') as file:
        fitted_umap_viz = pickle.load(file)
    reduced = fitted_umap_viz.transform(embeddings)
    print("--- UMAP Loaded in %s seconds \n--- Reduced dimensionality to %s ." % (
        (time.time() - start_time), reduced.shape))

    return reduced


def _new_umap_reduce(embeddings,args={}):
    start_time = time.time()
    params = {"n_neighbors": 10, "n_components": 384,
              "metric": 'cosine', "random_state": 0}
    for (k, v) in args.items():
        params[k] = v
    umap_data = umap.UMAP(**params).fit_transform(
        embeddings)
    print("--- UMAP fitted in %s seconds \n--- Reduced dimensionality to %s ." % (
        (time.time() - start_time), umap_data.shape))

    return umap_data


def load_umap_and_cluster(embeddings, umap_model,
                          viz_model="bert-german-dbmdz-uncased-sentence-stsb/umap_viz_100_19-neighbors_0.01-min-dist.pkl",
                          **kwargs):
    """
    This function takes embeddings, loads the given pretrained UMAP models, and performs the clustering.
    $ready to be vizualized,
    :param embeddings:
    :param kwargs:
    :return:
    """
    # Load the Model back from file
    start_time = time.time()

    viz_model_path = "../models/" + viz_model
    dim_reduction_model_path = "../models/" + umap_model

    with open(viz_model_path, 'rb') as file:
        fitted_umap_viz = pickle.load(file)

    with open(dim_reduction_model_path, 'rb') as file:
        fitted_umap_clustering = pickle.load(file)

    print("--- UMAP Loaded in %s seconds ---" % (time.time() - start_time))

    st = time.time()
    umap_data = fitted_umap_viz.transform(embeddings)

    umap_embeddings = fitted_umap_clustering.transform(embeddings)
    print(">> Reduced dimensionality from {} to {} ...".format(embeddings.shape[1], umap_embeddings.shape[1]),
          end="\r")
    # Overriding default parameters
    params = {"min_cluster_size": 3, "min_samples": 2, "alpha": 1.0, "cluster_selection_epsilon": 0.14,
              "allow_single_cluster": True,
              "metric": 'euclidean',
              "cluster_selection_method": 'leaf',
              "approx_min_span_tree": True}

    for (k, v) in kwargs.items():
        params[k] = v

    print(">> Clustering...", end="\r")
    print(umap_embeddings.shape)

    clusters = HDBSCAN(**params).fit_predict(umap_embeddings)
    print(">> --- Done in {:.1f} seconds ---".format(time.time() - st))
    print(">> Silhouette Coefficient: {}".format(metrics.silhouette_score(umap_embeddings, clusters)))
    return umap_data, clusters


def cluster_and_reduce(embeddings, one_day=False, n_neighbors=15, n_components_clustering=384, **kwargs):
    st = time.time()
    umap_data = umap.UMAP(n_neighbors=n_neighbors, n_components=3, metric='cosine', random_state=0).fit_transform(
        embeddings)
    print(">> Reducing dimensionality from {} to {} ...".format(embeddings.shape[1], str(n_components_clustering)),
          end="\r")
    if len(embeddings) > n_components_clustering:
        umap_embeddings = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.0,
                                    n_components=n_components_clustering, random_state=0,
                                    metric='cosine').fit_transform(embeddings)
    else:
        umap_embeddings = umap.UMAP(n_neighbors=n_neighbors,
                                    n_components=n_components_clustering, random_state=0,
                                    metric='cosine', init="random").fit_transform(embeddings)

    params = {"min_cluster_size": 3, "min_samples": 2,
              "alpha": 1.0, "cluster_selection_epsilon": 1, "metric": 'euclidean',
              "cluster_selection_method": 'leaf',
              "approx_min_span_tree": True}

    for (k, v) in kwargs.items():
        params[k] = v

    print(">> Clustering...", end="\r")
    clusters = HDBSCAN(**params).fit_predict(umap_embeddings)
    print(">> --- Done in {:.1f} seconds ---".format(time.time() - st))
    print(">> Silhouette Coefficient: {}".format(metrics.silhouette_score(umap_embeddings, clusters,metric='cosine')))

    return umap_data,umap_embeddings ,clusters




def cluster(embeddings,**kwargs):
    st = time.time()

    params = {"min_cluster_size": 3, "min_samples": 2,
              "alpha": 1.0, "cluster_selection_epsilon": 0.14, "metric":'euclidean',
              "cluster_selection_method": 'leaf',
              "approx_min_span_tree": True}

    for (k, v) in kwargs.items():
        params[k] = v

    print(">> Clustering...", end="\r")
    clusters = HDBSCAN(**params).fit_predict(embeddings)
    print(">> --- Done in {:.1f} seconds ---".format(time.time() - st))
    print(">> Silhouette Coefficient: {}".format(metrics.silhouette_score(embeddings, clusters,metric='cosine')))
    return  clusters

def scatter_plot(result, save_fig=False, hover_data=["created_at"], **kwargs):
    if "labels" in result.columns.to_list():
        result["labels"] = result.labels.apply(str)
    elif "topic_number" in result:
        result["labels"] = result.topic_number.apply(str)

    fig = px.scatter(result, x="x", y="y", hover_name="headline", hover_data=hover_data, color="labels",
                      **kwargs)
    fig.update_traces(marker=dict(size=9,
                                  line=dict(width=0.15,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig["layout"].pop("updatemenus")

    if save_fig:
        fig.update_layout(height=500)  # Dumping smaller images for convience
        fig.write_html("./tmp_scatter_plot.html")
    else:
        fig.update_layout(height=1000)
        fig.show()


def bar_plot(result, save_fig=False, **kwargs):

    if "labels" in result.columns.to_list():
        result["labels"] = result.labels.apply(str)
    elif "topic_number" in result:
        result["labels"] = result.topic_number.apply(str)

    res = result.groupby(['labels', 'created_at']).count().unstack()["headline"].T  # .plot(kind="bar",**kwargs)

    fig = px.bar(res)

    if save_fig:
        fig.update_layout(height=500)  # Dumping smaller images for convience
        fig.write_html("./tmp_scatter_plot.html")
    else:
        #        fig.update_layout(height=1000)
        fig.show()


################################

# Legacy functions below

################################


def c_tf_idf(documents, m, ngram_range=(1, 1), remove_stop_words=True):
    if remove_stop_words:
        def remove_stop_words(doc):
            for sword in utils.STOPWORDS:
                doc = doc.replace(sword, "")
                return doc

        documents = np.array(list(map(remove_stop_words, documents)))

    count = CountVectorizer(ngram_range=ngram_range).fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in
                   enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                   .Doc
                   .count()
                   .reset_index()
                   .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                   .sort_values("Size", ascending=False))
    return topic_sizes
