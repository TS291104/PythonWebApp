import re
import pandas as pd
import numpy as np
import networkx as nx
import random
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin_min
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from collections import deque
import os
import json
import hashlib
import yaml
import sys
import logging
import re
import copy
from google_search_results import GoogleSearchResults
from textblob import TextBlob
from joblib import dump, load
from peewee import *
from datetime import datetime
from crawler_orm import *

# Define hash function
def hash_digest(data):
    hashId = hashlib.md5()
    hashId.update(repr(data).encode('utf-8'))
    return hashId.hexdigest()

# Import known set of stopwords
stopwords = pd.read_table('model/stopwords.txt')['a'].tolist()

# Load pre-trained model
model = load("model/rf_intent_model.joblib")
model_vectorizer = load("model/intent_model_vectorizer.joblib")
model_label_encoder = load("model/intent_label_encoder.joblib")

# Hashing vectorizer for local semantic analysis of query responses
hvec = HashingVectorizer(analyzer='char_wb', ngram_range=(
    3, 3), n_features=500, stop_words='english')

# Exploration rate equivalent to expected error thresdhold. Bayesian adaptive control via posterior update on beta-binomial distribution
def theta(attempts, observations, alpha, beta): 
	return (attempts + alpha) / (observations + alpha + beta)

# Replace non-stopword terms with their part-of-speech token
def stopwordFiller(txt):
    terms = []
    blob = TextBlob(txt.replace("?", "").lower())
    for t in blob.pos_tags:
        if t[0] in stopwords:
            terms.append(t[0])
        else:
            terms.append("_%s_" % t[1])
    return(" ".join(terms))

# Replace stopword terms with their part-of-speech token
def reverseStopwordFiller(txt):
    terms = []
    blob = TextBlob(txt.replace("?", "").lower())
    for t in blob.pos_tags:
        if t[0] in stopwords:
            terms.append("_%s_" % t[1])
        else:
            terms.append(t[0])
    return(" ".join(terms))

# External API call
# Note: Future version should write files out to blob storage or a queue rather than a local drive
def execute_query(query_string, file_dir, root_token, api_key):
    query = GoogleSearchResults(
        {"q": query_string, 'serp_api_key': api_key, 'num': 10})
    json_results = query.get_json()
    json_results['query_root'] = root_token
    with open(file_dir + "/%s.json" % hash_digest(query_string), 'w') as outfile:
        json.dump(json_results, outfile)
    return json_results


# Attempts to initialize the database according to the ORM structure
# Note: This is only appropriate for a development environment and SQLite.
def initialize_database():
    try:
        database.connect()
        if database.get_tables() == []:
            database.create_tables(BaseModel.__subclasses__())
        database.close()
    except:
        print("Database already exists")


def index_paa(data, serp_request_id):
    for question in data['related_questions']:
        question['serp_request_id'] = serp_request_id
        paa = PaaResponse.select().where(
            PaaResponse.question == question['question'])
        if not paa:
            paa = PaaResponse(**question)
            paa.save()
        else:
            paa = paa.get()
        question['paa_id'] = paa.id
        RequestPaaResponse(**question).save()
    return True


def index_topics(data, serp_request_id):
    for topic in data['related_searches']:
        topic['serp_request_id'] = serp_request_id
        topic['topic_text'] = topic['query']
        tx = Topic.select().where(Topic.topic_text == topic['query'])
        if not tx:
            tx = Topic(**topic)
            tx.save()
        else:
            tx = tx.get()
        topic['topic_id'] = tx.id
        RequestTopicResponse(**topic).save()
    return True


def extract_paa_siblings(serp_request_id, serp_request_query, request):
    query = (PaaResponse
             .select(PaaResponse.id, PaaResponse.question)
             .join(RequestPaaResponse)
             .where(RequestPaaResponse.serp_request_id == serp_request_id)) # on=(PaaResponse.id == RequestPaaResponse.paa_id)
    qids = []
    queries = []
    paa = PaaResponse.select().where(PaaResponse.question == serp_request_query)
    if paa:
    	qids.append(paa.get().id)
    	queries.append(serp_request_query)
    edges = []
    qids.extend(set(list(map(lambda x: x['id'], query.dicts()))))
    queries.extend(set(list(map(lambda x: x['question'], query.dicts()))))
    for src in qids:
        for trg in qids:
            if src != trg:
                edges.append((src, trg))
    return qids, queries, edges


def compute_topic_metrics(request):
    # Link topics based on overlapping PAA connectivity
    qsr = SerpRequestLog.select(SerpRequestLog.id).where(
        SerpRequestLog.request_id == request.id)
    tedges = pd.DataFrame(list(RequestPaaResponse.select(RequestPaaResponse.paa_id, RequestTopicResponse.topic_id).join(RequestTopicResponse, on=(
        RequestTopicResponse.serp_request_id == RequestPaaResponse.serp_request_id)).where(RequestTopicResponse.serp_request_id << qsr, RequestPaaResponse.serp_request_id << qsr).distinct().dicts()))
    topic_edges = []
    for paa_id in tedges.paa_id.unique():
        subd = tedges[tedges.paa_id == paa_id]
        tids = subd.topic_id.unique()
        for src in tids:
            for trg in tids:
                if src != trg:
                    topic_edges.append((src, trg))

    # Compute topic structural ranking and network cluster membership
    Gt = nx.DiGraph(topic_edges)
    pgt = nx.pagerank(Gt)
    topic_clusters = list(
        map(lambda x: x, nx.community.greedy_modularity_communities(Gt)))

    # Extract metrics and update DB
    label_mappings = {'awareness': 'ml_awareness_score', 'advocacy': 'ml_advocacy_score',
                      'consideration': 'ml_consideration_score', 'conversion': 'ml_conversion_score', 'retention': 'ml_retention_score'}
    for row in Topic.select(Topic.id, Topic.topic_text).where(Topic.id << list(Gt.nodes())).distinct():
        tid = row.id
        cx = pd.DataFrame(list(RequestPaaMetrics.select().where(RequestPaaMetrics.paa_id << tedges[tedges.topic_id == tid].paa_id.unique(
        ).tolist()).dicts())).groupby('ml_intent_classification').apply(lambda x: x.shape[0])
        cx = dict(cx / cx.sum())
        tmp_res = {}
        for k, v in label_mappings.items():
            if k in cx:
                tmp_res[v] = cx[k]
            else:
                tmp_res[v] = 0.0
        tmp_res['request_id'] = request.id
        tmp_res['topic_id'] = tid
        tmp_res['topic_text'] = row.topic_text
        tmp_res['topic_ranking_score'] = pgt[tid]
        tmp_res['topic_cluster_id'] = [
            i for i, sx in enumerate(topic_clusters) if tid in sx][0]
        RequestTopicMetrics(**tmp_res).save()

    return True


def compute_paa_metrics(request):

    # Link PAA responses by common serp request
    qsr = SerpRequestLog.select(SerpRequestLog.id).where(SerpRequestLog.request_id == request.id)
    pedges = pd.DataFrame(list(RequestPaaResponse.select().where(RequestPaaResponse.serp_request_id << qsr).distinct().dicts()))
    paa_edges = []
    for sid in pedges.serp_request_id.unique():
        subd = pedges[pedges.serp_request_id == sid]
        pids = subd.paa_id.unique()
        for src in pids:
            for trg in pids:
                if src != trg:
                    paa_edges.append((src, trg))

    # Compute paa structural ranking and network cluster membership
    Gp = nx.Graph(paa_edges)
    pg = nx.pagerank(Gp)
    paa_clusters = list(map(lambda x: x, nx.community.greedy_modularity_communities(Gp)))

    # Extract metrics and update DB
    query = PaaResponse.select(PaaResponse.id, PaaResponse.question).where(PaaResponse.id << list(Gp.nodes())).distinct()
    pred_labels = model_label_encoder.inverse_transform(model.predict(
        model_vectorizer.transform(list(map(lambda x: stopwordFiller(x.question), query)))))
    for i, row in enumerate(query):
        pid = row.id
        tmp_res = {}
        tmp_res['paa_id'] = pid
        tmp_res['request_id'] = request.id
        tmp_res['paa_text'] = row.question
        tmp_res['paa_ranking_score'] = pg[pid]
        tmp_res['paa_cluster_id'] = [i for i, sx in enumerate(paa_clusters) if pid in sx][0]
        tmp_res['ml_intent_classification'] = pred_labels[i]
        RequestPaaMetrics(**tmp_res).save()

    return True


def execute_crawl(request, results_directory, cfg, verbose=True):

    # Initialize crawler parameters
    G = nx.Graph()
    q = request.request_query_text
    search_log_text, search_log_ids, corpus = [], [], set()
    attempts, itr, q_network_score = 0, 0, 0
    alpha, beta = 1, 20  # Error rate priors

    # Crawler loop
    crawl_ = True
    while (itr < request.serp_api_quota) & crawl_:

        # Stdout logging
        if verbose:
        	print("%s  %s ::: %s, %s - %s" % (itr, request.request_query_text, q, theta(attempts, itr, alpha, beta), q_network_score))

        # Execute SerpAPI request
        response_object = execute_query(q, results_directory, request.id, cfg['serp']['api_key'])
        itr += 1

        # Log SerpAPI query attributes
        deadend = 'related_questions' not in response_object
        slog = SerpRequestLog(serp_query_text=q,
                              request_id=request.id,
                              relevance_score=q_network_score,
                              network_score=q_network_score,
                              deadend=deadend,
                              generation=itr,
                              error_rate=theta(attempts, itr, alpha, beta),
                              created_at=datetime.now())
        slog.save()

        # Increment local log
        try:
            search_log_ids.append(PaaResponse.select(PaaResponse.id).where(PaaResponse.question == q).get().id)
            search_log_text.append(q)
        except:
            pass

        # If response is returned, then index observations.
        if not deadend:
            index_paa(response_object, slog.id)
            qnodes, queries, qedges = extract_paa_siblings(slog.id, q, request)
            corpus = corpus | set(queries)
            G.add_nodes_from(qnodes)
            G.add_edges_from(qedges)
        else:
            attempts += 1

        # Index related topics:
        if 'related_searches' in response_object:
        	index_topics(response_object, slog.id)

        # Find the next query to execute based on graph rank and semantic relevance
        while crawl_:

                # Select candidate PAA id with maximum rank coefficient that has not yet been executed
            try:
                pg = nx.pagerank(G)
                deque(map(lambda x: pg.pop(x, None), search_log_ids))
                qcid = max(pg, key=lambda k: pg[k])

                # To avoid getting stuck in local-minima, encourage exploration proportional at the known error rate
                exploration_threshold = 0.2 #theta(attempts, itr, alpha, beta)
                if random.uniform(0, 1) < exploration_threshold:
                    qcid = random.choice(list(pg.keys()))

                # Get text and network score for the final candidate
                cq_network_score = pg[qcid]
                qcandidate = PaaResponse.select(PaaResponse.question).where(PaaResponse.id == qcid).scalar()

                # Measure hash-similarity of candidate to executed searches
                if itr > 1:
	                Lx = hvec.transform(search_log_text)
	                LCx = hvec.transform(corpus)
	                qx = hvec.transform([qcandidate])
	                dx_mean = np.mean(pairwise_distances(qx, Lx, metric="cosine"))
	                pdist = pairwise_distances(qx, LCx, metric="cosine")
	                adaptive_threshold = np.mean(pdist) + 0.5 * np.std(pdist)
	                accept_candidate = dx_mean < adaptive_threshold
                else:
                	accept_candidate = True

                # If: candidate distance is below average distance, accept.
                # Else: remove candidate from graph
                if accept_candidate:
                    q_network_score = cq_network_score
                    q = qcandidate
                    break
                else:
                    G.remove_nodes_from([qcid])
                    attempts += 1

            except:
                attempts += 1
                pass

            # Evaluate posterior likelihood of error function
            crawl_ = theta(attempts, itr, alpha, beta) < 0.5

    # Post crawl evaluation
    try:
        compute_paa_metrics(request)
        compute_topic_metrics(request)
    except:
        # Add more informative logging and error capture
        pass

    return True
