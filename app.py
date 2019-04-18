"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from crawler_orm import *
from crawler_utils import *
with open('config.yml','r') as ymlfile:
    cfg = yaml.load(ymlfile)
warnings.filterwarnings('ignore')

from flask import Flask
app = Flask(__name__)

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app


@app.route('/')
def hello():
    """Renders a sample page."""
    return "Hello World!"





@app.route('/execute')
def execute():
    global request
    initialize_database()
    customer = Customer(created_at=datetime.now(), customer_name="Vennli", customer_id=hash_digest("Vennli"), updated_at = datetime.now())
    customer.save()
    preq = PlatformRequest(created_at=datetime.now(), created_by="Tyler Foxworthy",customer_id=customer.customer_id,platform_request_id=hash_digest(str(datetime.now())))
    preq.save()
    rcategory = RequestCategories(request_category_id=hash_digest("PrimaryObjective"),request_category_display_text="PrimaryObjective")
    rcategory.save()
    root_query_string = "content marketing strategy"
    request = RootQueryRequests(created_at=datetime.now(),
                            created_by="Tyler Foxworthy",
                            platform_request_id=preq.platform_request_id,
                            request_category_id=rcategory.request_category_id,
                            request_query_text=root_query_string,
                            serp_api_quota=50)
    request.save()
    results_directory = 'data'
    qdirectory = "%s/%s/%s/%s" % (results_directory,customer.customer_id,request.id,request.id)
    if not os.path.exists(qdirectory):
        os.makedirs(qdirectory)
    execute_crawl(request,qdirectory,cfg, False)
    return "DONE"

@app.route('/build_ppa_metrics')
def build_ppa_metrics():
    global resdf
    resdf = pd.DataFrame(list(RequestPaaMetrics.select().where(RequestPaaMetrics.request_id == request.id).distinct().dicts()))
    resdf.sort_values('paa_ranking_score',ascending=False).iloc[0:10]
    resdf.to_csv("data/request_ppa_metrics.csv")
    return "DONE"

@app.route('/build_ppa_intent_segmentation')
def build_ppa_intent_segmentation(): 
    global resdf
    plt.figure(figsize=(20, 12))
    cmap = sns.cubehelix_palette(as_cmap=True, light=.95)
    cluster_labels = resdf.groupby('paa_cluster_id').apply(lambda x: x.sort_values('paa_ranking_score',ascending=False).paa_text.iloc[0]).reset_index(name='paa_cluster_centroid')
    pdx = resdf.merge(cluster_labels,on='paa_cluster_id')
    pdx = pdx.groupby(('paa_cluster_centroid','ml_intent_classification')).apply(lambda x: x.paa_ranking_score.mean()).reset_index(name='avg_score')
    rdx = pdx.pivot(index='paa_cluster_centroid',columns='ml_intent_classification',values='avg_score')
    mask = rdx.isnull()
    with sns.axes_style("white"):
        sns.heatmap(rdx,mask=mask,cmap=cmap,annot=True)
        plt.tight_layout()
        plt.savefig("images/paa_intent_segmentation.png")
        plt.savefig("images/paa_intent_segmentation.pdf")    
    return "DONE"

@app.route('/build_request_topic_metrics')
def build_request_topic_metrics():
    global resdf
    resdf = pd.DataFrame(list(RequestTopicMetrics.select().where(RequestTopicMetrics.request_id == request.id).distinct().dicts()))
    resdf.to_csv("data/request_topic_metrics.csv")
    cluster_labels = resdf.groupby('topic_cluster_id').apply(lambda x: x.sort_values('topic_ranking_score',ascending=False).topic_id.iloc[0]).reset_index(name='topic_id')
    resdf = cluster_labels.merge(resdf)
    resdf.sort_values('topic_ranking_score',ascending=False).iloc[0:10]
    return "DONE"

@app.route('/build_topic_rank_distribution')
def build_topic_rank_distribution():
    global resdf
    plt.figure(figsize=(20,12))
    px = sns.barplot(data=resdf,x='topic_text',y='topic_ranking_score')
    loc, labels = plt.xticks()
    px.set_xticklabels(labels, rotation=90,fontsize=12)
    px.set_ylabel("Network Rank Score",fontsize=18)
    px.set_xlabel("Topic Cluster",fontsize=18)
    px.set_title("Key Topics by Network Rank",fontsize=18)
    plt.tight_layout()
    plt.savefig("images/topic_rank_distribution.pdf")
    plt.savefig("images/topic_rank_distribution.png")
    return "DONE"

@app.route('/build_topic_intent_segmentation')
def build_topic_intent_segmentation():
    global resdf
    rdx = pd.melt(resdf, id_vars=['topic_text'], value_vars=['ml_awareness_score','ml_advocacy_score','ml_conversion_score','ml_consideration_score','ml_retention_score']).reset_index()
    rdx = rdx.groupby(('topic_text','variable')).apply(lambda x: x.value.mean()).reset_index(name='value').pivot(index='topic_text',columns='variable',values='value').fillna(0)
    #%matplotlib inline
    plt.figure(figsize=(20, 12))
    cmap = sns.cubehelix_palette(as_cmap=True, light=.95)
    mask = rdx.isnull() 
    with sns.axes_style("white"):
        sns.heatmap(rdx,mask=mask,cmap=cmap,annot=True)
        plt.tight_layout()
        plt.savefig("images/topic_intent_segmentation.png")
        plt.savefig("images/topic_intent_segmentation.pdf")
    return "DONE"

@app.route('/build_crawler_graph_membership')
def build_crawler_graph_membership():
    qsr = SerpRequestLog.select(SerpRequestLog.id).where(
        SerpRequestLog.request_id == request.id)
    pedges = pd.DataFrame(list(RequestPaaResponse.select().where(
        RequestPaaResponse.serp_request_id << qsr).distinct().dicts()))
    paa_edges = []
    for sid in pedges.serp_request_id.unique():
        subd = pedges[pedges.serp_request_id == sid]
        pids = subd.paa_id.tolist()
        qsid = SerpRequestLog.select(SerpRequestLog.serp_query_text).where(SerpRequestLog.id == sid).get().serp_query_text
        paa = PaaResponse.select().where(PaaResponse.question == qsid)
    if paa:
        pids.append(paa.get().id)
    for src in pids:
        for trg in pids:
            if src != trg:
                paa_edges.append((src, trg))

    # Compute paa structural ranking and network cluster membership
    Gp = nx.Graph(paa_edges)
    pg = nx.pagerank(Gp)
    paa_clusters = list(
        map(lambda x: x, nx.community.greedy_modularity_communities(Gp)))

    f = plt.figure(figsize=(20,20))
    node_colors = [[i for i,sx in enumerate(paa_clusters) if tid in sx][0] for tid in list(Gp.nodes())]
    node_sizes  = [int(pg[i]*100000)*1.0 for i in Gp.nodes()]
    pos = nx.spring_layout(Gp)
    nx.draw(Gp,pos,node_color=node_colors,node_size=node_sizes)
    plt.title("Directed Crawler Graph: Colored By Community Membership)")
    f.savefig("images/crawler_graph_membership.pdf",bbox_inches='tight')
    f.savefig("images/crawler_graph_membership.png",bbox_inches='tight')
    return "DONE"

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
