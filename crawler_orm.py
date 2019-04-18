from peewee import *

database = SqliteDatabase('ouroboros.db', **{})

class BaseModel(Model):
    class Meta:
        database = database

class Customer(BaseModel):
    created_at = DateTimeField()
    customer_id = TextField(null=False) 
    customer_name = TextField(null=False)  
    updated_at = DateTimeField()

class PlatformRequest(BaseModel):
    created_at = DateTimeField()
    created_by = TextField()
    customer_id = TextField(null=False)  
    platform_request_id = TextField(null=False)  

class PaaResponse(BaseModel):
    question = TextField(null=False,unique=True)  
    displayed_link = TextField()
    snippet = TextField()  

class RequestCategories(BaseModel):
    request_category_display_text = TextField()  
    request_category_id = TextField(null=False)  

class RootQueryRequests(BaseModel):
    created_at = DateTimeField()
    created_by = TextField()  
    platform_request_id = TextField(null=False)  
    request_category_id = TextField(null=False)
    request_query_text = TextField(null=False)  
    serp_api_quota = IntegerField(null=False)

class SerpRequestLog(BaseModel):
    network_score = FloatField()
    error_rate = FloatField()
    created_at = DateTimeField()
    deadend = BooleanField()
    generation = IntegerField()
    relevance_score = FloatField()
    request_id = ForeignKeyField(RootQueryRequests)
    serp_query_text = TextField(null=False)  

class Topic(BaseModel):
    topic_text = TextField(null=False, unique=True)

class RequestPaaResponse(BaseModel):
    paa_id = ForeignKeyField(PaaResponse)
    serp_request_id = ForeignKeyField(SerpRequestLog)

class RequestTopicResponse(BaseModel):
    serp_request_id = ForeignKeyField(SerpRequestLog)
    topic_id = ForeignKeyField(Topic)

class RootQueryPerformance(BaseModel):
    error_rate = FloatField()
    request_id = ForeignKeyField(RootQueryRequests)
    semantic_quality_score = FloatField()
    serp_api_requests = IntegerField()
    crawl_depth = IntegerField()

class RequestTopicMetrics(BaseModel):
    request_id = ForeignKeyField(RootQueryRequests)
    topic_id = ForeignKeyField(Topic)
    topic_text = TextField(null=False)
    topic_ranking_score = FloatField()
    topic_cluster_id = FloatField()
    ml_awareness_score = FloatField(default=0.0)
    ml_advocacy_score = FloatField(default=0.0)
    ml_consideration_score = FloatField(default=0.0)
    ml_conversion_score = FloatField(default=0.0)
    ml_retention_score = FloatField(default=0.0)

class RequestPaaMetrics(BaseModel):
    request_id = ForeignKeyField(RootQueryRequests)
    paa_id = ForeignKeyField(PaaResponse)
    paa_text = TextField(null=False)
    paa_ranking_score = FloatField()
    paa_cluster_id = FloatField()
    ml_intent_classification = TextField()




