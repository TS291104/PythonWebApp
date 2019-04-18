CREATE TABLE ROOT_QUERY_REQUESTS (
	request_id integer,
	platform_request_id integer,
	request_category_id integer,
	request_query_text string,
	serp_api_quota integer,
	created_at datetime,
	created_by string
);

CREATE TABLE ROOT_QUERY_PERFORMANCE (
	request_id integer,
	serp_api_requests integer,
	deadend_count integer,
	error_rate float,
	crawl_depth integer,
	ppa_count integer,
	topic_count integer
);

CREATE TABLE SERP_REQUEST_LOG (
	serp_request_id integer,
	request_id integer,
	serp_query_text string,
	relevance_score float,
	network_score datetime,
	generation integer,
	deadend binary,
	error_rate float,
	created_at datetime
);

CREATE TABLE PAA_RESPONSE (
	paa_id integer,
	question string,
	snippet string,
	displayed_link string
);

CREATE TABLE TOPIC (
	topic_id integer,
	topic_text string
);

CREATE TABLE REQUEST_PAA_RESPONSE (
	serp_request_id integer,
	paa_id integer
);

CREATE TABLE REQUEST_TOPIC_RESPONSE (
	serp_request_id integer,
	topic_id integer
);

CREATE TABLE REQUEST_CATEGORIES (
	request_category_id integer,
	request_category_display_text string
);

CREATE TABLE PLATFORM_REQUEST (
	platform_request_id integer,
	customer_id integer,
	created_at datetime,
	created_by string
);

CREATE TABLE CUSTOMER (
	customer_id integer,
	customer_name string,
	created_at datetime,
	updated_at datetime
);

CREATE TABLE REQUEST_TOPIC_METRICS (
	request_id integer,
	topic_id integer,
	topic_text string,
	topic_ranking_score float,
	topic_cluster_id integer,
	ml_intent_classification string,
	ml_awareness_score float,
	ml_advocacy_score float,
	ml_consideration_score float,
	ml_retention_score float,
	ml_conversion_score float
);

CREATE TABLE REQUEST_PAA_METRICS (
	request_id integer,
	paa_id integer,
	paa_ranking_score float,
	paa_cluster_id integer,
	ml_intent_classification string
);

