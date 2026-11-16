-- B: Kafka engine + MVs. Redpanda reachable as redpanda:9092 from clickhouse net.
-- Run after schema.sql, when Redpanda is up.
CREATE TABLE IF NOT EXISTS rt.events_queue (payload String) ENGINE = Kafka
SETTINGS kafka_broker_list='redpanda:9092', kafka_topic_list='events.raw', kafka_group_name='ch-events', kafka_format='RawAsString';
CREATE TABLE IF NOT EXISTS rt.order_queue (payload String) ENGINE = Kafka
SETTINGS kafka_broker_list='redpanda:9092', kafka_topic_list='order_events.raw', kafka_group_name='ch-orders', kafka_format='RawAsString';
CREATE TABLE IF NOT EXISTS rt.kpi_queue (payload String) ENGINE = Kafka
SETTINGS kafka_broker_list='redpanda:9092', kafka_topic_list='agg.kpi', kafka_group_name='ch-kpi', kafka_format='RawAsString';
CREATE TABLE IF NOT EXISTS rt.intent_queue (payload String) ENGINE = Kafka
SETTINGS kafka_broker_list='redpanda:9092', kafka_topic_list='pred.intent', kafka_group_name='ch-intent', kafka_format='RawAsString';
CREATE TABLE IF NOT EXISTS rt.closed_queue (payload String) ENGINE = Kafka
SETTINGS kafka_broker_list='redpanda:9092', kafka_topic_list='session.closed', kafka_group_name='ch-closed', kafka_format='RawAsString';
CREATE TABLE IF NOT EXISTS ops.metrics_queue (payload String) ENGINE = Kafka
SETTINGS kafka_broker_list='redpanda:9092', kafka_topic_list='ops.metrics', kafka_group_name='ch-metrics', kafka_format='RawAsString';
CREATE TABLE IF NOT EXISTS rt.dlq_queue (payload String) ENGINE = Kafka
SETTINGS kafka_broker_list='redpanda:9092', kafka_topic_list='events.dlq', kafka_group_name='ch-dlq', kafka_format='RawAsString';

CREATE MATERIALIZED VIEW IF NOT EXISTS rt.mv_events TO rt.events AS
SELECT JSONExtractString(payload,'shop_id') shop_id, JSONExtractInt(payload,'event_id') event_id,
 JSONExtractInt(payload,'session_id') session_id, JSONExtractInt(payload,'seq_no') seq_no,
 JSONExtractInt(payload,'customer_id') customer_id, JSONExtractString(payload,'event_type') event_type,
 NULLIF(JSONExtractInt(payload,'product_id'),0) product_id, JSONExtractString(payload,'category') category,
 JSONExtractString(payload,'brand') brand, JSONExtractFloat(payload,'price') price,
 JSONExtractInt(payload,'qty') qty, JSONExtractInt(payload,'cart_size') cart_size,
 JSONExtractInt(payload,'order_id') order_id, JSONExtractFloat(payload,'amount_usd') amount_usd,
 JSONExtractString(payload,'currency') currency, JSONExtractFloat(payload,'amount_local') amount_local,
 JSONExtractString(payload,'device') device, JSONExtractString(payload,'source') source,
 JSONExtractString(payload,'country') country,
 parseDateTime64BestEffortOrNull(JSONExtractString(payload,'session_start_ts'),3,'UTC') session_start_ts,
 parseDateTime64BestEffort(JSONExtractString(payload,'event_ts'),3,'UTC') event_ts,
 parseDateTime64BestEffort(JSONExtractString(payload,'ts_raw'),3,'UTC') ts_raw,
 JSONExtractString(payload,'ym') ym FROM rt.events_queue;

CREATE MATERIALIZED VIEW IF NOT EXISTS rt.mv_kpi TO rt.kpi AS
SELECT parseDateTime64BestEffort(JSONExtractString(payload,'window_start'),3,'UTC') window_start,
 parseDateTime64BestEffort(JSONExtractString(payload,'window_end'),3,'UTC') window_end,
 JSONExtractString(payload,'shop_id') shop_id, JSONExtractInt(payload,'n_events') n_events,
 JSONExtractInt(payload,'n_sessions') n_sessions, JSONExtractInt(payload,'n_purchase') n_purchase,
 JSONExtractFloat(payload,'revenue_usd') revenue_usd, JSONExtractInt(payload,'n_add') n_add,
 JSONExtractInt(payload,'n_checkout') n_checkout FROM rt.kpi_queue;

CREATE MATERIALIZED VIEW IF NOT EXISTS rt.mv_intent TO rt.intent AS
SELECT JSONExtractString(payload,'shop_id') shop_id, JSONExtractInt(payload,'session_id') session_id,
 JSONExtractInt(payload,'seq_no') seq_no,
 parseDateTime64BestEffort(JSONExtractString(payload,'event_ts'),3,'UTC') event_ts,
 JSONExtractFloat(payload,'prob') prob, JSONExtractString(payload,'model_ver') model_ver FROM rt.intent_queue;

CREATE MATERIALIZED VIEW IF NOT EXISTS rt.mv_closed TO rt.session_closed AS
SELECT JSONExtractString(payload,'shop_id') shop_id, JSONExtractInt(payload,'session_id') session_id,
 JSONExtractInt(payload,'last_seq_no') last_seq_no,
 parseDateTime64BestEffort(JSONExtractString(payload,'last_event_ts'),3,'UTC') last_event_ts,
 JSONExtractInt(payload,'label_purchase') label_purchase, JSONExtractFloat(payload,'revenue_usd') revenue_usd FROM rt.closed_queue;

CREATE MATERIALIZED VIEW IF NOT EXISTS ops.mv_metrics TO ops.metrics AS
SELECT JSONExtractString(payload,'job') job,
 parseDateTime64BestEffort(JSONExtractString(payload,'ts'),3,'UTC') ts,
 JSONExtractString(payload,'status') status,
 JSONExtractFloat(payload,'lag_s') lag_s, JSONExtractFloat(payload,'throughput_eps') throughput_eps FROM ops.metrics_queue;
