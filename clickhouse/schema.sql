-- B: ClickHouse DDL (signed Int64 xxhash, UTC). Run: clickhouse-client --multiquery < clickhouse/schema.sql
CREATE DATABASE IF NOT EXISTS rt;
CREATE DATABASE IF NOT EXISTS ops;

CREATE TABLE IF NOT EXISTS rt.events (
  shop_id String, event_id Int64, session_id Int64, seq_no Int32, customer_id Int64,
  event_type String, product_id Nullable(Int64), category Nullable(String), brand Nullable(String),
  price Nullable(Float64), qty Nullable(Int32), cart_size Nullable(Int32),
  order_id Nullable(Int64), amount_usd Nullable(Float64), currency Nullable(String), amount_local Nullable(Float64),
  device Nullable(String), source Nullable(String), country Nullable(String),
  session_start_ts Nullable(DateTime64(3,'UTC')), event_ts DateTime64(3,'UTC'), ts_raw DateTime64(3,'UTC'),
  ym String, schema_version UInt8 DEFAULT 3
) ENGINE = MergeTree ORDER BY (shop_id, event_ts, session_id, seq_no);

CREATE TABLE IF NOT EXISTS rt.order_events (
  shop_id String, event_id Int64, order_id Int64, order_key String, customer_id Int64,
  seq_no Int32, event_type String, order_status String, n_items Int32,
  amount_usd Float64, currency String, amount_local Float64, customer_state String,
  estimated_delivery_ts Nullable(DateTime64(3,'UTC')), days_late Nullable(Float64),
  review_score Nullable(UInt8), event_ts DateTime64(3,'UTC'), ts_raw DateTime64(3,'UTC'), ym String
) ENGINE = MergeTree ORDER BY (shop_id, event_ts, order_id, seq_no);

CREATE TABLE IF NOT EXISTS rt.kpi (
  window_start DateTime64(3,'UTC'), window_end DateTime64(3,'UTC'), shop_id String,
  n_events UInt64, n_sessions UInt64, n_purchase UInt64, revenue_usd Float64, n_add UInt64, n_checkout UInt64
) ENGINE = MergeTree ORDER BY (window_start, shop_id);

CREATE TABLE IF NOT EXISTS rt.intent (
  shop_id String, session_id Int64, seq_no Int32, event_ts DateTime64(3,'UTC'),
  prob Float32, model_ver String DEFAULT 'intent_v1'
) ENGINE = MergeTree ORDER BY (event_ts, shop_id, session_id, seq_no);

CREATE TABLE IF NOT EXISTS rt.session_closed (
  shop_id String, session_id Int64, last_seq_no Int32, last_event_ts DateTime64(3,'UTC'),
  label_purchase UInt8, revenue_usd Float64
) ENGINE = MergeTree ORDER BY (last_event_ts, shop_id, session_id);

CREATE TABLE IF NOT EXISTS ops.metrics (
  job String, ts DateTime64(3,'UTC'), status String, lag_s Float64 DEFAULT 0, throughput_eps Float64 DEFAULT 0
) ENGINE = MergeTree ORDER BY (ts, job);

CREATE TABLE IF NOT EXISTS rt.dlq (
  shop_id String, session_id Int64, event_id Int64, seq_no Int32, event_type String,
  event_ts Nullable(DateTime64(3,'UTC')), failed_rules Array(String), kafka_ts DateTime64(3,'UTC')
) ENGINE = MergeTree ORDER BY (kafka_ts, shop_id);
