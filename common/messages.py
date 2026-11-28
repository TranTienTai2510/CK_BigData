from pyspark.sql import types as T


"""Kafka topic contract used across producers and consumers."""

TOPIC_EVENTS_RAW = "events.raw"
TOPIC_EVENTS_CLEAN = "events.clean"
TOPIC_ORDER_EVENTS_RAW = "order_events.raw"
TOPIC_KPI = "agg.kpi"
TOPIC_PRED_INTENT = "pred.intent"
TOPIC_SESSION_CLOSED = "session.closed"
TOPIC_METRICS = "ops.metrics"
TOPIC_DLQ = "events.dlq"

TOPICS = (
    TOPIC_EVENTS_RAW,
    TOPIC_EVENTS_CLEAN,
    TOPIC_ORDER_EVENTS_RAW,
    TOPIC_KPI,
    TOPIC_PRED_INTENT,
    TOPIC_SESSION_CLOSED,
    TOPIC_METRICS,
    TOPIC_DLQ,
)

SCHEMA_VERSION = 3


Q2_STATE_SCHEMA = T.StructType([
    T.StructField("session_state_json", T.StringType(), False),
    T.StructField("has_purchase", T.BooleanType(), False),
    T.StructField("purchase_seq_no", T.LongType(), True),
    T.StructField("revenue_usd", T.DoubleType(), False),
])

Q2_OUTPUT_SCHEMA = T.StructType([
    T.StructField("record_type", T.StringType(), False),
    T.StructField("shop_id", T.StringType(), False),
    T.StructField("session_id", T.LongType(), False),
    T.StructField("seq_no", T.IntegerType(), True),
    T.StructField("event_ts", T.StringType(), True),
    T.StructField("prob", T.FloatType(), True),
    T.StructField("model_ver", T.StringType(), True),
    T.StructField("last_pred_seq_no", T.IntegerType(), True),
    T.StructField("last_seq_no", T.IntegerType(), True),
    T.StructField("last_event_ts", T.StringType(), True),
    T.StructField("label_purchase", T.IntegerType(), True),
    T.StructField("revenue_usd", T.DoubleType(), True),
])
