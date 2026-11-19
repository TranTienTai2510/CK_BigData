"""Kafka topic contract — single source of truth (B owns the file, A reviews).

Every producer / consumer / ClickHouse Kafka-engine table MUST use these
constants instead of hardcoding topic strings. The `redpanda-init` service in
docker-compose.yml creates exactly TOPICS (8 topics); its inline list is kept
in sync manually because that init container does not mount project code.

Topic flow:
  simulator      -> events.raw, order_events.raw
  Q0 (validate)  -> events.clean, events.dlq   (from events.raw)
  Q1 (kpi)       -> agg.kpi                    (from events.clean)
  Q2 (intent, A) -> pred.intent, session.closed (from events.clean)
  listener       -> ops.metrics                 (streaming health heartbeat)
  ClickHouse Kafka engine ingests all of the above into rt.* / ops.*.
"""

TOPIC_EVENTS_RAW = "events.raw"
TOPIC_EVENTS_CLEAN = "events.clean"
TOPIC_ORDER_EVENTS_RAW = "order_events.raw"
TOPIC_KPI = "agg.kpi"
TOPIC_PRED_INTENT = "pred.intent"
TOPIC_SESSION_CLOSED = "session.closed"
TOPIC_METRICS = "ops.metrics"
TOPIC_DLQ = "events.dlq"

#: All topics that must exist (created by redpanda-init in docker-compose.yml).
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
