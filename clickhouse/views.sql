-- B: 15 views for Grafana (KPI, funnel, ops, SLA Olist, model placeholder for A).
-- 1
CREATE OR REPLACE VIEW rt.v_funnel_1m AS SELECT toStartOfMinute(event_ts) m, shop_id,
 countIf(event_type='page_view') views, countIf(event_type='add_to_cart') adds,
 countIf(event_type='checkout') checkouts, countIf(event_type='purchase') purchases FROM rt.events GROUP BY m, shop_id;
-- 2
CREATE OR REPLACE VIEW rt.v_kpi_1m AS SELECT toStartOfMinute(window_start) m, shop_id,
 sum(n_events) events, sum(n_sessions) sessions, sum(n_purchase) purchases, sum(revenue_usd) revenue FROM rt.kpi GROUP BY m, shop_id;
-- 3
CREATE OR REPLACE VIEW rt.v_events_1m AS SELECT toStartOfMinute(event_ts) m, shop_id, count() n FROM rt.events GROUP BY m, shop_id;
-- 4
CREATE OR REPLACE VIEW ops.v_lag AS SELECT job, max(ts) last_ts, avg(lag_s) lag_s, avg(throughput_eps) eps FROM ops.metrics GROUP BY job;
-- 5
CREATE OR REPLACE VIEW ops.v_throughput_1m AS SELECT toStartOfMinute(ts) m, job, avg(throughput_eps) eps FROM ops.metrics GROUP BY m, job;
-- 6
CREATE OR REPLACE VIEW rt.v_dlq_1m AS SELECT toStartOfMinute(kafka_ts) m, shop_id, count() n FROM rt.dlq GROUP BY m, shop_id;
-- 7 (SLA olist)
CREATE OR REPLACE VIEW rt.v_sla_daily AS SELECT toDate(event_ts) d, countIf(event_type='delivered') delivered,
 avgIf(days_late, event_type='delivered') avg_late, countIf(event_type='delivered' AND days_late>0) late_n FROM rt.order_events GROUP BY d;
-- 8
CREATE OR REPLACE VIEW rt.v_sla_by_state AS SELECT customer_state, countIf(event_type='delivered') delivered, avgIf(days_late,event_type='delivered') avg_late FROM rt.order_events GROUP BY customer_state;
-- 9
CREATE OR REPLACE VIEW rt.v_orders_1m AS SELECT toStartOfMinute(event_ts) m, event_type, count() n FROM rt.order_events GROUP BY m, event_type;
-- 10
CREATE OR REPLACE VIEW rt.v_review_dist AS SELECT review_score, count() n FROM rt.order_events WHERE event_type='review_created' GROUP BY review_score;
-- 11 (model panels; A fills rt.intent/session_closed via Q2; fake-data tolerant)
CREATE OR REPLACE VIEW rt.v_pred_1m AS SELECT toStartOfMinute(event_ts) m, count() n, avg(prob) avg_prob, quantiles(0.5,0.9,0.95)(prob) q FROM rt.intent GROUP BY m;
-- 12
CREATE OR REPLACE VIEW rt.v_online_eval_1m AS SELECT toStartOfMinute(c.last_event_ts) m, count() n_closed,
 avgIf(i.prob, c.label_purchase=1) avg_p_pos, avgIf(i.prob, c.label_purchase=0) avg_p_neg
 FROM rt.session_closed c LEFT JOIN rt.intent i
 ON c.shop_id=i.shop_id AND c.session_id=i.session_id AND c.last_pred_seq_no=i.seq_no
 GROUP BY m;
-- 13
CREATE OR REPLACE VIEW rt.v_top_categories AS SELECT category, count() views, countIf(event_type='purchase') purchases FROM rt.events WHERE category != '' GROUP BY category ORDER BY views DESC LIMIT 20;
-- 14
CREATE OR REPLACE VIEW rt.v_conversion_by_shop AS SELECT shop_id, countDistinct(session_id) sessions, countIf(event_type='purchase') purchases FROM rt.events GROUP BY shop_id;
-- 15
CREATE OR REPLACE VIEW ops.v_pipeline_health AS SELECT (SELECT count() FROM rt.events) events, (SELECT count() FROM rt.kpi) kpi_rows, (SELECT count() FROM rt.dlq) dlq, (SELECT count() FROM rt.intent) preds;
