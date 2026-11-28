from pathlib import Path
import re
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from common.messages import Q2_OUTPUT_SCHEMA


def main():
    schema_sql = (ROOT_DIR / "clickhouse" / "schema.sql").read_text(encoding="utf-8")
    ingest_sql = (ROOT_DIR / "clickhouse" / "kafka_ingest.sql").read_text(encoding="utf-8")
    views_sql = (ROOT_DIR / "clickhouse" / "views.sql").read_text(encoding="utf-8")
    eval_source = (ROOT_DIR / "analysis" / "live.py").read_text(encoding="utf-8")

    closed_table = re.search(
        r"CREATE TABLE IF NOT EXISTS rt\.session_closed\s*\((.*?)\)\s*ENGINE",
        schema_sql,
        flags=re.DOTALL | re.IGNORECASE,
    )
    assert closed_table is not None
    assert re.search(r"last_pred_seq_no\s+Nullable\(Int32\)", closed_table.group(1), re.I)
    assert re.search(r"last_seq_no\s+Int32", closed_table.group(1), re.I)
    assert re.search(
        r"ALTER TABLE rt\.session_closed\s+ADD COLUMN IF NOT EXISTS last_pred_seq_no",
        schema_sql,
        re.I,
    )

    closed_view = re.search(
        r"CREATE MATERIALIZED VIEW(?: IF NOT EXISTS)? rt\.mv_closed.*?FROM rt\.closed_queue;",
        ingest_sql,
        flags=re.DOTALL | re.IGNORECASE,
    )
    assert closed_view is not None
    assert re.search(r"DROP VIEW IF EXISTS rt\.mv_closed", ingest_sql, re.I)
    assert "last_pred_seq_no" in closed_view.group(0)
    assert "last_seq_no" in closed_view.group(0)

    online_view = re.search(
        r"CREATE OR REPLACE VIEW rt\.v_online_eval_1m.*?GROUP BY m;",
        views_sql,
        flags=re.DOTALL | re.IGNORECASE,
    )
    assert online_view is not None
    assert re.search(r"c\.last_pred_seq_no\s*=\s*i\.seq_no", online_view.group(0), re.I)
    assert "last_seq_no=i.seq_no" not in online_view.group(0)
    assert re.search(r"c\.last_pred_seq_no\s*=\s*i\.seq_no", eval_source, re.I)
    assert re.search(r"kafka_topic_list\s*=\s*'events\.clean'", ingest_sql, re.I)
    assert "RawAsString" not in ingest_sql
    assert re.search(r"CREATE MATERIALIZED VIEW IF NOT EXISTS rt\.mv_dlq", ingest_sql, re.I)

    output_fields = {field.name for field in Q2_OUTPUT_SCHEMA.fields}
    assert {"last_pred_seq_no", "last_seq_no", "label_purchase"}.issubset(output_fields)
    print("CLICKHOUSE_Q2_SCHEMA_OK")


if __name__ == "__main__":
    main()
