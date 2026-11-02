"""Data contracts.

Two layers:
  1. Source rules: checked before a source is mapped to the canonical schema.
     EVENT_RULES / LINK_RULES are the synthetic-source rules used by batch/preprocess.py
     (names kept for backward compatibility). REES46 source rules live in its adapter.
  2. CANONICAL_RULES: the contract of the unified schema (common/canonical.py). It is checked by
     every adapter and check_unified.py, and by streaming Q0 on Kafka payloads.

A rule is a Spark SQL boolean expression that MUST be true. A NULL result counts as a failure.
A canonical rule may be scoped to some shops: (expr, ("shop_a", ...)).
"""

EVENT_TYPES = ("page_view", "add_to_cart", "checkout", "purchase")  # synthetic source vocabulary

# ---------------------------------------------------------------- synthetic source rules (preprocess.py)
EVENT_RULES = {
    "type_known":            "event_type IN ('page_view','add_to_cart','checkout','purchase')",
    "ts_present":            "ts_raw IS NOT NULL",
    "product_iff_view_cart": "(product_id IS NOT NULL) = (event_type IN ('page_view','add_to_cart'))",
    "qty_iff_cart":          "(qty IS NOT NULL) = (event_type = 'add_to_cart') AND coalesce(qty, 1) > 0",
    "cart_size_iff_checkout": "(cart_size IS NOT NULL) = (event_type = 'checkout') AND coalesce(cart_size, 1) > 0",
    "payment_iff_purchase":  "(payment IS NOT NULL AND discount_pct IS NOT NULL AND amount_usd IS NOT NULL)"
                             " = (event_type = 'purchase')",
    "amount_positive":       "coalesce(amount_usd, 1) > 0",
    "discount_range":        "coalesce(discount_pct, 0) BETWEEN 0 AND 100",
}
LINK_RULES = {
    "purchase_has_order": "event_type <> 'purchase' OR order_id IS NOT NULL",
}

# ---------------------------------------------------------------- canonical contract (schema v2)
CANONICAL_RULES = {
    "type_known": "event_type IN ('page_view','add_to_cart','remove_from_cart','checkout','purchase')",
    "keys_present": "shop_id IS NOT NULL AND event_id IS NOT NULL AND session_id IS NOT NULL"
                    " AND customer_id IS NOT NULL AND seq_no >= 1",
    "ts_valid": "event_ts IS NOT NULL AND ts_raw IS NOT NULL AND event_ts <= ts_raw",
    "item_fields_iff_item_event": "(product_id IS NOT NULL AND category IS NOT NULL AND price IS NOT NULL)"
                                  " = (event_type IN ('page_view','add_to_cart','remove_from_cart'))",
    "price_non_negative": "coalesce(price, 0) >= 0",
    "qty_iff_cart_event": "(qty IS NOT NULL) = (event_type IN ('add_to_cart','remove_from_cart'))"
                          " AND coalesce(qty, 1) > 0",
    "cart_size_iff_checkout": "(cart_size IS NOT NULL) = (event_type = 'checkout') AND coalesce(cart_size, 1) > 0",
    "order_iff_purchase": "(order_id IS NOT NULL AND amount_usd IS NOT NULL AND size(items) > 0)"
                          " = (event_type = 'purchase')",
    "amount_non_negative": "coalesce(amount_usd, 0) >= 0",
    "items_valid": "event_type <> 'purchase'"
                   " OR forall(items, i -> i.product_id IS NOT NULL AND i.price >= 0 AND i.qty > 0)",
    "ym_consistent": "ym = date_format(event_ts, 'yyyy-MM')",
    # shop-scoped rules
    "payment_iff_purchase": ("(payment IS NOT NULL AND discount_pct IS NOT NULL) = (event_type = 'purchase')",
                             ("synthetic",)),
    "no_checkout_in_rees46": ("event_type <> 'checkout'", ("rees46_cosmetics",)),
    "currency_iff_amount_local": "(currency IS NULL) = (amount_local IS NULL)",
    "olist_purchase_only": ("event_type = 'purchase' AND currency = 'BRL' AND amount_local >= 0", ("olist",)),
}

# ---------------------------------------------------------------- order lifecycle contract (ORDER_EVENT_SCHEMA)
ORDER_EVENT_RULES = {
    "type_known": "event_type IN ('order_placed','payment_approved','handed_to_carrier','delivered','review_created')",
    "keys_present": "shop_id IS NOT NULL AND event_id IS NOT NULL AND order_id IS NOT NULL"
                    " AND customer_id IS NOT NULL AND seq_no >= 1",
    "ts_valid": "event_ts IS NOT NULL AND ts_raw IS NOT NULL AND event_ts >= ts_raw",
    "placed_is_first": "(event_type = 'order_placed') = (seq_no = 1)",
    "review_score_iff_review": "(review_score IS NOT NULL) = (event_type = 'review_created')"
                               " AND coalesce(review_score, 3) BETWEEN 1 AND 5",
    "days_late_only_delivered": "days_late IS NULL OR event_type = 'delivered'",
    "amount_non_negative": "coalesce(amount_usd, 0) >= 0 AND coalesce(amount_local, 0) >= 0",
    "ym_consistent": "ym = date_format(event_ts, 'yyyy-MM')",
}


def _scoped(expr, shops):
    if not shops:
        return expr
    in_list = ", ".join(f"'{s}'" for s in shops)
    return f"(shop_id NOT IN ({in_list})) OR ({expr})"


def failed_rules_col(rules):
    """Return a Spark Column: array<string> of rule names that failed for the row."""
    from pyspark.sql import functions as F
    checks = []
    for name, rule in rules.items():
        expr = _scoped(*rule) if isinstance(rule, tuple) else rule
        checks.append(F.when(~F.coalesce(F.expr(expr), F.lit(False)), F.lit(name)))
    return F.filter(F.array(*checks), lambda x: x.isNotNull())
