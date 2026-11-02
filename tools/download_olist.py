"""Download the Olist dataset with kagglehub and copy the 9 CSVs into data/raw_olist/.

  pip install kagglehub
  python tools/download_olist.py            # needs a Kaggle token (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)

kagglehub.load_dataset(..., file_path="") does not work for this dataset: it loads ONE file into a DataFrame and
needs its name. The pipeline needs the files on disk, so this script uses dataset_download (cached) + copy.
"""
import os
import shutil
import sys

import kagglehub

EXPECTED = ["olist_customers_dataset.csv", "olist_geolocation_dataset.csv", "olist_order_items_dataset.csv",
            "olist_order_payments_dataset.csv", "olist_order_reviews_dataset.csv", "olist_orders_dataset.csv",
            "olist_products_dataset.csv", "olist_sellers_dataset.csv", "product_category_name_translation.csv"]
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "raw_olist")
src = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
os.makedirs(out, exist_ok=True)
found = {f: os.path.join(root, f) for root, _, files in os.walk(src) for f in files if f in EXPECTED}
missing = sorted(set(EXPECTED) - set(found))
for f, p in found.items():
    shutil.copy2(p, os.path.join(out, f))
    print(f"{f:45s} {os.path.getsize(p) / 1e6:7.1f} MB")
if missing:
    sys.exit(f"missing in download: {missing}")
print(f"OK -> {out}")
