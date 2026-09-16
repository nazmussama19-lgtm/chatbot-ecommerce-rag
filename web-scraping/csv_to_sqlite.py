import sqlite3
from pathlib import Path

import pandas as pd

root = Path(__file__).parent.parent
csv_path = root / "app/resources/ecommerce_data_final.csv"
db_path = root / "app/db.sqlite"

# Taux de référence BCE du 20 mai 2024, date du scraping (1 EUR = 90,4715 INR)
EUR_INR = 90.4715

df = pd.read_csv(csv_path)
for column in ["product_link", "title", "brand"]:
    df[column] = df[column].str.strip()

# Les prix scrapés sont en roupies indiennes : conversion en euros, arrondie au centime
df["price"] = (df["price"] / EUR_INR).round(2)

with sqlite3.connect(db_path) as conn:
    conn.execute("DROP TABLE IF EXISTS product")
    conn.execute('''
    CREATE TABLE product (
        product_link TEXT,
        title TEXT,
        brand TEXT,
        price REAL,
        discount REAL,
        avg_rating REAL,
        total_ratings INTEGER
    );
    ''')
    df.to_sql("product", conn, if_exists="append", index=False)
conn.close()

print(f"{len(df)} produits insérés dans {db_path.name} (prix en euros)")
