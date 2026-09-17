import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from llm import chat

db_path = Path(__file__).parent / "db.sqlite"

# Garde le contexte envoyé au LLM sous la limite de tokens de Groq
MAX_ROWS = 10

PRODUCT_COLUMNS = {"product_link", "title", "brand", "price", "discount", "avg_rating", "total_ratings"}

sql_prompt = """You are an expert in understanding the database schema and generating SQL queries for a natural language question asked
pertaining to the data you have. The schema is provided in the schema tags.
<schema>
table: product

fields:
product_link - string (hyperlink to product)
title - string (name of the product)
brand - string (brand of the product)
price - float (price of the product in euros)
discount - float (discount on the product. 10 percent discount is represented as 0.1, 20 percent as 0.2, and such.)
avg_rating - float (average rating of the product. Range 0-5, 5 is the highest.)
total_ratings - integer (total number of ratings for the product)

</schema>
Make sure whenever you try to search for the brand name, the name can be in any case.
So, make sure to use %LIKE% to find the brand in condition. Never use "ILIKE".
Create a single SQL query for the question provided.
The question may be in French, but product titles and brands in the database are in English: translate keywords (e.g. "chaussures de course" -> "Running") before using them in LIKE conditions.
The query should have all the fields in SELECT clause (i.e. SELECT *)
Unless the question asks for a specific number of products, add LIMIT 10 to the query.
If the question is not about products of the catalog (for example store policies, payment, delivery or returns), do not write a query: answer exactly <SQL>NONE</SQL>.

Just the SQL query is needed, nothing more. Always provide the SQL in between the <SQL></SQL> tags."""


comprehension_prompt = """You are an expert in understanding the context of the question and replying based on the data pertaining to the question provided. You will be provided with Question: and Data:. The data will be in the form of an array or a dataframe or dict. Reply based on only the data provided as Data for answering the question asked as Question. Do not write anything like 'Based on the data' or any other technical words. Just a plain simple natural language response.
The Data would always be in context to the question asked. For example if the question is "Quelle est la note moyenne ?" and data is "4.3", then answer should be "La note moyenne est de 4,3." Make sure to note the column names to have some context, if needed, for your response.
Prices are in euros. Always answer in French, in one or two short sentences."""


def generate_sql_query(question):
    return chat([
        {"role": "system", "content": sql_prompt},
        {"role": "user", "content": question},
    ])


def run_query(query):
    # Certains modèles entourent le SQL de balises Markdown (```sql ... ```)
    query = re.sub(r"^```(?:sql)?|```$", "", query.strip(), flags=re.IGNORECASE)
    query = query.strip().rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\s", query, re.IGNORECASE) or ";" in query:
        return None
    # Base ouverte en lecture seule : une requête générée ne peut rien modifier
    with closing(sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)) as conn:
        return pd.read_sql_query(query, conn)


def short_link(url):
    # Les liens Flipkart embarquent des paramètres de tracking ; seul "pid" est utile
    if not isinstance(url, str):
        return url
    return url.split("&", 1)[0]


def data_comprehension(question, context):
    return chat(
        [
            {"role": "system", "content": comprehension_prompt},
            {"role": "user", "content": f"QUESTION: {question}. DATA: {context}"},
        ],
        max_tokens=2048,
    )


def format_number(value):
    # Séparateur de milliers à la française (espace fine insécable)
    return f"{int(value):,}".replace(",", " ")


def format_euros(value):
    return f"{value:.2f}".replace(".", ",") + " €"


def escape_markdown(text):
    return re.sub(r"([\\*_`$\[\]])", r"\\\1", str(text))


def format_products(df):
    count = len(df)
    intro = "Voici le produit que j'ai trouvé :" if count == 1 else f"Voici les {count} produits que j'ai trouvés :"
    blocks = [intro]
    for row in df.itertuples():
        details = [escape_markdown(row.brand).upper()]
        if pd.notna(row.price):
            details.append(format_euros(row.price))
        if pd.notna(row.discount) and row.discount > 0:
            details.append(f"−{round(row.discount * 100)} %")
        if pd.notna(row.avg_rating) and row.avg_rating > 0:
            rating = f"★ {row.avg_rating:.1f}".replace(".", ",")
            if pd.notna(row.total_ratings):
                rating += f" ({format_number(row.total_ratings)} avis)"
            details.append(rating)
        blocks.append(
            f"**{escape_markdown(row.title)}**  \n"
            f"{' · '.join(details)} · [Voir le produit]({row.product_link})"
        )
    return "\n\n".join(blocks)


def sql_chain(question):
    sql_query = generate_sql_query(question)
    matches = re.findall(r"<SQL>(.*?)</SQL>", sql_query, re.DOTALL)
    if len(matches) == 0:
        return "Je n'ai pas réussi à transformer votre demande en recherche. Pouvez-vous la reformuler ?"

    print(matches[0].strip())
    if matches[0].strip().upper() == "NONE":
        return None

    try:
        response = run_query(matches[0])
    except (sqlite3.Error, pd.errors.DatabaseError) as e:
        print("SQL error:", e)
        response = None
    if response is None:
        return "La recherche n'a pas abouti. Pouvez-vous reformuler votre demande ?"
    if response.empty:
        return "Aucun produit ne correspond à votre recherche."

    response = response.drop(columns=["index"], errors="ignore")
    if "product_link" in response.columns:
        # Le scraping contient quelques fiches en double : même produit, même lien
        response["product_link"] = response["product_link"].map(short_link)
        response = response.drop_duplicates(subset="product_link")
    response = response.head(MAX_ROWS)

    # Liste de produits : mise en forme directe, sans repasser par le LLM
    if PRODUCT_COLUMNS.issubset(response.columns):
        return format_products(response)

    return data_comprehension(question, response.to_dict(orient="records"))


if __name__ == "__main__":
    print(sql_chain("Les 3 chaussures les mieux notées"))
