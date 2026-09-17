from pathlib import Path

import chromadb
import pandas as pd
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from llm import chat

# Modèle all-MiniLM-L6-v2 exécuté en ONNX : pas besoin de PyTorch
ef = embedding_functions.DefaultEmbeddingFunction()

chroma_client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
collection_name_faq = "faqs"


def get_collection():
    return chroma_client.get_or_create_collection(
        name=collection_name_faq,
        embedding_function=ef,
    )


def ingest_faq_data(path):
    collection = get_collection()
    if collection.count() > 0:
        print(f"Collection: {collection_name_faq} already exists")
        return
    print("Ingesting FAQ data into Chromadb...")
    df = pd.read_csv(path)
    collection.add(
        documents=df["question"].to_list(),
        metadatas=[{"answer": ans} for ans in df["answer"].to_list()],
        ids=[f"id_{i}" for i in range(len(df))],
    )
    print(f"FAQ Data successfully ingested into Chroma collection: {collection_name_faq}")


def get_relevant_qa(query):
    # Le modèle d'embedding est faible en français : 4 entrées sur 11 laissent de la marge au LLM
    return get_collection().query(query_texts=[query], n_results=4)


def generate_answer(query, context):
    prompt = f'''Given the following context and question, generate answer based on this context only.
    If the answer is not found in the context, say "Je n'ai pas cette information." Don't try to make up an answer.
    Always answer in French, in one or two short sentences, addressing the customer as "vous".

    CONTEXT: {context}

    QUESTION: {query}
    '''
    return chat([{"role": "user", "content": prompt}])


def faq_chain(query):
    result = get_relevant_qa(query)
    context = "\n".join(r.get("answer") for r in result["metadatas"][0])
    return generate_answer(query, context)


if __name__ == "__main__":
    ingest_faq_data(Path(__file__).parent / "resources/faq_data.csv")
    print("Answer:", faq_chain("Est-ce que je peux payer en espèces ?"))
