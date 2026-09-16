import re

import numpy as np

from faq import ef
from llm import chat

ROUTES = {
    "faq": [
        "Quelle est la politique de retour des produits ?",
        "Est-ce que j'ai une réduction avec la carte de fidélité ?",
        "Comment suivre ma commande ?",
        "Quels moyens de paiement acceptez-vous ?",
        "Combien de temps faut-il pour être remboursé ?",
        "Puis-je annuler ou modifier ma commande ?",
        "Livrez-vous à l'étranger ?",
        "J'ai reçu un produit abîmé, que faire ?",
        "Comment utiliser un code promo ?",
        "Y a-t-il des soldes en ce moment ?",
        "What is the return policy of the products?",
        "How can I track my order?",
        "What payment methods are accepted?",
    ],
    "sql": [
        "Je veux des chaussures Nike avec 50 % de réduction.",
        "Avez-vous des chaussures à moins de 30 € ?",
        "Montre-moi les chaussures de running les mieux notées.",
        "Y a-t-il des chaussures Puma en promotion ?",
        "Quel est le prix des baskets Adidas ?",
        "Les 5 chaussures les moins chères de la marque Campus",
        "Chaussures de marche Skechers avec une note supérieure à 4",
        "I want to buy nike shoes that have 50% discount.",
        "Are there any shoes under 30 euros?",
        "What is the price of puma running shoes?",
    ],
    "chitchat": [
        "Bonjour",
        "Salut, ça va ?",
        "Merci",
        "Au revoir",
        "Qui es-tu ?",
        "Tu es un robot ?",
        "Que sais-tu faire ?",
        "Quel temps fait-il aujourd'hui ?",
        "Raconte-moi une blague",
        "Quelle est la capitale de la France ?",
        "Hello",
        "Thank you",
    ],
}

ROUTE_DESCRIPTIONS = """- faq: questions about the store's policies (returns, refunds, payment, delivery, order tracking, cancellation, promo codes)
- sql: searching or asking about products in the catalog (shoes, brands, prices, discounts, ratings)
- chitchat: greetings, thanks, questions about the assistant or anything unrelated to the store"""


def _embed(texts):
    vectors = np.array(ef(texts), dtype=float)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def classify_with_llm(query, candidates):
    prompt = (
        f"Classify the user message into one of these intents:\n{ROUTE_DESCRIPTIONS}\n\n"
        f"Message: {query}\n\nAnswer with a single word: faq, sql or chitchat."
    )
    answer = chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=5).lower()
    found = re.findall(r"faq|sql|chitchat", answer)
    return found[0] if found else candidates[0]


class SemanticRouter:
    """Choisit la route dont un exemple est le plus proche de la question (similarité cosinus).

    Le modèle d'embedding est entraîné surtout sur de l'anglais : quand deux routes
    obtiennent des scores trop proches, le LLM tranche.
    """

    def __init__(self, routes, margin=0.05, fallback=classify_with_llm):
        self.margin = margin
        self.fallback = fallback
        self.route_names = list(routes)
        self.names = np.array([name for name, utterances in routes.items() for _ in utterances])
        self.embeddings = _embed([u for utterances in routes.values() for u in utterances])

    def scores(self, query):
        similarities = self.embeddings @ _embed([query])[0]
        return {name: float(similarities[self.names == name].max()) for name in self.route_names}

    def __call__(self, query):
        ranked = sorted(self.scores(query).items(), key=lambda item: item[1], reverse=True)
        (best, best_score), (second, second_score) = ranked[0], ranked[1]
        if best_score - second_score >= self.margin or self.fallback is None:
            return best
        return self.fallback(query, [best, second])


def build_router():
    return SemanticRouter(ROUTES)


if __name__ == "__main__":
    router = SemanticRouter(ROUTES, fallback=None)
    print(router("Quelle est votre politique pour les produits défectueux ?"))
    print(router("Chaussures Puma roses entre 20 et 50 €"))
