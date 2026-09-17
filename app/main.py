import hashlib
from pathlib import Path

import groq
import streamlit as st

from faq import ingest_faq_data, faq_chain
from llm import has_api_key
from router import build_router
from sql import sql_chain

faqs_path = Path(__file__).parent / "resources/faq_data.csv"

SUGGESTIONS = [
    (":material/undo:", "Quelle est la politique de retour ?"),
    (":material/sell:", "Des Nike avec plus de 30 % de réduction"),
    (":material/star:", "Les 5 chaussures les mieux notées"),
    (":material/credit_card:", "Quels moyens de paiement acceptez-vous ?"),
]

CHITCHAT_ANSWER = (
    "Je suis l'assistant de la boutique. Je réponds aux questions sur les commandes, "
    "le paiement, la livraison et les retours, et je cherche pour vous dans le catalogue "
    "de chaussures de sport.\n\nEssayez par exemple : *Des Puma à moins de 40 € ?*"
)

AVATARS = {"user": ":material/person:", "assistant": ":material/footprint:"}

st.set_page_config(page_title="Assistant shopping", page_icon="👟")

st.html("""
<style>
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3rem; }
.hero { text-align: center; margin: 12vh 0 2rem; }
.hero-eyebrow {
    font-size: 0.75rem; letter-spacing: 0.14em; text-transform: uppercase;
    opacity: 0.55; margin-bottom: 0.75rem;
}
.hero h1 {
    font-family: "Instrument Serif", serif; font-weight: 400;
    font-size: clamp(2.6rem, 7vw, 3.6rem); line-height: 1.05; margin: 0; padding: 0;
}
.hero-sub { opacity: 0.65; max-width: 30rem; margin: 1rem auto 0; line-height: 1.55; }
.stack { text-align: center; font-size: 0.8rem; opacity: 0.45; margin: 2.5rem 0 0; }
.notice {
    text-align: center; font-size: 0.72rem; line-height: 1.5; opacity: 0.4;
    max-width: 28rem; margin: 0.4rem auto 0;
}
.stButton [data-testid="stMarkdownContainer"], .stButton [data-testid="stMarkdownContainer"] p {
    white-space: normal; overflow: visible; text-overflow: clip;
}
.topbar-title { font-family: "Instrument Serif", serif; font-size: 1.6rem; line-height: 2.4rem; }
</style>
""")


@st.cache_resource(show_spinner="Chargement des modèles (premier lancement uniquement)…")
def load(version):
    # `version` change quand la FAQ ou les exemples du routeur changent :
    # le cache est alors invalidé, même si Streamlit Cloud recharge le code à chaud
    ingest_faq_data(faqs_path)
    return build_router()


def data_version():
    files = [faqs_path, Path(__file__).parent / "router.py"]
    return hashlib.md5(b"".join(f.read_bytes() for f in files)).hexdigest()


def ask(router, query):
    try:
        route = router(query)
        if route == "faq":
            return faq_chain(query)
        if route == "sql":
            # sql_chain renvoie None quand la question ne porte pas sur le catalogue
            return sql_chain(query) or faq_chain(query)
        return CHITCHAT_ANSWER
    except groq.RateLimitError:
        return "Trop de demandes en ce moment. Réessayez dans une minute."
    except groq.APIError as e:
        print("Groq error:", e)
        return "Le modèle de langage est indisponible pour le moment. Réessayez plus tard."


def send_suggestion(question):
    st.session_state.pending = question


def reset_conversation():
    st.session_state.messages = []


if not has_api_key():
    st.error("La clé GROQ_API_KEY est absente. Ajoutez-la dans les secrets de l'appli (ou dans un fichier .env en local).")
    st.stop()

router = load(data_version())

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("Posez votre question…") or st.session_state.pop("pending", None)

if not st.session_state.messages and not query:
    st.html("""
    <div class="hero">
      <div class="hero-eyebrow">Assistant shopping</div>
      <h1>Trouvez la bonne paire.</h1>
      <p class="hero-sub">Posez une question sur la boutique ou décrivez ce que vous cherchez
      parmi 903 chaussures de sport.</p>
    </div>
    """)
    left, right = st.columns(2)
    for i, (icon, question) in enumerate(SUGGESTIONS):
        (left if i % 2 == 0 else right).button(
            question, icon=icon, width="stretch", on_click=send_suggestion, args=(question,)
        )
    st.html(
        '<p class="stack">GPT-OSS 120B via Groq · ChromaDB · SQLite</p>'
        '<p class="notice">Projet de démonstration sans but commercial, sans lien avec Flipkart. '
        "Catalogue issu de Flipkart (site marchand indien) en mai 2024, prix convertis en euros "
        "au taux BCE du 20 mai 2024.</p>"
    )
else:
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.html('<div class="topbar-title">Assistant shopping</div>', width="content")
        st.button("Effacer", icon=":material/refresh:", type="tertiary", on_click=reset_conversation)

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])

if query:
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Je cherche…"):
            response = ask(router, query)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
