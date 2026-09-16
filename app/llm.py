import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"

_client = None


def has_api_key():
    return bool(os.getenv("GROQ_API_KEY"))


def chat(messages, temperature=0.2, max_tokens=1024):
    # Client créé à la demande : l'import ne plante pas si la clé est absente
    global _client
    if _client is None:
        _client = Groq()
    completion = _client.chat.completions.create(
        model=os.getenv("GROQ_MODEL") or DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content
