import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"

# Modèles « à raisonnement » : raisonnement réduit au minimum pour garder des réponses rapides
# et ne pas épuiser max_tokens avant la réponse finale
REASONING_EFFORT = {
    "openai/gpt-oss": "low",
    "qwen/": "none",
}

_client = None


def has_api_key():
    return bool(os.getenv("GROQ_API_KEY"))


def get_model():
    return os.getenv("GROQ_MODEL") or DEFAULT_MODEL


def reasoning_params(model):
    for prefix, effort in REASONING_EFFORT.items():
        if model.startswith(prefix):
            return {"reasoning_effort": effort, "include_reasoning": False}
    return {}


def chat(messages, temperature=0.2, max_tokens=2048):
    # Client créé à la demande : l'import ne plante pas si la clé est absente
    global _client
    if _client is None:
        _client = Groq()
    model = get_model()
    completion = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=reasoning_params(model),
    )
    return completion.choices[0].message.content or ""
