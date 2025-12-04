# llm_client.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_llm(prompt: str) -> str:
    """
    Send a prompt to the LLM and return the response text.

    In Stage 2 the prompt already includes both:
      - the retrieved context
      - the user's question

    So this function just forwards the prompt to the model.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # or gpt-4o if you prefer
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions "
                    "using the provided context. If the context is not "
                    "enough, say that you are not sure."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content