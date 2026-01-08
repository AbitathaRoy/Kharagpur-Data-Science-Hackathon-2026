import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# We can initialize different clients based on need, 
# but Groq uses the EXACT same OpenAI python library!
def get_client(provider="groq"):
    if provider == "groq":
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
    else:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# model: "llama-3.3-70b-versatile"
# model: "qwen/qwen3-32b"
# model: "meta-llama/llama-4-scout-17b-16e-instruct"
# model: "gemma2-9b-it"
def call_llm(prompt: str, provider: str = "groq", model: str = "llama-3.3-70b-versatile") -> str:
    """
    Unified caller. 
    Defaulting to Groq + Llama 3.3 (Fast, Free-ish, Smart).
    """
    client = get_client(provider)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data extraction assistant. Output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}, 
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling {provider}: {e}")
        return "{}"