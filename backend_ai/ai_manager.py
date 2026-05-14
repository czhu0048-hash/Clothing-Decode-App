import os
import re
from google import genai
from google.genai import types
from .utils import get_context_from_data

MODEL_NAME = "gemini-2.5-flash"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


_local_knowledge = None

def get_knowledge():
    global _local_knowledge
    if _local_knowledge is None:
        _local_knowledge = get_context_from_data()
    return _local_knowledge

def build_system_instruction():
    
    knowledge = get_knowledge()
    return (
        "You are a sustainability expert for RedUse, an app that helps users make "
        "environmentally conscious choices about food and clothing. "
        f"Use the following knowledge base to answer questions:\n\n{knowledge}"
    )

async def analyze_sustainability(composition: str, brand: str | None = None) -> str:
    brand_context = f"Brand: {brand}" if brand else "Brand: Unknown"
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=(
            f"Analyze the environmental impact of this clothing item.\n"
            f"Composition: {composition}\n"
            f"{brand_context}\n\n"
            "Return a JSON object with EXACTLY these keys:\n"
            "{\n"
            '  "materials": [{"name": string, "percent": number, "is_recycled": boolean}],\n'
            '  "carbon_kg": number,\n'
            '  "water_liters": number,\n'
            '  "estimated_lifespan": string,\n'
            '  "care_warnings": [string],\n'
            '  "end_of_life_recommendation": {"option": string, "reason": string},\n'
            '  "badges": [string],\n'
            '  "faqs": [{"question": string}]\n'
            "}"
        ),
        config=types.GenerateContentConfig(
            system_instruction=build_system_instruction(),
            response_mime_type="application/json",
            temperature=0.2,
        )
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response.")
    return response.text