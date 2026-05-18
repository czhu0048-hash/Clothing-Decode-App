import os
import re
from google import genai
from google.genai import types
from .utils import get_context_from_data


MODEL_NAME = "gemini-3.1-flash-lite"
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
        "environmentally conscious choices about food and clothing.\n\n"
        "PRIORITY ORDER for answering:\n"
        "1. Use the knowledge base below first.\n"
        "2. If the knowledge base does not have specific data, use your general training knowledge.\n"
        "3. If still uncertain, use Google Search to find up-to-date figures.\n"
        "4. As a last resort, provide a reasonable estimate based on fibre type averages.\n\n"
        "NEVER say you cannot find data. NEVER return verbose explanations like "
        "'N/A - The knowledge base does not contain...'. "
        "Always return a best estimate with a number.\n\n"
        f"Knowledge base:\n\n{knowledge}"
    )


async def analyze_sustainability(composition: str, brand: str | None = None) -> str:
    brand_context = f"Brand: {brand}" if brand else "Brand: Unknown"

    prompt = (
        f"Analyze the environmental impact of this clothing item.\n"
        f"Composition: {composition}\n"
        f"{brand_context}\n\n"
        "RULES — follow these strictly:\n"
        "1. carbon_kg MUST be a number (float). You MUST always provide an estimate — never null or N/A.\n"
        "   Use Google Search if needed, otherwise estimate from fibre averages:\n"
        "   - Cotton: ~2-5 kg CO2e, Polyester: ~5-15 kg, Wool: ~10-25 kg, Linen: ~1-3 kg, Nylon: ~7-12 kg, Spandex: ~5-10 kg\n"
        "   - Weight the estimate by the percentage of each fibre.\n"
        "2. water_liters MUST be a number (float). You MUST always provide an estimate — never null or N/A.\n"
        "   Use Google Search if needed, otherwise estimate from fibre averages:\n"
        "   - Cotton: ~7,000-10,000L, Polyester: ~100-500L, Wool: ~2,000-5,000L, Linen: ~500-2,000L, Spandex: ~200-800L\n"
        "3. estimated_lifespan MUST be ONLY a short duration string. Examples: '3-5 years', '5-8 years'.\n"
        "   NEVER write a sentence. NEVER include 'Estimated life:' as a prefix. Just the duration range.\n"
        "4. lifespan_note is ONE short sentence (under 20 words) explaining why this composition affects durability.\n"
        "   Example: 'Wool is resilient but elastane degrades faster with heat and friction.'\n"
        "5. care_warnings MUST be a JSON array of short actionable strings. NEVER a plain string.\n"
        "   Include at least 2 specific care tips for the fibre types present.\n"
        "   NEVER return 'N/A' or knowledge-base-not-found messages here.\n"
        "6. badges should reflect fibre properties: e.g. 'organic', 'synthetic', 'biodegradable',\n"
        "   'microplastics', 'recycled', 'mixed-fibre', 'dry-clean-only'.\n"
        "7. brand_rating: if brand is known, give a brief 1-sentence ethical rating. If unknown, use null.\n"
        "8. faqs MUST be an array of objects each with BOTH 'question' AND 'answer' fields.\n"
        "   Provide 4 relevant questions with concise answers about this specific garment's composition.\n"
        "9. NEVER return explanatory sentences like 'not available in the knowledge base' for any field.\n\n"
        "Return a JSON object with EXACTLY these keys:\n"
        "{\n"
        '  "materials": [{"name": string, "percent": number, "is_recycled": boolean}],\n'
        '  "carbon_kg": number,\n'
        '  "water_liters": number,\n'
        '  "estimated_lifespan": string,\n'
        '  "lifespan_note": string,\n'
        '  "care_warnings": [string],\n'
        '  "brand_rating": string | null,\n'
        '  "end_of_life_recommendation": {"option": string, "reason": string},\n'
        '  "badges": [string],\n'
        '  "faqs": [{"question": string, "answer": string}]\n'
        "}"
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=build_system_instruction(),
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.2,
        )
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    cleaned = re.sub(r'^```json\s*|\s*```$', '', response.text.strip())
    return cleaned