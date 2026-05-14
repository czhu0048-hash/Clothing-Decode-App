from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")


import os
import json
import re

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

from .ai_manager import analyze_sustainability, client, MODEL_NAME

#  debug (TBR) 
print(f"DEBUG: Key found? {'Yes' if os.getenv('GEMINI_API_KEY') else 'No'}")
print(f"DEBUG: Using model: {MODEL_NAME}")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["https://reduseaus.me"] in production
    allow_methods=["*"],
    allow_headers=["*"],
)

#  routing 

@app.post("/api/decode")
async def decode_clothing(user_input: dict):
    composition = user_input.get("composition", "")
    brand = user_input.get("brand")
    if not composition:
        raise HTTPException(status_code=400, detail="No composition provided.")
    
    
    raw = ""
    try:
        raw = await analyze_sustainability(composition, brand)
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        result = json.loads(cleaned)
        result["brand"] = brand
        return result
    except json.JSONDecodeError:
        return {"raw_analysis": raw}
    except Exception as e:
        print(f"ERROR in decode_clothing: {e}")
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            raise HTTPException(status_code=429, detail="Gemini API quota exceeded. Please wait and try again.")
        raise HTTPException(status_code=500, detail=msg)


@app.post("/api/extract")
async def extract_label(file: UploadFile = File(...)):
    print(f"--- New Scan Request: {file.filename} ---")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image.")

    
    raw_text = ""
    try:
        image_bytes = await file.read()

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                (
                    "Extract the following fields from this clothing label image and return "
                    "strictly valid JSON with keys: brand, composition, made_in. "
                    "If a field is not visible, use null."
                ),
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=file.content_type or "image/jpeg"
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
        )

        if not response.text:
            return {"brand": None, "composition": None, "made_in": None}

        raw_text = response.text
        cleaned = re.sub(r"```(?:json)?|```", "", response.text).strip()
        return json.loads(cleaned)

    except json.JSONDecodeError:
        return {"brand": None, "composition": None, "made_in": None, "raw": raw_text}
    except Exception as e:
        print(f"CRITICAL ERROR in extract_label: {e}")
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            raise HTTPException(status_code=429, detail="Gemini API quota exceeded. Please wait and try again.")
        raise HTTPException(status_code=500, detail=msg)