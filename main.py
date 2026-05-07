from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import json
import io
import sys
import os
import base64
import re
import openai
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="Food Calorie Estimator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

CLASS_NAMES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio",
    "beef_tartare", "beet_salad", "beignets", "bibimbap",
    "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad",
    "cannoli", "caprese_salad", "carrot_cake", "ceviche",
    "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros",
    "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee",
    "croque_madame", "cup_cakes", "deviled_eggs", "donuts",
    "dumplings", "edamame", "eggs_benedict", "escargots",
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari",
    "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi",
    "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole",
    "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog",
    "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons",
    "miso_soup", "mussels", "nachos", "omelette",
    "onion_rings", "oysters", "pad_thai", "paella",
    "pancakes", "panna_cotta", "peking_duck", "pho",
    "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake",
    "risotto", "samosa", "sashimi", "scallops",
    "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese",
    "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu",
    "tuna_tartare", "waffles"
]

# Shim: model was saved with a Keras build that stores quantization_config=None
# in every Dense config; current Keras rejects that kwarg. Patch from_config directly
# because custom_objects doesn't override built-in layers in Keras 3.
_orig_dense_from_config = tf.keras.layers.Dense.from_config.__func__
def _dense_compat_from_config(cls, config):
    config = {k: v for k, v in config.items() if k != "quantization_config"}
    return _orig_dense_from_config(cls, config)
tf.keras.layers.Dense.from_config = classmethod(_dense_compat_from_config)

# Load TensorFlow model
print("Loading model...")
try:
    model = tf.keras.models.load_model("food_model_v2.keras")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

try:
    with open("calorie_map.json") as f:
        calorie_map = json.load(f)
except FileNotFoundError:
    print("calorie_map.json not found")
    sys.exit(1)

# OpenAI Vision client — only initialised when the env var is present
_openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = openai.AsyncOpenAI(api_key=_openai_api_key) if _openai_api_key else None

if openai_client:
    print("✅ OpenAI Vision fallback enabled")
else:
    print("⚠️  OPENAI_API_KEY not set — OpenAI Vision fallback disabled")

print("✅ Server ready!")

# Prompt that instructs GPT-4o to return strict JSON only
OPENAI_PROMPT = (
    "You are a nutrition expert. Identify the food in this image.\n\n"
    "Return ONLY a JSON object — no markdown, no code fences, no explanation.\n"
    "The JSON must contain exactly these fields:\n"
    '{\n'
    '  "food_name": "Name of the food in title case",\n'
    '  "confidence": 90,\n'
    '  "calories": 250,\n'
    '  "protein": 12,\n'
    '  "carbs": 30,\n'
    '  "fat": 8,\n'
    '  "serving_size": "100g"\n'
    '}\n\n'
    "Rules:\n"
    "- food_name: proper English name, title case\n"
    "- confidence: integer 0-100 representing your certainty\n"
    "- calories / protein / carbs / fat: integers per 100g serving\n"
    "- serving_size: always the string \"100g\"\n"
    "Return ONLY the JSON object, nothing else."
)


@app.get("/")
def home():
    return {"status": "Food Calorie API is running!", "hybrid_mode": openai_client is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()

    # ── Step 1: decode & preprocess image (unchanged) ────────────────────────
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image")

    img = img.resize((300, 300))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    # ── Step 2: TensorFlow prediction (unchanged logic) ───────────────────────
    try:
        preds = model.predict(arr, verbose=0)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    top3_indices = np.argsort(preds)[::-1][:3]
    top_confidence = float(preds[top3_indices[0]]) * 100

    # Build the TF result list exactly as before, just add source field
    tf_results = []
    for idx in top3_indices:
        food_key = CLASS_NAMES[idx]
        confidence = float(preds[idx]) * 100
        nutrition = calorie_map.get(food_key, {})
        tf_results.append({
            "food":       food_key.replace("_", " ").title(),
            "confidence": round(confidence, 1),
            "calories":   nutrition.get("calories", "N/A"),
            "protein":    nutrition.get("protein",  "N/A"),
            "carbs":      nutrition.get("carbs",    "N/A"),
            "fat":        nutrition.get("fat",      "N/A"),
            "source":     "tensorflow",
        })

    print(f"TF Model: {tf_results[0]['food']} | Confidence: {top_confidence:.1f}%")

    # ── Step 3: Hybrid decision ───────────────────────────────────────────────
    if top_confidence >= 70.0:
        print("✅ Using TensorFlow result")
        return {"predictions": tf_results}
    else:
        print("⚠️ TF confidence low, switching to OpenAI Vision")

    # TF confidence below threshold — try OpenAI Vision
    if openai_client is None:
        # No API key; return TF results as-is
        return {"predictions": tf_results}

    # ── Step 4: OpenAI Vision fallback ───────────────────────────────────────
    try:
        image_b64 = base64.standard_b64encode(contents).decode("utf-8")
        # Normalise content-type to a valid MIME type for the data URL
        ct = file.content_type
        media_type = "image/jpeg" if ct in ("image/jpeg", "image/jpg") else ct

        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":    f"data:{media_type};base64,{image_b64}",
                                "detail": "low",
                            },
                        },
                        {
                            "type": "text",
                            "text": OPENAI_PROMPT,
                        },
                    ],
                }
            ],
        )

        response_text = response.choices[0].message.content.strip()

        # Defensively strip markdown code fences if GPT-4o adds them
        if response_text.startswith("```"):
            response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
            response_text = re.sub(r"\s*```$", "", response_text.strip())

        openai_data = json.loads(response_text)

        # ── Step 5: return OpenAI result ──────────────────────────────────────
        openai_result = {
            "food":       str(openai_data.get("food_name", "Unknown Food")),
            "confidence": float(openai_data.get("confidence", 80)),
            "calories":   openai_data.get("calories",  "N/A"),
            "protein":    openai_data.get("protein",   "N/A"),
            "carbs":      openai_data.get("carbs",     "N/A"),
            "fat":        openai_data.get("fat",       "N/A"),
            "source":     "openai_vision",
        }
        print(f"✅ OpenAI Vision result: {openai_result['food']}")
        return {"predictions": [openai_result]}

    except Exception as e:
        # OpenAI failed — fall back to TF results, flag the error
        print(f"OpenAI Vision fallback failed: {e}")
        for result in tf_results:
            result["openai_error"] = True
        return {"predictions": tf_results}
