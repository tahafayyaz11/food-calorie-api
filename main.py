from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import json
import io
import sys

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

# Load model
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

print("✅ Server ready!")

@app.get("/")
def home():
    return {"status": "Food Calorie API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image")

    img = img.resize((300, 300))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    try:
        preds = model.predict(arr, verbose=0)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    top3_indices = np.argsort(preds)[::-1][:3]

    results = []
    for idx in top3_indices:
        food_name = CLASS_NAMES[idx]
        confidence = float(preds[idx]) * 100
        nutrition = calorie_map.get(food_name, {})
        results.append({
            "food": food_name.replace("_", " ").title(),
            "confidence": round(confidence, 1),
            "calories": nutrition.get("calories", "N/A"),
            "protein": nutrition.get("protein", "N/A"),
            "carbs": nutrition.get("carbs", "N/A"),
            "fat": nutrition.get("fat", "N/A"),
        })

    return {"predictions": results}