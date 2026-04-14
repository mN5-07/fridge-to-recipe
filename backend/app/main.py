# backend/app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import polars as pl
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

app = FastAPI(title="Fridge to Recipe API")

# CORS - allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== Load Data ======================
try:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    processed_path = PROJECT_ROOT / "processed"

    df = pl.read_parquet(processed_path / "recipes_cleaned.parquet")
    
    # Use only first 20,000 recipes for speed during development
    df = df.head(20000)
    
    embeddings = np.load(processed_path / "recipe_embeddings.npy")
    embeddings = embeddings[:20000]   # match the slice above

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"✅ Loaded {len(df):,} recipes (fast mode for development)")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    raise

# ====================== Request Model ======================
class RecommendRequest(BaseModel):
    ingredients: List[str]
    allergies: Optional[List[str]] = None
    top_k: int = 8


# ====================== Recommendation Logic ======================
def recommend_recipes_logic(user_ingredients: List[str], 
                           top_k: int = 8, 
                           allergy_list: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    
    if not user_ingredients:
        return []

    # Encode user query once
    user_text = "Recipe using: " + ", ".join(user_ingredients)
    user_emb = model.encode([user_text])[0]

    # Compute similarities
    similarities = np.dot(embeddings, user_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(user_emb) + 1e-9
    )

    # Get top candidates
    top_indices = np.argsort(similarities)[::-1][:300]

    results = []
    for idx in top_indices:
        recipe = df.row(idx, named=True)
        recipe_ings = recipe.get("cleaned_ingredients", [])

        if len(recipe_ings) < 3 or len(recipe_ings) > 15:
            continue

        # Fast fuzzy matching
        matches = 0
        missing = []
        for rec_ing in recipe_ings:
            best_score = max((fuzz.ratio(u.lower(), rec_ing.lower()) for u in user_ingredients), default=0)
            if best_score >= 72:
                matches += 1
            else:
                missing.append(rec_ing)

        coverage = matches / len(recipe_ings) if recipe_ings else 0

        if coverage < 0.30:
            continue

        final_score = 0.65 * float(similarities[idx]) + 0.35 * coverage

        # Allergy filter
        if allergy_list and any(
            any(fuzz.ratio(a.lower(), ing.lower()) > 85 for ing in recipe_ings)
            for a in allergy_list
        ):
            continue

        results.append({
            "name": str(recipe["name"]),
            "score": round(final_score, 4),
            "coverage_pct": round(coverage * 100, 1),
            "missing_ingredients": [str(ing) for ing in missing[:4]],
            "minutes": int(recipe.get("minutes", 0)) if recipe.get("minutes") is not None else "N/A"
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]


# ====================== Routes ======================
@app.get("/")
async def root():
    return {"message": "Fridge to Recipe API is running"}

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    try:
        results = recommend_recipes_logic(
            request.ingredients, 
            request.top_k, 
            request.allergies
        )
        return {"status": "success", "count": len(results), "recipes": results}
    except Exception as e:
        print(f"Error in /recommend: {e}")
        raise HTTPException(status_code=500, detail=str(e))