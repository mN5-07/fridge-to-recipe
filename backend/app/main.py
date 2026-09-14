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
import ast

app = FastAPI(title="Fridge to Recipe API - v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== Load Full Data ======================
try:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    processed_path = PROJECT_ROOT / "processed"

    df = pl.read_parquet(processed_path / "recipes_cleaned.parquet")

    # Use 80,000 recipes for good balance of quality + speed
    df = df.head(80000)

    embeddings = np.load(processed_path / "recipe_embeddings.npy")
    embeddings = embeddings[:80000]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"✅ v2 Backend loaded {len(df):,} recipes with full details")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    raise

# ====================== Request Models ======================
class RecommendRequest(BaseModel):
    ingredients: List[str]
    allergies: Optional[List[str]] = None
    top_k: int = 10
    max_minutes: Optional[int] = 90


# ====================== Ingredient Synonym Mapping ======================
# Maps ingredient names to a shared canonical form so different words for
# the same ingredient (e.g. "scallion" vs "green onion") are treated as
# equivalent during coverage matching. Grow this list as real misses come
# up through actual use -- this is intentionally not exhaustive yet.
INGREDIENT_SYNONYMS = {
    "scallion": "green onion",
    "spring onion": "green onion",
    "garbanzo bean": "chickpea",
    "garbanzo beans": "chickpea",
    "cilantro": "coriander",
    "eggplant": "aubergine",
    "bell pepper": "capsicum",
    # ADD MORE SYNONYMS AS NEEDED
}


def normalize_ingredient(ing: str) -> str:
    ing = ing.lower().strip()
    return INGREDIENT_SYNONYMS.get(ing, ing)


# ====================== Allergy Filtering ======================
# Keywords that indicate a dairy-derived ingredient even when the word
# "dairy" itself doesn't appear in the ingredient name.
DAIRY_KEYWORDS = {"butter", "milk", "cream", "cheese", "yogurt", "whey", "casein", "lactose"}


def has_allergy(recipe_ings: List[str], allergy_list: Optional[List[str]]) -> bool:
    """
    Returns True if any ingredient in recipe_ings conflicts with any
    allergen in allergy_list. Uses direct substring matching in both
    directions plus a fuzzy (token_sort_ratio) match to catch minor
    spelling/word-order variants. The 75% fuzzy threshold is not yet
    empirically validated -- see project notes. Known limitation: substring
    matching can false-positive (e.g. "egg" inside "eggplant").
    """
    if not allergy_list:
        return False

    allergy_lower = [a.lower() for a in allergy_list]

    for ing in recipe_ings:
        ing_lower = ing.lower()

        for allergy in allergy_lower:
            if (
                allergy in ing_lower
                or ing_lower in allergy
                or fuzz.token_sort_ratio(allergy, ing_lower) >= 75
            ):
                return True

        if "dairy" in allergy_lower and any(d in ing_lower for d in DAIRY_KEYWORDS):
            return True

    return False


# ====================== Matching Pass ======================
def run_matching_pass(
    candidate_indices,
    df,
    similarities,
    user_ingredients_normalized,
    allergy_list=None,
    max_minutes=90,
    min_ings=4,
    max_ings=16,
) -> List[Dict[str, Any]]:
    """
    Scores and filters candidate recipes against the user's normalized
    ingredient list. Called twice from recommend_recipes_logic: once with
    the default (strict) constraints, and again with relaxed constraints
    as a real fallback if the strict pass returns too few results.
    """
    results = []

    for idx in candidate_indices:
        recipe = df.row(idx, named=True)

        recipe_ings = [
            normalize_ingredient(ing)
            for ing in recipe.get("cleaned_ingredients", [])
        ]

        # Filter by ingredient-count bounds.
        if len(recipe_ings) < min_ings or len(recipe_ings) > max_ings:
            continue

        # Filter by preparation time.
        if recipe.get("minutes", 999) > max_minutes:
            continue

        # Filter allergens before doing any scoring work.
        if has_allergy(recipe_ings, allergy_list):
            continue

        matches = 0
        missing = []

        for rec_ing in recipe_ings:
            if any(
                u in rec_ing or rec_ing in u
                for u in user_ingredients_normalized
            ):
                matches += 1
            else:
                missing.append(rec_ing)

        coverage = matches / len(recipe_ings) if recipe_ings else 0.0

        # NOTE: these weights (0.55 / 0.35 / 0.10) are not empirically
        # validated -- carried over from the original version pending
        # real tuning against actual usage/feedback.
        final_score = (
            0.55 * float(similarities[idx])
            + 0.35 * coverage
            + 0.10 * (1.0 if recipe.get("minutes", 999) <= 45 else 0.8)
        )

        results.append({
            "id": int(recipe["id"]),
            "name": str(recipe["name"]),
            "score": round(final_score, 4),
            "coverage_pct": round(coverage * 100, 1),
            "missing_ingredients": [str(ing) for ing in missing[:5]],
            "minutes": int(recipe.get("minutes", 0)),
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)


# ====================== Smart Recommendation Logic (v2) ======================
def recommend_recipes_logic(user_ingredients: List[str],
                           top_k: int = 10,
                           allergy_list: Optional[List[str]] = None,
                           max_minutes: int = 90) -> List[Dict[str, Any]]:

    if not user_ingredients:
        return []

    user_text = f"Recipe that uses: {', '.join(user_ingredients)}"
    user_emb = model.encode([user_text])[0]

    similarities = np.dot(embeddings, user_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(user_emb) + 1e-9
    )

    candidate_indices = np.argsort(similarities)[::-1][:600]
    user_ingredients_normalized = [normalize_ingredient(u) for u in user_ingredients]

    # First pass: normal, strict constraints.
    results = run_matching_pass(
        candidate_indices=candidate_indices,
        df=df,
        similarities=similarities,
        user_ingredients_normalized=user_ingredients_normalized,
        allergy_list=allergy_list,
        max_minutes=max_minutes,
        min_ings=4,
        max_ings=16,
    )

    # Fallback: if too few results, retry with relaxed constraints.
    # NOTE: +30 minutes / 2-20 ingredient range are reasonable-sounding
    # defaults, not empirically validated -- revisit once real usage
    # shows whether this fallback actually produces useful results.
    if len(results) < 4:
        results = run_matching_pass(
            candidate_indices=candidate_indices,
            df=df,
            similarities=similarities,
            user_ingredients_normalized=user_ingredients_normalized,
            allergy_list=allergy_list,
            max_minutes=max_minutes + 30,
            min_ings=2,
            max_ings=20,
        )

    return results[:top_k]

# ====================== Routes ======================
@app.post("/recommend")
async def recommend(request: RecommendRequest):
    try:
        results = recommend_recipes_logic(
            request.ingredients,
            request.top_k,
            request.allergies,
            request.max_minutes or 90
        )
        return {"status": "success", "count": len(results), "recipes": results}
    except Exception as e:
        print(f"Error in /recommend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recipe/{recipe_id}")
async def get_recipe(recipe_id: int):
    try:
        recipe = df.filter(pl.col("id") == recipe_id).select([
            "id", "name", "ingredients", "steps", "minutes", "description"
        ]).to_dicts()

        if not recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")

        r = recipe[0]

        # Parse steps if stored as string
        steps = r["steps"]
        if isinstance(steps, str):
            try:
                steps = ast.literal_eval(steps)
            except:
                steps = [steps]

        return {
            "id": int(r["id"]),
            "name": str(r["name"]),
            "ingredients": r["ingredients"],   # original with quantities
            "steps": steps,
            "minutes": int(r["minutes"]) if r["minutes"] else None,
            "description": str(r.get("description", ""))
        }
    except Exception as e:
        print(f"Error fetching recipe {recipe_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Fridge to Recipe API v2 is running"}
    return {"message": "Fridge to Recipe API v2 is running"}