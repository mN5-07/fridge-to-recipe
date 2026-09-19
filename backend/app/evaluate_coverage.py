"""
evaluate_coverage.py

Measures average ingredient coverage of recommendations across a set of
realistic ingredient combinations, AND compares against a random-recipe
baseline -- so the resulting number has real context ("better than what?")
rather than being reported in isolation.

Requires the backend server to be running (uvicorn backend.app.main:app)
before this script is run, since it reuses main.py's data/normalization
logic directly (avoids re-implementing the synonym/normalization logic
a second time, which would risk the two drifting out of sync).
"""

import random
import requests
import sys
from pathlib import Path

# Allow importing from backend/app regardless of where this script is run from.
sys.path.insert(0, str(Path(__file__).parent))

from main import df, normalize_ingredient  # noqa: E402  (import after sys.path fix)

API_URL = "http://127.0.0.1:8000/recommend"

TEST_QUERIES = [
    ["chicken", "rice", "broccoli"],
    ["ground beef", "pasta", "tomato sauce"],
    ["eggs", "cheese", "spinach"],
    ["potatoes", "onion", "carrots"],
    ["tofu", "soy sauce", "garlic"],
    ["salmon", "lemon", "asparagus"],
    ["black beans", "corn", "bell pepper"],
    ["shrimp", "garlic", "butter"],
    ["chicken", "sweet potato", "red onion"],
    ["pork chops", "apple", "onion"],
]

RANDOM_SAMPLE_SIZE = 10  # random recipes per query, to compare against the real top-10
RANDOM_SEED = 42  # fixed seed so this is reproducible, not re-rollable until "nice" numbers appear


def coverage_pct(recipe_ings, user_ingredients_normalized):
    if not recipe_ings:
        return 0.0
    matches = sum(
        1 for ing in recipe_ings
        if any(u in ing or ing in u for u in user_ingredients_normalized)
    )
    return round((matches / len(recipe_ings)) * 100, 1)


def random_baseline_coverage(user_ingredients_normalized, rng):
    sample_indices = rng.sample(range(len(df)), RANDOM_SAMPLE_SIZE)
    coverages = []
    for idx in sample_indices:
        recipe = df.row(idx, named=True)
        recipe_ings = [normalize_ingredient(ing) for ing in recipe.get("cleaned_ingredients", [])]
        coverages.append(coverage_pct(recipe_ings, user_ingredients_normalized))
    return sum(coverages) / len(coverages)


def run_evaluation():
    rng = random.Random(RANDOM_SEED)

    system_top1 = []
    system_top10_avg = []
    baseline_avg = []

    for ingredients in TEST_QUERIES:
        response = requests.post(API_URL, json={"ingredients": ingredients, "top_k": 10})
        response.raise_for_status()
        recipes = response.json()["recipes"]

        if not recipes:
            print(f"⚠️  No results for {ingredients} -- skipping")
            continue

        top1 = recipes[0]["coverage_pct"]
        top10_avg = sum(r["coverage_pct"] for r in recipes) / len(recipes)

        user_ingredients_normalized = [normalize_ingredient(i) for i in ingredients]
        baseline = random_baseline_coverage(user_ingredients_normalized, rng)

        system_top1.append(top1)
        system_top10_avg.append(top10_avg)
        baseline_avg.append(baseline)

        print(f"{', '.join(ingredients):40} | system top-1: {top1:5.1f}% "
              f"| system top-10 avg: {top10_avg:5.1f}% | random baseline: {baseline:5.1f}%")

    n = len(system_top1)
    avg_top1 = sum(system_top1) / n
    avg_top10 = sum(system_top10_avg) / n
    avg_baseline = sum(baseline_avg) / n

    print("\n--- Summary ---")
    print(f"Queries evaluated: {n}")
    print(f"Average system top-1 coverage:       {avg_top1:.1f}%")
    print(f"Average system top-10 coverage:      {avg_top10:.1f}%")
    print(f"Average random-baseline coverage:    {avg_baseline:.1f}%")
    if avg_baseline > 0:
        improvement = ((avg_top1 - avg_baseline) / avg_baseline) * 100
        print(f"Improvement over random baseline (top-1 vs baseline): {improvement:.0f}%")


if __name__ == "__main__":
    run_evaluation()