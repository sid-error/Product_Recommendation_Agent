import os
import json
import sys

import pandas as pd
from google.adk.agents import LlmAgent

# ---------------------------------------------------------------------------
# Data layer — loaded once at import time
# ---------------------------------------------------------------------------
_csv_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "product_info.csv")
)
try:
    _df = pd.read_csv(_csv_path)
except Exception as e:
    _df = pd.DataFrame()
    print(f"[WARNING] Could not load product CSV: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Catalog tool functions (passed directly to LlmAgent as tools)
# ---------------------------------------------------------------------------

def get_taxonomy() -> str:
    """
    Returns the list of all available merchantDivision / category / subCategory /
    articleType combinations in the catalog.  Call this if you need to see what
    options exist before asking the user clarifying questions.
    """
    if _df.empty:
        return json.dumps({"error": "No catalog data."})
    cols = [c for c in ["merchantDivision", "category", "subCategory", "articleType"] if c in _df.columns]
    taxonomy = _df[cols].dropna().drop_duplicates()
    return taxonomy.to_json(orient="records")


def search_products(
    merchantDivision: str = None,
    category: str = None,
    subCategory: str = None,
    articleType: str = None,
    max_price: float = None,
) -> str:
    """
    Search the product catalog and return up to 5 matching products.

    All parameters are optional — pass only the ones the user has specified.
    Matching is case-insensitive and word-tokenized (e.g. 'soap bars' will match
    'solid soap bars').

    Args:
        merchantDivision: e.g. 'beauty', 'health & personal care', 'baby'
        category:         e.g. 'Bath & Body', 'Hair Care', 'Skin Care'
        subCategory:      e.g. 'cleansers', 'body washes', 'shampoo & conditioner'
        articleType:      e.g. 'solid soap bars', 'hand wash', 'shampoos'
        max_price:        Maximum offer price (float). Only used if user stated a budget.

    Returns:
        JSON array of matching products with keys:
        title, offerPrice, imageUrl, merchantDivision, category, subCategory, articleType
    """
    if _df.empty:
        return json.dumps({"error": "No catalog data."})

    def _contains(series: pd.Series, term: str) -> pd.Series:
        """
        Tokenize `term` into individual words (≥3 chars) and require ALL of
        them to appear somewhere in the series value (case-insensitive).
        This handles things like 'soap bars' matching 'solid soap bars'.
        """
        words = [
            w for w in term.lower().replace("&", " ").replace(" and ", " ").split()
            if len(w) >= 3
        ]
        if not words:
            return series.str.contains(term, case=False, na=False)
        mask = pd.Series(True, index=series.index)
        for w in words:
            mask &= series.str.contains(w, case=False, na=False)
        return mask

    results = _df.copy()

    if merchantDivision:
        results = results[_contains(results["merchantDivision"], merchantDivision)]
    if category:
        results = results[_contains(results["category"], category)]
    if subCategory:
        results = results[_contains(results["subCategory"], subCategory)]
    if articleType:
        results = results[_contains(results["articleType"], articleType)]
    if max_price is not None:
        results = results[results["offerPrice"] <= max_price]

    if results.empty:
        return json.dumps([])

    results = results.sort_values("offerPrice", ascending=True)
    top = results.head(5)
    cols = [c for c in ["title", "imageUrl", "offerPrice", "merchantDivision", "category", "subCategory", "articleType"] if c in top.columns]
    return top[cols].to_json(orient="records")


# ---------------------------------------------------------------------------
# Single conversational agent — the Runner handles session / turn memory
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a friendly and helpful Sales Assistant for a personal-care & beauty product catalog.

Your job is to help the user find products by having a natural conversation to understand what they are looking for, then searching the catalog and presenting the results beautifully.

## Catalog hierarchy (for your reference)
merchantDivision → category → subCategory → articleType

Available taxonomy (call get_taxonomy() if you need the full list):
- beauty / beauty & personal care / health & personal care / baby
  - Bath & Body → cleansers → solid soap bars | hand wash | body wash gels
  - Bath & Body → body washes → body lotions | body creams
  - Bath & Body → body scrubs → body scrubs
  - Bath & Body → deodorants & antiperspirants → deodorant | antiperspirant deodorant
  - Hair Care → shampoo & conditioner → shampoos | conditioners | 2-in-1 | deep conditioners
  - Hair Care → styling → hair serums | hair sprays & mists | creams, gels & lotions
  - Hair Care → hair oils → hair oils
  - Hair Care → hair masks & packs → hair masks & packs
  - Hair Care → hair loss products → hair regrowth treatments
  - Hair Care → hair care sets → hair care sets
  - Skin Care → face → cleansing creams & milks | creams & moisturisers
  - Personal Care → intimate care & hygiene → intimate care
  - Baby Care → bathing → body washes | baby shampoos
  - Baby Care → gift packs → gift packs

## Conversation rules

1. **Be conversational.** Ask the user questions naturally to narrow down what they want.
2. **Work top-down through the hierarchy.** Start broad (division/category) and drill down only as needed.
3. **Don't over-ask.** If the user gives enough detail, call the search tool immediately without asking more questions. For example:
   - "I want a Dove bar soap" → enough info to call search_products(articleType="solid soap bars", search_keyword or just search directly)
   - "I want a shampoo under ₹300" → ask for subCategory if needed, or search with category="Hair Care", subCategory="shampoo & conditioner", max_price=300
4. **Price:** Only ask for a budget if the user hasn't mentioned one. If they mention "under ₹X" or "below X", capture it as max_price.
5. **When you have enough info, call search_products()** with whatever parameters you know. Do NOT keep asking questions if you already have enough.
6. **After getting results from search_products(), you MUST:**
   - Write a short friendly sentence like "Here's what I found!" 
   - Then on its own line emit EXACTLY this marker (with the real JSON array from the tool, no changes):
     <!-- PRODUCTS_JSON: <paste the exact JSON array from search_products here> -->
   - Do NOT render any product titles, prices, or images yourself. The UI will do that.
7. **If search_products returns an empty list**, tell the user no products were found for those filters and suggest they broaden their search (e.g. remove the price limit or try a different articleType). Do NOT emit the PRODUCTS_JSON marker.
8. **Never show productId.**
9. **Do not hallucinate product names or prices.** Only show what came from search_products().
"""


def create_sales_agent() -> LlmAgent:
    """Returns the single sales agent used by the ADK Runner."""
    return LlmAgent(
        name="SalesAgent",
        description="A conversational sales assistant that helps users find products.",
        model="gemini-2.5-pro",   # upgraded to Gemini 2.5 Pro for better instruction following
        instruction=SYSTEM_PROMPT,
        tools=[search_products, get_taxonomy],
    )
