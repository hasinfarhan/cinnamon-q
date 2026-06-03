"""
Historical Sentiment Scoring — Gemini 2.5 Flash
=================================================
Generates time-varying LLM utility scores for the thesis dataset.

Rather than fetching only recent headlines (the old approach), this script
asks Gemini to recall what was happening with each asset month-by-month
across the 3-year history.  Gemini's training data covers events up to
early 2025, so it can give informed scores for the 2023-2025 period.

Strategy:
  - For each asset x month (~17 x 36 = ~612 calls), prompt Gemini:
      "In [Month Year], score [ASSET]'s crypto utility on 0-1 (Halal scale)."
  - Interpolate monthly scores to 15-minute bars
  - Save as crypto_deep_v3_sentiment.parquet
  - The model can then be re-trained with real signals

Cost estimate: 612 calls x ~300 tokens avg = ~180k tokens ≈ $0.05 at Gemini pricing.
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from google import genai
from google.genai import types as genai_types

# ── Config ─────────────────────────────────────────────────
DATA_PATH   = project_root / "data" / "processed" / "crypto_deep_v4.parquet"
OUTPUT_PATH = project_root / "data" / "processed" / "crypto_deep_v4_sentiment.parquet"
CACHE_PATH  = project_root / "data" / "processed" / "sentiment_cache.json"
MODEL       = "gemini-2.5-flash"
RATE_LIMIT_DELAY = 8.0   # seconds between calls

ASSET_DESCRIPTIONS = {
    "BTC":   "Bitcoin - the original cryptocurrency, digital gold, store of value",
    "ETH":   "Ethereum - smart contract platform, DeFi backbone, Proof of Stake",
    "SOL":   "Solana - high-speed blockchain, NFT and DeFi ecosystem",
    "AAVE":  "Aave - decentralised lending/borrowing protocol (DeFi)",
    "ALGO":  "Algorand - pure Proof of Stake blockchain, CBDC partnerships",
    "AVAX":  "Avalanche - smart contract platform, subnet architecture",
    "BAT":   "Basic Attention Token - browser advertising utility (Brave browser)",
    "BCH":   "Bitcoin Cash - Bitcoin fork focused on payments",
    "CRV":   "Curve Finance - stablecoin DEX, DeFi yield protocol",
    "GRT":   "The Graph - decentralised indexing protocol for blockchain data",
    "LINK":  "Chainlink - decentralised oracle network, real-world data on-chain",
    "LTC":   "Litecoin - payments-focused Bitcoin fork, digital silver",
    "MKR":   "Maker - governance token of MakerDAO, backs DAI stablecoin",
    "PAXG":  "PAX Gold - gold-backed token, each token = 1 troy oz of physical gold",
    "SUSHI": "SushiSwap - decentralised exchange (DEX fork of Uniswap)",
    "UNI":   "Uniswap - largest decentralised exchange, AMM pioneer",
    "YFI":   "Yearn Finance - yield aggregator and DeFi strategy vault",
}

SYSTEM_PROMPT = """You are a Shariah-compliant crypto investment analyst specialising in
Islamic Finance (Mudarabah principles). You score crypto assets on a UTILITY SCALE:

1.0 = Pure Utility / Real Adoption
  - Active developer ecosystem, real-world use cases
  - Institutional partnerships, protocol upgrades, regulatory clarity
  - Genuine economic activity (not speculation-driven)

0.5 = Neutral / Mixed
  - General market movements, some utility but also speculative interest
  - Moderate developer activity, growing but unproven use cases

0.0 = Pure Hype / Speculation (Maisir)
  - Meme-driven, celebrity promotions, "to the moon" narrative
  - No fundamental utility, pure trading instrument
  - Gambling-like speculative frenzy

Respond ONLY with valid JSON object where each key is an asset symbol and value has score+reasoning:
{"BTC": {"score": 0.75, "reasoning": "..."}, "ETH": {"score": 0.9, "reasoning": "..."}, ...}"""


def score_all_assets_for_month(client, assets: list, year: int, month: int) -> dict:
    """
    Score ALL assets in a single API call for a given month.
    This reduces 612 calls (17 assets x 36 months) to just 36 calls (1 per month).
    Well within any quota tier.
    """
    month_name = datetime(year, month, 1).strftime("%B %Y")

    asset_lines = "\n".join(
        f"- {a}: {ASSET_DESCRIPTIONS.get(a, a + ' cryptocurrency')}"
        for a in assets
    )

    prompt = f"""In {month_name}, score each of these crypto assets on the Halal Utility Scale (0.0-1.0).

Assets to score:
{asset_lines}

For each asset consider what was actually happening in {month_name}:
protocol upgrades, partnerships, developer activity, adoption metrics, market narrative.

Return a single JSON object with one key per asset symbol:
{{"AAVE": {{"score": float, "reasoning": "one sentence"}}, "ALGO": {{"score": float, "reasoning": "one sentence"}}, ...}}

Include ALL {len(assets)} assets. Scores must be between 0.0 and 1.0."""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        result = json.loads(response.text.strip())

        out = {}
        for asset in assets:
            entry = result.get(asset, {})
            score = float(entry.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            out[asset] = {
                "score": score,
                "reasoning": entry.get("reasoning", ""),
                "error": None
            }
        return out

    except Exception as e:
        # Return neutral defaults for all assets on error
        return {a: {"score": 0.5, "reasoning": "API error", "error": str(e)} for a in assets}


def main():
    print("=" * 65)
    print("  HISTORICAL SENTIMENT SCORING — Gemini 2.5 Flash")
    print("=" * 65)

    # ── Init Gemini ────────────────────────────────────────
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not found in .env")
        sys.exit(1)
    client = genai.Client(api_key=gemini_key)
    print(f"Gemini client initialised (model: {MODEL})")

    # ── Load data ──────────────────────────────────────────
    print(f"\nLoading {DATA_PATH.name}...")
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    assets = sorted(df["ticker"].unique().tolist())
    print(f"  Rows: {len(df):,}   Assets: {assets}")

    # ── Date range ─────────────────────────────────────────
    start = df["timestamp"].min().replace(day=1)
    end   = df["timestamp"].max()

    months = []
    cur = start
    while cur <= end:
        months.append((cur.year, cur.month))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    # BATCH approach: 1 API call per month (all 17 assets together)
    # Total calls: 36 months  — well within any quota tier
    total_calls = len(months)
    print(f"  Date range: {months[0]} to {months[-1]}  ({len(months)} months)")
    print(f"  Total API calls: {total_calls} (batch: all assets per month)")
    print(f"  Estimated time: ~{total_calls * RATE_LIMIT_DELAY / 60:.0f} min")

    # ── Load cache (resume support) ────────────────────────
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        # Count only good (non-error) cached entries
        good_cached = sum(1 for v in cache.values() if not v.get("error"))
        print(f"  Resuming from cache: {good_cached} good scores already done")

    # ── Score each month (all assets in one call) ──────────
    print("\nScoring all assets by month (batch mode)...")
    scores_table = {asset: {} for asset in assets}
    done = 0
    errors = 0

    for year, month in months:
        month_name = datetime(year, month, 1).strftime("%Y-%m")

        # Check if ALL assets for this month are already cached (good)
        all_cached = all(
            f"{a}_{year}_{month:02d}" in cache and not cache[f"{a}_{year}_{month:02d}"].get("error")
            for a in assets
        )
        if all_cached:
            for asset in assets:
                key = f"{asset}_{year}_{month:02d}"
                scores_table[asset][(year, month)] = cache[key]["score"]
            done += 1
            print(f"  [{done}/{total_calls}] {month_name}: cached")
            continue

        # Make one batch call for all assets this month
        batch_results = score_all_assets_for_month(client, assets, year, month)

        month_had_error = False
        for asset, result in batch_results.items():
            key = f"{asset}_{year}_{month:02d}"
            scores_table[asset][(year, month)] = result["score"]
            cache[key] = result
            if result.get("error"):
                month_had_error = True

        # Save cache after every month
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)

        done += 1
        if month_had_error:
            errors += 1
            first_err = next(r["error"] for r in batch_results.values() if r.get("error"))
            print(f"  [{done}/{total_calls}] {month_name}: ERROR — {first_err[:80]}")
        else:
            sample = [(a, batch_results[a]["score"]) for a in assets[:4]]
            sample_str = "  ".join(f"{a}={s:.2f}" for a, s in sample)
            print(f"  [{done}/{total_calls}] {month_name}: {sample_str} ...")

        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nScoring complete. {done} months, {errors} errors.")

    # ── Build per-row utility scores ───────────────────────
    print("\nInterpolating monthly scores to 15-min bars...")

    def get_score(row):
        asset = row["ticker"]
        ts    = row["timestamp"]
        year, month = ts.year, ts.month

        asset_scores = scores_table.get(asset, {})
        score = asset_scores.get((year, month), None)

        if score is None:
            # Find nearest month
            keys = sorted(asset_scores.keys())
            if not keys:
                return 0.5
            nearest = min(keys, key=lambda k: abs(k[0]*12 + k[1] - year*12 - month))
            score = asset_scores[nearest]

        return score

    df["utility_score"] = df.apply(get_score, axis=1)

    # Add small gaussian noise within each month for training diversity
    np.random.seed(42)
    df["utility_score"] = (df["utility_score"] + np.random.normal(0, 0.03, size=len(df))).clip(0.0, 1.0)

    # ── Stats ──────────────────────────────────────────────
    print("\nUtility Score Summary (mean per asset):")
    summary = df.groupby("ticker")["utility_score"].agg(["mean", "min", "max", "std"]).round(3)
    print(summary.to_string())

    # ── Save ───────────────────────────────────────────────
    print(f"\nSaving to {OUTPUT_PATH.name}...")
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved: {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size // 1024 // 1024} MB)")
    print(f"  Rows: {len(df):,}   Columns: {len(df.columns)}")

    print("\n" + "=" * 65)
    print("  DONE — now re-train the model using crypto_deep_v3_sentiment.parquet")
    print("=" * 65)


if __name__ == "__main__":
    main()
