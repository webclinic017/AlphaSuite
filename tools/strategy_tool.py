import random
import json
import traceback
from typing import Dict, Optional

from langchain_core.prompts import PromptTemplate
from tools.file_wrapper import LLMClient, remove_json_marker

# --- Strategy Ingredients ---
INDICATORS = [
    "RSI (Relative Strength Index)", "MACD (Moving Average Convergence Divergence)",
    "Bollinger Bands", "Stochastic Oscillator", "ADX (Average Directional Index)",
    "Ichimoku Cloud", "Fibonacci Retracement", "Volume Profile", "VWAP (Volume Weighted Average Price)",
    "ATR (Average True Range)", "Parabolic SAR", "Donchian Channels", "Keltner Channels",
    "EMA (Exponential Moving Average)", "SMA (Simple Moving Average)",
    "Laguerre RSI", "Hull Moving Average (HMA)", "Money Flow Index (MFI)",
    "Volume Pressure", "MPWR (Market Power)"
]

MARKETS = [
    "Large Cap Tech Stocks (e.g., NVDA, AAPL)",
    "Broad Market Indices (e.g., SPY, QQQ)",
    "Small Cap Equities (e.g., IWM)",
    "Cryptocurrency Majors (e.g., BTC, ETH)",
    "Forex Majors (e.g., EUR/USD)",
    "Commodity ETFs (e.g., GLD, USO)",
    "Treasury Bonds (e.g., TLT)",
    "High Volatility Stocks (e.g., TSLA, AMD)",
    "Dividend Aristocrats"
]

TIMEFRAMES = [
    "15-minute", "1-hour", "4-hour", "Daily", "Weekly"
]

CONCEPTS = [
    "Mean Reversion", "Trend Following", "Breakout Trading", "Momentum",
    "Volatility Contraction", "Gap Fill", "Supply and Demand Zones",
    "Liquidity Sweeps", "Divergence", "Statistical Arbitrage",
    "Regime Detection (Hidden Markov Models)", "Wasserstein Clustering (Optimal Transport)",
    "Sentiment Analysis (NLP)", "Triple Confirmation System", "Trend + Power"
]

RISK_MANAGEMENT_STYLES = [
    "Fixed Fractional Sizing",
    "Kelly Criterion (Half-Kelly)",
    "Volatility Targeting (ATR-based)",
    "Max Drawdown Constraint",
    "Time-based Exits",
    "Trailing Stop (Chandelier Exit)"
]

NARRATIVE_ANGLES = [
    "The Contrarian: Betting against the crowd",
    "The Data Scientist: Pure statistical edge",
    "The Behavioralist: Exploiting human psychology",
    "The Institutionalist: Following big money flow",
    "The Minimalist: Trading with naked charts/few indicators"
]

def generate_strategy_blueprint(llm_client: LLMClient) -> Optional[Dict]:
    """
    Generates a structured technical blueprint for a strategy, designed to be consumed 
    by a Coding Agent to generate a valid Python strategy class.
    """
    # 1. Select Ingredients
    indicator = random.choice(INDICATORS)
    market = random.choice(MARKETS)
    timeframe = random.choice(TIMEFRAMES)
    concept = random.choice(CONCEPTS)
    
    indicator2 = random.choice(INDICATORS) if random.random() > 0.5 else None
    indicators_str = f"{indicator}" + (f" and {indicator2}" if indicator2 and indicator2 != indicator else "")

    print(f"Generating strategy blueprint for: {concept} on {market} ({timeframe})...")

    prompt = f"""
    Act as a Lead Quantitative Architect. Your goal is to design a concrete, programmatic specification for a new trading strategy.
    
    **Context:**
    - Core Concept: {concept}
    - Target Market: {market}
    - Timeframe: {timeframe}
    - Indicators: {indicators_str}

    **Requirement:**
    The output must be a JSON object that maps directly to a Python class inheriting from `BaseStrategy`.
    
    **Output JSON Structure:**
    {{{{
        "strategy_name": "snake_case_name (e.g., rsi_mean_reversion)",
        "class_name": "CamelCaseName (e.g., RsiMeanReversionStrategy)",
        "description": "A brief technical description of the edge.",
        "parameters": {{{{
            "param_name": {{{{ "type": "int|float", "default": value, "tuning_range": [min, max] }}}}
        }}}},
        "required_features": ["list", "of", "column", "names", "needed", "for", "ml", "model"],
        "logic_add_features": "Pseudocode for 'add_strategy_specific_features' method. How to calculate indicators using pandas/ta-lib.",
        "logic_get_setup_mask": "Pseudocode for 'get_setup_mask' method. Boolean logic for entry (True = setup found)."
    }}}}
    
    Ensure the logic is mathematically sound and uses standard libraries (pandas, numpy, talib).
    Return ONLY the JSON object.
    """

    try:
        # Use PromptTemplate to handle the escaping of braces for JSON
        final_prompt = PromptTemplate.from_template(prompt).format()
        response_content = llm_client.get_response(final_prompt).strip()
        json_str = remove_json_marker(response_content)
        blueprint_data = json.loads(json_str)

        if isinstance(blueprint_data, dict) and "class_name" in blueprint_data:
            print(f"Successfully generated Strategy Blueprint: {blueprint_data['class_name']}")
            return blueprint_data
        else:
            print(f"Error: LLM response was not a valid blueprint. Response: {blueprint_data}")
            return None

    except Exception as e:
        print(f"Error generating strategy blueprint: {e}")
        print(traceback.format_exc())
        return None
