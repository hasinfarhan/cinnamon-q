"""
LLM Crypto Signal Generator - Utility vs. Hype Scoring

This module provides an LLM-based signal generator that scores crypto news
headlines on a "Halal Utility Scale" (0.0 = Hype/Speculation, 1.0 = Utility/Adoption).

This aligns with Mudarabah principles by favoring fundamental utility over
pure speculation (Maisir/Gambling).
"""

import json
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


class LLMCryptoSignal:
    """
    LLM-based crypto signal generator that scores news headlines for utility.
    
    This class uses OpenAI to analyze crypto news headlines and assign a
    "Utility Score" from 0.0 (Pure Hype/Speculation) to 1.0 (Fundamental Utility).
    
    Scoring Criteria:
    - 1.0 (Utility): Partnerships, mainnet launches, code audits, regulatory clarity,
                     adoption by institutions, technical improvements
    - 0.0 (Hype): "To the moon", meme references, burn events, celebrity tweets,
                  pure speculation without fundamentals
    - 0.5 (Neutral): General market news, price movements without context
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize the LLM Crypto Signal generator.
        
        Args:
            model: OpenAI model to use (default: "gpt-4o-mini" for cost efficiency)
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
            cache_enabled: Whether to use local cache for headlines (default: True)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file or "
                "pass it as api_key parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.cache_enabled = cache_enabled
    
    def _create_prompt(self, headline: str, symbol: str) -> str:
        """
        Create the system prompt for scoring a headline.
        
        Args:
            headline: News headline to score
            symbol: Crypto symbol (e.g., "BTC", "ETH")
        
        Returns:
            Formatted prompt string
        """
        system_prompt = """You are a Sharia-compliant crypto analyst specializing in distinguishing between fundamental utility and speculative hype.

Your task is to analyze crypto news headlines and assign a "Utility Score" from 0.0 to 1.0.

Scoring Guidelines:
- 1.0 (Pure Utility): Real-world adoption, partnerships with established companies, mainnet launches, code audits, regulatory clarity, technical improvements, institutional adoption
- 0.8-0.9 (Strong Utility): Development milestones, protocol upgrades, governance decisions, ecosystem growth
- 0.5-0.7 (Neutral/Mixed): General market news, price movements with some context, routine updates
- 0.2-0.4 (Hype Elements): Celebrity endorsements, social media trends, minor events with excessive marketing
- 0.0 (Pure Hype/Speculation): "To the moon", meme references, burn events without utility, pure price speculation, gambling references

Return ONLY valid JSON in this exact format:
{"score": 0.75, "reason": "Brief explanation"}

Do not include any other text, markdown, or formatting."""
        
        user_prompt = f"""Analyze this headline for {symbol}:

"{headline}"

Return the Utility Score as JSON."""
        
        return system_prompt, user_prompt
    
    def score_headline(self, headline: str, symbol: str) -> Dict[str, any]:
        """
        Score a single headline using OpenAI.
        
        Args:
            headline: News headline to score
            symbol: Crypto symbol (e.g., "BTC", "ETH")
        
        Returns:
            Dictionary with 'score' (float) and 'reason' (str)
            On error, returns {'score': 0.5, 'reason': 'Error message'}
        """
        try:
            system_prompt, user_prompt = self._create_prompt(headline, symbol)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent scoring
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Validate score is in range [0, 1]
            score = float(result.get('score', 0.5))
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
            return {
                'score': score,
                'reason': result.get('reason', 'No reason provided')
            }
            
        except json.JSONDecodeError as e:
            print(f"  ⚠ Warning: Failed to parse JSON response for headline: {e}")
            return {'score': 0.5, 'reason': 'JSON parsing error'}
        except Exception as e:
            print(f"  ⚠ Warning: Error scoring headline: {e}")
            return {'score': 0.5, 'reason': str(e)}
    
    def score_headlines(
        self,
        headlines: List[Dict[str, any]],
        symbol: str
    ) -> float:
        """
        Score multiple headlines and return average utility score.
        
        Args:
            headlines: List of headline dictionaries with 'title' key
            symbol: Crypto symbol (e.g., "BTC", "ETH")
        
        Returns:
            Average utility score (float, 0.0 to 1.0)
            Returns 0.5 (neutral) if no headlines or all errors
        """
        if not headlines:
            return 0.5
        
        scores = []
        for i, item in enumerate(headlines, 1):
            headline = item.get('title', '')
            if not headline:
                continue
            
            print(f"    Scoring headline {i}/{len(headlines)}: {headline[:60]}...")
            result = self.score_headline(headline, symbol)
            scores.append(result['score'])
        
        if not scores:
            return 0.5
        
        avg_score = sum(scores) / len(scores)
        print(f"    ✓ Average utility score: {avg_score:.3f} ({len(scores)} headlines)")
        return avg_score

