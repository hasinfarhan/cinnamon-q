# CryptoPanic API Setup Guide

## Overview

CryptoPanic is a crypto-specific news aggregator that provides better quality headlines for cryptocurrency assets compared to general market news sources like yfinance.

**Benefits:**
- Crypto-specific news (not general market news)
- Better coverage for altcoins
- Real-time updates
- Free tier available

## Setup Instructions

### 1. Get Your API Key

1. Visit [CryptoPanic Developer Portal](https://cryptopanic.com/developers/api/)
2. Register for a free account
3. Navigate to API section
4. Generate your API key (free tier)

### 2. Add to Environment

Add your API key to `.env` file:

```bash
CRYPTOPANIC_API_KEY=your_api_key_here
```

### 3. Verify Setup

Test the integration:

```bash
python -c "from src.data.news import NewsFetcher; f = NewsFetcher(); print(f'Found {len(f.get_headlines(\"BTC\", limit=5))} headlines')"
```

## How It Works

The `NewsFetcher` class:
1. **Tries CryptoPanic first** (if API key is set)
2. **Falls back to yfinance** (if CryptoPanic fails or no key)
3. **Combines results** from both sources
4. **Removes duplicates** and sorts by timestamp

## API Limits

**Free Tier:**
- 100 requests per day
- Sufficient for daily sentiment generation

**Paid Tiers:**
- Higher rate limits
- More features

## Troubleshooting

### Issue: "Missing auth_token parameter"
**Solution:** Add `CRYPTOPANIC_API_KEY` to your `.env` file.

### Issue: "0 headlines found"
**Solution:** 
- Verify API key is correct
- Check API key hasn't expired
- Ensure you're using the correct coin symbol (BTC, not BTC-USD)

### Issue: "Rate limit exceeded"
**Solution:** 
- Wait for rate limit reset (daily for free tier)
- Consider upgrading to paid tier
- Reduce frequency of news fetching

## References

- [CryptoPanic API Documentation](https://cryptopanic.com/developers/api/)
- [CryptoPanic Website](https://cryptopanic.com/)

