"""
Trading environment of the hierarchical agent (thesis Sections 2.5 and 3.3).

Each 4-hour step:
  1. Signal generation (the learned policy) outputs one conviction score per asset;
     scores are squashed into (-1, 1) by tanh.
  2. Portfolio construction turns the scores into weights with a downside
     semicovariance (Sortino) solver, then applies the exposure limits.
  3. A 1% no-trade band suppresses small rebalancing trades.
  4. Rule-based risk gates can override the book: the drawdown circuit breaker
     (with trend-side re-entry), the macro panic gate and the regime-uncertainty gate.
  5. The portfolio earns the next bar's asset returns, net of a 0.1% transaction cost
     on turnover and a 3% per annum borrowing cost on short positions.

The per-step reward (Section 3.3) is used only for training.
"""

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

MAX_LONG = 0.80          # gross long exposure limit
MAX_SHORT = 0.60         # gross short exposure limit
MAX_PER_ASSET = 0.10     # per-asset limit, either direction
NO_TRADE_BAND = 0.01     # minimum per-asset weight change that is traded
TREND_DEAD_ZONE = 0.025  # |BTC / 200-day SMA - 1| at or below this counts as no clear trend


class TradingEnv(gym.Env):
    """Fourteen-asset long/short environment on 4-hour bars.

    Observation (226 values): per asset [return, developer activity, sentiment,
    correlation with BTC, EWMA volatility, RSI, MACD, Bollinger width, ATR, price/SMA,
    4h / 1d / 7d returns] (14 x 13), portfolio value and cash weight (2), current
    weights (14), macro inputs (4), crypto market inputs (4), calendar features (6) and
    each coin's own 200-day trend (14).

    `training=True` is used while training: episodes start at a random bar and the
    inference-only rules (trend-side re-entry after the circuit breaker and the
    regime-uncertainty gate) are off. Evaluation uses `training=False` and starts at bar 0.
    """

    metadata = {"render_modes": []}

    def __init__(self, data: Dict, training: bool = True, initial_cash: float = 100000.0,
                 fee_rate: float = 0.001, warmup_bars: int = 1200, min_episode_bars: int = 1000):
        super().__init__()
        self.training = training
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.warmup_bars = warmup_bars
        self.min_episode_bars = min_episode_bars

        for k in ['returns', 'volatility', 'commit', 'utility', 'price', 'corr', 'garch', 'rsi', 'macd',
                  'bbw', 'atr', 'sma_ratio', 'ret_4h', 'ret_1d', 'ret_7d', 'macro', 'regime',
                  'coin_trend', 'avail']:
            setattr(self, k, data[k])
        self.asset_list = list(data['asset_list'])
        self.timestamps = data['timestamps']
        self.n_steps = int(data['n_steps'])
        self.n_assets = len(self.asset_list)
        self.bars_per_day = int(data['bars_per_day'])

        ts = pd.to_datetime(self.timestamps)
        h = ts.hour.values.astype(np.float32)
        d = ts.dayofweek.values.astype(np.float32)
        m = (ts.month.values - 1).astype(np.float32)
        self.time_features = np.stack([
            np.sin(2 * np.pi * h / 24).astype(np.float32), np.cos(2 * np.pi * h / 24).astype(np.float32),
            np.sin(2 * np.pi * d / 7).astype(np.float32), np.cos(2 * np.pi * d / 7).astype(np.float32),
            np.sin(2 * np.pi * m / 12).astype(np.float32), np.cos(2 * np.pi * m / 12).astype(np.float32),
        ], axis=1)

        # The policy acts in a wide raw range; step() maps it into (-1, 1) with tanh.
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n_assets,), dtype=np.float32)
        n_obs = self.n_assets * 13 + 2 + self.n_assets + 4 + 4 + 6 + self.n_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        # Risk gates
        self.breaker_threshold = 0.15                 # circuit breaker fires at a 15% drawdown
        self.breaker_recovery = 0.07
        self.breaker_cooldown_bars = self.bars_per_day  # one day in cash
        self.macro_threshold = 0.50                   # macro score below this scales positions down
        self.short_borrow_rate = 0.03 / (365 * self.bars_per_day)

        self.current_step = 0
        self.portfolio_value = initial_cash
        self.peak_equity = float(initial_cash)
        self.current_weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0
        self.prev_weights = self.current_weights.copy()
        self.breaker_active = False
        self.breaker_cooldown = 0
        self.reentry_dir = 0
        self.last_components = None

    # ── helpers ────────────────────────────────────────────────────────────
    def listed(self, t: Optional[int] = None) -> np.ndarray:
        """True for coins that are listed and tradable at step t."""
        t = self.current_step if t is None else t
        return np.asarray(self.avail[min(t, len(self.avail) - 1)], dtype=bool)

    def btc_trend(self, t: Optional[int] = None) -> float:
        """BTC price / 200-day SMA - 1, clipped to [-0.5, 0.5]."""
        t = self.current_step if t is None else t
        return float(self.regime[min(t, len(self.regime) - 1), 3]) - 0.5

    # ── portfolio construction ─────────────────────────────────────────────
    def construct_portfolio(self, alpha_scores: np.ndarray) -> np.ndarray:
        """Conviction scores -> weights (last entry = cash).

        Raw weights are the inverse of the 20-day downside semicovariance (10% diagonal
        shrinkage) times the scores. Assets whose weight would take the opposite sign of
        their score are dropped, and the book is scaled by one common factor so that
        initial gross exposure = MAX_LONG x mean absolute conviction. The per-asset and
        side limits are then applied; they can only reduce exposure.
        """
        alpha = np.clip(alpha_scores, -1.0, 1.0).astype(np.float64)
        av = self.listed()
        alpha = np.where(av, alpha, 0.0)
        idx = np.flatnonzero(av)

        W = self.bars_per_day * 20
        t = self.current_step
        hist = self.returns[max(0, t - W):t].astype(np.float64)[:, idx]

        if len(idx) == 0:
            raw_w = np.zeros(self.n_assets)
        elif len(hist) >= 10:
            downside = np.minimum(hist, 0.0)
            semi_cov = (downside.T @ downside) / len(hist)
            semi_cov += max(1e-4, 0.10 * float(np.diag(semi_cov).mean())) * np.eye(len(idx))
            raw_w = np.zeros(self.n_assets)
            try:
                raw_w[idx] = np.linalg.inv(semi_cov) @ alpha[idx]
            except np.linalg.LinAlgError:
                raw_w[idx] = alpha[idx] / np.sqrt(np.maximum(np.diag(semi_cov), 1e-8))
        else:
            # fewer than 10 bars of history: scale by each asset's recent volatility
            vol = self.volatility[t].astype(np.float64)
            raw_w = alpha / np.clip(vol, 1e-8, None)

        mean_conviction = float(np.mean(np.abs(alpha[idx]))) if len(idx) else 0.0
        kept = np.where(np.sign(raw_w) == np.sign(alpha), raw_w, 0.0)
        total = float(np.abs(kept).sum())
        w = kept / total * (MAX_LONG * mean_conviction) if total > 1e-8 else np.zeros_like(kept)
        w = np.clip(w, -MAX_PER_ASSET, MAX_PER_ASSET)
        longs, shorts = np.clip(w, 0.0, None), np.clip(w, None, 0.0)
        if longs.sum() > MAX_LONG:
            longs = longs * (MAX_LONG / longs.sum())
        if -shorts.sum() > MAX_SHORT:
            shorts = shorts * (MAX_SHORT / -shorts.sum())
        weights = (longs + shorts).astype(np.float32)
        cash = float(np.clip(1.0 - np.abs(weights).sum(), 0.0, 1.0))
        return np.concatenate([weights, [cash]]).astype(np.float32)

    def apply_no_trade_band(self, weights: np.ndarray) -> np.ndarray:
        """Keep the held weight of any asset whose target moved by less than the band,
        then re-impose the side limits."""
        w = weights.astype(np.float64).copy()
        prev = np.asarray(self.prev_weights[:-1], dtype=np.float64)
        keep = np.abs(w[:-1] - prev) < NO_TRADE_BAND
        w[:-1] = np.where(keep, prev, w[:-1])
        longs, shorts = np.clip(w[:-1], 0.0, None), np.clip(w[:-1], None, 0.0)
        long_sum, short_sum = longs.sum(), -shorts.sum()
        if long_sum > MAX_LONG:
            longs *= MAX_LONG / long_sum
        if short_sum > MAX_SHORT:
            shorts *= MAX_SHORT / short_sum
        w[:-1] = longs + shorts
        w[-1] = np.clip(1.0 - np.abs(w[:-1]).sum(), 0.0, 1.0)
        return w.astype(np.float32)

    # ── risk gates ─────────────────────────────────────────────────────────
    def apply_risk_gates(self, weights: np.ndarray) -> np.ndarray:
        """Rule-based overrides of the constructed book (Section 2.5.3).

        Drawdown circuit breaker: at a 15% drawdown from the reference peak the book goes
        to cash for one day and the reference peak is reset to the current value. At
        evaluation, trading resumes only once BTC is clearly above or below its 200-day
        SMA, and only on that side until the trend flips.
        Macro panic gate: below a macro score of 0.5 (VIX above about 45), positions are
        scaled down linearly, to no less than 10%.
        Regime-uncertainty gate (evaluation only): cash while BTC is within 2.5% of its
        200-day SMA.
        """
        weights = weights.copy()

        if self.peak_equity > 1e-8:
            drawdown = (self.peak_equity - float(self.portfolio_value)) / self.peak_equity
            if self.breaker_active:
                self.breaker_cooldown = max(0, self.breaker_cooldown - 1)
                recovered = self.breaker_cooldown == 0 and drawdown < self.breaker_recovery
                if recovered and not self.training:
                    sig = self.btc_trend()
                    if abs(sig) <= TREND_DEAD_ZONE:
                        recovered = False                 # no clear trend yet: stay in cash
                    else:
                        self.reentry_dir = int(np.sign(sig))
                if recovered:
                    self.breaker_active = False
                else:
                    weights[:] = 0.0
                    weights[-1] = 1.0
                    return weights

            if not self.breaker_active and drawdown > self.breaker_threshold:
                self.breaker_active = True
                self.breaker_cooldown = self.breaker_cooldown_bars
                self.peak_equity = float(self.portfolio_value)
                self.reentry_dir = 0
                weights[:] = 0.0
                weights[-1] = 1.0
                return weights

            if self.reentry_dir != 0:
                sig = self.btc_trend()
                if sig * self.reentry_dir < -TREND_DEAD_ZONE:
                    self.reentry_dir = 0                  # trend flipped: no restriction
                else:
                    if self.reentry_dir < 0:
                        weights[:-1] = np.minimum(weights[:-1], 0.0)   # shorts only
                    else:
                        weights[:-1] = np.maximum(weights[:-1], 0.0)   # longs only
                    weights[-1] = float(np.clip(1.0 - np.abs(weights[:-1]).sum(), 0.0, 1.0))

        macro_score = float(self.macro[self.current_step, 3])
        if macro_score < self.macro_threshold:
            weights[:-1] *= max(0.10, macro_score / self.macro_threshold)
            weights[-1] = float(np.clip(1.0 - float(np.abs(weights[:-1]).sum()), 0.0, 1.0))

        if not self.training and self.current_step < len(self.regime):
            if abs(float(self.regime[self.current_step, 3]) - 0.5) <= TREND_DEAD_ZONE:
                weights[:-1] = 0.0
                weights[-1] = 1.0
                return weights

        return weights

    # ── reward (Section 3.3) ───────────────────────────────────────────────
    def reward(self, r: float, drawdown: float, alpha: np.ndarray, av: np.ndarray) -> Tuple[float, Dict]:
        """C1 + C2 + C4 + C6.

        C1 = r, or 1.35 r when r < 0 and the drawdown exceeds 0.5%   (net return, loss aversion)
        C2 = -0.005 * drawdown                                       (drawdown drag)
        With D = sign(s_BTC) (0 within the dead zone) and s_i a coin's own 200-day trend,
        the per-coin trend term for a position x_i is
            |s_i| * x_i * D             if the coin's trend agrees with the market,
            |s_BTC| * min(x_i * D, 0)   if it disagrees (only positions against the market count),
            |s_BTC| * x_i * D           if the coin has no 200-day history yet.
        C6 = 2.0 * mean over listed coins with x = portfolio weights;
        C4 = 0.005 * mean over listed coins with x = conviction scores.
        """
        c1 = r * 1.35 if (r < 0.0 and drawdown > 0.005) else r
        c2 = -drawdown * 0.005
        c4 = c6 = 0.0
        t = min(self.current_step, len(self.regime) - 1)
        sb = float(self.regime[t, 3]) - 0.5
        if abs(sb) > TREND_DEAD_ZONE:
            D = float(np.sign(sb))
            s = np.asarray(self.coin_trend[min(t, len(self.coin_trend) - 1)], dtype=np.float64) - 0.5
            d = np.where(np.abs(s) > TREND_DEAD_ZONE, np.sign(s), 0.0)
            no_hist = (s == 0.0)
            agree = (d == D) & ~no_hist

            def term(x):
                xd = x * D
                v = np.where(agree, np.abs(s) * xd, abs(sb) * np.minimum(xd, 0.0))
                v = np.where(no_hist, abs(sb) * xd, v)
                return float(np.mean(v[av]))

            w = np.asarray(self.current_weights[:self.n_assets], dtype=np.float64)
            c4 = 0.005 * term(np.asarray(alpha, dtype=np.float64))
            c6 = 2.0 * term(w)
        return c1 + c2 + c4 + c6, {"C1": c1, "C2": c2, "C4": c4, "C6": c6}

    # ── gym API ────────────────────────────────────────────────────────────
    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        if self.current_step >= self.n_steps:
            return obs
        t, i = self.current_step, 0
        for a in range(self.n_assets):
            obs[i:i + 13] = [self.returns[t, a], self.commit[t, a], self.utility[t, a], self.corr[t, a],
                             self.garch[t, a], self.rsi[t, a], self.macd[t, a], self.bbw[t, a],
                             self.atr[t, a], self.sma_ratio[t, a], self.ret_4h[t, a], self.ret_1d[t, a],
                             self.ret_7d[t, a]]
            i += 13
        obs[i] = self.portfolio_value / self.initial_cash
        obs[i + 1] = self.current_weights[-1]
        i += 2
        obs[i:i + self.n_assets] = self.current_weights[:-1]
        i += self.n_assets
        obs[i:i + 4] = self.macro[t]
        i += 4
        obs[i:i + 4] = self.regime[t]
        i += 4
        obs[i:i + 6] = self.time_features[t]
        i += 6
        obs[i:i + self.n_assets] = np.asarray(self.coin_trend[t], dtype=np.float32) - 0.5
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if self.n_steps > self.warmup_bars + self.min_episode_bars:
            self.current_step = np.random.randint(self.warmup_bars, self.n_steps - self.min_episode_bars)
        else:
            self.current_step = self.warmup_bars
        self.portfolio_value = self.initial_cash
        self.current_weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0
        self.prev_weights = self.current_weights.copy()
        self.peak_equity = float(self.initial_cash)
        self.breaker_active = False
        self.breaker_cooldown = 0
        self.reentry_dir = 0
        self.last_components = None
        return self._get_observation(), {'portfolio_value': float(self.portfolio_value),
                                         'step': int(self.current_step)}

    def step(self, action: np.ndarray):
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        action = np.tanh(action.flatten()[:self.n_assets]).astype(np.float32)

        weights = self.construct_portfolio(action)
        weights = self.apply_no_trade_band(weights)
        weights = self.apply_risk_gates(weights)
        av = self.listed()
        if not av.all():                          # never hold a coin that is not listed yet
            weights[:-1][~av] = 0.0
            weights[-1] = float(np.clip(1.0 - np.abs(weights[:-1]).sum(), 0.0, 1.0))
        self.current_weights = weights

        if self.current_step >= self.n_steps - 1:
            return self._get_observation(), 0.0, True, False, {
                'portfolio_value': float(self.portfolio_value), 'step': int(self.current_step)}

        asset_returns = np.zeros(self.n_assets, dtype=np.float32)
        p0, p1 = self.price[self.current_step, :], self.price[self.current_step + 1, :]
        mask = p0 > 1e-8
        asset_returns[mask] = (p1[mask] / p0[mask]) - 1.0
        gross_return = np.dot(weights[:-1], asset_returns)

        turnover = np.sum(np.abs(weights - self.prev_weights))
        transaction_cost = turnover * self.fee_rate
        short_exposure = float(np.sum(np.abs(np.minimum(weights[:-1], 0.0))))
        short_borrow_cost = short_exposure * self.short_borrow_rate
        net_return = gross_return - transaction_cost - short_borrow_cost

        self.portfolio_value *= (1.0 + net_return)
        self.peak_equity = max(self.peak_equity, float(self.portfolio_value))
        self.prev_weights = weights.copy()

        r = float(net_return)
        drawdown = max(0.0, (self.peak_equity - float(self.portfolio_value)) / self.peak_equity) \
            if self.peak_equity > 1e-8 else 0.0
        reward, self.last_components = self.reward(r, drawdown, action[:self.n_assets], av)

        self.current_step += 1
        observation = self._get_observation()
        terminated = self.current_step >= self.n_steps - 1
        info = {
            'portfolio_value': float(self.portfolio_value),
            'peak_equity': float(self.peak_equity),
            'current_drawdown': float(drawdown),
            'portfolio_return': float(net_return),
            'gross_return': float(gross_return),
            'transaction_cost': float(transaction_cost),
            'short_borrow_cost': float(short_borrow_cost),
            'turnover': float(turnover),
            'long_exposure': float(np.sum(np.maximum(weights[:-1], 0.0))),
            'short_exposure': short_exposure,
            'cash_weight': float(weights[-1]),
            'step': int(self.current_step),
            'reward_components': self.last_components,
            'breaker_active': self.breaker_active,
        }
        return observation, reward, terminated, False, info
