# Product Blueprint
Status: Reconstructed; incomplete and requires user review.

## Product concept
A mobile-friendly market intelligence workspace combining market data, probabilistic forecasts, multi-timeframe technical context, macro/news context, risk analysis, and validation transparency.

## Proposed user experience areas

### Market Overview
- Current market context and selected instrument.
- Data-source health, timestamps, and market status.
- Multi-timeframe summary.
- Volatility/regime context.
- Visible uncertainty and stale-data warnings.

### Forecast Workspace
- Forecast distribution and fan chart.
- Forecast horizon controls.
- Point estimate plus quantiles/ranges.
- Clearly defined probability event and horizon.
- Model version, inputs, data timestamps, and caveats.

### Technical Analysis
- Trend and structure context.
- Volatility and momentum features.
- Support/resistance and swing structure.
- Indicators with explanation and dependency/correlation awareness.
- Avoid presenting indicator count as evidence strength.

### Macro & News
- Economic calendar with local timezone conversion.
- ECB/Federal Reserve policy and communications.
- Inflation, labor, growth, rates, yields and risk sentiment.
- Geopolitical/event risk with sources and timestamps.
- Separate observed facts from interpretation.

### Risk & Scenario Analysis
- Risk-per-trade and position sizing calculator.
- Stop/target scenario analysis.
- Drawdown, exposure and reward/risk views.
- Broker contract, pip value, spread, commission and slippage inputs must be broker-specific and verified.

### Backtest & Model Health
- Walk-forward results.
- Baselines and benchmark comparisons.
- Calibration and reliability.
- Drawdown and cost-adjusted metrics.
- Version comparison and validation status.

### Data & Source Transparency
- Provider-level values and timestamps.
- Source status/fallback state.
- Feed disagreement/outlier warnings.
- Consensus calculation method and uncertainty, when implemented.

## UI direction
Previously preferred: dark trading-terminal style, serious and authentic chart-centric presentation rather than generic AI imagery. Design tokens, screen mockups, navigation, responsive behavior, accessibility and visual references need a dedicated review.

## Mobile-first constraints
- Optimize for Android screen sizes and limited bandwidth.
- Provide compact API response variants.
- Keep critical status visible without oversized chart payloads.
- Readable controls; avoid tiny text.
- Support typed/manual workflows when screenshots/uploads are unavailable.

## Not yet specified
Exact navigation, screen hierarchy, chart library, UI component inventory, theme tokens, notification settings, accounts, subscriptions, and accessibility requirements.