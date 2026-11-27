# Project Constitution & Vision

Status: Initial reconstructed specification. User requirements and proposals are distinguished.

## Vision
Build a serious, explainable, research-driven market intelligence and probabilistic forecasting application. Its ambition is to help traders understand what markets are doing, what evidence supports an interpretation, how uncertain the analysis is, and when there is not enough evidence to justify a directional conclusion.

The project began as the EUR/USD Distribution Forecaster and has expanded toward broker-independent, multi-source market intelligence with a future multi-asset architecture.

## Mission
- Observe and compare market data from multiple independent sources.
- Assess data quality, freshness, and disagreement.
- Explain market context across time horizons.
- Produce probabilistic forecasts with calibrated uncertainty.
- Evaluate evidence and risk before presenting trade-oriented scenarios.
- Clearly communicate limitations and reasons to abstain.

## Non-negotiable principles
1. Evidence over confidence theater. More indicators do not automatically mean stronger evidence.
2. Uncertainty is a first-class output. Forecasts should show distributions/ranges, not only point estimates.
3. Explainability. Users should understand factors and assumptions.
4. Abstention is legitimate. HOLD/no actionable conclusion is valid.
5. Data provenance. Retain provider identity, timestamps, and quality status.
6. Separate intelligence and execution. Global research consensus is not a broker executable price.
7. Validation before trust. Poorly performing models must not be marketed as reliable.
8. Risk controls before trading automation. Execution remains disabled during research and validation.
9. Preserve decision history.
10. Mobile-first practicality: interfaces and workflows should account for constrained screen space and limited bandwidth.

## Intended users
Initial focus: individual traders seeking understandable, disciplined, quantitative market analysis.
Future proposal: users across trading platforms/brokers and potentially multi-asset traders.

## Scope
Current proof of concept: EUR/USD research and forecasting, diagnostics, multi-timeframe analysis, backend API, validation tools, and eventual mobile UI.
Long-term proposal: additional forex pairs, indices, commodities, crypto, and cross-asset analytics.

The EUR/USD proof of concept validates architecture and research discipline—not a permanent single-asset ceiling.

## Trust boundaries
- Decision-support/research software, not a profit guarantee.
- Never imply a probability is certainty or a guaranteed recommendation.
- Never hide poor model performance.
- Third-party research quotes are not executable prices.
- Live execution must remain disabled until explicit separate approval and extensive testing.
- Forecasts should be suppressed/downgraded when data or validation gates fail.

## Status labels
USER-REQUIREMENT, ACCEPTED-DECISION, PROPOSAL, IMPLEMENTED-VERIFIED, PARTIAL, UNVERIFIED, RESEARCH-NEEDED, DEPRECATED.

## Historical note
The original concept used long-horizon distribution forecasting anchored to economic equilibrium ideas, then expanded toward short-term trading intelligence, technical/macro evidence, risk management, data validation, and broker/platform independence. Future updates must preserve this evolution.