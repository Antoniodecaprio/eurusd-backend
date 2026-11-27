# Market-Data Intelligence

## Reality of FX feeds
Forex is OTC and decentralized: providers may differ in liquidity, quote conventions, time zones, spreads and candle boundaries. A provider's EUR/USD candles are not universal truth.

## Documented current setup
Twelve Data primary when configured; Yahoo Finance fallback. Fallback improves availability but is **not** simultaneous independent corroboration. M15, H1 and D1 are supported; H4 is derived from H1. D1 is provider-defined; broker/session alignment is unverified.

## Adapter/normalization vision
Adapters per provider; retain raw provenance and normalize symbol, OHLC, timestamps, timeframe, candle/session boundaries, volume meaning, validity, freshness and usage/licensing metadata.

## Quality controls
Check malformed OHLC, duplicates, timestamp ordering, staleness, gaps, jumps/outliers, partial data, rate limits, and comparability. Align candles before measuring provider disagreement. Gaps can span weekends/closures; raw elapsed-gap counts are not session-aware missing-bar counts.

## Future consensus
Simultaneously collect feeds (where licensing permits), align equivalent candles, score quality using evaluated evidence, measure disagreement, use robust aggregation/quality weights only if justified, and widen uncertainty/abstain when feeds conflict. Clearly distinguish failover from consensus. Do not silently switch sources inside a series.

## Broker comparison protocol (not yet verified)
Compare Headway MT5 with research data using timestamp-aligned candles, server timezone, candle boundaries, quote basis and representative periods including DST changes. Matching broad trends is insufficient. Broker-alignment flag is documented as unverified/false.