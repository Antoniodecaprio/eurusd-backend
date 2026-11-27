# EUR/USD Distribution Forecaster — Documentation Hub

> Documentation status: Initial reconstruction; subject to review against historical conversations and current code.
> Active development branch: v6-development

This directory is the project's structured, version-controlled knowledge base. It is Wiki-style documentation stored in the repository because direct GitHub Wiki publishing is not currently available through the connected GitHub tools.

## Documentation rules

Every page distinguishes:
- USER REQUIREMENT — explicitly requested or confirmed by the user.
- ACCEPTED DECISION — agreed architectural/product direction.
- PROPOSAL — recommendation awaiting approval.
- VERIFIED IMPLEMENTATION — checked against current source code or deployment.
- UNVERIFIED / RESEARCH — needs evidence or testing.

Do not convert a proposal into a requirement, claim implementation without verification, or erase history when a decision changes.

## Pages
1. [Project Constitution & Vision](01-CONSTITUTION.md)
2. [Product Blueprint](02-PRODUCT-BLUEPRINT.md)
3. [Market Data & Cross-Source Intelligence](03-MARKET-DATA.md)
4. [Quantitative Models & Forecasting](04-QUANT-MODELS.md)
5. [Evidence, Decisions & Risk](05-DECISION-RISK.md)
6. [Validation & Research Protocol](06-VALIDATION.md)
7. [Backend, API & Deployment](07-BACKEND.md)
8. [Platform Independence & Integrations](08-INTEGRATIONS.md)
9. [Roadmap](09-ROADMAP.md)
10. [Architecture Decision Register](10-DECISIONS.md)
11. [Known Limitations & Open Questions](11-LIMITATIONS.md)

## Current development checkpoint
- Repository: Antoniodecaprio/eurusd-backend
- Branch: v6-development
- FastAPI backend: v6_app.py
- Staging: https://eurusd-v6-staging.onrender.com
- Trading and broker execution are disabled.
- Twelve Data primary / Yahoo Finance fallback exist; this is NOT simultaneous independent-source consensus.
- Broker candle alignment remains unverified.

See Roadmap for immediate work sequence.