# Architecture

## System Shape

This project has the same core shape as a small production ML system.

### Offline flow

1. Download raw cricket data
2. Parse match JSON into team-perspective rows
3. Build snapshot features and future-looking labels
4. Audit data quality and leakage
5. Train classification and regression models
6. Run unsupervised and advanced experiments
7. Save artifacts for later use

### Online flow

1. Load saved artifacts
2. Load latest team snapshots
3. Score every team or a requested team
4. Return dominance, downfall-risk, and surprise signals

## Dataset Contract

The core training table is `world_cricket_snapshots.csv`. Each row represents one team immediately after one completed T20I match.

### Features

- recent form: rolling win rates, rolling NRR, rolling runs
- match context: opponent, venue, city, tournament, toss decision
- schedule context: rest days, quarter, season, opponent recent form
- match performance: runs, wickets, run rate, NRR

### Labels

- regression target: `future_win_rate_next_5`
- classification target: `dominant_next_cycle`
- supporting decision labels: `downfall_next_cycle`, `surprise_candidate`, `outlook_label`

## Why The Granularity Matters

Using post-match team snapshots lets one dataset support all of Stage-1:

- classification asks whether a team is about to enter a dominant phase
- regression asks how strong the next run of matches will be
- unsupervised learning clusters style and form states
- failure analysis examines mistakes by team, form bucket, and scheduling conditions
- the API serves the latest snapshot per team as an operational prediction surface
