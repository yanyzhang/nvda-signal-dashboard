# NVDA Signal Dashboard

Investigating whether NVDA's price movement in winter 2025 aligned 
with public attention, AI news sentiment, or broader semiconductor momentum.

## Time Window
January 15, 2025 – March 15, 2025

## Datasets
- NVDA, AMD, ARM, ASML, SOXX daily stock prices
- GDELT GKG Tone Timeline (AI news sentiment)
- Wikimedia Pageviews API (public attention)

## How to Run
pip install -r requirements.txt
python app.py

## Publish to Plotly Cloud
1. Install cloud support: `pip install "dash[cloud]"`
2. Start app in debug mode: `python app.py`
3. Open the app in your browser, open Dash Dev Tools, select Plotly Cloud, sign in, and publish.
