# Publish This Dashboard to Plotly Cloud

## 1) Install Cloud publish support
```bash
pip install "dash[cloud]"
```

## 2) Run the app locally (debug mode required)
```bash
python app.py
```

Open the app URL shown in terminal (usually http://127.0.0.1:8050/).

## 3) Publish from Dash Dev Tools
1. Open Dash Dev Tools panel.
2. Click **Plotly Cloud** and sign in.
3. Enter app name `nvda-signal-dashboard`.
4. Choose your team/workspace.
5. Click **Publish App**.

## Notes
- Plotly Cloud direct-upload limit: 80 MiB.
- Dash dev-tools publish limit: 200 MiB.
- The app reads local CSV files from the project root, so keep data files in this folder.
