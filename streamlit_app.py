"""Entry point for Streamlit Community Cloud.

The app itself lives in app.py. Streamlit Community Cloud's default main file
path is `streamlit_app.py`, so this thin shim just delegates to it — no code is
duplicated, and `app.py` stays the single source of truth.
"""

from app import main

main()
