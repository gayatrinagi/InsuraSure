from app import app  # your Flask instance is named "app" in app.py
# Expose "application" or "app" for WSGI servers. Render + Gunicorn uses "app".
# If you ever need "application", uncomment the next line:
# application = app