import os 
from app import app

if _name_ == "_main_":
  port = int(os.environ.get("PORT", 4000))
  app.run(host="0.0.0.0", port=port)
