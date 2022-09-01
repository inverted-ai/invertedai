import os
from dotenv import load_dotenv
from invertedai_drive.drive import Drive, Config
from invertedai_drive.utils import Jupyter_Render


load_dotenv()
dev = os.environ.get("DEV", False)
iai = Drive()
if dev:
    dev_url = os.environ.get("DEV_URL", "http://localhost:8000")

__all__ = [Drive, Config, Jupyter_Render]
