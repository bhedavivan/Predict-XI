import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_env():
    """Load .env file manually, looking relative to this script's directory."""
    env_path = os.path.join(_SCRIPT_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

load_env()

API_TOKEN = os.getenv("API_TOKEN", "")
BASE_URL = "https://api.football-data.org/v4"

# The Fixtures dropdown / live-API allow-list = the top-flight leagues the free
# football-data.org tier serves, in strength order, plus the cups. Sourced from
# the canonical registry (leagues.py) so names never diverge from the rest of the
# app (this is where "La Liga" vs "Primera Division" used to disagree).
from leagues import fixtures_leagues
LEAGUE_CODES = fixtures_leagues()
