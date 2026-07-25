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

LEAGUE_CODES = {
    "WC": "FIFA World Cup",
    "CL": "UEFA Champions League",
    "BL1": "Bundesliga",
    "DED": "Eredivisie",
    "BSA": "Campeonato Brasileiro Série A",
    "PD": "Primera Division",
    "FL1": "Ligue 1",
    "ELC": "Championship",
    "PPL": "Primeira Liga",
    "EC": "European Championship",
    "SA": "Serie A",
    "PL": "Premier League",
}