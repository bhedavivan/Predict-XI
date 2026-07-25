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

# All 12 leagues from Predict-XI plus additional major leagues
LEAGUE_CODES = {
    "PL": "Premier League",
    "SA": "Serie A",
    "BL1": "Bundesliga",
    "PD": "Primera Division",
    "FL1": "Ligue 1",
    "DED": "Eredivisie",
    "PPL": "Primeira Liga",
    "ELC": "Championship",
    "BSA": "Campeonato Brasileiro Série A",
    "CL": "UEFA Champions League",
    "EC": "European Championship",
    "WC": "FIFA World Cup",
    "FL2": "Ligue 2",
    "PD2": "Segunda Division",
    "BL2": "2. Bundesliga",
    "ELC2": "League One",
    "ELC3": "League Two",
    "SA2": "Serie B",
    "PPL2": "Liga Portugal 2",
    "DED2": "Eerste Divisie",
    "BSA2": "Campeonato Brasileiro Série B",
}
