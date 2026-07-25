#!/usr/bin/env python3
"""
Predict-XI — Flask web UI.
A polished front-end over the from-scratch match-outcome model.
Serves on http://localhost:5000
"""

import os
import json
import urllib.parse
from datetime import datetime

from flask import Flask, render_template_string, request, redirect

# Resolve paths relative to this script so it runs from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from config import LEAGUE_CODES
from api_client import fetch_upcoming_matches, MissingTokenError
from data_processor import prepare_prediction_features, load_data, compute_team_stats
from model_trainer import MatchPredictorModel

app = Flask(__name__)
app.jinja_env.filters['urlencode'] = lambda s: urllib.parse.quote(str(s), safe='')


# ─── Shared design system (injected into every page) ──────────────────────

BASE_CSS = r"""
:root {
  --bg: #060b16;
  --bg-2: #0b1424;
  --panel: rgba(22, 33, 54, 0.72);
  --panel-solid: #16213a;
  --border: rgba(120, 150, 200, 0.16);
  --text: #e8eefc;
  --muted: #93a4c4;
  --faint: #64748b;
  --home: #38e8b0;   /* green */
  --draw: #f5c451;   /* amber */
  --away: #6aa8ff;   /* blue  */
  --accent: #7c5cff; /* violet */
  --accent-2: #22d3ee;
  --danger: #ff6b7d;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  color: var(--text);
  min-height: 100vh;
  background:
    radial-gradient(1100px 600px at 15% -10%, rgba(124,92,255,0.22), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,0.16), transparent 55%),
    radial-gradient(800px 700px at 50% 120%, rgba(56,232,176,0.10), transparent 60%),
    var(--bg);
  background-attachment: fixed;
}
a { color: inherit; text-decoration: none; }
.wrap { max-width: 1040px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }

.nav { display: flex; align-items: center; justify-content: space-between; margin-bottom: 2.2rem; }
.brand { display: flex; align-items: center; gap: .6rem; font-weight: 800; font-size: 1.15rem; letter-spacing: -0.02em; }
.brand .logo {
  width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center;
  background: linear-gradient(135deg, var(--accent), var(--accent-2)); font-size: 1.1rem;
  box-shadow: 0 6px 20px rgba(124,92,255,0.45);
}
.brand small { color: var(--muted); font-weight: 500; }
.nav-links { display: flex; gap: 1.2rem; font-size: .92rem; color: var(--muted); }
.nav-links a:hover { color: var(--text); }

.hero { margin-bottom: 2rem; }
.hero h1 {
  font-size: clamp(2.1rem, 5vw, 3.2rem); font-weight: 850; letter-spacing: -0.03em; line-height: 1.05;
  background: linear-gradient(120deg, #fff 20%, var(--accent-2) 60%, var(--home));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: var(--muted); margin-top: .7rem; font-size: 1.05rem; max-width: 42rem; }

.card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 18px;
  padding: 1.5rem; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 20px 50px rgba(2, 6, 20, 0.45);
}
.card + .card { margin-top: 1.25rem; }
.card h2 { font-size: 1.05rem; font-weight: 700; margin-bottom: 1rem; display: flex; align-items: center; gap: .5rem; }
.card h2 .tag { font-size: .68rem; font-weight: 700; color: var(--accent-2); border: 1px solid rgba(34,211,238,.35);
  padding: .12rem .5rem; border-radius: 999px; letter-spacing: .04em; text-transform: uppercase; }

.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .9rem; }
.stat {
  background: rgba(10, 17, 33, 0.55); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.1rem;
  position: relative; overflow: hidden;
}
.stat::after { content:""; position:absolute; inset:0 auto 0 0; width:3px;
  background: linear-gradient(var(--accent), var(--accent-2)); opacity:.9; }
.stat .v { font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; }
.stat .l { color: var(--muted); font-size: .78rem; margin-top: .15rem; }
.stat .sub { color: var(--faint); font-size: .72rem; margin-top: .3rem; }

label { display: block; color: var(--muted); font-size: .82rem; font-weight: 600; margin-bottom: .4rem; }
select, input[type=text], button {
  width: 100%; padding: .8rem 1rem; border-radius: 12px; border: 1px solid var(--border);
  background: rgba(8, 14, 28, 0.75); color: var(--text); font-size: .98rem; font-family: inherit;
}
select:focus, input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(124,92,255,.22); }
.btn {
  border: none; cursor: pointer; font-weight: 700; color: #08101f;
  background: linear-gradient(135deg, var(--accent-2), var(--home));
  transition: transform .12s ease, box-shadow .2s ease; box-shadow: 0 10px 24px rgba(34,211,238,.28);
}
.btn:hover { transform: translateY(-1px); box-shadow: 0 14px 30px rgba(34,211,238,.4); }
.btn.violet { background: linear-gradient(135deg, var(--accent), #b06bff); color: #fff; box-shadow: 0 10px 24px rgba(124,92,255,.4); }
.btn.ghost { background: rgba(255,255,255,.04); color: var(--text); border: 1px solid var(--border); box-shadow: none; }
.row { display: grid; grid-template-columns: 1fr auto 1fr; gap: .8rem; align-items: end; }
@media (max-width: 620px){ .row { grid-template-columns: 1fr; } .vs-mid { display:none; } }
.vs-mid { text-align: center; color: var(--faint); font-weight: 800; padding-bottom: .8rem; }

.alert { padding: .9rem 1.1rem; border-radius: 12px; margin-bottom: 1.2rem; font-size: .92rem; }
.alert.err { background: rgba(255,107,125,.12); border: 1px solid rgba(255,107,125,.4); color: #ffc2ca; }
.alert.warn { background: rgba(245,196,81,.12); border: 1px solid rgba(245,196,81,.4); color: #ffe9ad; }
.alert.ok { background: rgba(56,232,176,.12); border: 1px solid rgba(56,232,176,.4); color: #b6ffe6; }

.pill { display:inline-flex; align-items:center; gap:.4rem; font-size:.75rem; font-weight:700; padding:.25rem .6rem;
  border-radius: 999px; border:1px solid var(--border); color: var(--muted); }
.dot { width:8px; height:8px; border-radius:50%; }
.muted { color: var(--muted); }
.foot { margin-top: 2.5rem; text-align:center; color: var(--faint); font-size:.8rem; }
.foot code { color: var(--muted); }
"""


def head(title):
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;850&display=swap" rel="stylesheet">'
        f'<title>{title}</title><style>{BASE_CSS}</style></head><body><div class="wrap">'
    )


NAV = r"""
<div class="nav">
  <a class="brand" href="/"><span class="logo">⚽</span><span>Predict-XI<br><small>match outcome engine</small></span></a>
  <div class="nav-links"><a href="/">Dashboard</a><a href="/#predict">Predict</a><a href="/#leagues">Fixtures</a></div>
</div>
"""

FOOT = r"""
<div class="foot">Predict-XI · softmax regression + Elo, built from scratch (Python stdlib only)<br>
Data: <code>football-data.org</code> (live fixtures) · <code>footballcsv</code> (training)</div>
</div></body></html>
"""


# ─── Templates ────────────────────────────────────────────────────────────

HOME_TEMPLATE = head("Predict-XI · Dashboard") + NAV + r"""
<div class="hero">
  <h1>Predict the match<br>before it's played.</h1>
  <p>A from-scratch machine-learning engine rates every team with a running Elo and
     softmax-regression model trained on {{ "{:,}".format(metrics.get('training_samples', 0)) }}
     matches across {{ n_leagues }} leagues.</p>
</div>

{% if error %}<div class="alert err">{{ error }}</div>{% endif %}
{% if warning %}<div class="alert warn">{{ warning }}</div>{% endif %}

<div class="card">
  <h2>Model performance <span class="tag">live</span></h2>
  {% if metrics %}
  <div class="stat-grid">
    <div class="stat"><div class="v"><span class="cnt" data-to="{{ (metrics.accuracy*100)|round(1) }}">0</span>%</div>
      <div class="l">Accuracy</div><div class="sub">temporal holdout</div></div>
    <div class="stat"><div class="v">+<span class="cnt" data-to="{{ ((metrics.accuracy-metrics.baseline_accuracy)*100)|round(1) }}">0</span></div>
      <div class="l">Lift vs baseline</div><div class="sub">over always-home</div></div>
    <div class="stat"><div class="v"><span class="cnt" data-dec="3" data-to="{{ metrics.log_loss }}">0</span></div>
      <div class="l">Log-loss</div><div class="sub">{{ 'calibrated' if metrics.log_loss < 1.0986 else 'ok' }} (&lt;1.099)</div></div>
    <div class="stat"><div class="v"><span class="cnt" data-to="{{ metrics.training_samples }}">0</span></div>
      <div class="l">Matches trained</div><div class="sub">5 seasons</div></div>
    <div class="stat"><div class="v"><span class="cnt" data-to="{{ metrics.get('n_teams', teams|length) }}">0</span></div>
      <div class="l">Teams rated</div><div class="sub">by Elo</div></div>
  </div>
  {% else %}
  <p class="muted">No trained model found. Run <code>python main.py --source csv --train eng.1 es.1 de.1 it.1 fr.1</code>
     to train one, then refresh.</p>
  {% endif %}
</div>

<div class="card" id="predict">
  <h2>Predict a matchup</h2>
  {% if teams %}
  <form method="GET" action="/predict">
    <div class="row">
      <div><label for="home">Home team</label>
        <input list="teamlist" id="home" name="home" placeholder="e.g. Arsenal FC" autocomplete="off" required></div>
      <div class="vs-mid">vs</div>
      <div><label for="away">Away team</label>
        <input list="teamlist" id="away" name="away" placeholder="e.g. Chelsea FC" autocomplete="off" required></div>
    </div>
    <datalist id="teamlist">
      {% for t in teams %}<option value="{{ t.name }}">{{ t.tier }} · Elo {{ t.elo|round|int }}</option>{% endfor %}
    </datalist>
    <button class="btn violet" type="submit" style="margin-top:1rem;">Run prediction →</button>
  </form>
  {% else %}
  <p class="muted">Team ratings load once a model is trained.</p>
  {% endif %}
</div>

<div class="card" id="leagues">
  <h2>Browse live fixtures</h2>
  <form method="GET" action="/fixtures">
    <label for="league">League</label>
    <select name="league" id="league">
      <option value="">— choose a league —</option>
      {% for code, name in leagues.items() %}<option value="{{ code }}">{{ name }} ({{ code }})</option>{% endfor %}
    </select>
    <button class="btn" type="submit" style="margin-top:1rem;">View upcoming fixtures →</button>
  </form>
  <p class="muted" style="margin-top:.8rem; font-size:.82rem;">Requires a free football-data.org API token in <code>.env</code>.</p>
</div>
""" + FOOT + r"""
<script>
document.querySelectorAll('.cnt').forEach(function(el){
  var to = parseFloat(el.dataset.to) || 0, dec = parseInt(el.dataset.dec || '0');
  var start = null, dur = 900;
  function step(ts){ if(!start) start = ts; var p = Math.min((ts-start)/dur, 1);
    var val = to * (1 - Math.pow(1-p, 3));
    el.textContent = dec ? val.toFixed(dec) : Math.round(val).toLocaleString();
    if(p < 1) requestAnimationFrame(step); }
  requestAnimationFrame(step);
});
</script>
</body></html>
"""

PREDICT_TEMPLATE = head("Predict-XI · Prediction") + NAV + r"""
{% if error %}<div class="alert err">{{ error }}</div>{% endif %}
{% if warning %}<div class="alert warn">{{ warning }}</div>{% endif %}

{% if prediction %}
<div class="card">
  <div style="display:grid; grid-template-columns:1fr auto 1fr; gap:1rem; align-items:center; text-align:center; margin-bottom:1.5rem;">
    <div>
      <div style="font-size:1.35rem; font-weight:800;">{{ home_team }}</div>
      <div class="pill" style="margin-top:.5rem;"><span class="dot" style="background:var(--home)"></span>{{ home_info.tier }} · Elo {{ home_info.elo|round|int }}</div>
    </div>
    <div style="color:var(--faint); font-weight:800; font-size:1.1rem;">VS</div>
    <div>
      <div style="font-size:1.35rem; font-weight:800;">{{ away_team }}</div>
      <div class="pill" style="margin-top:.5rem;"><span class="dot" style="background:var(--away)"></span>{{ away_info.tier }} · Elo {{ away_info.elo|round|int }}</div>
    </div>
  </div>

  <div style="text-align:center; margin-bottom:1.3rem;">
    <div class="muted" style="font-size:.8rem; letter-spacing:.05em; text-transform:uppercase;">Predicted outcome</div>
    <div style="font-size:2rem; font-weight:850; margin-top:.2rem;
      color: {% if prediction.prediction=='Home Win' %}var(--home){% elif prediction.prediction=='Away Win' %}var(--away){% else %}var(--draw){% endif %};">
      {{ prediction.prediction }}</div>
    <span class="pill" style="margin-top:.4rem;">{{ confidence_label }} confidence · {{ (top_prob*100)|round|int }}%</span>
  </div>

  <!-- stacked probability bar -->
  <div style="display:flex; height:16px; border-radius:999px; overflow:hidden; border:1px solid var(--border); margin-bottom:1.2rem;">
    <div class="seg" style="width:0; background:var(--home);" data-w="{{ (p_home*100)|round(1) }}"></div>
    <div class="seg" style="width:0; background:var(--draw);" data-w="{{ (p_draw*100)|round(1) }}"></div>
    <div class="seg" style="width:0; background:var(--away);" data-w="{{ (p_away*100)|round(1) }}"></div>
  </div>

  <div class="stat-grid">
    {% for name, prob, col in bars %}
    <div class="stat" style="border-color: {% if name==prediction.prediction %}{{ col }}{% else %}var(--border){% endif %};">
      <div class="v" style="color:{{ col }};">{{ (prob*100)|round(1) }}%</div>
      <div class="l">{{ name }}</div>
      <div style="height:6px; border-radius:999px; background:rgba(255,255,255,.07); margin-top:.5rem; overflow:hidden;">
        <div class="minibar" style="height:100%; width:0; background:{{ col }};" data-w="{{ (prob*100)|round(1) }}"></div>
      </div>
    </div>
    {% endfor %}
  </div>

  <p class="muted" style="margin-top:1.2rem; font-size:.85rem;">
    Elo edge: <strong style="color:var(--text)">{{ home_team if elo_diff>=0 else away_team }}</strong>
    by {{ elo_diff|abs|round|int }} points{% if not both_known %} · ⚠ limited data for one team, prediction may be weak{% endif %}.
  </p>
</div>

<div style="display:flex; gap:.8rem; margin-top:1.25rem; flex-wrap:wrap;">
  <a class="btn violet" href="/#predict" style="width:auto; padding:.7rem 1.4rem;">← Predict another</a>
  <a class="btn ghost" href="/" style="width:auto; padding:.7rem 1.4rem;">Dashboard</a>
</div>

{% elif not model_exists %}
<div class="card">
  <h2>No trained model</h2>
  <p class="muted">Train a model first, then come back to predict.</p>
  <form method="POST" action="/train" style="margin-top:1rem;">
    <input type="hidden" name="redirect" value="/predict?home={{ home_team|urlencode }}&away={{ away_team|urlencode }}">
    <button class="btn violet" type="submit" style="width:auto; padding:.7rem 1.4rem;">Train model now</button>
  </form>
</div>
{% endif %}
""" + FOOT + r"""
<script>
window.addEventListener('load', function(){
  setTimeout(function(){
    document.querySelectorAll('.seg').forEach(function(e){ e.style.transition='width 1s cubic-bezier(.2,.8,.2,1)'; e.style.width=e.dataset.w+'%'; });
    document.querySelectorAll('.minibar').forEach(function(e){ e.style.transition='width 1.1s cubic-bezier(.2,.8,.2,1)'; e.style.width=e.dataset.w+'%'; });
  }, 80);
});
</script>
</body></html>
"""

FIXTURES_TEMPLATE = head("Predict-XI · Fixtures") + NAV + r"""
<div class="hero"><h1 style="font-size:2rem;">{{ league_name }}</h1><p>Upcoming fixtures · {{ league_code }}</p></div>
{% if error %}<div class="alert err">{{ error }}</div>{% endif %}
<div class="card">
  <h2>Fixtures</h2>
  {% if fixtures %}
  <div style="display:flex; flex-direction:column; gap:.6rem;">
    {% for m in fixtures %}
    <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem; padding:.9rem 1rem;
      background:rgba(10,17,33,.5); border:1px solid var(--border); border-radius:12px;">
      <div>
        <div style="font-weight:700;">{{ m.home_team }} <span class="muted">vs</span> {{ m.away_team }}</div>
        <div class="muted" style="font-size:.8rem; margin-top:.15rem;">{{ m.date }}</div>
      </div>
      <a class="btn" style="width:auto; padding:.55rem 1.1rem; font-size:.88rem;"
         href="/predict?home={{ m.home_team|urlencode }}&away={{ m.away_team|urlencode }}">Predict</a>
    </div>
    {% endfor %}
  </div>
  {% else %}
  <p class="muted">No upcoming fixtures found — the season may be between rounds, or the API token isn't set.</p>
  {% endif %}
</div>
<div style="margin-top:1.25rem;"><a class="btn ghost" href="/" style="width:auto; padding:.7rem 1.4rem;">← Dashboard</a></div>
""" + FOOT

TRAINING_TEMPLATE = head("Predict-XI · Training") + NAV + r"""
{% if error %}<div class="alert err">{{ error }}</div>{% endif %}
{% if success %}
<div class="alert ok">{{ success }}</div>
<div class="card">
  <h2>Training results</h2>
  <div class="stat-grid">
    <div class="stat"><div class="v">{{ (accuracy*100)|round(1) }}%</div><div class="l">Accuracy</div></div>
    <div class="stat"><div class="v">{{ train_samples }}</div><div class="l">Train samples</div></div>
    <div class="stat"><div class="v">{{ test_samples }}</div><div class="l">Test samples</div></div>
  </div>
</div>
{% if redirect_url %}<div style="margin-top:1.25rem;"><a class="btn violet" href="{{ redirect_url }}" style="width:auto; padding:.7rem 1.4rem;">→ Continue to prediction</a></div>{% endif %}
{% endif %}
<div style="margin-top:1.25rem;"><a class="btn ghost" href="/" style="width:auto; padding:.7rem 1.4rem;">← Dashboard</a></div>
""" + FOOT


# ─── Helpers ──────────────────────────────────────────────────────────────

def _load_json(name):
    try:
        with open(os.path.join(SCRIPT_DIR, name)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_metrics():
    return _load_json("model_metrics.json") or {}


def build_team_stats():
    """Team stats for predictions. Prefer the committed team_stats.json, then
    the full processed_data.json, then the API-path matches_data.json."""
    stats = _load_json("team_stats.json")
    if stats:
        return stats
    processed = _load_json("processed_data.json")
    if processed and processed.get("team_stats"):
        return processed["team_stats"]
    rows = load_data()
    return compute_team_stats(rows) if rows else {}


def model_exists():
    return os.path.exists(os.path.join(SCRIPT_DIR, "model.json"))


def get_model():
    model = MatchPredictorModel()
    return model if model.load() else None


def team_tier(elo):
    if elo >= 1650:
        return "Elite"
    if elo >= 1560:
        return "Strong"
    if elo >= 1470:
        return "Mid-table"
    return "Underdog"


def team_list(stats):
    out = []
    for name, s in stats.items():
        if not name or s.get("matches_played", 0) <= 0:
            continue
        elo = s.get("elo", 1500)
        out.append({"name": name, "elo": elo, "tier": team_tier(elo)})
    out.sort(key=lambda t: t["elo"], reverse=True)
    return out


def team_info(stats, name):
    s = stats.get(name, {})
    elo = s.get("elo", 1500)
    return {"elo": elo, "tier": team_tier(elo), "known": bool(s)}


# ─── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def home():
    stats = build_team_stats()
    return render_template_string(
        HOME_TEMPLATE,
        metrics=load_metrics(),
        teams=team_list(stats),
        leagues=LEAGUE_CODES,
        n_leagues=len(LEAGUE_CODES),
        model_exists=model_exists(),
        error=request.args.get("error", ""),
        warning=request.args.get("warning", ""),
    )


@app.route("/fixtures")
def fixtures():
    league_code = request.args.get("league", "").strip()
    if not league_code:
        return redirect("/?error=" + urllib.parse.quote("Please select a league."))
    if league_code not in LEAGUE_CODES:
        return redirect("/?error=" + urllib.parse.quote(f"Unknown league code: {league_code}"))

    try:
        raw = fetch_upcoming_matches(league_code)
    except MissingTokenError as e:
        return redirect("/?error=" + urllib.parse.quote(str(e)))
    except Exception as e:
        return redirect("/?error=" + urllib.parse.quote(f"API error: {e}"))

    fixtures_list = []
    for m in raw:
        date_str = m.get("utcDate", "")
        try:
            date_str = datetime.fromisoformat(date_str.replace("Z", "+00:00")).strftime("%b %d, %Y · %H:%M")
        except (ValueError, AttributeError):
            pass
        fixtures_list.append({
            "home_team": m.get("homeTeam", {}).get("name", "Unknown"),
            "away_team": m.get("awayTeam", {}).get("name", "Unknown"),
            "date": date_str,
        })

    return render_template_string(
        FIXTURES_TEMPLATE,
        league_code=league_code,
        league_name=LEAGUE_CODES.get(league_code, league_code),
        fixtures=fixtures_list,
        error="",
    )


@app.route("/predict")
def predict():
    home_team = request.args.get("home", "").strip()
    away_team = request.args.get("away", "").strip()

    if not home_team or not away_team:
        return render_template_string(
            PREDICT_TEMPLATE, home_team=home_team or "?", away_team=away_team or "?",
            prediction=None, model_exists=model_exists(),
            error="Please provide both a home and away team.", warning="",
        )

    model = get_model()
    if model is None:
        return render_template_string(
            PREDICT_TEMPLATE, home_team=home_team, away_team=away_team,
            prediction=None, model_exists=False, error="", warning="",
        )

    stats = build_team_stats()
    features = prepare_prediction_features(home_team, away_team, stats)
    result = model.predict(features)

    probs = result["probabilities"]
    p_home = probs.get("Home Win", 0.0)
    p_draw = probs.get("Draw", 0.0)
    p_away = probs.get("Away Win", 0.0)
    top_prob = max(p_home, p_draw, p_away)
    conf = "High" if top_prob >= 0.55 else ("Moderate" if top_prob >= 0.42 else "Low")

    h_info = team_info(stats, home_team)
    a_info = team_info(stats, away_team)

    return render_template_string(
        PREDICT_TEMPLATE,
        home_team=home_team, away_team=away_team, prediction=result,
        model_exists=True, error="", warning="",
        p_home=p_home, p_draw=p_draw, p_away=p_away,
        top_prob=top_prob, confidence_label=conf,
        home_info=h_info, away_info=a_info,
        elo_diff=h_info["elo"] - a_info["elo"],
        both_known=h_info["known"] and a_info["known"],
        bars=[("Home Win", p_home, "var(--home)"), ("Draw", p_draw, "var(--draw)"),
              ("Away Win", p_away, "var(--away)")],
    )


@app.route("/train", methods=["POST"])
def train():
    from main import train_model_csv, TrainingError

    redirect_url = request.form.get("redirect", "")
    top5 = ["eng.1", "es.1", "de.1", "it.1", "fr.1"]
    seasons = ["2021-22", "2022-23", "2023-24"]
    try:
        _, metrics = train_model_csv(seasons, top5, cv_folds=3, model_type="logreg")
        return render_template_string(
            TRAINING_TEMPLATE, success="Model trained successfully!",
            accuracy=metrics.get("accuracy", 0), test_samples=metrics.get("test_samples", 0),
            train_samples=metrics.get("train_samples", 0), redirect_url=redirect_url, error="",
        )
    except (TrainingError, MissingTokenError) as e:
        return render_template_string(TRAINING_TEMPLATE, success="", accuracy=0,
                                      test_samples=0, train_samples=0, redirect_url="", error=str(e))
    except Exception as e:
        return render_template_string(TRAINING_TEMPLATE, success="", accuracy=0,
                                      test_samples=0, train_samples=0, redirect_url="", error=f"Training failed: {e}")


if __name__ == "__main__":
    print("Starting Predict-XI web UI  ->  http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
