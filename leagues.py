"""
Canonical league registry — the SINGLE source of truth for every top-flight
league the app knows about: its canonical display name, strength rank (1 =
strongest), country, data source, and the season-table rules the simulator uses.

Before this module, league names lived in three disagreeing places (config.py
said "Primera Division", app.py said "La Liga") and dropdowns were ordered
alphabetically or by an Elo that is invalid across leagues. Everything now reads
from here, so a name or a rank changes in exactly one spot.

Scope = the world's top-flight leagues only (no second/lower divisions). Ranks
are data-anchored for the European leagues (cached ClubElo, PL first) and curated
for the rest by continental standing + squad value — a display ordering, not a
model input, so a curated order is acceptable; edit any `rank` to re-order.

IMPORTANT: this module must import nothing from the project (config imports it,
and almost everything imports config — keep it dependency-free to avoid cycles).
"""

from collections import namedtuple

# source: "fd"    -> football-data.co.uk mmz4281 feed (rich stats), via csv_code
#         "fdnew" -> football-data.co.uk "new leagues" feed, via csv_code
#         "tm"    -> scraped from Transfermarkt gesamtspielplan, via tm_code
League = namedtuple(
    "League",
    "code name rank country csv_code source tm_code calendar_year live_api",
)

# Rank order is the list order below. European leagues 1-? follow cached avg
# ClubElo (PL>PD>BL1>SA>FL1>PPL>DED>BEL1>...); non-European and newly-scraped
# leagues are placed by continental standing + squad value. All adjustable.
_L = [
    # code    name                         country        csv     source   tm_code  cal    live
    ("PL",   "Premier League",             "England",     "eng.1", "fd",    None,   False, True),
    ("PD",   "La Liga",                    "Spain",       "es.1",  "fd",    None,   False, True),
    ("BL1",  "Bundesliga",                 "Germany",     "de.1",  "fd",    None,   False, True),
    ("SA",   "Serie A",                    "Italy",       "it.1",  "fd",    None,   False, True),
    ("FL1",  "Ligue 1",                    "France",      "fr.1",  "fd",    None,   False, True),
    ("PPL",  "Primeira Liga",              "Portugal",    "pt.1",  "fd",    None,   False, True),
    ("DED",  "Eredivisie",                 "Netherlands", "nl.1",  "fd",    None,   False, True),
    ("BEL1", "Belgian Pro League",         "Belgium",     "be.1",  "fd",    None,   False, False),
    ("BSA",  "Brasileirão Série A",        "Brazil",      "br.1",  "fdnew", None,   True,  True),
    ("TUR1", "Süper Lig",                  "Turkey",      "tr.1",  "fd",    None,   False, False),
    ("ARG1", "Primera División (Arg)",     "Argentina",   "ar.1",  "fdnew", None,   True,  False),
    ("RUS1", "Russian Premier League",     "Russia",      "ru.1",  "fdnew", None,   False, False),
    ("SAU1", "Saudi Pro League",           "Saudi Arabia","sau.1", "tm",    "SA1",  False, False),
    ("MEX1", "Liga MX",                    "Mexico",      "mx.1",  "fdnew", None,   False, False),
    ("GRE1", "Super League Greece",        "Greece",      "gr.1",  "fd",    None,   False, False),
    ("SUI1", "Swiss Super League",         "Switzerland", "ch.1",  "fdnew", None,   False, False),
    ("AUT1", "Austrian Bundesliga",        "Austria",     "at.1",  "fdnew", None,   False, False),
    ("SCO1", "Scottish Premiership",       "Scotland",    "sco.1", "fd",    None,   False, False),
    ("DEN1", "Danish Superliga",           "Denmark",     "dk.1",  "fdnew", None,   False, False),
    ("UKR1", "Ukrainian Premier League",   "Ukraine",     "ukr.1", "tm",    "UKR1", False, False),
    ("CRO1", "SuperSport HNL",             "Croatia",     "cro.1", "tm",    "KR1",  False, False),
    ("CZE1", "Chance Liga",                "Czechia",     "cze.1", "tm",    "TS1",  False, False),
    ("SRB1", "Super liga Srbije",          "Serbia",      "srb.1", "tm",    "SER1", False, False),
    ("NOR1", "Eliteserien",                "Norway",      "no.1",  "fdnew", None,   True,  False),
    ("POL1", "Ekstraklasa",                "Poland",      "pl.1",  "fdnew", None,   False, False),
    ("SWE1", "Allsvenskan",                "Sweden",      "se.1",  "fdnew", None,   True,  False),
    ("ROU1", "Liga I",                     "Romania",     "ro.1",  "fdnew", None,   False, False),
    ("KOR1", "K League 1",                 "South Korea", "kor.1", "tm",    "RSK1", True,  False),
    ("JPN1", "J1 League",                  "Japan",       "jp.1",  "fdnew", None,   True,  False),
    ("MLS",  "Major League Soccer",        "USA",         "us.1",  "fdnew", None,   True,  False),
    ("COL1", "Primera A",                  "Colombia",    "col.1", "tm",    "COL1", True,  False),
    ("QAT1", "Qatar Stars League",         "Qatar",       "qat.1", "tm",    "QSL",  False, False),
    ("UAE1", "UAE Pro League",             "UAE",         "uae.1", "tm",    "UAE1", False, False),
    ("ISR1", "Ligat ha'Al",               "Israel",      "isr.1", "tm",    "ISR1", False, False),
    ("CHI1", "Primera División (Chi)",     "Chile",       "chi.1", "tm",    "CLPD", True,  False),
    ("EGY1", "Egyptian Premier League",    "Egypt",       "egy.1", "tm",    "EGY1", False, False),
    ("HUN1", "NB I",                       "Hungary",     "hun.1", "tm",    "UNG1", False, False),
    ("URU1", "Primera División (Uru)",     "Uruguay",     "uru.1", "tm",    "URU1", True,  False),
    ("RSA1", "Betway Premiership",         "South Africa","rsa.1", "tm",    "SFA1", False, False),
    ("AUS1", "A-League Men",               "Australia",   "aus.1", "tm",    "AUS1", False, False),
    ("CHN1", "Chinese Super League",       "China",       "cn.1",  "fdnew", None,   True,  False),
    ("ECU1", "LigaPro Serie A",            "Ecuador",     "ecu.1", "tm",    "EC1N", True,  False),
    ("MAR1", "Botola Pro",                 "Morocco",     "mar.1", "tm",    "MAR1", False, False),
    ("PAR1", "Primera División (Par)",     "Paraguay",    "par.1", "tm",    "PR1A", True,  False),
    ("PER1", "Liga 1",                     "Peru",        "per.1", "tm",    "TDeA+TDeC", True, False),
    ("IND1", "Indian Super League",        "India",       "ind.1", "tm",    "IND1", False, False),
    ("FIN1", "Veikkausliiga",             "Finland",     "fi.1",  "fdnew", None,   True,  False),
    ("IRL1", "League of Ireland",          "Ireland",     "ie.1",  "fdnew", None,   True,  False),
]

REGISTRY = [League(c, n, i + 1, ctry, csv, src, tm, cal, api)
            for i, (c, n, ctry, csv, src, tm, cal, api) in enumerate(_L)]

BY_CODE = {lg.code: lg for lg in REGISTRY}
BY_CSV = {lg.csv_code: lg for lg in REGISTRY}
TOP_FLIGHT_CODES = frozenset(BY_CODE)
# Training csv-codes, strongest first (so a capped training run keeps the best).
TRAIN_CSV_CODES = [lg.csv_code for lg in REGISTRY]
# csv_code -> our code (drop-in replacement for the old LEAGUE_CODE_MAP).
CODE_BY_CSV = {lg.csv_code: lg.code for lg in REGISTRY}
# Transfermarkt-sourced leagues: csv_code -> tm gesamtspielplan code(s).
TM_RESULT_COMPS = {lg.csv_code: lg.tm_code for lg in REGISTRY if lg.source == "tm"}
# csv_codes whose season is a single calendar year (South American, Nordic, …).
CALENDAR_CSV_CODES = frozenset(lg.csv_code for lg in REGISTRY if lg.calendar_year)

# Cups are not strength-ranked leagues but must still resolve to a name and stay
# available on the Fixtures page.
CUP_CODES = {
    "CL": "UEFA Champions League",
    "EC": "European Championship",
    "WC": "FIFA World Cup",
}


def display_name(code):
    """Canonical display name for a league or cup code; falls back to the code."""
    lg = BY_CODE.get(code)
    if lg:
        return lg.name
    return CUP_CODES.get(code, code or "Other")


def league_rank(code):
    """Strength rank (1 = strongest); unknown codes sort last."""
    lg = BY_CODE.get(code)
    return lg.rank if lg else 999


def fixtures_leagues():
    """Ordered {code: name} for the Fixtures page: the live-API top flights in
    strength order, then the cups. Replaces config.LEAGUE_CODES."""
    out = {lg.code: lg.name for lg in REGISTRY if lg.live_api}
    out.update(CUP_CODES)
    return out


# ─── Simulator: per-league season-table rules ──────────────────────────────
# relegation_slots / cl_slots (top continental qualification) / uel_slots
# (secondary European band, European leagues only) / playoff_slots (a
# promotion/European playoff band above the drop). Approximate and open to edit
# — European berths shift yearly by coefficient and cup-winner routes; the
# simulator reports these as "by final league position (approx)".
LeagueRules = namedtuple("LeagueRules",
                         "relegation_slots cl_slots uel_slots playoff_slots")
DEFAULT_RULES = LeagueRules(relegation_slots=3, cl_slots=4, uel_slots=2, playoff_slots=0)

LEAGUE_RULES = {
    "PL":   LeagueRules(3, 5, 2, 0),
    "PD":   LeagueRules(3, 5, 2, 0),
    "BL1":  LeagueRules(2, 4, 2, 1),   # 16th plays a relegation playoff
    "SA":   LeagueRules(3, 4, 2, 0),
    "FL1":  LeagueRules(3, 3, 2, 1),   # 16th playoff; 3 direct UCL-ish
    "PPL":  LeagueRules(3, 2, 3, 0),
    "DED":  LeagueRules(3, 2, 2, 0),
    "BEL1": LeagueRules(3, 2, 2, 0),
    "SCO1": LeagueRules(1, 1, 3, 1),   # 11th playoff, small league
}


# Confederation per country, and the labels for its top/secondary continental
# club competitions — so the simulator shows "Libertadores" for a Brazilian
# league, not "UCL". The slot COUNTS come from LEAGUE_RULES (approximate); these
# are just the names. (Second-competition label "" => that league shows no
# secondary continental tier.)
_CONFED_BY_COUNTRY = {
    # UEFA
    "England": "UEFA", "Spain": "UEFA", "Germany": "UEFA", "Italy": "UEFA",
    "France": "UEFA", "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA",
    "Turkey": "UEFA", "Greece": "UEFA", "Russia": "UEFA", "Switzerland": "UEFA",
    "Austria": "UEFA", "Scotland": "UEFA", "Denmark": "UEFA", "Ukraine": "UEFA",
    "Croatia": "UEFA", "Czechia": "UEFA", "Serbia": "UEFA", "Norway": "UEFA",
    "Poland": "UEFA", "Sweden": "UEFA", "Romania": "UEFA", "Hungary": "UEFA",
    "Israel": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Colombia": "CONMEBOL",
    "Chile": "CONMEBOL", "Uruguay": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL",
    # AFC
    "Saudi Arabia": "AFC", "South Korea": "AFC", "Japan": "AFC", "China": "AFC",
    "Qatar": "AFC", "UAE": "AFC", "India": "AFC", "Australia": "AFC",
    # CAF
    "Egypt": "CAF", "South Africa": "CAF", "Morocco": "CAF",
    # CONCACAF
    "Mexico": "CONCACAF", "USA": "CONCACAF",
}

_CONFED_LABELS = {
    "UEFA": ("UCL", "UEL"),
    "CONMEBOL": ("Libertadores", "Sudamericana"),
    "AFC": ("AFC CL", ""),
    "CAF": ("CAF CL", ""),
    "CONCACAF": ("CCC", ""),   # CONCACAF Champions Cup
}


def tier_labels(code):
    """(top_continental_label, secondary_label) for a league, by confederation.
    Defaults to the European UCL/UEL when the country isn't mapped."""
    lg = BY_CODE.get(code)
    conf = _CONFED_BY_COUNTRY.get(lg.country) if lg else None
    return _CONFED_LABELS.get(conf, ("UCL", "UEL"))


def league_rules(code, n_teams):
    """Rules for a league, clamped so the qualification + relegation bands always
    leave at least one mid-table place (guards tiny/hypothetical fields)."""
    r = LEAGUE_RULES.get(code) or DEFAULT_RULES
    n = n_teams or 20
    cl = min(r.cl_slots, max(1, n // 3))
    uel = min(r.uel_slots, max(0, n - cl - 2))
    rel = min(r.relegation_slots, max(1, n - cl - uel - 1))
    playoff = min(r.playoff_slots, 1) if n >= 8 else 0
    return LeagueRules(rel, cl, uel, playoff)
