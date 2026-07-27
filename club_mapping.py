"""
Maps our training-data club names (football-data.co.uk conventions, e.g.
"Man City", "Ath Madrid") to Transfermarkt club records, so squad market
values can be attached to teams.

Matching is league-constrained and tiered:
  1. exact        — normalized names identical
  2. containment  — one normalized name contains the other, uniquely
  3. high-similarity (>= 0.90)
  4. MANUAL_CLUB_MAP — everything else, hand-verified

Tier 4 exists because string similarity is not a safe decision rule here, in
BOTH directions. Verified failures from the actual data:

  WRONG but scored high          CORRECT but scored low
  ---------------------------    ------------------------------------
  Ath Madrid  -> Real Madrid     M'gladbach -> Borussia Monchengladbach (.56)
  Man City    -> Swansea City    Wolves     -> Wolverhampton Wanderers  (.43)
  Rennes      -> FC Nantes       Sp Lisbon  -> Sporting CP              (.43)
  U. Cluj     -> CFR Cluj        OFI Crete  -> Omilos Filathlon Irakliou(.32)
  Hearts      -> Rangers FC      AEK        -> Athlitiki Enosi Konst.   (.46)

A wrong join silently attaches another club's squad value and the model
still returns confident probabilities — the same failure shape as the
Fixtures team-name bug. Unmapped is always safer than mis-mapped: callers
treat a missing club as "no squad value", which the model handles via an
explicit indicator feature.
"""

import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, Optional

# Our league code -> Transfermarkt domestic_competition_id. Only leagues
# Transfermarkt actually covers; our lower divisions (eng.2-5, es.2, de.2,
# it.2, fr.2, sco.2-4) have no counterpart and are intentionally absent.
LEAGUE_TO_TM_COMP = {
    "PL": "GB1", "PD": "ES1", "BL1": "L1", "SA": "IT1", "FL1": "FR1",
    "DED": "NL1", "PPL": "PO1", "BEL1": "BE1", "TUR1": "TR1", "GRE1": "GR1",
    "RUS1": "RU1", "POL1": "PL1", "AUT1": "A1", "SUI1": "C1", "DEN1": "DK1",
    "ROU1": "RO1", "BSA": "BRA1", "SCO1": "SC1", "MEX1": "MEX1",
}

_DROP_TOKENS = r"\b(fc|afc|cf|sc|cd|ac|sv|ss|as|us|rc|sk|sp|ca|ec|se|cr|club|de|do|the|team)\b"

# Hand-verified. our_name -> exact Transfermarkt club name within the same
# league. Every entry below was checked against the league's candidate list
# by hand; do not add entries from a similarity score alone.
MANUAL_CLUB_MAP = {
    # England
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    # Spain — three clubs here are routinely confused by string matching
    "Ath Bilbao": "Athletic Bilbao",
    "Ath Madrid": "Atlético de Madrid",   # NOT Real Madrid
    "Espanol": "RCD Espanyol Barcelona",
    # Germany
    "Ein Frankfurt": "Eintracht Frankfurt",
    "M'gladbach": "Borussia Mönchengladbach",
    # Italy — Hellas Verona, not Chievo Verona (both are Verona clubs)
    "Verona": "Hellas Verona",
    # France
    "Rennes": "Stade Rennais FC",
    # Netherlands
    "For Sittard": "Fortuna Sittardia Combinatie",
    # Portugal
    "Sp Lisbon": "Sporting CP",
    "Aves": "Desportivo Aves (- 2020)",
    # Belgium
    "Club Brugge": "Club Brugge KV",
    "St Truiden": "Sint-Truidense VV",
    "St. Gilloise": "Union Saint-Gilloise",
    # Scotland
    "Hearts": "Heart of Midlothian FC",
    # Austria
    "BW Linz": "FC Blau-Weiss Linz",
    # Switzerland
    "Grasshoppers": "Grasshopper Club Zurich",
    # Turkey
    "Ad. Demirspor": "Adana Demirspor",
    "Bodrumspor": "Bodrum FK",
    "Buyuksehyr": "Basaksehir FK",        # formerly Istanbul Buyuksehir Bld.
    "Erzurum BB": "Erzurumspor FK",
    # Greece — acronyms that expand to full Greek names
    "AEK": "Athlitiki Enosi Konstantinoupoleos",
    "Aris": "Aris Thessalonikis",
    "Levadeiakos": "APO Levadiakos Football Club",
    "OFI Crete": "Omilos Filathlon Irakliou FC",
    "PAOK": "Panthessalonikios Athlitikos Omilos Konstantinoupoliton",
    "Volos NFC": "Volou Neos Podosferikos Syllogos",
    # Russia — several Moscow clubs; Torpedo is a decoy for all of them
    "CSKA Moscow": "PFK CSKA Moskva",
    "Dynamo Moscow": "FK Dinamo Moskva",
    "Lokomotiv Moscow": 'Футбольный клуб "Локомотив" Москва',
    "Spartak Moscow": "FK Spartak Moskva",
    "Pari NN": "FK Nizhny Novgorod",
    "Yenisey": "Enisey Krasnoyarsk",
    # Brazil — three distinct "Atletico" clubs, keep them apart
    "Athletico-PR": "Clube Atlético Paranaense",
    "Atletico-MG": "Clube Atlético Mineiro",
    "Botafogo RJ": "S. A. F. Botafogo",
    "Flamengo RJ": "Clube de Regatas do Flamengo",
    "Mirassol": "Mirasol Futebol Clube",
    # Mexico — Chivas is Deportivo Guadalajara, NOT Atlas Guadalajara
    # (two separate clubs from the same city)
    "Atl. San Luis": "Atlético de San Luis",
    "Guadalajara Chivas": "Deportivo Guadalajara",
    # Poland
    "GKS Katowice": "GKS GieKSa Katowice Spółka Akcyjna",
    "Termalica B-B.": "Termalica Bruk-Bet Nieciecza Klub Sportowy",
    # Romania
    "Csikszereda M. Ciuc": "FK Csikszereda Miercurea Ciuc",
    "Din. Bucuresti": "FC Dinamo 1948",
    "Dinamo Bucuresti": "FC Dinamo 1948",
    "FC Rapid Bucuresti": "FC Rapid 1923",
    "Poli Iasi": "ACSM Politehnica Iasi",
    "U Craiova": "Universitatea Craiova",
    "Univ. Craiova": "Universitatea Craiova",
    "U. Cluj": "FC Universitatea Cluj",   # NOT CFR Cluj, a different club
    "Viitorul Constanta": "FCV Farul Constanta",  # 2021 merger continuation
}

# Clubs confirmed ABSENT from their league's Transfermarkt roster (defunct,
# relegated out of the covered tier, or never in the first division during
# the covered window). Listed explicitly so they read as "checked and not
# available" rather than "overlooked".
KNOWN_ABSENT = {
    "A. Lustenau", "Admira", "Mattersburg", "St. Polten",          # Austria
    "Lausanne Ouchy", "Schaffhausen", "Vaduz", "Xamax",            # Switzerland
    "Monarcas", "Veracruz",                                        # Mexico
    "America MG", "Atletico GO", "Avai", "Criciuma", "Cuiaba", "Goias",  # Brazil
    "LKS Lodz", "Leczna", "Legnica", "Podbeskidzie", "Ruch Chorzow",
    "Warta Poznan",                                                # Poland
    "Academica Clinceni", "Astra", "Calarasi", "Chindia Targoviste",
    "Concordia", "FC Voluntari", "Gaz Metan Medias", "Gloria Buzau",
    "Mioveni", "U Craiova 1948", "Rodina Moscow",                  # Romania/Russia
}


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ").replace("-", " ").replace(".", " ")
    s = re.sub(_DROP_TOKENS, " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def build_club_mapping(our_teams: Dict[str, dict], tm_clubs: list) -> Dict[str, str]:
    """Return {our_team_name: transfermarkt_club_id}.

    `our_teams` is team_stats-shaped ({name: {"league": code, ...}}).
    Teams that cannot be resolved confidently are simply omitted.
    """
    by_comp = defaultdict(list)
    for row in tm_clubs:
        by_comp[row.get("domestic_competition_id", "")].append(
            (row.get("club_id", ""), row.get("name", ""))
        )

    mapping: Dict[str, str] = {}
    for team, info in our_teams.items():
        comp = LEAGUE_TO_TM_COMP.get(info.get("league", ""))
        if not comp:
            continue
        candidates = by_comp.get(comp, [])
        if not candidates:
            continue

        manual = MANUAL_CLUB_MAP.get(team)
        if manual:
            hit = [cid for cid, nm in candidates if nm == manual]
            if hit:
                mapping[team] = hit[0]
            continue
        if team in KNOWN_ABSENT:
            continue

        n = normalize(team)
        norms = [(cid, nm, normalize(nm)) for cid, nm in candidates]

        exact = [cid for cid, _, nn in norms if nn == n]
        if len(exact) == 1:
            mapping[team] = exact[0]
            continue

        contained = [cid for cid, _, nn in norms if n and nn and (n in nn or nn in n)]
        if len(contained) == 1:
            mapping[team] = contained[0]
            continue

        best, score = None, 0.0
        for cid, _, nn in norms:
            s = SequenceMatcher(None, n, nn).ratio()
            if s > score:
                best, score = cid, s
        # 0.90 is deliberately strict: everything below it goes to a human,
        # because the 0.7-0.85 band is exactly where the wrong matches live.
        if best and score >= 0.90:
            mapping[team] = best

    return mapping
