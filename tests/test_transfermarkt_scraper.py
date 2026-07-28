"""Tests for the Transfermarkt scraper's HTML parser (no network).

The parser is the fragile part — Transfermarkt markup can shift — so it's pinned
against a realistic fixture of the competition 'startseite' items table. The
polite HTTP layer is not unit-tested here (it hits the live site by design).
"""

from transfermarkt_scraper import (parse_club_values, _value_to_eur,
                                    _reference_season, parse_gesamtspielplan)


# A trimmed but structurally faithful items table: two club rows, each with an
# average-value column then the TOTAL squad-value column (the larger one), plus
# a header row that must be ignored.
FIXTURE = """
<table class="items">
  <thead><tr><th>club</th><th>ø</th><th>Total</th></tr></thead>
  <tbody>
    <tr class="odd">
      <td class="zentriert">1</td>
      <td class="hauptlink no-border-links">
        <a href="/ipswich-town/startseite/verein/677/saison_id/2025" title="Ipswich Town">Ipswich Town</a>
      </td>
      <td class="rechts">&euro;6.81m</td>
      <td class="rechts"><a href="/ipswich-town/kader/verein/677">&euro;217.95m</a></td>
    </tr>
    <tr class="even">
      <td class="zentriert">2</td>
      <td class="hauptlink no-border-links">
        <a href="/1-fc-nuremberg/startseite/verein/4/saison_id/2025" title="1.FC Nuremberg">1.FC Nuremberg</a>
      </td>
      <td class="rechts">&euro;1.20m</td>
      <td class="rechts"><a href="/1-fc-nuremberg/kader/verein/4">&euro;48.00m</a></td>
    </tr>
  </tbody>
</table>
"""


class TestValueParsing:
    def test_units(self):
        assert _value_to_eur("312.35", "m") == 312_350_000
        assert _value_to_eur("1.20", "bn") == 1_200_000_000
        assert _value_to_eur("850", "k") == 850_000

    def test_thousands_separator_stripped(self):
        assert _value_to_eur("1,250.00", "k") == 1_250_000


class TestParseClubValues:
    def test_extracts_id_name_and_total(self):
        rows = parse_club_values(FIXTURE)
        assert len(rows) == 2
        by_id = {cid: (name, val) for cid, name, val in rows}
        # Total (not the average) is taken, id from /verein/<id>/, name from title.
        assert by_id["677"] == ("Ipswich Town", 217_950_000)
        assert by_id["4"] == ("1.FC Nuremberg", 48_000_000)

    def test_header_row_ignored(self):
        # No tr.odd/tr.even header, so the <thead> can't leak in as a club.
        names = {name for _, name, _ in parse_club_values(FIXTURE)}
        assert "club" not in names

    def test_out_of_band_values_dropped(self):
        # A stray tiny value (below the sanity floor) is not a squad total.
        html = FIXTURE.replace("&euro;217.95m", "&euro;0.05m").replace("&euro;6.81m", "&euro;0.01m")
        rows = parse_club_values(html)
        ids = {cid for cid, _, _ in rows}
        assert "677" not in ids   # both Ipswich values now below the floor

    def test_empty_html(self):
        assert parse_club_values("<html>no table here</html>") == []


class TestReferenceSeason:
    def test_returns_int_year(self):
        r = _reference_season()
        assert isinstance(r, int) and 2000 < r < 2100


# A faithful trim of a gesamtspielplan page: row 1 prints its date; row 2 is the
# same date (no date link → must carry forward); row 3 is unplayed (no score).
GSP_FIXTURE = """
<tr>
  <td class="hide-for-small">Fri <a href="/aktuell/waspassiertheute/aktuell/new/datum/2023-08-11">8/11/23</a></td>
  <td class="zentriert hide-for-small">2:00 PM</td>
  <td class="text-right no-border-rechts hauptlink"><a title="Al-Ahli SFC" href="/x/spielplan/verein/18487/saison_id/2023">Al-Ahli</a></td>
  <td class="zentriert no-border-links"><a title="Al-Ahli SFC" href="/x/spielplan/verein/18487/saison_id/2023"><img/></a></td>
  <td class="zentriert hauptlink">&nbsp;<a title="" class="ergebnis-link" id="1" href="/x/index/spielbericht/4120140">3:1</a>&nbsp;</td>
  <td class="zentriert no-border-rechts"><a title="Al-Hazem SC" href="/y/spielplan/verein/9131/saison_id/2023"><img/></a></td>
  <td class="no-border-links hauptlink"><a title="Al-Hazem SC" href="/y/spielplan/verein/9131/saison_id/2023">Al-Hazem</a></td>
</tr>
<tr>
  <td class="hide-for-small"></td>
  <td class="zentriert hide-for-small">4:00 PM</td>
  <td class="text-right no-border-rechts hauptlink"><a title="Al-Nassr FC" href="/n/spielplan/verein/18544/saison_id/2023">Al-Nassr</a></td>
  <td class="zentriert hauptlink">&nbsp;<a title="" class="ergebnis-link" id="2" href="/n/index/spielbericht/4120141">2:0</a>&nbsp;</td>
  <td class="no-border-links hauptlink"><a title="Al-Fateh SC" href="/f/spielplan/verein/9932/saison_id/2023">Al-Fateh</a></td>
</tr>
<tr>
  <td class="hide-for-small">Sat <a href="/aktuell/waspassiertheute/aktuell/new/datum/2023-08-12">8/12/23</a></td>
  <td class="zentriert hauptlink"><a title="" class="ergebnis-link" id="3" href="/z/index/spielbericht/0">-:-</a></td>
  <td class="no-border-links hauptlink"><a title="Al-Ittihad Club" href="/i/spielplan/verein/1/saison_id/2023">Al-Ittihad</a></td>
  <td class="no-border-links hauptlink"><a title="Al-Hilal SFC" href="/h/spielplan/verein/2/saison_id/2023">Al-Hilal</a></td>
</tr>
"""


class TestParseGesamtspielplan:
    def test_extracts_played_matches_with_carried_date(self):
        rows = parse_gesamtspielplan(GSP_FIXTURE)
        assert rows == [
            ("2023-08-11", "Al-Ahli SFC", "Al-Hazem SC", 3, 1),
            ("2023-08-11", "Al-Nassr FC", "Al-Fateh SC", 2, 0),   # date carried forward
        ]

    def test_unplayed_rows_are_skipped(self):
        rows = parse_gesamtspielplan(GSP_FIXTURE)
        assert all("Al-Hilal SFC" not in (h, a) for _, h, a, _, _ in rows)  # -:- dropped

    def test_empty_html(self):
        assert parse_gesamtspielplan("<html>nothing</html>") == []
