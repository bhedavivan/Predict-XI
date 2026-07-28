# Squad-value coverage audit

Every team classified against the Transfermarkt squad-value source. **601 of 1110** teams map to a Transfermarkt club.

Three buckets:

- **Mapped (601)** — has a Transfermarkt club_id, so it carries a point-in-time squad value.
- **Name-mismatch residual (166)** — in a Transfermarkt-covered league but the name didn't match safely. Hand-fixable via `MANUAL_CLUB_MAP` (no fuzzy matching — each must be verified against the league's TM roster). Mostly abbreviated names in non-top-5 leagues.
- **Structural (343)** — the club's league has NO Transfermarkt counterpart (TM ships first divisions only). These **cannot** get a TM value; the model flags them `has_squad_value=0`.

## Name-mismatch residual, by league (hand-fixable)

- **ARG1** (26): All Boys, Argentinos Jrs, Arsenal Sarandi, Atl. Rafaela, Atl. Tucuman, Central Cordoba, Chacarita Juniors, Colon Santa Fe, Crucero del Norte, Dep. Riestra, Estudiantes L.P., Estudiantes Rio Cuarto, Gimnasia L.P., Gimnasia Mendoza, Ind. Rivadavia, Independiente, Nueva Chicago, Olimpo Bahia Blanca, Patronato, Quilmes, San Martin S.J., San Martin T., Sarmiento Junin, Talleres Cordoba, Temperley, Union de Santa Fe
- **ROU1** (21): Academica Clinceni, Astra, Bistrita, CS Turnu Severin, Calarasi, Ceahlaul, Chindia Targoviste, Concordia, Corona Brasov, Daco-Getica Bucuresti, FC Brasov, FC Voluntari, Gaz Metan Medias, Gloria Buzau, Mioveni, Navodari, Pandurii, Poli Timisoara, Targu Mures, U Craiova 1948, Vaslui
- **SWE1** (20): AFC Eskilstuna, AIK, Atvidabergs, Brage, Dalkurd, Falkenbergs, GAIS, Gefle, Goteborg, Helsingborg, Jonkopings, Landskrona, Ljungskile, Malmo FF, Orebro, Ostersunds, Sundsvall, Syrianska, Trelleborgs, Varberg
- **NOR1** (17): Bodo/Glimt, HamKam, Honefoss, Jerv, KFUM Oslo, Kongsvinger, Lillestrom, Mjondalen, Moss, Odd, Ranheim, Sandnes, Sogndal, Stabaek, Stromsgodset, Tromso, Valerenga
- **JPN1** (14): FC Tokyo, Hokkaido Consadole Sapporo, Iwata, Kofu, Kumamoto, Machida, Montedio Yamagata, Oita Trinita, Omiya Ardija, Sagan Tosu, Tokushima, Urawa Reds, Vegalta Sendai, Yamaga
- **BSA** (13): America MG, Atletico GO, Avai, CSA, Criciuma, Cuiaba, Figueirense, Goias, Joinville, Nautico, Ponte Preta, Portuguesa, Santa Cruz
- **POL1** (12): GKS Belchatow, LKS Lodz, Leczna, Legnica, Podbeskidzie, Polonia Warszawa, Ruch, Ruch Chorzow, Sandecja Nowy S., Warta Poznan, Zaglebie Sosnowiec, Zawisza
- **RUS1** (7): FK Anzi Makhackala, M. Saransk, Rodina Moscow, Spartak Nalchik, T. Moscow, Volga N. Novgorod, Volgar-Astrakhan
- **AUT1** (7): A. Lustenau, Admira, Grodig, Mattersburg, Neustadt, St. Polten, Wacker Innsbruck
- **MEX1** (6): Atlante, Chiapas, Dorados de Sinaloa, Lobos BUAP, Monarcas, Veracruz
- **MLS** (5): Atlanta Utd, Chivas USA, Inter Miami, Los Angeles FC, New York Red Bulls
- **SUI1** (5): Aarau, Lausanne Ouchy, Schaffhausen, Vaduz, Xamax
- **BEL1** (3): Bergen, Germinal, Mouscron-Peruwelz
- **TUR1** (3): Akhisar Belediyespor, Mersin Idman Yurdu, Osmanlispor
- **GRE1** (3): Kallonis, Niki Volos, OFI
- **FL1** (2): Evian Thonon Gaillard, St Etienne
- **PL** (1): QPR
- **BL1** (1): Nurnberg

## Structural — leagues Transfermarkt does not cover

- **ENG5** (50 teams) — no TM competition
- **CHN1** (45 teams) — no TM competition
- **PD2** (41 teams) — no TM competition
- **SA2** (30 teams) — no TM competition
- **ELC3** (26 teams) — no TM competition
- **FL2** (24 teams) — no TM competition
- **FIN1** (23 teams) — no TM competition
- **IRL1** (22 teams) — no TM competition
- **BL2** (19 teams) — no TM competition
- **ELC2** (19 teams) — no TM competition
- **SCO4** (16 teams) — no TM competition
- **ELC** (12 teams) — no TM competition
- **SCO3** (9 teams) — no TM competition
- **SCO2** (7 teams) — no TM competition