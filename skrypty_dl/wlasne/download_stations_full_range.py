"""
Pobiera natywne dane 6h (tresolution=360) dla 22 stacji z
`download_stations_native6h.py`, ale w PELNYM zakresie potrzebnym do
pelnego skanu po t0 (P_days=3350, krok 6h) uzytego w 20260819a.ipynb dla
Moskwy/Oulu - nie tylko wycinek 2011-2019.

Dlaczego ten skrypt istnieje (2026-08-20):
Zadanie z planu Homoli "rozszerzenie skanu na 22 stacje" (patrz
20260819.txt pkt 4) zaklada uzycie TEGO SAMEGO zakresu kandydatow t0 co dla
Moskwy/Oulu: [eq.index.min(), eq.index.max() - P_days] =
[2005-01-01, 2015-11-30] (przy P_days=3350, dt_days=0 - patrz
20260819a.ipynb komorka 2), co wymaga danych CR pokrywajacych CALY zakres
2005-01-01 .. 2025-01-31 (koniec katalogu EQ, usgs_m4_2005_2025.csv).
Dotychczasowe pobrania 22 stacji (`csv_data_allstations/`,
`csv_data_stations_native6h/`) pokrywaja TYLKO 2011-2019 (9 lat < 3350 dni
= 9.17 roku) - sprawdzone bezposrednio (2026-08-20, sesja Claude'a):
`sed -n '2p'`/`tail -1` na wszystkich 22 plikach `*_6h.csv` pokazuje
identyczny zakres 2011-01-01..2019-12-31 dla kazdej stacji. Przy takim
zakresie okno P_days=3350 w ogole nie miesci sie w danych (0 poprawnych
kandydatow t0), wiec naiwne rozszerzenie skanu na te dane byloby bez sensu.

Sprawdzono tez (zapytanie NMDB allstations, tresolution=roczna, 1950-2010),
ze poszczegolne stacje maja BARDZO rozna glebokosc historyczna - od 1950s
(HRMS, JUNG, MOSC, THUL) po dopiero 2007 (PSNM). Ten skrypt pobiera per
stacje od max(2005-01-01, faktyczny poczatek danych stacji) do 2025-01-31 -
stacje z pozniejszym startem beda mialy odpowiednio mniej kandydatow t0 w
skanie (run_t0_scan_parallel/cosmoseismic_stat i tak odrzuca kandydatow bez
pelnego pokrycia danych przez filtr NaN), ale to jest lepsze niz brak
jakichkolwiek poprawnych kandydatow.

WAZNE: pisze do NOWEGO katalogu (`data/csv_data_stations_full6h/`), NIE
nadpisuje `csv_data_stations_native6h/` - te pliki (2011-2019) sa nadal
uzywane przez 20260719b.ipynb (meta-analiza porownawcza 22 stacji) i nie
powinny sie zmienic pod tamtym notebookiem.

Ten sam wzorzec co `download_stations_native6h.py` (1 zapytanie na
stacje/rok, retry, sleep grzecznosciowy, wznawialne przez pomijanie
istniejacych plikow) - jedyna zmiana to YEAR_BEGIN per stacja i dluzszy
YEAR_END.

Uzycie:
    python3 download_stations_full_range.py
Da sie bezpiecznie przerwac i wznowic (pomija stacje z istniejacym plikiem
wyjsciowym). UWAGA: to jest 20 stacji (bez MOSC/OULU - patrz nizej) x do 21
lat = potencjalnie ~350-420 zapytan do NMDB (z 2s odstepem miedzy nimi) -
realistycznie liczyc sie z dlugim czasem dzialania (rzedu godzin, nie
minut), ostroznie odpalac np. przez `nohup`/`screen`/w tle.
"""

import datetime
import os
import time
from io import StringIO
from urllib import request

import pandas as pd

# MOSC i OULU CELOWO pominiete - maja juz dedykowane pliki pelnej historii
# (data/mosc_data.csv 1960-2025, data/oulu_5min_data.csv 1970-2025), ktore
# 20260819a.ipynb juz uzywa do skanu Moskwy/Oulu. Pobieranie ich tutaj
# jeszcze raz (inna sciezka zapytania NMDB) byloby zbedne i mogloby dac
# nieznacznie inne wartosci (patrz uwaga o allstations vs pojedyncze
# zapytanie w download_stations_native6h.py) - tylko 20 pozostalych stacji.
STATIONS = [
    "THUL", "LMKS", "APTY", "SOPB", "JUNG1", "SOPO", "FSMT",
    "JUNG", "NEWK", "PWNK", "MXCO", "NAIN", "HRMS", "TERA", "INVK", "ATHN",
    "AATB", "PSNM", "NANM", "MCRL",
]

# Przyblizony pierwszy rok z realnymi danymi per stacja - z sondy NMDB
# allstations (tresolution=roczna, 1950-2010, sesja 2026-08-20). To tylko
# DOLNE OGRANICZENIE zapytan (nie ufamy mu w 100% - stacja mogla zaczac w
# trakcie roku) - i tak jest floorowane do max(., 2005), bo starsze dane nie
# sa potrzebne do tego skanu (zakres kandydatow t0 zaczyna sie 2005-01-01).
STATION_FIRST_YEAR = {
    "THUL": 1957, "LMKS": 1981, "APTY": 2000, "SOPB": 1997,
    "JUNG1": 1986, "SOPO": 1964, "FSMT": 2000, "JUNG": 1958,
    "NEWK": 1964, "PWNK": 2000, "MXCO": 1990, "NAIN": 2000, "HRMS": 1957,
    "TERA": 1968, "INVK": 2000, "ATHN": 2000, "AATB": 1973, "PSNM": 2007,
    "NANM": 1997, "MCRL": 1950,
}

SCAN_RANGE_YEAR_BEGIN = 2005  # zakres kandydatow t0 zaczyna sie 2005-01-01
YEAR_END = 2025  # katalog EQ konczy sie 2025-01-31

TRESOLUTION = 360  # minuty = 6h, natywna rozdzielczosc mosc_data.csv
UNITS = 0  # 0 = counts, jak w oryginalnych pobraniach Moskwy/Oulu/native6h
SLEEP_BETWEEN_REQUESTS = 2  # sekundy - uprzejmosciowy odstep miedzy zapytaniami do serwera NMDB
MAX_RETRIES = 3

BASE_URL = (
    "https://www.nmdb.eu/nest/draw_graph.php?formchk=1&stations[]={station}"
    "&tabchoice=revori&dtype=corr_for_efficiency&tresolution={tres}"
    "&force=1&yunits={yunits}&date_choice=bydate"
    "&start_day={sd}&start_month={sm}&start_year={sy}"
    "&start_hour=0&start_min=0&end_day={ed}&end_month={em}&end_year={ey}"
    "&end_hour=0&end_min=0&output=ascii"
)

# Sciezka wzgledna do data/ - skrypt jest w skrypty_dl/wlasne/, wiec dwa
# poziomy wyzej do korzenia repo. NOWY katalog - patrz docstring wyzej.
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "data", "csv_data_stations_full6h")
os.makedirs(DATA_FOLDER, exist_ok=True)


def fetch_year(station, year):
    """Pobiera 1 rok danych (1 sty {year} 00:00 .. 1 sty {year+1} 00:00) dla jednej stacji."""
    url = BASE_URL.format(
        station=station, tres=TRESOLUTION, yunits=UNITS,
        sd=1, sm=1, sy=year,
        ed=1, em=1, ey=year + 1,
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with request.urlopen(url, timeout=120) as response:
                return response.read().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"    proba {attempt}/{MAX_RETRIES} nieudana ({e})")
            if attempt < MAX_RETRIES:
                time.sleep(5)
    return ""


def parse_nmdb_ascii_to_df(data, station):
    """Parsuje odpowiedz 1 kawalka do DataFrame (kolumny: timestamp, value).
    Ta sama, juz sprawdzona logika co w download_stations_native6h.py /
    download_allstations_history.py."""
    start_tag, end_tag = "<pre><code>", "</code>"
    start_idx, end_idx = data.find(start_tag), data.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        print(f"    {station}: nie znaleziono tagow <pre><code> ... </code>")
        return None

    content = data[start_idx + len(start_tag):end_idx].strip()
    lines = content.splitlines()

    header_line = None
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not header_line and not line.startswith("20"):
            header_line = line.replace(" ", ";").replace("\t", ";")
            while ";;" in header_line:
                header_line = header_line.replace(";;", ";")
            continue
        if line[:4].isdigit():
            clean_line = line.replace(";", "; ")
            while ";;" in clean_line:
                clean_line = clean_line.replace(";;", ";")
            data_lines.append(clean_line.strip())

    if not header_line or not data_lines:
        return None

    csv_text = "timestamp;" + header_line + "\n" + "\n".join(data_lines)
    df = pd.read_csv(
        StringIO(csv_text), sep=";", skipinitialspace=True,
        na_values=["null"], parse_dates=["timestamp"],
    )
    value_col = [c for c in df.columns if c != "timestamp"][0]
    return df[["timestamp", value_col]].rename(columns={value_col: "value"})


def download_station(station):
    out_path = os.path.join(DATA_FOLDER, f"{station.lower()}_full_6h.csv")
    if os.path.exists(out_path):
        print(f"{station}: plik juz istnieje ({out_path}), pomijam")
        return

    year_begin = max(SCAN_RANGE_YEAR_BEGIN, STATION_FIRST_YEAR.get(station, SCAN_RANGE_YEAR_BEGIN))
    print(f"{station}: pobieranie {year_begin}-{YEAR_END}...")
    chunks = []
    for year in range(year_begin, YEAR_END + 1):
        print(f"  rok {year}...")
        data = fetch_year(station, year)
        if not data:
            print(f"  {station} {year}: brak odpowiedzi, pomijam ten rok")
            continue
        df_chunk = parse_nmdb_ascii_to_df(data, station)
        if df_chunk is not None:
            chunks.append(df_chunk)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if not chunks:
        print(f"{station}: brak danych w calym zakresie, nie zapisuje pliku")
        return

    full = pd.concat(chunks, ignore_index=True)
    full = full.drop_duplicates(subset="timestamp").sort_values("timestamp")
    full = full.rename(columns={"timestamp": "datetime"})
    full.to_csv(out_path, index=False)

    delta_median = full["datetime"].diff().median()
    print(f"{station}: zapisano {len(full)} wierszy -> {out_path}")
    print(f"  zakres: {full['datetime'].min()} .. {full['datetime'].max()}")
    print(f"  mediana odstepu czasowego: {delta_median} "
          f"({'OK, zgadza sie z tresolution=360 (6h)' if delta_median == pd.Timedelta(hours=6) else 'UWAGA: NIE zgadza sie z oczekiwanym 6h!'})")


def main():
    for station in STATIONS:
        download_station(station)


if __name__ == "__main__":
    main()
