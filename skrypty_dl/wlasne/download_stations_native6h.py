"""
Pobiera natywne dane 6h (tresolution=360) POJEDYNCZO dla kazdej z 22 stacji
neutron monitorow wybranych 2026-07-17 (>=95% kompletnosci w oknie testu,
patrz 20260717b.ipynb), zamiast zbiorczego endpointu allstations.

Dlaczego ten skrypt istnieje (2026-07-19):
`download_allstations_history.py` pobiera wszystkie stacje naraz przez
`allstations=1`, ale serwer NMDB w tym trybie SAM ogranicza zwracana
rozdzielczosc do ~2h niezaleznie od zadanego `tresolution` (nazwa pliku
wynikowego "*_60min.csv" jest przez to MYLACA - realna rozdzielczosc to 2h).
Test kosmo-sejsmiczny (test znaku, wrazliwy blisko mediany - patrz
20260719.txt) jest na to czuly: dla Moskwy porownanie natywnego 6h
(mosc_data.csv, sigma=3.12) vs 2h->6h resample z allstations (sigma_mc=2.48)
pokazalo ~6.5x roznice w PCDF przy zaledwie 4 zmienionych glosach na 334.

Ten skrypt pobiera kazda stacje OSOBNO (jak juz dziala dla MOSC/OULU w
skrypty_dl/cudze/Pobieranie_Oulu_+_Mosc.ipynb) z tresolution=360 wprost -
pojedyncze zapytanie o 1 stacje NIE ma (o ile wiadomo) problemu z
przycinaniem rozdzielczosci, ktory dotyczy trybu allstations. Po pobraniu
skrypt drukuje realna mediane odstepu czasowego dla kazdej stacji - jesli
wyjdzie cos innego niz 6h, to znak, ze jednak i tu serwer cos przycina i
trzeba to zbadac (nie zakladamy z gory, ze problem na pewno zniknal).

Ten sam zakres lat (2011-2019) i te same parametry dtype/units co oryginalny
download allstations, dla porownywalnosci: dtype=corr_for_efficiency,
units=0 (sprawdzone 2026-07-19, ze sa identyczne w obu pobraniach).

Wyjscie: jeden plik CSV na stacje,
data/csv_data_stations_native6h/{stacja}_6h.csv, kolumny datetime,value -
ten sam format co mosc_data.csv/oulu_5min_data.csv, wiec da sie go wczytac
tym samym prostym loaderem (pd.read_csv(path, parse_dates=["datetime"])).

Uzycie:
    python3 download_stations_native6h.py
Da sie bezpiecznie przerwac i wznowic - stacje z juz istniejacym plikiem
wyjsciowym sa pomijane (patrz `if os.path.exists(out_path)` w download_station).
"""

import datetime
import os
import time
from io import StringIO
from urllib import request

import pandas as pd

STATIONS = [
    "OULU", "THUL", "LMKS", "APTY", "SOPB", "JUNG1", "SOPO", "FSMT", "MOSC",
    "JUNG", "NEWK", "PWNK", "MXCO", "NAIN", "HRMS", "TERA", "INVK", "ATHN",
    "AATB", "PSNM", "NANM", "MCRL",
]

YEAR_BEGIN, YEAR_END = 2011, 2019
TRESOLUTION = 360  # minuty = 6h, natywna rozdzielczosc mosc_data.csv
UNITS = 0  # 0 = counts, jak w oryginalnych pobraniach Moskwy/Oulu
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
# poziomy wyzej do korzenia repo.
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "data", "csv_data_stations_native6h")
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
    """
    Parsuje odpowiedz 1 kawalka do DataFrame (kolumny: timestamp, value).
    Ta sama logika co parse_nmdb_ascii_to_df w download_allstations_history.py
    (sprawdzona, dziala dla formatu odpowiedzi NMDB), przystosowana do
    pojedynczej kolumny (1 stacja) zamiast wielu.
    """
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
    # Pojedyncza stacja -> dokladnie 1 kolumna danych obok timestamp. Nazwa tej
    # kolumny wg NMDB moze byc rozna (np. sama nazwa stacji) - bierzemy ja
    # pozycyjnie (pierwsza po timestamp), zamiast zakladac konkretna nazwe.
    value_col = [c for c in df.columns if c != "timestamp"][0]
    return df[["timestamp", value_col]].rename(columns={value_col: "value"})


def download_station(station):
    out_path = os.path.join(DATA_FOLDER, f"{station.lower()}_6h.csv")
    if os.path.exists(out_path):
        print(f"{station}: plik juz istnieje ({out_path}), pomijam")
        return

    print(f"{station}: pobieranie {YEAR_BEGIN}-{YEAR_END}...")
    chunks = []
    for year in range(YEAR_BEGIN, YEAR_END + 1):
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

    # Diagnostyka natychmiast po pobraniu - sprawdza, czy realna rozdzielczosc
    # zgadza sie z zadanym tresolution=360 (6h), zamiast zakladac to na slepo
    # (patrz motywacja w docstringu modulu - dokladnie to samo zalozenie
    # zawiodlo dla trybu allstations).
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
