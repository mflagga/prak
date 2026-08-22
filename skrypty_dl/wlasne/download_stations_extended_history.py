"""
Rozszerza historie CR dla 20 stacji NMDB (wszystkie oprocz MOSC/OULU, ktore
juz maja dedykowane pliki pelnej historii) WSTECZ, do ich FAKTYCZNEGO
pierwszego roku danych (nie floorowane do 2005, jak w
`download_stations_full_range.py`) - prerekwizyt dla mapy efektu
cykl-do-cyklu (20260822c.ipynb), ktora potrzebuje danych CR pokrywajacych
WIELE cykli slonecznych dla jak najwiekszej liczby stacji, nie tylko
2005-2025.

Kontekst (2026-08-22, patrz 20260822.txt): Homola zasugerowal (komentarz
24.07) porownanie optymalnych t0 miedzy cyklami slonecznymi - "globalne
mapy efektu cykl do cyklu". Katalog EQ zostal juz dzisiaj rozszerzony do
1965-2025 (`download_usgs_extended_history.py`). Dane CR mosc/oulu juz
siegaja odpowiednio do 1960/1970. Pozostale 20 stacji maja dotychczas
pobrane TYLKO 2011-2019 (`csv_data_stations_native6h/`) lub 2005-2025
(`csv_data_stations_full6h/`, pobrane 20.08 pod skan Moskwy/Oulu-stylu) -
za malo dla wiekszosci cykli. Sonda NMDB z 20.08 (patrz
`download_stations_full_range.py`, STATION_FIRST_YEAR) pokazala, ze WIEKSZOSC
stacji ma dane siegajace duzo dalej (lata 50./60./70./80./90.) - ten skrypt
je faktycznie sciaga.

WAZNE: pisze do NOWEGO katalogu (`data/csv_data_stations_extended/`), NIE
nadpisuje `csv_data_stations_full6h/` ani `csv_data_stations_native6h/` -
tamte pliki sa nadal uzywane przez istniejace notebooki (20260719b.ipynb,
20260820/22 - test wykluczenia 2006-2009) i nie powinny sie zmienic.
20260822c.ipynb sam sprawdza, czy plik w NOWYM katalogu istnieje, i jesli
nie - spada z powrotem na `csv_data_stations_full6h/` (2005-2025) dla danej
stacji, wiec notebook da sie uruchomic czesciowo takze PRZED ukonczeniem
tego pobrania.

Ten sam wzorzec co `download_stations_full_range.py` (1 zapytanie na
stacje/rok, retry, sleep grzecznosciowy, wznawialne przez pomijanie
istniejacych plikow) - jedyna zmiana to brak floorowania YEAR_BEGIN do 2005.

Uzycie:
    python3 download_stations_extended_history.py
Bezpiecznie przerywalne/wznawialne (pomija stacje z istniejacym plikiem
wyjsciowym). UWAGA: to jest 20 stacji x do ~76 lat (najstarsza: MCRL od 1950)
= ~870 zapytan lacznie (z 2s odstepem miedzy nimi) - wyraznie wiecej niz
poprzednie pobranie (~350-420 zapytan na 20.08) - realistycznie liczyc sie z
czasem dzialania rzedu GODZINY, moze wiecej. Bezpiecznie odpalac w tle
(`nohup`/`screen`), nie czekac interaktywnie.
"""

import os
import time
from io import StringIO
from urllib import request

import pandas as pd

# Ta sama lista i te same przyblizone pierwsze lata co
# `download_stations_full_range.py` (sonda NMDB allstations, tresolution=
# roczna, 1950-2010, sesja 2026-08-20) - skopiowane bez zmian, TERAZ
# uzywane BEZ floorowania do 2005.
STATIONS = [
    "THUL", "LMKS", "APTY", "SOPB", "JUNG1", "SOPO", "FSMT",
    "JUNG", "NEWK", "PWNK", "MXCO", "NAIN", "HRMS", "TERA", "INVK", "ATHN",
    "AATB", "PSNM", "NANM", "MCRL",
]

STATION_FIRST_YEAR = {
    "THUL": 1957, "LMKS": 1981, "APTY": 2000, "SOPB": 1997,
    "JUNG1": 1986, "SOPO": 1964, "FSMT": 2000, "JUNG": 1958,
    "NEWK": 1964, "PWNK": 2000, "MXCO": 1990, "NAIN": 2000, "HRMS": 1957,
    "TERA": 1968, "INVK": 2000, "ATHN": 2000, "AATB": 1973, "PSNM": 2007,
    "NANM": 1997, "MCRL": 1950,
}

YEAR_END = 2025  # katalog EQ rozszerzony konczy sie 2025-01-31

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

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "data", "csv_data_stations_extended")
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
    """Ta sama, juz sprawdzona logika co w download_stations_full_range.py /
    download_stations_native6h.py / download_allstations_history.py."""
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
        if not header_line and not line.startswith("20") and not line.startswith("19"):
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
    out_path = os.path.join(DATA_FOLDER, f"{station.lower()}_extended_6h.csv")
    if os.path.exists(out_path):
        print(f"{station}: plik juz istnieje ({out_path}), pomijam")
        return

    year_begin = STATION_FIRST_YEAR.get(station)
    print(f"{station}: pobieranie {year_begin}-{YEAR_END} ({YEAR_END - year_begin + 1} lat)...")
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

    print(f"{station}: zapisano {len(full)} wierszy -> {out_path}")
    print(f"  zakres: {full['datetime'].min()} .. {full['datetime'].max()}")


def main():
    for station in STATIONS:
        download_station(station)


if __name__ == "__main__":
    main()
