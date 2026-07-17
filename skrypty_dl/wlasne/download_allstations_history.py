"""
Pobiera historyczne dane (2011-2019) dla wszystkich stacji neutron monitorow
z NMDB (nest.nmdb.eu), do meta-analizy efektu kosmo-sejsmicznego na wielu
niezaleznych stacjach CR (rozszerzenie serii Moskwa/Oulu/Auger z 20260715-17).

Historia wersji tego skryptu (wazne dla zrozumienia dlaczego jest tak, a nie
prosciej):
1. v1: chunking po 30 dni (jak w skrypty_dl/cudze/Pobieranie_Oulu_+_Mosc.ipynb),
   sklejanie TEKSTOWE wierszy pod jednym naglowkiem wzietym z pierwszego
   kawalka. Blad: zestaw aktywnie raportujacych stacji rozni sie miedzy
   kawalkami tego samego roku -> wiersze z innych kawalkow maja inna liczbe
   pol niz naglowek, dane przesuniete kolumnowo (5/9 lat nie parsowalo sie
   wcale, ParserError).
2. v2: 1 zapytanie na caly rok (zamiast chunkowania) - eliminuje problem
   niezgodnosci kolumn (jeden naglowek na caly rok), ale odkryto inny fakt:
   serwer NMDB w trybie allstations SAM dopasowuje zwracana rozdzielczosc do
   rozpietosci czasowej zapytania (cel: ~350-370 punktow), ignorujac
   tresolution ponad pewien prog - caly rok w 1 zapytaniu dawal tylko
   ~365 punktow (1/dzien) zamiast ~4400 (~2h), duza strata rozdzielczosci.
3. v3 (ta wersja): wracamy do chunkowania po 30 dni (dla dobrej rozdzielczosci
   ~2h), ale KAZDY kawalek parsowany do wlasnego pandas.DataFrame i lączony
   przez pd.concat, ktory dopasowuje kolumny PO NAZWIE stacji, nie pozycji -
   brakujaca stacja w danym kawalku dostaje NaN, a nie przesuniecie danych.

Uzycie:
    python3 download_allstations_history.py
"""

import datetime
import os
from io import StringIO
from urllib import request

import pandas as pd

BASE_URL = (
    "https://www.nmdb.eu/nest/draw_graph.php?"
    "formchk=1&allstations=1&tabchoice=revori&dtype=corr_for_efficiency"
    "&output=ascii&tresolution={tres}&yunits={yunits}"
    "&date_choice=bydate&start_day={sd}&start_month={sm}&start_year={sy}"
    "&start_hour=0&start_min=0&end_day={ed}&end_month={em}&end_year={ey}"
    "&end_hour=23&end_min=59"
)

# Sciezka wzgledna do data/ - skrypt jest w skrypty_dl/wlasne/, wiec dwa
# poziomy wyzej do korzenia repo.
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "data", "csv_data_allstations")
os.makedirs(DATA_FOLDER, exist_ok=True)

YEAR_BEGIN, YEAR_END = 2011, 2019
TRESOLUTION = 60  # minuty - serwer w trybie allstations i tak zwraca ~2h na 30-dniowy kawalek
STEP_DAYS = 30
UNITS = 0  # 0 = counts


def fetch_period(start_date, end_date, tresolution):
    url = BASE_URL.format(
        tres=tresolution, yunits=UNITS,
        sd=start_date.day, sm=start_date.month, sy=start_date.year,
        ed=end_date.day, em=end_date.month, ey=end_date.year,
    )
    try:
        with request.urlopen(url, timeout=120) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Blad pobierania {start_date}-{end_date}: {e}")
        return ""


def parse_nmdb_ascii_to_df(data):
    """Parsuje odpowiedz 1 kawalka do DataFrame (kolumny = timestamp + stacje)."""
    start_tag, end_tag = "<pre><code>", "</code>"
    start_idx, end_idx = data.find(start_tag), data.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        print("Nie znaleziono tagow <pre><code> ... </code>")
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
    return df


def nmdb_allstations(years, tresolution, step_days):
    delta = datetime.timedelta(days=step_days)

    for year in years:
        date = datetime.date(year, 1, 1)
        year_end_date = datetime.date(year, 12, 31)

        chunks = []
        while date <= year_end_date:
            next_date = min(date + delta - datetime.timedelta(days=1), year_end_date)
            print(f"Rok {year}: pobieranie {date} .. {next_date}")
            data = fetch_period(date, next_date, tresolution)
            if data:
                df_chunk = parse_nmdb_ascii_to_df(data)
                if df_chunk is not None:
                    chunks.append(df_chunk)
            date = next_date + datetime.timedelta(days=1)

        if not chunks:
            print(f"Brak danych dla roku {year}")
            continue

        # pd.concat dopasowuje kolumny po nazwie - brakujaca stacja w danym
        # kawalku dostaje NaN, zamiast przesuwac pozostale wartosci.
        full = pd.concat(chunks, ignore_index=True, sort=False)
        full = full.drop_duplicates(subset="timestamp").sort_values("timestamp")

        out_path = os.path.join(DATA_FOLDER, f"{year}_allstations_60min.csv")
        full.to_csv(out_path, sep=";", index=False)
        print(f"Zapisano {out_path} ({len(full)} wierszy, {full.shape[1] - 1} stacji)")


def main():
    nmdb_allstations(range(YEAR_BEGIN, YEAR_END + 1), TRESOLUTION, STEP_DAYS)


if __name__ == "__main__":
    main()
