"""
Rozszerza katalog trzesien ziemi USGS (M>=4.0) WSTECZ, z 2005 do 1965 -
przygotowanie pod sugestie Homoli (przeczytana 22.08.2026, patrz
20260822.txt): "Ciekawe bedzie porownanie optymalnych t0 dla roznych dekad
i roznych stacji ... wybrac epoki wg cykli slonecznych, dla kazdego cyklu
wziac ekstremalne PPDF i odpowiadajace mu t0 ... globalne mapy efektu
cykl do cyklu".

Dlaczego ten skrypt istnieje (2026-08-22):
Dotychczasowy katalog EQ (`usgs_m4_2005_2025.csv`, pobrany 2026-07-15,
patrz `download_usgs_catalog.sh`) obejmuje TYLKO 2005-2025 (~2 cykle
sloneczne: koncowka 23, caly 24, poczatek 25) - za malo, zeby zrobic
porownanie cykl-do-cyklu, o ktore prosi Homola. Dane CR (mosc_data.csv
1960-2025, oulu_5min_data.csv 1970-2025) juz siegaja wystarczajaco daleko -
to katalog EQ jest waskim gardlem.

Dlaczego DOKLADNIE 1965 jako granica: to (prawdopodobnie nieprzypadkowo) ten
sam punkt startowy, ktorego uzyl sam artykul w swojej analizie cykl-do-cyklu
(sekcja 4/Fig. 4, patrz 20260819.txt pkt 7 - "analiza cykli slonecznych
1965-2015"). Pokrywa sie z ustanowieniem globalnej sieci sejsmografow WWSSN
(1961-1967), po ktorej katalogi M>=4.0 sa uwazane za w miare kompletne
globalnie - przed tym dane sa rzadsze/mniej wiarygodne. Sensowniej pojsc za
tym samym punktem, niz wymyslac wlasny.

UWAGA METODOLOGICZNA do zapamietania przy interpretacji wynikow: kompletnosc
katalogu M>=4.0 NIE jest jednorodna w calym zakresie 1965-2025 - starsze
dekady (zwlaszcza lata 60./70.) maja gorsze pokrycie instrumentalne niz
dekady po 1990 (sieci cyfrowe). Przy porownywaniu "sily efektu" miedzy
cyklami trzeba to brac pod uwage (mniej kompletny katalog = inna
charakterystyka statystyki testu, niezaleznie od realnego sygnalu CR-EQ).

Podejscie (ten sam wzorzec co `download_usgs_catalog.sh` - zapytania
MIESIECZNE do USGS FDSN event API, bo pojedyncze zapytanie ma limit ~20000
zdarzen a caly zakres 2005-2025 to ~290000 zdarzen; dla starszych dekad
zdarzen/rok jest pravdopodobnie mniej, ale miesieczna granulacja zostaje dla
bezpieczenstwa i spojnosci):
1. Pobiera TYLKO brakujacy zakres 1965-01-01 .. 2005-01-01 (NIE powtarza
   pobrania 2005-2025 - to juz mamy w `usgs_m4_2005_2025.csv`) -
   zapytanie na miesiac, zapisywane jako osobne pliki-kawalki w
   `data/usgs_data/usgs_extended_chunks/{rok}_{miesiac}.csv` - wznawialne
   (pomija miesiace z juz istniejacym plikiem).
2. Gdy WSZYSTKIE oczekiwane kawalki (1965-01 .. 2004-12, 480 miesiecy) sa na
   dysku, laczy je z istniejacym `usgs_m4_2005_2025.csv` w jeden plik
   `data/usgs_data/usgs_m4_1965_2025.csv` (posortowany po `time`,
   odduplikowany po `id`) - NIE nadpisuje `usgs_m4_2005_2025.csv`, ktory
   zostaje nietkniety dla notebookow, ktore go juz uzywaja.

Uzycie:
    python3 download_usgs_extended_history.py
Bezpiecznie przerywalne/wznawialne (pomija miesiace z istniejacym plikiem
kawalka). Szacunek: 480 zapytan miesiecznych x (~1s odstepu
grzecznosciowego + czas odpowiedzi USGS) - rzedu 15-30 minut, nie godzin,
ale i tak nie odpalac bez zastanowienia (odpala Maciek).
"""

import glob
import os
import time
from io import StringIO
from urllib import request

import pandas as pd

MIN_MAGNITUDE = 4.0
YEAR_BEGIN = 1965  # patrz uzasadnienie w docstringu (WWSSN / Fig.4 artykulu)
YEAR_END_EXCLUSIVE = 2005  # istniejacy katalog zaczyna sie dokladnie tu, nie powielamy

SLEEP_BETWEEN_REQUESTS = 1.0  # sekundy - uprzejmosciowy odstep miedzy zapytaniami do USGS
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120

BASE_URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query"
    "?format=csv&starttime={start}&endtime={end}&minmagnitude={minmag}"
)

# Sciezki wzgledne do data/usgs_data - skrypt jest w skrypty_dl/wlasne/, wiec
# dwa poziomy wyzej do korzenia repo.
USGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "usgs_data")
CHUNKS_DIR = os.path.join(USGS_DIR, "usgs_extended_chunks")
EXISTING_CATALOG_PATH = os.path.join(USGS_DIR, "usgs_m4_2005_2025.csv")
FINAL_OUTPUT_PATH = os.path.join(USGS_DIR, "usgs_m4_1965_2025.csv")
os.makedirs(CHUNKS_DIR, exist_ok=True)


def month_range(year_begin, year_end_exclusive):
    """(rok, miesiac) dla kazdego miesiaca od year_begin-01 do (year_end_exclusive-1)-12 wlacznie."""
    for year in range(year_begin, year_end_exclusive):
        for month in range(1, 13):
            yield year, month


def next_month(year, month):
    return (year + 1, 1) if month == 12 else (year, month + 1)


def fetch_month(year, month):
    """Pobiera 1 miesiac (starttime <= t < endtime) dla catalog M>=MIN_MAGNITUDE."""
    end_year, end_month = next_month(year, month)
    start = f"{year}-{month:02d}-01"
    end = f"{end_year}-{end_month:02d}-01"
    url = BASE_URL.format(start=start, end=end, minmag=MIN_MAGNITUDE)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with request.urlopen(url, timeout=REQUEST_TIMEOUT) as response:
                return response.read().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"    proba {attempt}/{MAX_RETRIES} nieudana ({e})")
            if attempt < MAX_RETRIES:
                time.sleep(5)
    return ""


def download_chunk(year, month):
    out_path = os.path.join(CHUNKS_DIR, f"{year}_{month:02d}.csv")
    if os.path.exists(out_path):
        return
    print(f"{year}-{month:02d}: pobieranie...")
    text = fetch_month(year, month)
    if not text.strip():
        print(f"  {year}-{month:02d}: pusta odpowiedz, pomijam (NIE zapisuje pliku - zostanie ponowione przy nastepnym uruchomieniu)")
        return
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    n_rows = max(0, text.count("\n") - 1)
    print(f"  {year}-{month:02d}: zapisano {n_rows} wierszy -> {out_path}")
    time.sleep(SLEEP_BETWEEN_REQUESTS)


def merge_final_catalog():
    expected_months = list(month_range(YEAR_BEGIN, YEAR_END_EXCLUSIVE))
    chunk_paths = [os.path.join(CHUNKS_DIR, f"{y}_{m:02d}.csv") for y, m in expected_months]
    missing = [p for p in chunk_paths if not os.path.exists(p)]
    if missing:
        print(f"\nBrakuje jeszcze {len(missing)}/{len(chunk_paths)} kawalkow - laczenie pominiete. "
              f"Uruchom skrypt ponownie, zeby dokonczyc pobieranie.")
        return
    if not os.path.exists(EXISTING_CATALOG_PATH):
        print(f"\nBRAK {EXISTING_CATALOG_PATH} - nie moge polaczyc z istniejacym katalogiem 2005-2025.")
        return

    print("\nWszystkie kawalki 1965-2004 pobrane - laczenie z istniejacym katalogiem 2005-2025...")
    frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(CHUNKS_DIR, "*.csv"))) if os.path.getsize(p) > 0]
    frames.append(pd.read_csv(EXISTING_CATALOG_PATH))
    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset="id").sort_values("time")
    full.to_csv(FINAL_OUTPUT_PATH, index=False)
    print(f"Zapisano {len(full)} zdarzen -> {FINAL_OUTPUT_PATH}")
    print(f"Zakres: {full['time'].min()} .. {full['time'].max()}")


def main():
    for year, month in month_range(YEAR_BEGIN, YEAR_END_EXCLUSIVE):
        download_chunk(year, month)
    merge_final_catalog()


if __name__ == "__main__":
    main()
