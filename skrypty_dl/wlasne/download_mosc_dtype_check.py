"""
Diagnostyczne pobranie danych Moskwy (MOSC) w CZTERECH wariantach
dtype x units z NMDB, na potrzeby sprawdzenia hipotezy z 20260717.txt
(pkt "NIEROZSTRZYGNIETE", punkt 1) / 20260819.txt: czy niezgodnosc naszej
replikacji artykulu dla Moskwy (u nas N+=206,N-=128 vs artykul N+=218,
N-=113, przy identycznym przepisie P=1675,d=5,m=4.0,dt=15,t0=14 lis 2013)
bierze sie z innego dtype/units NMDB niz to, ktorego (prawdopodobnie)
uzyl artykul.

Nasz dotychczasowy `mosc_data.csv` (i wszystkie inne pobrania w tym repo)
uzywaja dtype=corr_for_efficiency, units=0 - patrz
skrypty_dl/cudze/Pobieranie_Oulu_+_Mosc.ipynb (komorka 10) i
skrypty_dl/wlasne/download_stations_native6h.py. Artykul NIE podaje wprost
jakiego dtype/units uzyl (sprawdzone pdftotext na
zrodla/GlownyArtykul.pdf - brak wzmianki o "pressure"/"efficiency"), wiec
to tylko hipoteza do sprawdzenia empirycznie, nie potwierdzona wprost w
tekscie.

Pobiera TYLKO okno potrzebne do jednego konkretnego testu (nie cala
historie): CR od t0=2013-11-14 do t0+P_days=2018-06-16 (lata 2013-2018,
z zapasem na brzegach roku). To male, szybkie pobranie (5 lat x 4
warianty = 20 zapytan rocznych), w przeciwienstwie do pelnych pobran
historii w innych skryptach tego katalogu.

Wyjscie: data/mosc_dtype_check/mosc_{dtype_skrot}_{units}.csv (kolumny
datetime,value - ten sam format co mosc_data.csv), np.
mosc_eff_0.csv (= nasz dotychczasowy wariant, do potwierdzenia ze sie
zgadza z mosc_data.csv na tym samym oknie), mosc_eff_1.csv,
mosc_press_0.csv, mosc_press_1.csv.

Uzycie:
    python3 download_mosc_dtype_check.py
Bezpieczne do przerwania/wznowienia - warianty z istniejacym plikiem sa
pomijane.
"""

import os
import time
from io import StringIO
from urllib import request

import pandas as pd

STATION = "MOSC"
YEAR_BEGIN, YEAR_END = 2013, 2018
TRESOLUTION = 360  # 6h - jak we wszystkich innych pobraniach tego repo
SLEEP_BETWEEN_REQUESTS = 2
MAX_RETRIES = 3

VARIANTS = [
    ("corr_for_efficiency", 0),  # = nasz dotychczasowy standard (kontrola spojnosci)
    ("corr_for_efficiency", 1),
    ("corr_for_pressure", 0),
    ("corr_for_pressure", 1),
]

BASE_URL = (
    "https://www.nmdb.eu/nest/draw_graph.php?formchk=1&stations[]={station}"
    "&tabchoice=revori&dtype={dtype}&tresolution={tres}"
    "&force=1&yunits={yunits}&date_choice=bydate"
    "&start_day={sd}&start_month={sm}&start_year={sy}"
    "&start_hour=0&start_min=0&end_day={ed}&end_month={em}&end_year={ey}"
    "&end_hour=0&end_min=0&output=ascii"
)

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mosc_dtype_check")
os.makedirs(DATA_FOLDER, exist_ok=True)


def dtype_short(dtype):
    return "eff" if dtype == "corr_for_efficiency" else "press"


def fetch_year(dtype, units, year):
    """Pobiera 1 rok danych (1 sty {year} .. 1 sty {year+1}) dla danego wariantu."""
    url = BASE_URL.format(
        station=STATION, dtype=dtype, tres=TRESOLUTION, yunits=units,
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


def parse_nmdb_ascii_to_df(data, label):
    """Ta sama logika parsowania co w download_stations_native6h.py."""
    start_tag, end_tag = "<pre><code>", "</code>"
    start_idx, end_idx = data.find(start_tag), data.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        print(f"    {label}: nie znaleziono tagow <pre><code> ... </code>")
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


def download_variant(dtype, units):
    out_path = os.path.join(DATA_FOLDER, f"mosc_{dtype_short(dtype)}_{units}.csv")
    if os.path.exists(out_path):
        print(f"{dtype}/units={units}: plik juz istnieje ({out_path}), pomijam")
        return

    print(f"\n=== dtype={dtype}, units={units} ===")
    chunks = []
    for year in range(YEAR_BEGIN, YEAR_END + 1):
        print(f"  rok {year}...")
        data = fetch_year(dtype, units, year)
        if not data:
            print(f"  {year}: brak odpowiedzi, pomijam ten rok")
            continue
        df_chunk = parse_nmdb_ascii_to_df(data, f"{dtype}/{units}/{year}")
        if df_chunk is not None:
            chunks.append(df_chunk)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if not chunks:
        print(f"{dtype}/units={units}: brak danych w calym zakresie, nie zapisuje pliku")
        return

    full = pd.concat(chunks, ignore_index=True)
    full = full.drop_duplicates(subset="timestamp").sort_values("timestamp")
    full = full.rename(columns={"timestamp": "datetime"})
    full.to_csv(out_path, index=False)

    delta_median = full["datetime"].diff().median()
    print(f"{dtype}/units={units}: zapisano {len(full)} wierszy -> {out_path}")
    print(f"  zakres: {full['datetime'].min()} .. {full['datetime'].max()}")
    print(f"  mediana odstepu czasowego: {delta_median}")


def main():
    for dtype, units in VARIANTS:
        download_variant(dtype, units)


if __name__ == "__main__":
    main()
