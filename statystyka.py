"""Wspolna implementacja statystyki z sekcji 4 artykulu Homoli i in. (2023).

PO CO TEN MODUL. Do 05.09.2026 ta sama funkcja byla skopiowana w dziewieciu
notebookach (`_znaki`/`przygotuj_szereg` w 20260902a-20260905a), a kazda kopia
mogla sie rozjechac niezauwazona. Tutaj jest jedna implementacja z jawnym
przelacznikiem konwencji, walidowana raz.

DWIE KONWENCJE. Roznica miedzy nimi zostala ustalona 05.09.2026 przez
przeczytanie kodu zespolu z archiwum CREDO (`zrodla/kod/pdf_values8.py`) i
odtworzenie jego zapisanego wyjscia kosz po koszu (20260905c.ipynb, komorka 2;
dziennik 20260905.txt pkt 13a-13g):

  STARA   - to, czym liczylismy do 04.09 wlacznie. Krata koszy zaczyna sie
            dokladnie w t0, przesuniecie EQ wzgledem CR wynosi dokladnie dt,
            mediany liczone na wszystkich koszach, pelny filtr rownania (3),
            remisy A*B = 0 odrzucane z mianownika.
  CREDO   - konwencja, ktorej faktycznie uzywa kod zespolu, a wiec i artykul:
            okno zaczyna sie o jeden kosz PRZED t0; krata jest zakotwiczona nie
            w t0, tylko w PIERWSZYM DOSTEPNYM REKORDZIE (dla CR - pierwszej
            probce, dla EQ - pierwszym trzesieniu), co bierze sie z
            pd.Grouper(freq="5D", origin="start") w ich kodzie; mediany
            liczone po filtrze; filtr skrocony (tylko nCR > 0 i |dCR| > 0);
            remisy zostaja w mianowniku.

Praktyczna konsekwencja kotwiczenia na danych: efektywne przesuniecie EQ
wzgledem CR NIE jest rowne dt. Dla punktu sekcji 3 Moskwy wynosi 14,794688
dnia zamiast 15. Funkcje zwracaja te wielkosc, zeby dalo sie ja raportowac.

CO Z CZYM JEST ZGODNE. `znaki` (szybka, wektorowa) w konwencji STARA odtwarza
co do bitu funkcje `_znaki` z 20260902a-20260904a, a `stat_ref` w konwencji
STARA odtwarza `cosmoseismic_stat` z 20260824f/20260905a, ktora policzyla
wszystkie pliki `results/finetune_*.csv`. W konwencji CREDO obie odtwarzaja
zapisana tabele koszy zespolu. Notebook 20260906a sprawdza to wszystko jawnie.

UWAGA O DWOCH IMPLEMENTACJACH. W konwencji STARA szybka i referencyjna NIE sa
wymienne co do bitu: filtr rownania (3) odrzuca kosze, dla ktorych Sm jest
bitowo rowne medianie, a szum sumowania na 13. miejscu znaczacym rozstrzyga te
remisy inaczej w obu implementacjach (dziennik 20260905.txt pkt 10f). W
konwencji CREDO tego filtra nie ma, wiec efekt jest mniejszy - ale nie zerowy,
bo remis moze przesunac Np o 1. Nigdy nie mieszac obu implementacji w jednym
porownaniu.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

NS_DZIEN = np.int64(86400 * 10 ** 9)


class Konwencja(NamedTuple):
    """Piec przelacznikow oddzielajacych nasza konwencje od konwencji zespolu.

    kotwica       "dane"       krata zaczyna sie w pierwszym dostepnym rekordzie
                  "nominalna"  krawedzie dokladnie w t0 + k*d
    wiodacy_kosz  czy okno zaczyna sie o jeden kosz przed t0
    mediany       "po_filtrze" / "wszystkie"
    filtr         "skrocony"   nCR(i) > 0 i |dCR| > 0  (kod zespolu)
                  "pelny"      nCR(i) > 0, nCR(i-1) > 0, Sm > 0  (rownanie (3))
    remisy        "w_N"        kosze o A*B = 0 zostaja w mianowniku
                  "odrzucone"  wypadaja
    """
    kotwica: str = "dane"
    wiodacy_kosz: bool = True
    mediany: str = "po_filtrze"
    filtr: str = "skrocony"
    remisy: str = "w_N"


CREDO = Konwencja()
STARA = Konwencja(kotwica="nominalna", wiodacy_kosz=False, mediany="wszystkie",
                  filtr="pelny", remisy="odrzucone")


class Okno(NamedTuple):
    """Parametry okna wraz z policzonymi raz offsetami krawedzi."""
    P_days: int
    d_days: int
    dt_days: int
    konw: Konwencja
    n_koszy: int
    offsety: np.ndarray      # (n_koszy + 1,) w nanosekundach, od kotwicy
    przed_ns: np.int64       # ile przed t0 zaczyna sie okno CR (0 albo d)
    dt_ns: np.int64
    P_ns: np.int64


def okno(P_days, d_days, dt_days, konw=CREDO):
    n = int(P_days // d_days) + (1 if konw.wiodacy_kosz else 0)
    krok = np.int64(d_days) * NS_DZIEN
    return Okno(P_days=P_days, d_days=d_days, dt_days=dt_days, konw=konw,
                n_koszy=n, offsety=krok * np.arange(n + 1, dtype=np.int64),
                przed_ns=(np.int64(d_days) * NS_DZIEN if konw.wiodacy_kosz else np.int64(0)),
                dt_ns=np.int64(dt_days) * NS_DZIEN,
                P_ns=np.int64(P_days) * NS_DZIEN)


# --------------------------------------------------------------------------
# Przygotowanie danych. Postac (czasy, wartosci+pad, obecnosc+pad) jest
# przepisana bez zmian z 20260902a-20260904a, zeby stara konwencja dawala
# dokladnie te same liczby co tam. Pad na koncu jest po to, zeby
# np.add.reduceat mogl dostac indeks rowny dlugosci tablicy.
# --------------------------------------------------------------------------
def przygotuj_szereg(s):
    t = s.index.values.astype("datetime64[ns]").astype(np.int64)
    v = s.to_numpy(dtype=float)
    ok = ~np.isnan(v)
    return t, np.append(np.where(ok, v, 0.0), 0.0), np.append(ok.astype(np.int64), 0)


def przygotuj_katalog(kat):
    if isinstance(kat, pd.DataFrame):
        kat = kat.set_index("time")["mag"]
    return przygotuj_szereg(kat.sort_index())


def _sumy_koszy(wartosci, idx):
    s = np.add.reduceat(wartosci, idx)[:-1]
    s[idx[1:] == idx[:-1]] = 0
    return s


# --------------------------------------------------------------------------
# SZYBKA IMPLEMENTACJA WEKTOROWA
# --------------------------------------------------------------------------
def znaki(cr_prep, eq_prep, t0_ns, ok_okno, pelne=False):
    """Zwraca (Np, Nm, N, dt_efektywne_ns) albo None, gdy okna nie da sie ulozyc.

    N jest MIANOWNIKIEM testu dwumianowego: przy remisy="w_N" zawiera takze
    kosze o A*B = 0, wiec Np + Nm moze byc od niego mniejsze - dokladnie tak,
    jak liczy to kod zespolu (PDF = comb(N, n) * 0.5**N przy n = liczba C == +1).

    pelne=True dokłada (p_A, p_B) - udzial koszy o A > 0 i o B > 0 WSROD
    WAZNYCH. Wsrod wszystkich koszy okna oba wynosza 0,5 z definicji mediany;
    odchylenie pojawia sie dopiero po odfiltrowaniu, i to jest material do
    hipotezy H2 (czy sama selekcja koszy nie lamie zalozenia p = 0,5).
    """
    k = ok_okno.konw
    ct, cv, cn = cr_prep
    et, ev, _ = eq_prep

    start_cr_nom = np.int64(t0_ns) - ok_okno.przed_ns
    start_eq_nom = start_cr_nom + ok_okno.dt_ns
    if k.kotwica == "dane":
        i = int(np.searchsorted(ct, start_cr_nom, side="left"))
        j = int(np.searchsorted(et, start_eq_nom, side="left"))
        if i >= len(ct) or j >= len(et):
            return None
        start_cr, start_eq = np.int64(ct[i]), np.int64(et[j])
    else:
        start_cr, start_eq = start_cr_nom, start_eq_nom

    # Gorne granice dokladnie jak w pdf_values8.py: CR do t0 + P, EQ do t0 + dt + P.
    kon_cr = np.int64(t0_ns) + ok_okno.P_ns
    kon_eq = kon_cr + ok_okno.dt_ns
    ci = np.minimum(np.searchsorted(ct, start_cr + ok_okno.offsety, side="left"),
                    np.searchsorted(ct, kon_cr, side="right"))
    ei = np.minimum(np.searchsorted(et, start_eq + ok_okno.offsety, side="left"),
                    np.searchsorted(et, kon_eq, side="right"))

    suma, liczba = _sumy_koszy(cv, ci), _sumy_koszy(cn, ci)
    with np.errstate(invalid="ignore", divide="ignore"):
        cr_vals = np.where(liczba > 0, suma / np.maximum(liczba, 1), np.nan)
    sm_vals = _sumy_koszy(ev, ei)

    nCR_i, nCR_im1 = cr_vals[1:], cr_vals[:-1]
    dCR, Sm = np.abs(nCR_i - nCR_im1), sm_vals[1:]

    with np.errstate(invalid="ignore", divide="ignore"):
        if k.filtr == "skrocony":
            dane_ok = (nCR_i > 0) & (dCR > 0)
        else:
            dane_ok = (nCR_i > 0) & (nCR_im1 > 0) & (Sm > 0) & ~np.isnan(dCR)
        if k.mediany == "po_filtrze":
            if not dane_ok.any():
                return None
            med_Sm, med_dCR = np.nanmedian(Sm[dane_ok]), np.nanmedian(dCR[dane_ok])
        else:
            med_Sm, med_dCR = np.nanmedian(Sm), np.nanmedian(dCR)
        A = Sm / med_Sm - 1.0
        B = dCR / med_dCR - 1.0
        AB = A * B
        wazne = dane_ok & ~np.isnan(AB)
        if k.remisy == "odrzucone":
            wazne = wazne & (A != 0) & (B != 0)

    N = int(wazne.sum())
    if N == 0:
        return None
    ab = AB[wazne]
    wynik = (int((ab > 0).sum()), int((ab < 0).sum()), N, int(start_eq - start_cr))
    if not pelne:
        return wynik
    return wynik + (float((A[wazne] > 0).mean()), float((B[wazne] > 0).mean()))


def krzywa(cr_prep, eq_prep, siatka_ns, ok_okno, max_jitter_d=None):
    """-log10(PPDF) dla kazdego t0 z siatki.

    Zwraca (y, Np, Nm, N, dt_efektywne_d). Ostatnia tablica ma sens tylko przy
    kotwica="dane" i jest tam wazna diagnostyka: pokazuje, gdzie dziura w
    danych przesunela krate wzgledem nominalnego t0 (patrz era 1968 Moskwy,
    dziennik 20260905.txt pkt 13f).

    max_jitter_d - HIGIENA SKANU, nie zmiana statystyki. Kotwiczenie na danych
    (origin="start" w kodzie zespolu) przy skanie po CALEJ historii cicho
    PRZENOSI okno tam, gdzie sa dane: jesli w chwili t0 stacja ma dziure, krata
    zaczyna sie dopiero po niej i liczymy okno przesuniete o tyle, ile trwala
    dziura - a wynik i tak wyglada na policzony. Zespol tego nie mial jak
    zauwazyc, bo liczyl w oknach, ktore mial pokryte danymi. Przy podanym
    progu punkty, dla ktorych |dt_efektywne - dt| przekracza prog, dostaja
    N = 0 i NaN - tak samo jak punkty bez danych w starej konwencji. Sensowny
    prog to szerokosc kosza (d_days): przesuniecie mniejsze niz kosz nie
    zmienia przydzialu danych do koszy w sposob, ktorego nie da sie obronic.
    None (domyslnie) = bez progu, czyli wiernie jak kod zespolu."""
    n = len(siatka_ns)
    Np = np.zeros(n, dtype=np.int64)
    Nm = np.zeros(n, dtype=np.int64)
    Nn = np.zeros(n, dtype=np.int64)
    dt_d = np.full(n, np.nan)
    prog_ns = None if max_jitter_d is None else abs(float(max_jitter_d)) * float(NS_DZIEN)
    odrzucone = 0
    for i, t0_ns in enumerate(siatka_ns):
        r = znaki(cr_prep, eq_prep, t0_ns, ok_okno)
        if r is None:
            continue
        dt_d[i] = r[3] / float(NS_DZIEN)
        if prog_ns is not None and abs(r[3] - int(ok_okno.dt_ns)) > prog_ns:
            odrzucone += 1
            continue
        Np[i], Nm[i], Nn[i] = r[0], r[1], r[2]
    y = np.full(n, np.nan)
    ok = Nn > 0
    with np.errstate(divide="ignore"):
        y[ok] = -np.log10(binom.pmf(Np[ok], Nn[ok], 0.5))
    krzywa.ostatnio_odrzuconych = odrzucone
    return y, Np, Nm, Nn, dt_d


def ramka_krzywej(t0_index, y, Np, Nm, Nn, dt_d, n_koszy):
    """Krzywa w postaci ramki o kolumnach ZGODNYCH z plikami finetune_*.csv
    (N, N_valid, Np, Nm, PPDF, PCDF, sigma, t0), plus neglog10_PPDF i
    dt_efektywne_d. Dzieki tej zgodnosci notebooki czytajace stare pliki
    dzialaja po podmianie samej nazwy pliku."""
    ok = Nn > 0
    pcdf = np.full(len(y), np.nan)
    sig = np.full(len(y), np.nan)
    ppdf = np.full(len(y), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppdf[ok] = binom.pmf(Np[ok], Nn[ok], 0.5)
        pcdf[ok] = binom.sf(Np[ok] - 1, Nn[ok], 0.5)
        sig[ok] = norm.isf(pcdf[ok])
    return pd.DataFrame(dict(N=n_koszy, N_valid=Nn, Np=Np, Nm=Nm, PPDF=ppdf,
                             PCDF=pcdf, sigma=sig, t0=t0_index,
                             neglog10_PPDF=y, dt_efektywne_d=dt_d))


def punkt(cr_prep, eq_prep, t0, ok_okno):
    """Pelny slownik wynikow dla jednego t0 (przyjmuje Timestamp albo ns)."""
    t0_ns = int(pd.Timestamp(t0).value) if not isinstance(t0, (int, np.integer)) else int(t0)
    r = znaki(cr_prep, eq_prep, t0_ns, ok_okno)
    if r is None:
        return dict(N_valid=0, Np=0, Nm=0, remisow=0, PPDF=np.nan, PCDF=np.nan,
                    sigma=np.nan, dt_efektywne_d=np.nan)
    Np, Nm, N, dt_ns = r
    ppdf = float(binom.pmf(Np, N, 0.5))
    pcdf = float(binom.sf(Np - 1, N, 0.5))
    return dict(N_valid=N, Np=Np, Nm=Nm, remisow=N - Np - Nm, PPDF=ppdf, PCDF=pcdf,
                sigma=float(norm.isf(pcdf)), dt_efektywne_d=dt_ns / float(NS_DZIEN))


# --------------------------------------------------------------------------
# IMPLEMENTACJA REFERENCYJNA (pandas). Wolna, ale czytelna - sluzy do
# walidacji szybkiej i do wszystkiego, co musi sie zgadzac z plikami
# finetune_*.csv policzonymi przez cosmoseismic_stat.
# --------------------------------------------------------------------------
def kosze_ref(cr, eq, t0, P_days, d_days, dt_days, konw=CREDO):
    """Zwraca slownik z tablicami na kosz - do porownan kosz-w-kosz."""
    t0 = pd.Timestamp(t0)
    n_koszy = int(P_days // d_days) + (1 if konw.wiodacy_kosz else 0)
    start_cr_nom = t0 - pd.Timedelta(days=d_days) if konw.wiodacy_kosz else t0
    start_eq_nom = start_cr_nom + pd.Timedelta(days=dt_days)
    kon_cr = t0 + pd.Timedelta(days=P_days)
    kon_eq = t0 + pd.Timedelta(days=dt_days + P_days)

    cr = cr[(cr.index >= start_cr_nom) & (cr.index <= kon_cr)]
    eq = eq[(eq.index >= start_eq_nom) & (eq.index <= kon_eq)]
    if len(cr) == 0 or len(eq) == 0:
        return None
    if konw.kotwica == "dane":
        start_cr, start_eq = cr.index[0], eq.index[0]
    else:
        start_cr, start_eq = start_cr_nom, start_eq_nom

    kr_cr = pd.date_range(start_cr, periods=n_koszy + 1, freq=pd.Timedelta(days=d_days))
    kr_eq = pd.date_range(start_eq, periods=n_koszy + 1, freq=pd.Timedelta(days=d_days))
    cr_cats = pd.cut(cr.index, kr_cr, right=False)
    cr_vals = cr.groupby(cr_cats, observed=False).mean().reindex(cr_cats.categories).to_numpy()
    eq_cats = pd.cut(eq.index, kr_eq, right=False)
    sm_vals = (eq.groupby(eq_cats, observed=False).sum()
               .reindex(eq_cats.categories, fill_value=0.0).to_numpy())

    nCR_i, nCR_im1 = cr_vals[1:], cr_vals[:-1]
    dCR, Sm = np.abs(nCR_i - nCR_im1), sm_vals[1:]

    with np.errstate(invalid="ignore", divide="ignore"):
        if konw.filtr == "skrocony":
            dane_ok = (nCR_i > 0) & (dCR > 0)
        else:
            dane_ok = (nCR_i > 0) & (nCR_im1 > 0) & (Sm > 0) & ~np.isnan(dCR)
        if konw.mediany == "po_filtrze":
            if not dane_ok.any():
                return None
            med_Sm, med_dCR = np.nanmedian(Sm[dane_ok]), np.nanmedian(dCR[dane_ok])
        else:
            med_Sm, med_dCR = np.nanmedian(Sm), np.nanmedian(dCR)
        A, B = Sm / med_Sm - 1.0, dCR / med_dCR - 1.0
        C = np.sign(A * B)
        wazne = dane_ok & ~np.isnan(A * B)
        if konw.remisy == "odrzucone":
            wazne = wazne & (A != 0) & (B != 0)

    N = int(wazne.sum())
    if N == 0:
        return None
    Np, Nm = int((C[wazne] > 0).sum()), int((C[wazne] < 0).sum())
    ppdf, pcdf = float(binom.pmf(Np, N, 0.5)), float(binom.sf(Np - 1, N, 0.5))
    return dict(N=n_koszy, N_valid=N, Np=Np, Nm=Nm, remisow=N - Np - Nm,
                PPDF=ppdf, PCDF=pcdf, sigma=float(norm.isf(pcdf)),
                med_Sm=float(med_Sm), med_dCR=float(med_dCR),
                start_cr=start_cr, start_eq=start_eq,
                dt_efektywne_d=(start_eq - start_cr) / pd.Timedelta(days=1),
                jitter_cr_h=(start_cr - start_cr_nom) / pd.Timedelta(hours=1),
                jitter_eq_h=(start_eq - start_eq_nom) / pd.Timedelta(hours=1),
                kraw_cr=kr_cr[1:-1], kraw_eq=kr_eq[1:-1],
                cr_mean=nCR_i, Sm=Sm, dCR=dCR, A=A, B=B, C=C, wazne=wazne)


_SKALARY = ("N", "N_valid", "Np", "Nm", "remisow", "PPDF", "PCDF", "sigma",
            "med_Sm", "med_dCR", "dt_efektywne_d", "jitter_cr_h", "jitter_eq_h")


def stat_ref(cr, eq, t0, P_days, d_days, m, dt_days, konw=CREDO):
    """Sygnatura zgodna z mc_parallel.run_t0_scan_parallel (argument `m` jest
    ignorowany - prog magnitudy nakladamy przy wczytywaniu katalogu, tak samo
    jak robil to cosmoseismic_stat)."""
    r = kosze_ref(cr, eq, t0, P_days, d_days, dt_days, konw)
    if r is None:
        return {k: (0 if k in ("N", "N_valid", "Np", "Nm", "remisow") else np.nan)
                for k in _SKALARY}
    return {k: r[k] for k in _SKALARY}


def stat_ref_fn(konw):
    """Funkcja o sygnaturze stat_fn dla run_t0_scan_parallel, z zaszyta konwencja."""
    from functools import partial
    return partial(stat_ref, konw=konw)
