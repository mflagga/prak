"""Równoległa (multiprocessing) wersja skanu Monte Carlo z 20260715.ipynb.

Wydzielone do osobnego pliku, bo funkcje przekazywane do
ProcessPoolExecutor muszą być importowalne przez procesy potomne
(nie mogą być zdefiniowane w komórce notebooka pod Windows/spawn;
na Linuksie fork by wystarczył, ale trzymamy się rozwiązania
przenośnego i użytecznego też poza Jupyterem, np. do przyszłego n=1e5).

Od 20260819 zawiera też `run_t0_scan_parallel` — ten sam wzorzec
(fork-based ProcessPoolExecutor) zastosowany do pełnego skanu po t0
(zamiast po przesunięciach MC), bo liczba kandydatów t0 (~16k na stację
przy oknie 3350d/kroku 6h) jest podobnego rzędu wielkości co n_sims
w skanach MC, które już się tu dobrze skalują.
"""

import numpy as np
import pandas as pd

_state = {}


def _init_worker(cr, eq, t0, P_days, m, dt_days, full_d_scan_fn, circular_shift_fn):
    _state["cr"] = cr
    _state["eq"] = eq
    _state["t0"] = t0
    _state["P_days"] = P_days
    _state["m"] = m
    _state["dt_days"] = dt_days
    _state["full_d_scan"] = full_d_scan_fn
    _state["circular_shift_eq"] = circular_shift_fn


def _mc_worker(seed_seq):
    rng = np.random.default_rng(seed_seq)
    eq_shift = _state["circular_shift_eq"](_state["eq"], rng)
    sim = _state["full_d_scan"](
        _state["cr"], eq_shift, _state["t0"], _state["P_days"],
        _state["m"], _state["dt_days"], range(1, 31),
    )
    return np.nanmin(list(sim.values()))


def run_mc_parallel(cr, eq, t0, P_days, m, dt_days, n_sims, full_d_scan_fn,
                     circular_shift_fn, base_seed=42, n_workers=None, save_path=None):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import os

    n_workers = n_workers or os.cpu_count()
    seeds = np.random.SeedSequence(base_seed).spawn(n_sims)

    # Python 3.14 zmienił domyślną metodę na "forkserver" (wymaga
    # bezpiecznego importu __main__, czego notebooki nie gwarantują).
    # Wymuszamy "fork" - na Linuksie bezpieczne, bo nie mamy wątków
    # przed startem puli procesów.
    ctx = multiprocessing.get_context("fork")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(cr, eq, t0, P_days, m, dt_days, full_d_scan_fn, circular_shift_fn),
    ) as ex:
        results = list(ex.map(_mc_worker, seeds, chunksize=max(1, n_sims // (n_workers * 4))))

    mc_minima = np.array(results)

    # Zapis na dysk od razu po zakończeniu - żeby po ewentualnym zamknięciu
    # notebooka bez zapisu (albo crashu przy próbie narysowania wykresu itp.)
    # nie trzeba było liczyć od nowa całego (długiego, np. n=1e5) przebiegu.
    if save_path is not None:
        np.save(save_path, mc_minima)

    return mc_minima


_t0_state = {}


def _init_t0_worker(cr, eq, P_days, d_days, m, dt_days, stat_fn):
    _t0_state["cr"] = cr
    _t0_state["eq"] = eq
    _t0_state["P_days"] = P_days
    _t0_state["d_days"] = d_days
    _t0_state["m"] = m
    _t0_state["dt_days"] = dt_days
    _t0_state["stat_fn"] = stat_fn


def _t0_worker(t0):
    s = _t0_state
    r = s["stat_fn"](s["cr"], s["eq"], t0, s["P_days"], s["d_days"], s["m"], s["dt_days"])
    r["t0"] = t0
    return r


def run_t0_scan_parallel(cr, eq, t0_candidates, P_days, d_days, m, dt_days,
                          stat_fn, n_workers=None, save_path=None):
    """Pełny skan po t0 (kandydaci = t0_candidates, np. pd.date_range co 6h),
    dla ustalonych P_days/d_days/m/dt_days. `stat_fn` to funkcja typu
    cosmoseismic_stat(cr, eq, t0, P_days, d_days, m, dt_days) -> dict
    (musi zwracać przynajmniej PPDF/PCDF/sigma/Np/Nm - patrz notebooki
    20260819*). Zwraca DataFrame, jeden wiersz na kandydata t0.

    Uwaga wydajnościowa: `cr`/`eq` warto przyciąć PRZED wywołaniem tej
    funkcji do minimalnego zakresu dat potrzebnego dla WSZYSTKICH
    kandydatów t0 (cosmoseismic_stat robi pd.cut na całym cr.index przy
    każdym wywołaniu) - patrz komentarz w 20260819a.ipynb.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import os

    n_workers = n_workers or os.cpu_count()
    t0_list = list(t0_candidates)
    ctx = multiprocessing.get_context("fork")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_t0_worker,
        initargs=(cr, eq, P_days, d_days, m, dt_days, stat_fn),
    ) as ex:
        results = list(ex.map(_t0_worker, t0_list, chunksize=max(1, len(t0_list) // (n_workers * 4))))

    df = pd.DataFrame(results)

    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df


def run_jobs_parallel(job_fn, jobs, n_workers=None, save_path=None,
                      opis="zadan", raport_co=1):
    """Uruchamia `job_fn(job)` dla każdego elementu `jobs` w puli procesów.

    Celowo BEZ żadnej wiedzy o analizie: `job_fn` ma zwrócić listę słowników
    (albo pojedynczy słownik), a ta funkcja tylko rozdziela zadania, raportuje
    postęp i skleja wynik w DataFrame. Cała logika merytoryczna zostaje w
    notebooku — zgodnie z konwencją repo, że notebooki są samodzielne.

    Dodane 20260902 na potrzeby kalibracji surogatowej (20260902a.ipynb), gdzie
    jedno zadanie = jeden katalog surogatowy przeliczony przez cały łańcuch
    sekcji 4. `run_t0_scan_parallel` się do tego nie nadaje, bo zrównolegla po
    t0 przy USTALONYM katalogu — dla 200 katalogów tworzyłaby 200 pul.

    Uwagi:
    - Kontekst `fork` (jak w pozostałych funkcjach tego modułu): `job_fn`
      zdefiniowana w komórce notebooka jest wtedy przekazywalna, a duże obiekty
      globalne (serie CR, katalog) procesy potomne dziedziczą bez kopiowania.
      Pod Pythonem 3.14 `fork` trzeba brać jawnie — domyślny kontekst się
      zmienił (patrz komentarz przy `run_mc_parallel`).
    - `save_path`: zapis następuje DOPIERO po zebraniu wszystkich wyników
      (pojedyncze zadanie jest tu tanie, więc atomowy zapis per zadanie jak w
      `run_t0_scan_parallel` nie jest potrzebny).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    import os
    import time

    n_workers = n_workers or os.cpu_count()
    jobs = list(jobs)
    ctx = multiprocessing.get_context("fork")

    start = time.time()
    wyniki = []
    zrobione = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = {ex.submit(job_fn, j): j for j in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            wyniki.extend(r if isinstance(r, list) else [r])
            zrobione += 1
            if raport_co and (zrobione % raport_co == 0 or zrobione == len(jobs)):
                minelo = time.time() - start
                zostalo = minelo / zrobione * (len(jobs) - zrobione)
                print(f"  {zrobione}/{len(jobs)} {opis}; "
                      f"minelo {minelo / 60:.1f} min, zostalo ~{zostalo / 60:.1f} min",
                      flush=True)

    df = pd.DataFrame(wyniki)
    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"Zapisano {save_path} ({len(df)} wierszy)")
    return df
