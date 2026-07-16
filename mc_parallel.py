"""Równoległa (multiprocessing) wersja skanu Monte Carlo z 20260715.ipynb.

Wydzielone do osobnego pliku, bo funkcje przekazywane do
ProcessPoolExecutor muszą być importowalne przez procesy potomne
(nie mogą być zdefiniowane w komórce notebooka pod Windows/spawn;
na Linuksie fork by wystarczył, ale trzymamy się rozwiązania
przenośnego i użytecznego też poza Jupyterem, np. do przyszłego n=1e5).
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
