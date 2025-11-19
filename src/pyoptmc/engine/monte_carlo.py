import numpy as np
from tqdm import tqdm
from pyoptmc.structures.base import StructureMC
from pyoptmc.model.market_process import BlackScholes
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from multiprocessing import cpu_count


__all__ = ['MonteCarlo']


def _run_one_time_caller(
        batch_size: int,
        option: StructureMC,
        process: BlackScholes,
        request_greeks: bool = False,
):
    _coordinator = process.coordinator(option, process)
    df = _coordinator.df

    if not request_greeks:
        def _calc(seed):
            eps = _coordinator.generate_eps(seed, batch_size)
            path = _coordinator.paths_given_eps(eps)
            return option.pv_log_paths(path, df)
        return _calc

    fd_steps = dict(ds=0.01, dr=0.01, dv=0.005)
    ds, dr, dv = fd_steps['ds'], fd_steps['dr'], fd_steps['dv']
    ds_sq = ds * ds

    def _calc(seed):
        eps = _coordinator.generate_eps(seed, batch_size)
        base_path = _coordinator.paths_given_eps(eps)
        shifted_path = _coordinator.shift(
            paths=base_path, ds=ds, dr=dr, dv=dv, eps=eps
        )

        # base PV
        pv = option.pv_log_paths(base_path, df)

        # S: delta, gamma
        pv_s_plus = option.pv_log_paths(shifted_path['S plus'], df)
        pv_s_minus = option.pv_log_paths(shifted_path['S minus'], df)
        delta = (pv_s_plus - pv_s_minus) / (2 * ds) / option.spot
        gamma = (pv_s_plus + pv_s_minus - 2 * pv) / (
            ds_sq * option.spot * option.spot
        )

        # R: rho
        pv_r_plus = option.pv_log_paths(
            shifted_path['R plus'], shifted_path['DF plus']
        )
        rho = (pv_r_plus - pv) / 10.0

        # V: vega
        pv_v_plus = option.pv_log_paths(shifted_path['V plus'], df)
        pv_v_minus = option.pv_log_paths(shifted_path['V minus'], df)
        vega = pv_v_plus - pv_v_minus

        # Theta
        pv_next_day = option.pv_log_paths(
            shifted_path['Paths next day'], shifted_path['DF next day']
        )
        theta = pv_next_day - pv

        return pv, delta, gamma, rho, vega, theta

    _calc.__doc__ = (
        "Run 1 time of Monte Carlo simulation given a random seed.\n"
        "Parameters:\n"
        f"batch_size={batch_size}, option={option}, process={process}, "
        f"request_greeks={request_greeks}"
    )
    return _calc


def joblib_caller(calc, seed_sequence, *,
                  n_jobs=None,
                  backend="loky",
                  prefer=None,
                  batch_size="auto",
                  verbose=0,
                  show_progress=True,
                  progress_desc="Running Monte Carlo",
                  chunk_size=None):

    seed_list = list(seed_sequence)
    total = len(seed_list)

    if total == 0:
        return []

    if n_jobs is None or n_jobs <= 0:
        n_jobs = cpu_count()

    wrapped_calc = wrap_non_picklable_objects(calc)

    if chunk_size is None:
        chunk_size = max(1, total // 50)

    results = []

    with Parallel(
        n_jobs=n_jobs,
        backend=backend,
        prefer=prefer,
        batch_size=batch_size,
        verbose=verbose,
    ) as parallel:
        if show_progress:
            with tqdm(total=total, desc=progress_desc) as pbar:
                for start in range(0, total, chunk_size):
                    chunk = seed_list[start:start + chunk_size]
                    chunk_res = parallel(
                        delayed(wrapped_calc)(s) for s in chunk
                    )
                    results.extend(chunk_res)
                    pbar.update(len(chunk))
        else:
            results = parallel(
                delayed(wrapped_calc)(s) for s in seed_list
            )

    return results

class MonteCarlo:
    most_recent_entropy = property(
        lambda self: self._most_recent_entropy,
        lambda self, v: None, lambda self: None,
        "The most recently used entropy."
    )

    @property
    def caller(self):
        return self._caller

    @caller.setter
    def caller(self, val):
        self._caller = val

    @caller.deleter
    def caller(self):
        self._caller = None

    caller.__doc__ = \
        """Default caller used to implement Monte Carlo simulation.
        if set to None, *joblib.Parallel* will be used."""

    def __init__(self, batch_size: int, num_iter: int, caller=None):
        """A Monte Carlo engine for valuing path-dependent options.

        Parameters regarding the simulation are specified here. This engine implements
        vectorization, which significantly enhances algorithm speed.

        Parameters
        ----------
        batch_size : int
            An integer telling the engine how many paths should be generated in each
            iteration.
        num_iter : int
            Number of iterations. The product of *batch_size* and *num_iter* is the total
            number of paths generated."""
        self.batch_size = batch_size
        self.num_iter = num_iter
        self._most_recent_entropy = None
        self._caller = caller

    def calc(self, option: StructureMC, process: BlackScholes,
             request_greeks=False, entropy=None, caller=None, caller_args=None):
        ss = np.random.SeedSequence(entropy)
        self._most_recent_entropy = ss.entropy
        subs = ss.spawn(self.num_iter)

        _calc = _run_one_time_caller(
            self.batch_size, option, process, request_greeks
        )

        if caller_args is None:
            caller_args = {}

        if caller is None:
            caller = self._caller
        if caller is None:
            caller = joblib_caller

        if not callable(caller):
            raise TypeError("caller must be callable or None")

        res = caller(
            _calc, subs,
            n_jobs=cpu_count(),
            show_progress=True,
            progress_desc="Monte Carlo Greeks",
        )
        res_mean = np.mean(res, axis=0)

        if not request_greeks:
            return res_mean

        _pv, _delta, _gamma, _rho, _vega, _theta = res_mean
        return dict(PV=_pv, Delta=_delta, Gamma=_gamma,
                    Rho=_rho, Vega=_vega, Theta=_theta)

    def single_iter_caller(
            self, option: StructureMC, process: BlackScholes,
            request_greeks=False
    ):
        return _run_one_time_caller(
            batch_size=self.batch_size, option=option,
            process=process,
            request_greeks=request_greeks
        )
