import numpy as np
from akshare import spot

from pyoptmc.tools.helper import (
    up_ko_t_and_surviving_paths,
    down_ko_t_and_surviving_paths,
    double_ko_and_surviving_paths,
    down_ki_paths,
    arr_scalar_converter,
    PayoffWrapper,
    merge_days,
    merge_days_tuple,
    check_ko_path,
    check_up_settle_idx,
)
from pyoptmc.tools.enum import SettlementType, BarrierType
from pyoptmc.structures.base import StructureMC
from pyoptmc.structures._docs import _pv_log_paths_docs
from pyoptmc._decorators import DocstringWriter
from enum import Enum

__all__ = ['SingleBarrierAccumulator', 'DoubleBarrierAccumulator']



class SingleBarrierAccumulator(StructureMC):
    def __init__(self,
                 spot,
                 settlement_type,
                 barrier_type,
                 ko_calc_type, ko_barrier, ko_payoff,
                 ki_payoff,
                 unhit_payoff,
                 ob_days):
        self.ko_barrier = arr_scalar_converter(ko_barrier, ob_days)
        self.log_ko_barrier = np.log(self.ko_barrier/spot)
        ko_payoff_helper = PayoffWrapper(ob_days, ko_payoff, ko_calc_type)
        self.ko_payoffs = ko_payoff_helper.get_payoff_vec()
        self.unhit_payoffs = [ki_payoff + unhit_payoff for _ in ob_days]
        self._spot = spot
        self._settlement_type = settlement_type
        self._barrier_type = barrier_type
        self._sim_t_array = np.append([0], ob_days)
        barrier_lookup = {
            BarrierType.UP_OUT: up_ko_t_and_surviving_paths,
            BarrierType.DOWN_OUT: down_ko_t_and_surviving_paths,
        }
        self.func = barrier_lookup.get(barrier_type)


    def _set_spot(self, val):
        if val <= 0:
            raise ValueError("Spot price should be positive.")
        self._spot = val
        self.log_ko_barrier = np.log(self.ko_barrier/val)


    @DocstringWriter(_pv_log_paths_docs)
    def pv_log_paths(self, log_paths, df):
        ko_t, ko_mask, nko_mask = self.func(
            log_paths, self.log_ko_barrier, return_idx=True
        )

        spots = np.exp(log_paths) * self.spot
        n_paths, n_steps = spots.shape
        is_deferred = (self._settlement_type == SettlementType.AT_MATURITY)

        nko_matrix = np.array([f(spots[:, i]) for i, f in enumerate(self.unhit_payoffs)]).T
        ko_matrix = np.array([f(spots[:, i]) for i, f in enumerate(self.ko_payoffs)]).T

        pv_total = 0

        if np.any(ko_mask):
            settled_ko = nko_matrix[ko_mask]
            ko = ko_matrix[ko_mask]
            t_ko = ko_t[ko_mask]

            step_indices = np.arange(n_steps)
            alive_mask = step_indices <= t_ko[:, None]
            actual_rebates = ko[np.arange(len(t_ko)), t_ko]

            if is_deferred:
                sum_coupons = np.sum(settled_ko * alive_mask, axis=1)
                pv_ko = np.sum((sum_coupons + actual_rebates) * df[t_ko])
            else:
                sum_coupons_discounted = np.sum(settled_ko * alive_mask * df, axis=1)
                pv_ko = np.sum(sum_coupons_discounted + (actual_rebates * df[t_ko]))

            pv_total += pv_ko

        if np.any(nko_mask):
            c_nko = nko_matrix[nko_mask]
            if is_deferred:
                pv_nko = np.sum((np.sum(c_nko, axis=1)) * df[-1])
            else:
                pv_nko = np.sum(np.sum(c_nko * df, axis=1))

            pv_total += pv_nko
        return pv_total / n_paths



class DoubleBarrierAccumulator(StructureMC):
    def __init__(
            self, spot, settlement_type,
             up_calc_type, up_barrier, up_payoff,
             down_calc_type, down_barrier, down_payoff,
             unhit_payoff,
             ob_days
    ):
        """
        敲入敲出均终止型累计

        Parameters
        ----------
        spot
        settlement_type
        up_calc_type
        up_barrier
        up_payoff
        down_calc_type
        down_barrier
        down_payoff
        unhit_payoff
        ob_days
        """
        self.up_barrier = arr_scalar_converter(up_barrier, ob_days)
        self.log_up_barrier = np.log(self.up_barrier / spot)
        self.down_barrier = arr_scalar_converter(down_barrier, ob_days)
        self.log_down_barrier = np.log(self.down_barrier / spot)
        self._spot = spot
        up_payoff_helper = PayoffWrapper(ob_days, up_payoff, up_calc_type)
        self.up_payoffs = up_payoff_helper.get_payoff_vec()
        down_payoff_helper = PayoffWrapper(ob_days, down_payoff, down_calc_type)
        self.down_payoffs = down_payoff_helper.get_payoff_vec()
        self.unhit_payoffs = [unhit_payoff for _ in ob_days]
        self.ob_days = ob_days
        self._sim_t_array = np.append([0], ob_days)
        self._settlement_type = settlement_type

    def update_sim_array(self):
        ob_days = np.array(self.ob_days)
        ob_days = ob_days +1
        self._sim_t_array =  np.append([0], ob_days)

    def _set_spot(self, val):
        if val <= 0:
            raise ValueError("Spot price should be positive.")
        self._spot = val
        self.log_up_barrier = np.log(self.up_barrier/val)
        self.log_down_barrier = np.log(self.down_barrier/val)

    @DocstringWriter(_pv_log_paths_docs)
    def pv_log_paths(self, log_paths, df):
        up_t, down_t, up_idx, down_idx, survive_idx = double_ko_and_surviving_paths(
            log_paths,
            self.log_up_barrier,
            self.log_down_barrier,
            return_idx=True
        )

        n_paths, n_steps = log_paths.shape
        is_deferred = (self._settlement_type == SettlementType.AT_MATURITY)
        pv_total = 0
        n_paths, n_days = log_paths.shape
        calc_df = np.ones(n_steps) if is_deferred else df
        if np.any(up_idx):
            spots_up = np.exp(log_paths[up_idx]) * self._spot
            t_hit = up_t[up_idx].astype(int)

            unhit_m = np.array([f(spots_up[:, i]) for i, f in enumerate(self.unhit_payoffs)]).T
            rebate_m = np.array([f(spots_up[:, i]) for i, f in enumerate(self.up_payoffs)]).T

            alive_mask = self.ob_days <= t_hit[:, None]
            actual_rebates = rebate_m[np.arange(len(t_hit)), t_hit]

            running_pv = np.sum(unhit_m * alive_mask * calc_df, axis=1)

            if is_deferred:
                pv_up = np.sum((running_pv + actual_rebates) * df[t_hit])
            else:
                pv_up = np.sum(running_pv + (actual_rebates * df[t_hit]))

            pv_total += pv_up

        if np.any(down_idx):
            spots_down = np.exp(log_paths[down_idx]) * self._spot
            t_hit = down_t[down_idx].astype(int)

            unhit_m = np.array([f(spots_down[:, i]) for i, f in enumerate(self.unhit_payoffs)]).T
            rebate_m = np.array([f(spots_down[:, i]) for i, f in enumerate(self.down_payoffs)]).T
            alive_mask = self.ob_days <= t_hit[:, None]
            actual_rebates = rebate_m[np.arange(len(t_hit)), t_hit]

            running_pv = np.sum(unhit_m * alive_mask * calc_df, axis=1)

            if is_deferred:
                pv_down = np.sum((running_pv + actual_rebates) * df[t_hit])
            else:
                pv_down = np.sum(running_pv + (actual_rebates * df[t_hit]))

            pv_total += pv_down

        if np.any(survive_idx):
            spots_survive = np.exp(log_paths[survive_idx]) * self._spot
            unhit_m = np.array([f(spots_survive[:, i]) for i, f in enumerate(self.unhit_payoffs)]).T
            running_pv = np.sum(unhit_m * calc_df, axis=1)
            pv_survive = np.sum(running_pv * df[-1])
            pv_total += pv_survive

        return pv_total / n_paths

if __name__ == "__main__":
    import datetime
    import pyoptmc as opt

    calendar = opt.Calendar()
    start_date = datetime.date(2026, 1, 7)

    ob_dates = calendar.periodic(start_date, '1D', 121, "next")
    mc = opt.MonteCarlo(100, 1000000)
    bs = opt.BlackScholes(0.03, 0.03, 0.3, 244)

    initial_price = 4062.0
    ko_barrier = 4300.0
    ki_barrier = 3800.0
    ko_value = 100.0
    ki_value = 3850.0

    phx_acc = opt.PhoenixAccumulatorProd(
        start_date, ob_dates, initial_price, opt.AccumulatorType.Accumulator,
        ko_barrier, opt.QuantityCalcType.REMAINING_INCLUDE_TERMINATE, opt.PayoffType.FIX,
        100.0, 1.0, ki_barrier, opt.QuantityCalcType.FULL,
        opt.PayoffType.FLOAT, ki_value, 2.0,
        opt.PayoffType.FLOAT, 4000.0, 1.0, calendar
    )
    res = phx_acc.value(start_date, initial_price, True, mc, bs, request_greeks=False) * 10.0
    print(res)