import datetime
# from abc import ABC, abstractmethod
import numpy as np
import pyoptmc.structures as structures
import pyoptmc.tools.payoffs as pay
from functools import partial
from pyoptmc.tools.payoffs import plain_vanilla, cash, cash_or_nothing
from pyoptmc.tools.helper import arr_scalar_converter
from pyoptmc.tools.enum import AccumulatorType, QuantityCalcType, PayoffType, SettlementType
from pyoptmc.dateutil import Calendar
from scipy.optimize import fsolve
from numpy import array, any, argmax
from typing import List

__all__ = ['PhoenixAccumulatorProd',
    'SnowballProd', 'PhoenixProd', 'PhoenixAccumulatorProd']


def _interval_coupon(
        coupon_rate, principal,
        last_payment_date: datetime.date,
        next_payment_date: datetime.date,
        day_counter=365
):
    """Returns the amount of interval coupon between two payment dates.
    Principle, coupon rate, and day counter are mandatory."""
    td = (next_payment_date - last_payment_date).days
    return coupon_rate * principal * td / day_counter


def _update_day_arr(arr, offset, *more):
    """Here, arr is an ascending array of scalars and offset is a scalar. The function
    returns the positive part of arr - offset. If more is passed in, it also returns
    more[arr > offset]"""

    if not offset:
        return arr, *more
    nn = []
    for v in arr:
        d = v - offset
        if d >= 0:
            nn.append(d)

    if more:
        return nn, *(m[len(arr) - len(nn):] for m in more)
    return nn

def _compute_coupons(dates, start_date, spot, coupon_rate):
    first = (dates[0] - start_date).days / 365.0
    diffs = [(dates[i] - dates[i - 1]).days / 365.0 for i in range(1, len(dates))]
    ttms = np.array([first] + diffs, dtype=float)
    coupons = ttms * coupon_rate * spot

    return coupons


def _check_ob_dates(ob_dates, calendar):
    """Check if all dates in ob_dates are trading days. If True, return ob_dates
    as-is. Otherwise raise ValueError."""
    for date in ob_dates:
        date = _check_is_trading(date, calendar)
        if not calendar.is_trading(date):
            raise ValueError("%s does not trade" % str(date))
    return ob_dates


def _check_is_trading(date, calendar):
    """Check if *start* is trading."""
    if not isinstance(date, datetime.date):
        raise TypeError("{} is not a datetime.date object".format(date))
    if not calendar.is_trading(date):
        raise ValueError("given non-trading day: {}".format(date))
    return date


def _check_payoff(payoff):
    """Check if payoff is a *Payoff* object."""
    if not isinstance(payoff, pay.Payoff):
        raise TypeError("payoff must be a Payoff object")
    return payoff


def _check_calendar(calendar):
    """Check if calendar is a Calendar object."""
    if not isinstance(calendar, Calendar):
        raise TypeError("calendar must be Calendar object")
    return calendar


# accumulator requires payoff wrapper...
def _update_hit_payoff(payoff_type, value, multiple, option_type):
    def payoff(s):
        payoff_val = 0.0
        if payoff_type == PayoffType.FIX:
            payoff_val = cash(s, value * multiple)
        elif payoff_type == PayoffType.FLOAT:
            payoff_val =  multiple * plain_vanilla(s, value, option_type)
        elif payoff_type == PayoffType.NONE:
            payoff_val = cash(s, 0.0)
        else:
            raise ValueError("Unknown payoff type: {}".format(payoff_type))
        return payoff_val
    return payoff



class PhoenixAccumulatorProd:
    def __init__(self,
                 start_date,
                 ob_dates,
                 initial_price,
                 accumulator_type: AccumulatorType,
                 ko_barrier: float,
                 ko_calc_type: QuantityCalcType,
                 ko_payoff_type: PayoffType,
                 ko_value: float,
                 ko_multiple: float,
                 ki_barrier: float,
                 ki_calc_type: QuantityCalcType,
                 ki_payoff_type: PayoffType,
                 ki_value: float,
                 ki_multiple: float,
                 unhit_payoff_type: PayoffType,
                 unhit_value: float,
                 unhit_multiple: float,
                 cal: Calendar = None
                 ):
        _inputs = locals()
        _inputs.pop("self")
        self._inputs = _inputs
        self._acc_type = accumulator_type
        if cal is None:
            cal = Calendar()
        self.start_date = _check_is_trading(start_date, cal)
        ob_dates = _check_ob_dates(ob_dates, cal)
        self.ob_days = cal.to_scalar(ob_dates, self.start_date)
        self.ko_barrier = ko_barrier
        self.ki_barrier = ki_barrier
        self.calendar = cal
        self.ko_value = ko_value
        self.ki_value = ki_value
        self.unhit_payoff_type = unhit_payoff_type
        self.unhit_value = unhit_value
        self.unhit_multiple = unhit_multiple
        self.ko_calc_type = ko_calc_type
        self.ki_calc_type = ki_calc_type
        ko_option_type = "call"
        ki_option_type = "put"
        if accumulator_type == AccumulatorType.Deccumulator:
            ko_option_type = "put"
            ki_option_type = "call"
        self.unhit_option_type = ko_option_type
        self.unhit_payoff = self._update_unhit_payoff()
        self.ko_payoff = _update_hit_payoff(ko_payoff_type, ko_value, ko_multiple, ko_option_type)
        self.ki_payoff = _update_hit_payoff(ki_payoff_type, ki_value, -ki_multiple, ki_option_type)

    def _update_unhit_payoff(self):
        if self.unhit_payoff_type == PayoffType.FLOAT:
            def payoff(s):
                if self._acc_type == AccumulatorType.Accumulator:
                    cash_amount = self.unhit_multiple * (self.ko_value - self.unhit_value)
                    c2 = self.unhit_multiple * (self.ki_value - self.unhit_value)
                else:
                    cash_amount = self.unhit_multiple * (self.unhit_value - self.ko_value)
                    c2 = self.unhit_multiple * (self.unhit_value - self.ki_value)
                payoff_val = (
                        self.unhit_multiple * plain_vanilla(s, self.unhit_value, self.unhit_option_type)
                        - self.unhit_multiple * plain_vanilla(s, self.ko_barrier, self.unhit_option_type)
                        - cash_or_nothing(s, self.ko_barrier, cash_amount, self.unhit_option_type)
                )

                if self._acc_type == AccumulatorType.Accumulator:
                    if self.unhit_value < self.ki_barrier:
                        payoff_val += cash_or_nothing(s, self.ki_barrier, c2, self.unhit_option_type)
                else:
                    if self.unhit_value > self.ki_barrier:
                        payoff_val += cash_or_nothing(s, self.ki_barrier, c2, self.unhit_option_type)

                return payoff_val
            return payoff

        elif self.unhit_payoff_type == PayoffType.FIX:
            def payoff(s):
                coupon = self.unhit_value * self.unhit_multiple
                payoff_val = cash_or_nothing(
                    s, self.ki_barrier, coupon, self.unhit_option_type) - \
                    cash_or_nothing(s, self.ko_barrier, coupon, self.unhit_option_type)
                return payoff_val
            return payoff
        elif self.unhit_payoff_type == PayoffType.NONE:
            def payoff(s):
                return cash(s, 0.0)
            return payoff
        else:
            raise ValueError("Unknown Unhit PayoffType")

    def to_structure(self, valuation_date, spot):
        valuation_date = _check_is_trading(valuation_date, self.calendar)
        td = self.calendar.num_trading_days_between(
            start=self.start_date, end=valuation_date, count_end=True
        )
        ob_days = _update_day_arr(self.ob_days, td)
        ob_days = np.array(ob_days[0])
        obj = None
        if self._acc_type == AccumulatorType.Accumulator:
            obj = structures.DoubleBarrierAccumulator(spot, SettlementType.AT_OBSERVATION,
                                                    self.ko_calc_type, self.ko_barrier, self.ko_payoff,
                                                    self.ki_calc_type, self.ki_barrier, self.ki_payoff,
                                                    self.unhit_payoff, ob_days)

        else:
            obj = structures.DoubleBarrierAccumulator(spot, SettlementType.AT_OBSERVATION,
                                                   self.ki_calc_type, self.ki_barrier, self.ki_payoff,
                                                   self.ko_calc_type, self.ko_barrier, self.ko_payoff,
                                                   self.unhit_payoff, ob_days)
        return obj

    def value(self, valuation_date, spot, cal_beginning=False, *args, **kwargs):
        structure =  self.to_structure(valuation_date, spot)
        if cal_beginning:
            structure.update_sim_array()
        return structure.calc_value(
            *args, **kwargs)

    def single_call(self, valuation_date, spot, cal_beginning=False, *args, **kwargs):
        structure =  self.to_structure(valuation_date, spot)
        if cal_beginning:
            structure.update_sim_array()
        return structure.calc_single_batch( *args, **kwargs)
# class KnockInAccumulatorProd:
#     def __init__(self):
#         pass
#
#     def to_structure(self, valuation_date, spot):
#         pass
#
#     def value(self, valuation_date, spot, *args, **kwargs):
#         return self.to_structure(valuation_date, spot).calc_value(
#             *args, **kwargs)
#
# class KnockOutAccumulatorProd:
#     def __init__(self):
#         pass
#
#     def to_structure(self, valuation_date, spot):
#         pass
#
#     def value(self, valuation_date, spot, *args, **kwargs):
#         return self.to_structure(valuation_date, spot).calc_value(
#             *args, **kwargs)

class PhoenixProd:
    def __init__(
            self,
            start_date,
            end_date,
            initial_price,
            settlement_barrier,
            settlement_dates,
            settlement_coupon_rate,
            ko_barrier,
            ko_ob_dates,
            ki_barrier,
            ki_ob_dates,
            calendar: Calendar = None
    ):
        _inputs = locals()
        _inputs.pop("self")
        self._inputs = _inputs

        if calendar is None:
            calendar = Calendar()
        # check values
        self.ki_barrier = ki_barrier
        self.ko_barrier = ko_barrier
        self.settlement_barrier = settlement_barrier
        self.settlement_coupon_rate = settlement_coupon_rate
        calendar = _check_calendar(calendar)
        start_date = _check_is_trading(start_date, calendar)
        ko_ob_dates = _check_ob_dates(ko_ob_dates, calendar)
        settlement_dates = _check_ob_dates(settlement_dates, calendar)
        try:
            # check if ki_ob_dates are trading
            ki_ob_dates = _check_ob_dates(ki_ob_dates, calendar)
        except TypeError:
            # return default daily dates
            if ki_ob_dates != "daily":
                raise ValueError(
                    "ki_ob_dates must either be an array of "
                    "trading days or 'daily, got {}".format(ki_ob_dates)
                )
            else:
                end = ko_ob_dates[-1]
                ki_ob_dates = calendar.trading_days_between(
                    start=start_date, end=end, endpoints=True
                )[1:]
        self.ob_days_out = calendar.to_scalar(ko_ob_dates, start_date)
        self.ob_days_in = calendar.to_scalar(ki_ob_dates, start_date)
        self.ob_days_settled = calendar.to_scalar(settlement_dates, start_date)
        self.calendar = calendar
        self.settlement_coupons = _compute_coupons(settlement_dates, start_date, initial_price,
                                                   settlement_coupon_rate)
        self.start_date = start_date

    def to_structure(self, valuation_date, spot, ki_flag):
        if ki_flag:
            self.ki_barrier = 0.0
        valuation_date = _check_is_trading(valuation_date, self.calendar)
        td = self.calendar.num_trading_days_between(
            start=self.start_date, end=valuation_date, count_end=True
        )
        ob_days_out = _update_day_arr(self.ob_days_out, td)
        ob_days_out = np.array(ob_days_out[0])
        ob_days_settled = _update_day_arr(self.ob_days_settled, td)
        ob_days_settled = np.array(ob_days_settled[0])
        ob_days_in = _update_day_arr(self.ob_days_in, td)
        ob_days_in = np.array(ob_days_in[0])
        obj = structures.StandardPhoenix(spot, self.ko_barrier, self.ki_barrier, self.settlement_barrier,
                                         ob_days_in, ob_days_out, ob_days_settled,
                                         self.settlement_coupons,
                                         0.0 * np.ones(len(ob_days_out)),
                                         0.0)

        return obj


    def value(self, valuation_date, spot, ki_flag, *args, **kwargs):
        return self.to_structure(valuation_date, spot, ki_flag).calc_value(
            *args, **kwargs)

class SnowballProd:
    """A snowball structure is an autocallable structured product with snowballing
    coupon payments.

    Parameters
    ----------
    start_date : datetime.date
        A datetime.date object indicating the starting day of the
        structured product. It must be a trading as determined by *calendar*
    initial_price : scalar
        The price of the underlying asset on *start_date*.
    ko_barriers : scalar or array_like
        The knock-out level of the structure. It can either be
        a scalar or be an array of numbers. If a scalar is passed in, it will be
        treated as the time-invariant barrier level. If an array is passed in,
        it must match the length of *ko_ob_dates*.
    ko_ob_dates : array_like
        The observation dates for knock-out. It must be an array
        of datetime.date objects. All of these dates must be trading dates as
        determined by *calendar*.
    ki_barriers : scalar or array_like
        Similar to *ko_barriers*. It controls the level of knock-in barrier.
    ki_ob_dates : array_like or "daily"
        similar to *ko_ob_dates*. *"daily"* indicates daily observation for the
        knock-in event.
    ki_payoff : Payoff
        Controls payoff which applies when, during the life of the contract,
        a knock-in event occurs while a knock-out does not.
    ko_coupon_rate : scalar
        the coupon rate that applies in the event of a knock-out.
    maturity_coupon_rate : scalar
        this rate applies when there is neither knock-out nor knock-in during
        the entire life of the contract.
    calendar : Calendar
        *Calendar* object. If *None* a default calendar will be used.

    Note
    ----
    The day count convention for coupon payment is *ACT/365*.
    The maturity is ``ko_ob_dates[-1]``.
    Notional principal is equal to ``initial_price``.

    Examples
    --------
    .. ipython:: python

        import datetime
        # instantiate a Calendar object so we can use it to generate periodic
        # datetime.date array
        calendar = qm.Calendar()
        # this should be a trading day
        start = datetime.date(2019, 1, 31)
        assert calendar.is_trading(start)
        # monthly trading dates excluding the start
        monthly_dates = calendar.periodic(start, "1m", 13)[1:]
        # a short put
        short_put = - qm.Payoff(qm.plain_vanilla, option_type="put",
                                strike=100)
        # instantiate the structured product
        option = qm.SnowballProd(
            start_date=start, initial_price=100, ko_barriers=105,
            ko_ob_dates=monthly_dates, ki_barriers=80, ki_ob_dates="daily",
            ki_payoff=short_put, ko_coupon_rate=0.15,
            maturity_coupon_rate=0.15
        )
        mc = qm.MonteCarlo(100, 1000)
        bs = qm.BlackScholes(0.03, 0, 0.25, 252)
        # value the contract given day and spot price
        option.value(datetime.date(2019, 5, 7), 102, False, mc, bs)"""

    def __init__(
            self, start_date, initial_price, ko_barriers,
            ko_ob_dates, ki_barriers, ki_ob_dates, ki_payoff,
            ko_coupon_rate, maturity_coupon_rate,
            calendar: Calendar = None
    ):
        _inputs = locals()
        _inputs.pop("self")
        self._inputs = _inputs

        if calendar is None:
            calendar = Calendar()
        # check values
        self.calendar = _check_calendar(calendar)
        self.start_date = _check_is_trading(start_date, calendar)
        self.ki_payoff = _check_payoff(ki_payoff)
        self.ko_ob_dates = _check_ob_dates(ko_ob_dates, calendar)
        try:
            # check if ki_ob_dates are trading
            self.ki_ob_dates = _check_ob_dates(ki_ob_dates, calendar)
        except TypeError:
            # try "daily" if ki_ob_dates are not datetime.date objects
            if ki_ob_dates != "daily":
                raise ValueError(
                    "ki_ob_dates must either be an array of "
                    "trading days or 'daily, got {}".format(ki_ob_dates)
                )
            else:
                end = ko_ob_dates[-1]
                ki_ob_dates = calendar.trading_days_between(
                    start=start_date, end=end, endpoints=True
                )[1:]
        self.ki_ob_dates = ki_ob_dates
        # those that need not be checked
        self.initial_price = initial_price
        self.ko_barriers = arr_scalar_converter(ko_barriers, ko_ob_dates)
        self.ki_barriers = arr_scalar_converter(ki_barriers, ki_ob_dates)
        self.ko_coupon_rate = ko_coupon_rate
        self.maturity_coupon_rate = maturity_coupon_rate
        # inferred and calculated values
        # these values and arrays can be directly passed to the structure
        # constructor
        self.maturity_date = ko_ob_dates[-1]
        self.ob_days_out = calendar.to_scalar(ko_ob_dates, start_date)
        self.ob_days_in = calendar.to_scalar(ki_ob_dates, start_date)
        _frozen = partial(
            _interval_coupon, principal=initial_price,
            last_payment_date=start_date
        )
        self._maturity_coupon_pmt = _frozen(
            coupon_rate=maturity_coupon_rate,
            next_payment_date=self.maturity_date
        )
        # convert a constant into a Payoff object, so it can be passed to the
        # structure constructor
        self.nk_payoff = pay.Payoff(
            pay.constant_payoff, self._maturity_coupon_pmt
        )
        # infer from given values the rebate array that can be passed to the
        # structure constructor
        self.ko_rebate = [_frozen(
            coupon_rate=ko_coupon_rate, next_payment_date=d
        ) for d in ko_ob_dates]

        self._frozen = _frozen

    def to_structure(self, valuation_date, spot, ki_flag):
        """Return the structure used to value the product."""
        valuation_date = _check_is_trading(valuation_date, self.calendar)
        td = self.calendar.num_trading_days_between(
            start=self.start_date, end=valuation_date, count_end=True
        )
        ob_days_out, rebate_out, barrier_out = _update_day_arr(
            self.ob_days_out, td, self.ko_rebate, self.ko_barriers
        )
        ob_days_in, barrier_in = _update_day_arr(
            self.ob_days_in, td, self.ki_barriers
        )
        if not ki_flag:
            obj = structures.UpOutDownIn(
                spot=spot, ob_days_out=ob_days_out, rebate_out=rebate_out,
                ob_days_in=ob_days_in, payoff_in=self.ki_payoff,
                upper_barrier_out=barrier_out, lower_barrier_in=barrier_in,
                payoff_nk=self.nk_payoff
            )
        else:
            obj = structures.UpOut(
                spot=spot, ob_days=ob_days_out, rebate=rebate_out,
                payoff=self.ki_payoff, barrier=barrier_out
            )
        return obj

    def value(self, valuation_date, spot, ki_flag, *args, **kwargs):
        """Value the product given a date and a spot price. *args* and *kwargs*
        are positional and keyword arguments forwarded to
        :meth:`pyoptmc.engine.monte_carlo.MonteCarlo.calc`

        Parameters
        ----------
        valuation_date : datetime.date
            the valuate date. It must be a trading day.
        spot : scalar
            the spot price.
        ki_flag: bool
            whether to mark the product as knock-in. If *True*, the
            structure is an up-and-out option.
        """
        return self.to_structure(valuation_date, spot, ki_flag).calc_value(
            *args, **kwargs)

    def find_coup_rate(self, engine, process, target_pv,
                       entropy=None, caller=None):
        """Give a target PV, find the coupon rate.

        *entropy* and *caller* are forwarded to
        :meth:`pyoptmc.engine.monte_carlo.MonteCarlo.calc`
        """
        e = entropy

        def _call(c):
            nonlocal e
            if e is None:
                e = engine.most_recent_entropy
            inputs = self._inputs
            inputs['ko_coupon_rate'] = c[0]
            inputs['maturity_coupon_rate'] = c[0]
            s = self.__class__(**inputs)
            diff = s.value(self.start_date, self.initial_price, False,
                           engine, process, entropy=e,
                           caller=caller) - target_pv
            return diff

        rate = fsolve(_call, array([self.ko_coupon_rate]))[0]
        return dict(result=rate, diff=_call([rate]))

    def backtest(self, daily_underlying_asset_prices):
        """backtest the performance of the product in the option holders' view
        with given underlying asset price series, find out the actual holding period and its final payoff

        Parameters
        ----------
        daily_underlying_asset_prices: array_like
            the simulated or historical daily underlying asset price series for performance backtest, the length of the
            price series must be greater than the option's life

        Returns
        -------
        end_date : datetime.date
            the actual end date of the trade, could possibly be earlier than the maturity date
        payout: double
            the actual payout of the product to the option holder
        """
        strct = self.to_structure(self.start_date, self.initial_price, False)
        ko_ob_days = strct.ob_days_out
        ki_ob_days = strct.ob_days_in
        ko_barriers = strct.upper_barrier_out
        ki_barriers = strct.log_barrier_in
        n_days = ko_ob_days[-1]
        rebates = strct.rebate_out

        prices = daily_underlying_asset_prices / daily_underlying_asset_prices[0] * self.initial_price
        knocked_out = any(prices[ko_ob_days] > ko_barriers)
        if knocked_out:
            ko_index = argmax(prices[ko_ob_days] > ko_barriers)
            return self.ko_ob_dates[ko_index], rebates[ko_index]
        knocked_in = any(prices[ki_ob_days] < ki_barriers)
        if knocked_in:
            return self.maturity_date, self.ki_payoff(prices[n_days])
        else:
            return self.maturity_date, rebates[-1]


class SingleBarrierOption:
    _structure = structures.SingleBarrierOption
    _out = True

    def __init__(self, start, barrier, rebate, ob_dates, payoff, calendar):
        self.calendar = _check_calendar(calendar)
        self.start = _check_is_trading(start, calendar)
        self.barrier = arr_scalar_converter(barrier, ob_dates)

        if self.__class__._out:
            self.rebate = arr_scalar_converter(rebate, ob_dates)
        else:
            if hasattr(rebate, "__iter__"):
                raise TypeError(
                    "rebate of knock-in options should be a scalar"
                )
            self.rebate = rebate

        self.ob_dates = ob_dates
        self.payoff = _check_payoff(payoff)
        # scalars
        self.ob_days = calendar.to_scalar(ob_dates, start)

    def to_structure(self, valuation_date=None, spot=None):
        td = self.calendar.num_trading_days_between(self.start, valuation_date)
        ob_days, rebate, barrier = _update_day_arr(
            valuation_date, td, self.rebate, self.barrier
        )
        return self.__class__._structure(
            spot=spot, barrier=barrier, rebate=rebate,
            ob_days=ob_days, payoff=-self.payoff
        )

    def value(self, valuation_date, spot, *args, **kwargs):
        return self.to_structure(valuation_date, spot
                                 ).calc_value(*args, **kwargs)


class UpOut(SingleBarrierOption):
    _structure = structures.UpOut


class DownOut(SingleBarrierOption):
    _structure = structures.DownOut


class UpIn(SingleBarrierOption):
    _structure = structures.UpIn
    _out = False


class DownIn(SingleBarrierOption):
    _structure = structures.DownIn
    _out = False


if __name__ == "__main__":
    from pyoptmc.products import *
    from pyoptmc.dateutil.date import Calendar
    from pyoptmc import MonteCarlo, BlackScholes
    from pyoptmc import Payoff
    calendar = Calendar()

    start_date = datetime.date(2025, 11, 5)
    ko_ob_dates = calendar.periodic(start_date, '1M', 13, "next")[1:]

    mc = MonteCarlo(100, 1000000)
    bs = BlackScholes(0.03, 0, 0.265, 244)


    phx = PhoenixProd(
        start_date= start_date,
        end_date = ko_ob_dates[-1],
        initial_price = 100.0,
        settlement_barrier =80.0,
        settlement_dates = ko_ob_dates,
        settlement_coupon_rate = 0.15,
        ko_barrier = 100.0,
        ko_ob_dates = ko_ob_dates,
        ki_barrier = 0.0,
        ki_ob_dates = "daily",
        ko_coupon_rate = 0.0,
        maturity_coupon_rate = 0.0,
        calendar = calendar)

    print(phx.value(start_date, 100.0, True, mc, bs))