# Monte Carlo Method for Option Pricing

> UPDATE: 19th November 2025

```bash
    pip install pyoptmc --upgrade
```

This is a package for pricing
path-dependent options using Monte Carlo Simulation
under Black-Scholes market dynamics.

This package relies heavily on **NumPy** for the implementation
of *vectorization*, which significantly boosts algorithm
speed. It also uses **joblib** to implement parallel computation.

## Example

```python
import datetime
import pyoptmc as opt
calendar = opt.Calendar()

start_date = datetime.date(2025, 11, 5)
ko_ob_dates = calendar.periodic(start_date, '1M', 13, "next")[1:]

mc = opt.MonteCarlo(100, 1000000)
bs = opt.BlackScholes(0.03, 0, 0.265, 244)
end_date = ko_ob_dates[-1]

dcn = opt.PhoenixProd(
    start_date= start_date,
    end_date = end_date,
    initial_price = 100.0,
    settlement_barrier = 80.0,
    settlement_dates = ko_ob_dates,
    settlement_coupon_rate = 0.15,
    ko_barrier = 100.0,
    ko_ob_dates = ko_ob_dates,
    ki_barrier = 80.0,
    ki_ob_dates = "daily",
    calendar = calendar)

print(dcn.value(start_date, 100.0, True, mc, bs, request_greeks=True))

# {'PV': np.float64(-0.16274331381605528), 'Delta': np.float64(0.2390255106366182), 'Gamma': np.float64(-0.031593825237697534), 'Rho': np.float64(0.020117531945007528), 'Vega': np.float64(-0.23208919587396326), 'Theta': np.float64(0.04311655912199197)}

fcn = opt.PhoenixProd(
    start_date=start_date,
    end_date=end_date,
    initial_price=100.0,
    settlement_barrier=0.0,
    settlement_dates=ko_ob_dates,
    settlement_coupon_rate=0.15,
    ko_barrier=100.0,
    ko_ob_dates=ko_ob_dates,
    ki_barrier=80.0,
    ki_ob_dates=[end_date],
    calendar=calendar)

print(fcn.value(start_date, 100.0, False, mc, bs, request_greeks=True))
# {'PV': np.float64(1.6827002150401873), 'Delta': np.float64(-0.007450531928672407), 'Gamma': np.float64(-0.011540440160383967), 'Rho': np.float64(0.011854131363408455), 'Vega': np.float64(-0.1869648625597277), 'Theta': np.float64(0.017297330302293138)}
```

