import numpy as np
from pyoptmc.structures.base import StructureMC
from pyoptmc.structures._docs import _pv_log_paths_docs
from pyoptmc._decorators import DocstringWriter

class FixedStrike(StructureMC):
    def __init__(self, spot, ob_days, payoff, strike, avgfunc):
        self._spot = spot
        self.ob_days = ob_days
        self._sim_t_array = np.append([0], ob_days)
        self.payoff = payoff
        self.avgfunc = avgfunc

    @DocstringWriter(_pv_log_paths_docs)
    def pv_log_paths(self, log_paths, df):
        pass



class FloatingStrike(StructureMC):
    def __init__(self, spot, ob_days, payoff, avgfunc):

        self._spot = spot
        self.ob_days = ob_days
        self._sim_t_array = np.append([0], ob_days)
        self.payoff = payoff
        self.avgfunc = avgfunc

    @DocstringWriter(_pv_log_paths_docs)
    def pv_log_paths(self, log_paths, df):
        pass