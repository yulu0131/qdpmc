
dependencies = ("numpy", "joblib")
missing_dependencies = []
for dependency in dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append("{}: {}".format(dependency, e))
        del e
if missing_dependencies:
    raise ImportError(
        "Unable to import required packages: \n" + "\n".join(missing_dependencies)
    )


__version__ = "0.14.2"

del dependencies, dependency, missing_dependencies

from pyoptmc.engine import *
from pyoptmc.model import *
from pyoptmc.tools import *
from pyoptmc.structures import *
from pyoptmc.dateutil import Calendar
from pyoptmc.products.products import PhoenixProd
