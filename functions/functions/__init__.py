import types
import inspect

from . import analysis, utils, plot, effective_rates, \
              load_data, sys_setup, kymo, xyz_reader, \
              sPGdeposition, midcell_transport, directionality, \
              box, constriction,  cylinder, \
              cmplx_scatter, simconfig # import all your modules here
 #spt, huet_analysis

# Optional: selectively import functions you want top-level
from .utils import *
from .plot import *
from .analysis import *
from .sys_setup import *
from .kymo import *
from .xyz_reader import *
from .sPGdeposition import *
from .midcell_transport import *
from .directionality import *
from .box import *
from .constriction import *
# from .spt import *
from .cmplx_scatter import *

# Automatically collect function names
definitions = {}

for mod in [utils, plot, analysis, sys_setup, kymo, xyz_reader, effective_rates,
           load_data, sPGdeposition, directionality, constriction,
           midcell_transport, cmplx_scatter]:  #spt, 
    fname = mod.__name__.split('.')[-1]
    defs = {
        name: obj
        for name, obj in inspect.getmembers(mod)
        if isinstance(obj, types.FunctionType) and not name.startswith('_')
    }
    definitions[fname] = list(defs.keys())

# Make it accessible via the package
__all__ = [*utils.__all__, *plot.__all__, *analysis.__all__] if all(
    hasattr(mod, '__all__') for mod in [utils, plot, analysis]
) else []
