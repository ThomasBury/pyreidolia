from . import plot
from . import mask

__version__ = "0.0.1"
__all__ = ["plot", "mask"]

# bound to upper level
from .plot import *
from .mask import *

# Authors declaration
__author__ = "Thomas Bury, Afonso Alves, Daniel Staudegger"
