"""Backward-compatible shim: re-exports everything from config.utils."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "config"))
from config.utils import *
