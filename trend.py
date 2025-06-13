"""Trend indicator re-export module.

Historically the project referenced ``backtesting_framework.indicators.trend``
while most implementations live in ``basic_trend.py``.  To maintain backwards
compatibility without duplicating code, this stub simply re-exports everything
from ``basic_trend``.
"""

from .basic_trend import *  # noqa: F403, F401

# Derive the public names dynamically – anything imported above that does not
# start with an underscore is considered part of the public interface.

__all__ = [name for name in globals() if not name.startswith('_')] 