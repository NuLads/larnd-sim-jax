"""
Optimize module for larnd-sim-jax parameter fitting.

This module provides:
- Simulation strategies (LUTSimulation, ParametrizedSimulation, etc.)
- Loss strategies (GenericLossStrategy, ProbabilisticLossStrategy)
- Data loading utilities (TracksDataset, DataLoader)
- Parameter ranges for optimization

Import directly from submodules:
    from optimize.strategies import LUTSimulation, ...
    from optimize.dataio import TracksDataset, DataLoader
    from optimize.ranges import ranges
"""

# Note: We don't import anything here to avoid circular import issues.
# Users should import directly from submodules.

__all__ = [
    'strategies',
    'dataio', 
    'ranges',
]
