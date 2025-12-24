from .nsa import nsa_func

# Backwards compatibility alias while migrating to the new implementation
parallel_nsa = nsa_func

__all__ = [
    'nsa_func',
    'parallel_nsa',
]
