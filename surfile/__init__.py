"""
Surfile (better name???)
________
Package for surface and profile processing

Containers
________
    - profile
    - surface

Utilities
________
    - remover: levelling and form removal operations
    - filter: filtering of profile and surfaces
    - cutter: cutting edges and sections of topographies
    - extractor: profile extraction from surface

Processings
________
    - roughness: PSD / roughness
    - morph: morphological features

GitHub: @andeledea
"""

# from . import cutter
# from . import filter
# from . import form
# from . import profile
# from . import surface

__all__ = ['cutter', 'remover.py', 'filter', 'extractor', 'surface', 'profile']
