"""
Surfile
________
Package for surface and profile processing

Containers
________
    - profile
    - surface

Example:
    sur = surface.Surface()
    sur.openFile(fname, interp=False, bplt=True)

Utilities
________
    - remover: levelling and form removal operations
    - filter: filtering of profile and surfaces
    - cutter: cutting edges and sections of topographies
    - extractor: profile extraction from surface

Example:
    The utilities can be used in two ways:
        - interactive: the program asks the user the parameters every time
            extractor.ComplexExtractor.profile(sur, width=2, bplt=True)
        - programmed: the user sets the parameters at the beginning
            ex = extractor.ComplexExtractor()
            ex.template()
            ex.apply(sur, bplt=True)

Processings
________
    - roughness: PSD / roughness
    - morph: morphological features
    -slope: slope distribution analysis

GitHub: @andeledea, @...
"""
