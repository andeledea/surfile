"""
--------
SurfILE: package for surface and profile processing

Data structures
---------------
- profile
- surface
    
>>> sur = surface.Surface()
>>> sur.openFile(fname, interp=False, bplt=True)

Utilities
--------
- stitcher: stitching of topography for augmented FOV
- cutter: cropping topographies and cutting profiles
- extractor: profile extraction from surface

The utilities can be used in two ways:
    
- interactive: the program asks the user the parameters every time
>>> extractor.ComplexExtractor.profile(sur, width=2, bplt=True)

- programmed: the user sets the parameters at the beginning
>>> ex = extractor.ComplexExtractor()
>>> ex.template()
>>> ex.apply(sur, bplt=True)

Processings
-----------
    - texture: PSD / roughness / slopeDistribution
    - filter: filtering of profile and surfaces (alpha state)
    - analysis: gratings, readius calculation\n

Structure
---------

```mermaid
graph RL;
    A[surfile.analysis]--> C & D & E[surfile.profile] & G;
    B[surfile.cutter]--> D & E & G;
    C[surfile.extractor]--> B & D & E & G;
    D[surfile.geometry]--> B & E & G;
    M[surfile.stitcher]--> G;
    G[surfile.surface]--> E;
    H[surfile.texture]--> D & E & G;

    subgraph DATA STRUCTURES
        E
        G
    end

    subgraph FORM OPERATORS
        D
    end

    subgraph UTILITIES
        C
        M
        B
    end

    subgraph DATA PROCESSING
        A
        H
    end
```

Dependencies
------------
This package depends on the following packages
matplotlib
- scipy
- numpy
- circle_fit
- open3d
- csaps (optional)
- igor (optional)
- alive_pogress

To install all packages run:
>>> pip install matplotlib circle_fit alive_progress open3d csaps igor

To view documentation interactively run:
>>> pip install pdoc
>>> python3 -m pdoc surfile --math --mermaid
and open the localhost server in the browser.

References
----------
For more information about the package, please read our paper on ... 
If you use this package in one of your projects, please cite: DOI: ...

\nGitHub: @andeledea, @...
"""

__docformat__ = 'numpy'
__name__ = 'surfile'
