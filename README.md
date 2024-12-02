# SurfILE: surface and profile analysis
Python library for analyzing topographic features of surfaces from
- areal surface data -- __Surf__ -ace
- profiles -- prof- __ILE__

<!-- PROJECT LOGO -->
<p align="center">
<img src="https://github.com/andeledea/surfile/blob/PLTsub/resources/Surf-Logo.png" alt="drawing" width="300"/>
</p>

<!-- ABOUT THE PROJECT -->
## About The Project
The project is intended to be an easy to use tool box providing multiple algorithms to analyze areal surface topographies: areal height maps and profiles.

A scientific paper was published and can be found here:
https://doi.org/10.3390/metrology4040041
<!-- GOAL -->
### Goal
The project aims to provide easy-to-use Python programs for analysing areal surface topographies and profiles:
- as a tool box of algorithms to estimate geometry parameters of spheres, cylinders, gratings (height and pitch), etc
- in the field of surface metrology for users
    - who want the flexibility to adapt their data processing to their own needs by modifying and adding code as required
    - who need to process large amounts of measurement data for automatic analysis (such as 20 to 100 topography maps of repetitive measurements)
- for beginners as well as for those experts who need a quick and easy way to solve their tasks

### Usage
To use the package clone this repository
```bash
git clone https://github.com/andeledea/surfile.git
```
or
```bash
gh repo clone andeledea/surfile
```
Check the dependencies
```bash
pip install matplotlib circle_fit alive_progress open3d csaps igor numpy scipy
```
Try this simple example
```python
from surfile import surface
import matplotlib.pyplot as plt

if __name__ == "__main__":    
    sur = surface.Surface()
    sur.openFile("/path/to/data.(asc/txt/...)", bplt=True)

    sur.pltC()
    plt.show()
```
and you are using the package!!

### Documentation
The package is documented in docstrings in NumPy format and the documentation was exported in html format in the folder docs using pdoc. It is possible to view the documentation interactively in the browser by running pdoc
```bash
pip install pdoc
python -m pdoc surfile --math --mermaid

Note
----
To ignore external Imports when generating the docs:

fish >>> set -Ux PDOC_ALLOW_EXEC 1
bash >>> export PDOC_ALLOW_EXEC=1
```

<!-- CONTRIBUTING -->
## Contributing
When contributing to this repository, please first discuss the change you wish to make via "issue" before making a change.
Please consider contributing to the documentation of the package as well.

<!-- LICENCE -->
## License
GNU GPLv3
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

See <https://www.gnu.org/licenses/>.
