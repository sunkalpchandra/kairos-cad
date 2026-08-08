"""Static dashboard over KAIROS run artifacts.

The dashboard is built, not served: `build_dashboard.py` reads the artifacts on
disk, bundles them into JSON, and inlines that into a single HTML file. Opening
the file needs no process, no network, and no build step.

`bundle` (pure python, either interpreter) gathers metrics; `mesh` (FreeCAD only)
tessellates solids for the 3D viewer.
"""
