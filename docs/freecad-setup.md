# FreeCAD setup

KAIROS drives FreeCAD ≥ 0.21 (tested against **1.1.3**) through its Python
modules. FreeCAD is not pip-installable; the modules ship inside the
application bundle with their own Python interpreter.

## macOS

```bash
brew install --cask freecad
```

This installs `/Applications/FreeCAD.app`. Two interpreters now matter:

| Interpreter | Use |
| --- | --- |
| your `python3` | pure-python code and tests (`pytest -m "not cad"`) |
| `/Applications/FreeCAD.app/Contents/Resources/bin/python` | everything touching CAD |

The bundled interpreter must be used for CAD work because the binary modules
(`FreeCAD.so`, `Part.so`, ...) are built against its exact Python ABI. The
`Makefile` resolves it automatically:

```bash
make test-cad        # CAD integration tests
make generate-data   # procedural dataset
```

## Linux

Install the `freecad` (or `freecad-python3`) package with your package
manager. `kairos.cad.backend` searches these library directories:

- `$KAIROS_FREECAD_LIB` (explicit override)
- `/Applications/FreeCAD.app/Contents/Resources/lib` (macOS)
- `/usr/lib/freecad-python3/lib`, `/usr/lib/freecad/lib`,
  `/usr/local/lib/freecad/lib`

If your distribution installs elsewhere:

```bash
export KAIROS_FREECAD_LIB=/path/to/dir/containing/FreeCAD.so
```

## How discovery works

`kairos.cad.backend.load_freecad()`:

1. tries `import FreeCAD` directly (already running under the bundled
   interpreter, or paths preconfigured);
2. otherwise appends each candidate directory to `sys.path` and retries;
3. raises `BackendUnavailableError` with the last import error if all fail.

`freecad_available()` wraps this in a bool and is what the test suite uses to
auto-skip `cad`-marked tests.

## Headless notes

- Everything KAIROS does is GUI-free: no `FreeCADGui`, no `ViewObject`
  access. Rendering uses a software rasterizer (`kairos/cad/rendering.py`).
- The bundled interpreter needs `pytest` once for CAD tests:

  ```bash
  /Applications/FreeCAD.app/Contents/Resources/bin/python -m pip install pytest
  ```

- `numpy` is already bundled with FreeCAD.
