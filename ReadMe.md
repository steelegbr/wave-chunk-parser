# Wave Chunk Parser

Parses wave files chunks in order to extract metadata and audio. Also provides an option to write a bunch of given chunks back into a coherent file.

## Numpy on macOS

We need openblas to make Numpy work on macOS:

    pip uninstall numpy
    brew install openblas
    OPENBLAS="$(brew --prefix openblas)" pip install numpy --no-cache-dir