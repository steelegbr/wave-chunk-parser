# Wave Chunk Parser

Parses wave files chunks in order to extract metadata and audio. Also provides an option to write a bunch of given chunks back into a coherent file.

# Installation

    pip install wave-chunk-parser

# Reading a file / IO stream

    from wave_chunk_parser.chunks import RiffChunk

    with open("file.wav", "rb") as file:
        riff_chunk = RiffChunk.from_file(file)

From there you can access the sub chunks from riff_chunk.sub_chunks. The data chunk uses a numpy array to hold the vectors of audio samples.

Format (fmt) and data chunks are critical. Cart chunk is optional but provides those markers we need for handling broadcast audio.

# Writing a file / IO stream

You will need to build the chunks individually and supply them in a list of:

    chunks = [FormatChunk, DataChunk, CartChunk]

The format chunk must come before the data chunk (we need to know how to en/decode it). Cart chunk can appear anywhere in the list or even not exist at all.

To get a blob for writing to disk (or wherever), simply:

    riff_chunk = RiffChunk(chunks)
    blob = riff_chunk.to_bytes()

The to_bytes function exists on every chunk type. So, if you particularly desire, you can turn a standalone format chunk into a blob.

## Numpy on macOS

We need openblas to make Numpy work on macOS:

    pip uninstall numpy
    brew install openblas
    OPENBLAS="$(brew --prefix openblas)" pip install numpy --no-cache-dir
