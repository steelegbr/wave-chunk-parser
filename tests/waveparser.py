"""
Parse Wave file

"""

import sys
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import glob
import hexdump
import builtins
from datetime import datetime

from wave_chunk_parser.chunks import (
    RiffChunk,
    CuePoint,
    CueChunk,
    ListChunk,
    LabelChunk,
    GenericChunk,
    InfoChunk,
    NoteChunk,
)


def wave_parse(wave_file, args):
    """
    chunks parse with wave_chunk_parser lib
    """
    print(f'=====> Parse "{wave_file}"')
    fwave = builtins.open(wave_file, "rb")

    riff_chunk = RiffChunk.from_file(fwave)

    for chunk in riff_chunk.sub_chunks:
        print(f"Chunk type: {chunk.get_name}", end="")

        if chunk.get_name == b"fmt ":
            print()
            print(f"  format: {chunk.format}")
            print(f"  bits per sample: {chunk.bits_per_sample}")
            print(f"  sample rate: {chunk.sample_rate}")
            print(f"  channels: {chunk.channels}")
            print(f"  byte rate: {chunk.byte_rate}")
            print(f"  block align: {chunk.block_align}")
            if chunk.extension == None:
                print(f"  extended: no")
            else:
                print(f"  extended: {len(chunk.extension)} bytes")
                for hex in hexdump.hexdump(chunk.extension, "generator"):
                    print("   ", hex)

        elif chunk.get_name == b"data":
            print(f", len: {len(chunk.samples)}")

        elif chunk.get_name == b"cue ":
            print()
            for cp in chunk.cue_points:
                print(
                    f"  ID {cp.id} : position:{cp.position} , data_chunk_id:{cp.data_chunk_id} , "
                    f"chunk_start: {cp.chunk_start} , block_start:{cp.block_start} , sample_offset:{cp.sample_offset}"
                )
        elif chunk.get_name == b"LIST":

            print(f", Type: {chunk.get_type}")
            for sc in chunk.sub_chunks:

                if chunk.get_type == b"INFO":
                    print(f"  Sub-Chunk : {sc.get_name} : {sc.info}")

                else:
                    if sc.get_name == b"labl":
                        print(f'  ID {sc.id} : label: "{sc.label}"')

                    elif sc.get_name == b"note":
                        print(f'  ID {sc.id} : note: "{sc.note}"')

                    elif sc.get_name == b"ltxt":
                        print(
                            f"  ID {sc.id} : labeled text: sample_length:{sc.sample_length} , purpose:{sc.purpose}"
                        )

                    else:
                        print(f"  Sub-Chunk : {sc.get_name}")
                        for hex in hexdump.hexdump(sc.datas, "generator"):
                            print(" ", hex)
        else:
            if isinstance(chunk, GenericChunk):
                print(f", len: {len(chunk.datas)}")
                dump_lines = args.dump_lines
                if args.dump_lines != "":
                    for num, hex in enumerate(
                        hexdump.hexdump(chunk.datas, "generator")
                    ):
                        if int(args.dump_lines) != 0 and num >= int(args.dump_lines):
                            break
                        print(" ", hex)
                    if num * 16 < len(chunk.datas):
                        print("  ...")

    print()
    return riff_chunk


def test_rewrite_wav(wave_filein, wave_fileout):
    """
    Test rewrite file : output file must be identical
    """
    fwave = builtins.open(wave_filein, "rb")
    fwave_out = builtins.open(wave_fileout, "wb")

    riff_chunk = RiffChunk.from_file(fwave)
    blob = riff_chunk.to_bytes()

    fwave_out.write(blob)
    fwave_out.close()
    fwave.close()


def test_set_cue_points(fname_in, fname_out):
    """
    Test : add cue points and modify infos
    """

    fwave = builtins.open(fname_in, "rb")
    riff_chunk = RiffChunk.from_file(fwave)

    # get sample rate from format chunk
    fmt = riff_chunk.get_chunk("fmt ")
    rate = fmt.sample_rate

    #
    # Add or create two cue points with their labels (LIST ASSOC)
    #

    # get or create LIST ASSOC, retrieve last id
    chunk_list_assoc = riff_chunk.get_chunk(
        ListChunk.HEADER_LIST, ListChunk.HEADER_ASSOC
    )
    if not chunk_list_assoc:
        chunk_list_assoc = ListChunk(ListChunk.HEADER_ASSOC, [])
    assocs = chunk_list_assoc.sub_chunks

    # get or create Cue chunk
    chunk_cue = riff_chunk.get_chunk(CueChunk.HEADER_CUE)
    if not chunk_cue:
        chunk_cue = CueChunk([])
        id_cue = 1
    else:
        id_cue = chunk_cue.cue_points[-1].id + 1

    # add/create 2 cue points and their labels

    pos = int(1.0 * rate)
    chunk_cue.cue_points.append(CuePoint(id_cue, pos, b"data", 0, 0, pos))
    assocs.append(LabelChunk(id_cue, "First added"))

    id_cue += 1

    pos = int(2.0 * rate)
    chunk_cue.cue_points.append(CuePoint(id_cue, pos, b"data", 0, 0, pos))
    assocs.append(LabelChunk(id_cue, "Second added"))

    # add or replace cue and list chunks
    riff_chunk.replace_chunk(chunk_cue)
    riff_chunk.replace_chunk(chunk_list_assoc)

    # # force comment on second cue point
    # chunk_label.append(NoteChunk(2, 'Comment on second point'))

    #
    # Add or replace LIST INFO
    #

    chunk_list_info = riff_chunk.get_chunk(b"LIST", b"INFO")
    if not chunk_list_info:
        chunk_list_info = ListChunk(ListChunk.HEADER_INFO, [])

    # add or replace InfoChunk
    chunk_list_info.replace_chunk(InfoChunk("INAM", "My New Title"))
    chunk_list_info.replace_chunk(InfoChunk("ISFT", "WaveChunkParser Software"))
    chunk_list_info.replace_chunk(
        InfoChunk("ICRD", datetime.now().strftime("%Y-%m-%d"))
    )
    riff_chunk.replace_chunk(chunk_list_info)

    # write file
    try:
        fwave_out = builtins.open(fname_out, "wb")

        blob = riff_chunk.to_bytes()

        fwave_out.write(blob)
        fwave_out.close()
    except PermissionError as _e:
        print("FAILED to rewrite file :", _e)


def main():
    """Main program entry"""

    #
    # commands parser
    #
    parser = ArgumentParser(
        description="Parse Wave file", formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument("wavefile", nargs="*", help="Wave file")
    parser.add_argument("--test", help="test name to execute and exit")
    parser.add_argument("--fileout", help="outfilename needed for some tests")
    parser.add_argument(
        "--dump-lines",
        "-n",
        default="2",
        help="limit number of lines for datas dumped  (default:%(default)s)",
    )

    # parse arguments
    args = parser.parse_args()

    if args.test == "rewrite":
        if not args.fileout:
            sys.exit("Test need --fileout")
        print(f'Test rewrite "{args.wavefile[0]}" to "{args.fileout}"')
        test_rewrite_wav(args.wavefile[0], args.fileout)
        return

    if args.test == "setcue":
        if not args.fileout:
            sys.exit("Test need --fileout")
        print(f'Test set cue points "{args.wavefile[0]}" to "{args.fileout}"')
        test_set_cue_points(args.wavefile[0], args.fileout)
        return

    for glob_name in args.wavefile:
        glob_name = glob_name.rstrip("\r\n")
        for fname in glob.glob(glob_name):
            try:
                if os.path.isfile(fname):
                    _, ext = os.path.splitext(fname)
                    if ext.lower() == ".wav":
                        wave_parse(fname, args)
                elif os.path.isdir(fname):
                    for root, _, files in os.walk(fname, topdown=False):
                        for fname in files:
                            _, ext = os.path.splitext(fname)
                            if ext.lower() == ".wav":
                                wave_parse(os.path.join(root, fname), args)
            except Exception as _e:
                print("FAILED:", _e)


if __name__ == "__main__":
    # protect main from IOError occuring with a pipe command
    try:
        main()
    except IOError as _e:
        if _e.errno not in [22, 32]:
            raise _e
    except KeyboardInterrupt:
        print()
