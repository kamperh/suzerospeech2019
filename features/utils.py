"""
Utility functions

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2019
"""

def read_vad(vad_fn, frame_indices=True):
    """
    Read a voice activity detection (VAD) file and return a dictionary.

    The dictionary has utterance labels as keys and as values the speech
    regions as lists of tuples of (start, end) frame, with the end excluded.
    """
    vad_dict = {}
    with open(vad_fn) as f:
        for line in f:
            utt_key, start, end = line.strip().split()
            start = float(start)
            end = float(end)
            if utt_key not in vad_dict:
                vad_dict[utt_key] = []

            if frame_indices:
                # Convert time to frames
                start = int(round(start * 100))
                end = int(round(end * 100)) + 1  # end index excluded
            vad_dict[utt_key].append((start, end))
    return vad_dict

