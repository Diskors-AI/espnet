import argparse
from pathlib import Path
from praatio import textgrid
import soundfile as sf
from collections import defaultdict
import random
from typing import Tuple, Optional

train_dir_name = "tr_no_dev"
dev_dir_name = "dev"
test_dir_name = "eval1"


def extract_phoneme_durations(
    tg: textgrid.Textgrid,
    start_time: float,
    end_time: float,
    sample_rate: int,
    hop_size: int,
) -> Optional[Tuple[list, list]]:
    """Extract phoneme durations for the specified time range."""
    phones_tier = tg.getTier("phones")
    if not phones_tier:
        return None, None

    phones = []
    durations = []
    total_frames = int((end_time - start_time) * sample_rate / hop_size) + 1
    prev_end = start_time

    for phone_start, phone_end, phone_label in phones_tier.entries:
        # Only consider phones within the current utterance
        if phone_start < start_time or phone_end > end_time:
            continue

        # Add the current phone
        phones.append(phone_label)
        phone_duration = int((phone_end - phone_start) * sample_rate / hop_size)
        durations.append(phone_duration)
        prev_end = phone_end

    # Check if there are any durations
    if not durations:
        return None, None

    # Adjust last duration to match total frames if needed
    if sum(durations) < total_frames:
        durations[-1] += total_frames - sum(durations)

    return phones, durations


def create_stage2_data(
    alignments_dir: str, corpus_dir: str, output_dir: str, hop_size: int = 256
) -> None:
    """
    Create the necessary data structure for ESPnet stage 2 using TextGrid alignments
    and audio files.
    """

    alignments_dir = Path(alignments_dir)
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)

    # Create output directories if they do not exist
    train_dir = output_dir / train_dir_name
    dev_dir = output_dir / dev_dir_name
    test_dir = output_dir / test_dir_name
    for d in [train_dir, dev_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # In-memory data structures to track what's written
    written_entries = {
        train_dir_name: {"wav_scp": set(), "text": set()},
        dev_dir_name: {"wav_scp": set(), "text": set()},
        test_dir_name: {"wav_scp": set(), "text": set()},
    }

    # Dictionary to hold utt2spk and spk2utt
    utt2spk_dict = defaultdict(dict)  # Stores utt2spk mappings for each set
    spk2utt_dict = defaultdict(
        lambda: defaultdict(list)
    )  # Stores spk2utt mappings for each set

    # Open output files for each set
    files = {
        train_dir_name: {
            "wav_scp": open(train_dir / "wav.scp", "w"),
            "text": open(train_dir / "text", "w"),
            "segments": open(train_dir / "segments", "w"),
            "durations": open(train_dir / "durations", "w"),
            "utt2spk": open(train_dir / "utt2spk", "w"),
            "spk2utt": open(train_dir / "spk2utt", "w"),
        },
        dev_dir_name: {
            "wav_scp": open(dev_dir / "wav.scp", "w"),
            "text": open(dev_dir / "text", "w"),
            "segments": open(dev_dir / "segments", "w"),
            "durations": open(dev_dir / "durations", "w"),
            "utt2spk": open(dev_dir / "utt2spk", "w"),
            "spk2utt": open(dev_dir / "spk2utt", "w"),
        },
        test_dir_name: {
            "wav_scp": open(test_dir / "wav.scp", "w"),
            "text": open(test_dir / "text", "w"),
            "segments": open(test_dir / "segments", "w"),
            "durations": open(test_dir / "durations", "w"),
            "utt2spk": open(test_dir / "utt2spk", "w"),
            "spk2utt": open(test_dir / "spk2utt", "w"),
        },
    }

    # Process each TextGrid file
    for tg_path in sorted(alignments_dir.glob("*.TextGrid")):
        tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
        basename = tg_path.stem
        wav_path = corpus_dir / f"{basename}.wav"

        if not wav_path.exists():
            print(f"WAV file not found for {basename}, skipping.")
            continue

        with sf.SoundFile(wav_path) as audio:
            orig_sr = audio.samplerate
            num_samples = audio.frames
            duration = num_samples / orig_sr

        speaker = basename.split("_")[0]
        utterances_tier = tg.getTier("utterances")

        if not utterances_tier:
            print(f"No 'utterances' tier found in {basename}, skipping.")
            continue

        for i, (start, end, text) in enumerate(utterances_tier.entries):
            if not text.strip():
                continue

            if start >= duration or end > duration + 1e-3:
                continue

            segment_id = f"{basename}_{i:04d}"
            phones, phone_durations = extract_phoneme_durations(
                tg, start, end, orig_sr, hop_size
            )

            if phones is None or phone_durations is None:
                print(
                    f"Skipping {segment_id} due to missing phoneme information. Text: {text}, Start: {start}, End: {end}"
                )
                continue

            # Randomly assign to tr_no_dev/dev/eval1
            rnd = random.randint(1, 100)
            if rnd <= 80:
                set_name = train_dir_name
            elif 81 <= rnd <= 90:
                set_name = dev_dir_name
            else:
                set_name = test_dir_name

            # Ensure not to write duplicates
            if basename not in written_entries[set_name]["wav_scp"]:
                files[set_name]["wav_scp"].write(f"{basename} {wav_path}\n")
                written_entries[set_name]["wav_scp"].add(basename)

            files[set_name]["text"].write(f"{segment_id} {text}\n")
            files[set_name]["segments"].write(
                f"{segment_id} {basename} {start} {end}\n"
            )
            files[set_name]["durations"].write(
                f"{segment_id} {' '.join(map(str, phone_durations))}\n"
            )

            # Track utt2spk and spk2utt in-memory for consistency
            utt2spk_dict[set_name][segment_id] = speaker
            spk2utt_dict[set_name][speaker].append(segment_id)

    # Write utt2spk and spk2utt for each set
    for set_name, f in files.items():
        for utt, speaker in utt2spk_dict[set_name].items():
            f["utt2spk"].write(f"{utt} {speaker}\n")

        for speaker, utts in spk2utt_dict[set_name].items():
            f["spk2utt"].write(f"{speaker} {' '.join(sorted(utts))}\n")

    # Close all files
    for set_name in files:
        for file in files[set_name].values():
            file.close()

    print(f"Data preparation completed. Files saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for ESPnet stage 2 using TextGrid alignments."
    )
    parser.add_argument(
        "--alignments_dir",
        type=str,
        required=True,
        help="Path to the directory containing TextGrid files.",
    )
    parser.add_argument(
        "--corpus_dir",
        type=str,
        required=True,
        help="Path to the directory containing WAV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where data will be saved.",
    )
    parser.add_argument(
        "--hop_size", type=int, default=256, help="Hop size for STFT (default: 256)."
    )

    args = parser.parse_args()

    create_stage2_data(
        alignments_dir=args.alignments_dir,
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        hop_size=args.hop_size,
    )
