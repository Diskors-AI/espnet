import argparse
from pathlib import Path
import re
from praatio import textgrid
import soundfile as sf
from collections import defaultdict
import random
from typing import Tuple, Optional, Dict, List

train_dir_name = "tr_no_dev"
dev_dir_name = "dev"
test_dir_name = "eval1"
train_dir_name_phn = "tr_no_dev_phn"
dev_dir_name_phn = "dev_phn"
test_dir_name_phn = "eval1_phn"

# Define regex to match markers such as {laugh}, {cough}, etc.
MARKER_PATTERN = re.compile(r"\{[^}]+\}")


def filter_markers(text):
    """Remove markers enclosed in curly braces from text."""
    return MARKER_PATTERN.sub("", text).strip()


def extract_phoneme_durations(
    tg: textgrid.Textgrid, start_time: float, end_time: float
) -> Optional[Tuple[List[str], List[float]]]:
    """Extract phoneme durations for the specified time range, excluding markers."""
    phones_tier = tg.getTier("phones")
    if not phones_tier:
        return None, None

    phones = []
    durations = []

    for phone_start, phone_end, phone_label in phones_tier.entries:
        if phone_start < start_time or phone_end > end_time:
            continue
        if phone_label in {"spn", "sil", ""}:
            continue
        phones.append(phone_label)
        # This calculates phone durations in seconds, as expected by ESPNet GAN VITS TTS training
        phone_duration = phone_end - phone_start
        durations.append(phone_duration)

    if not durations:
        return None, None

    return phones, durations


def add_entry(
    data_entries,
    set_name: str,
    token_type: str,
    basename: str,
    wav_path: Path,
    segment_id: str,
    text: str,
    phones: List[str],
    phone_durations: List[float],
    speaker: str,
    start: float,
    end: float,
):
    if basename not in [entry[0] for entry in data_entries[set_name]["wav_scp"]]:
        data_entries[set_name]["wav_scp"].append((basename, wav_path))

    entry_text = " ".join(phones) if token_type == "phn" else text
    data_entries[set_name]["text"].append((segment_id, entry_text))
    data_entries[set_name]["segments"].append((segment_id, basename, start, end))
    data_entries[set_name]["durations"].append(
        (segment_id, " ".join(f"{d:.6f}" for d in phone_durations))
    )
    data_entries[set_name]["utt2spk"].append((segment_id, speaker))
    data_entries[set_name]["spk2utt"][speaker].append(segment_id)


def gather_data(
    alignments_dir: str,
    corpus_dir: str,
    data_percentage: float,
    token_type: str,
) -> Dict[str, Dict[str, List]]:
    alignments_dir = Path(alignments_dir)
    corpus_dir = Path(corpus_dir)

    suffix = "_phn" if token_type == "phn" else ""
    set_names = {
        "train": train_dir_name + suffix,
        "dev": dev_dir_name + suffix,
        "test": test_dir_name + suffix,
    }

    data_entries = {
        name: {
            "wav_scp": [],
            "text": [],
            "segments": [],
            "durations": [],
            "utt2spk": [],
            "spk2utt": defaultdict(list),
        }
        for name in set_names.values()
    }

    all_textgrids = sorted(alignments_dir.glob("*.TextGrid"))
    subset_size = int(len(all_textgrids) * data_percentage)
    selected_textgrids = random.sample(all_textgrids, subset_size)

    for tg_path in selected_textgrids:
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
            text = filter_markers(text)
            if not text.strip() or start >= duration or end > duration + 1e-3:
                continue

            segment_id = f"{basename}_{i:04d}"
            phones, phone_durations = extract_phoneme_durations(tg, start, end)

            if phones is None or phone_durations is None:
                print(f"Skipping {segment_id} due to missing phoneme information.")
                continue

            rnd = random.randint(1, 100)
            set_key = "train" if rnd <= 80 else "dev" if rnd <= 90 else "test"
            set_name = set_names[set_key]

            add_entry(
                data_entries,
                set_name,
                token_type,
                basename,
                wav_path,
                segment_id,
                text,
                phones,
                phone_durations,
                speaker,
                start,
                end,
            )

    return data_entries


def write_data(data_entries: Dict[str, Dict[str, List]], output_dir: str) -> None:
    """Write the gathered data to files in the specified output directory."""
    output_dir = Path(output_dir)

    for set_name, entries in data_entries.items():
        set_dir = output_dir / set_name
        set_dir.mkdir(parents=True, exist_ok=True)

        with (
            open(set_dir / "wav.scp", "w") as wav_scp,
            open(set_dir / "text", "w") as text,
            open(set_dir / "segments", "w") as segments,
            open(set_dir / "durations", "w") as durations,
            open(set_dir / "utt2spk", "w") as utt2spk,
            open(set_dir / "spk2utt", "w") as spk2utt,
        ):
            for basename, wav_path in sorted(entries["wav_scp"]):
                wav_scp.write(f"{basename} {wav_path}\n")
            for segment_id, text_content in sorted(entries["text"]):
                text.write(f"{segment_id} {text_content}\n")
            for segment_id, basename, start, end in sorted(entries["segments"]):
                segments.write(f"{segment_id} {basename} {start} {end}\n")
            for segment_id, duration in sorted(entries["durations"]):
                durations.write(f"{segment_id} {duration}\n")
            for segment_id, speaker in sorted(entries["utt2spk"]):
                utt2spk.write(f"{segment_id} {speaker}\n")
            for speaker, utts in sorted(entries["spk2utt"].items()):
                spk2utt.write(f"{speaker} {' '.join(sorted(utts))}\n")

    print(f"Data preparation completed. Files saved to {output_dir}")


def create_stage2_data(
    alignments_dir: str,
    corpus_dir: str,
    output_dir: str,
    token_type: str,
    data_percentage: float = 1.0,
) -> None:
    data_entries = gather_data(
        alignments_dir=alignments_dir,
        corpus_dir=corpus_dir,
        data_percentage=data_percentage,
        token_type=token_type,
    )
    write_data(data_entries, output_dir)


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
        "--data_percentage",
        type=float,
        default=1.0,
        help="Percentage of the data to use for training (0.0 < p <= 1.0).",
    )
    parser.add_argument(
        "--token_type",
        type=str,
        choices=["phn", "char"],
        required=True,
        help="Whether to generate phoneme-based ('phn') or character-based ('char') folders.",
    )

    args = parser.parse_args()

    create_stage2_data(
        alignments_dir=args.alignments_dir,
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        data_percentage=args.data_percentage,
        token_type=args.token_type,
    )
