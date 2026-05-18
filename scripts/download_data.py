"""Download training data for openwakeword custom model training.

Downloads:
- MIT environmental impulse responses (room acoustics)
- AudioSet background noise clips
- Free Music Archive clips
- Pre-computed openwakeword features (ACAV100M)
- Validation set features
- Piper TTS sample generator + voice model

Usage:
    python scripts/download_data.py --output-dir ./training_data
    python scripts/download_data.py --output-dir ./training_data --fma-hours 10
    python scripts/download_data.py --output-dir ./training_data --skip piper fma
"""

import argparse
import logging
import subprocess
from pathlib import Path

import datasets
import numpy as np
import scipy.io.wavfile
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

AUDIOSET_REPO = "agkphysics/AudioSet"
PIPER_REPO = "https://github.com/rhasspy/piper-sample-generator"
PIPER_VOICE_URL = "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"
FEATURES_URL = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
VALIDATION_URL = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"

ALL_STEPS = ["piper", "rir", "audioset", "fma", "features"]


def save_audio_dataset_as_wav(dataset, output_dir: Path, replace_ext: str | None = None):
    for row in tqdm(dataset):
        name = row["audio"]["path"].split("/")[-1]
        if replace_ext:
            name = Path(name).stem + ".wav"
        audio = (row["audio"]["array"] * 32767).astype(np.int16)
        scipy.io.wavfile.write(str(output_dir / name), 16000, audio)


def download_piper(output_dir: Path):
    piper_dir = output_dir / "piper-sample-generator"
    if piper_dir.exists():
        logger.info("piper-sample-generator already exists, skipping clone")
    else:
        logger.info("Cloning piper-sample-generator...")
        subprocess.run(["git", "clone", PIPER_REPO, str(piper_dir)], check=True)

    voice_path = piper_dir / "models" / "en_US-libritts_r-medium.pt"
    if voice_path.exists():
        logger.info("Piper voice model already exists, skipping download")
    else:
        logger.info("Downloading piper voice model...")
        voice_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["wget", "-O", str(voice_path), PIPER_VOICE_URL], check=True)


def download_mit_rirs(output_dir: Path):
    rir_dir = output_dir / "mit_rirs"
    rir_dir.mkdir(parents=True, exist_ok=True)

    if any(rir_dir.iterdir()):
        logger.info("MIT RIRs directory is not empty, skipping")
        return

    logger.info("Downloading MIT environmental impulse responses...")
    ds = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
    )
    save_audio_dataset_as_wav(ds, rir_dir)


def download_audioset(output_dir: Path, n_hours: int):
    audioset_dir = output_dir / "audioset_16k"
    audioset_dir.mkdir(parents=True, exist_ok=True)

    if any(audioset_dir.iterdir()):
        logger.info("AudioSet directory is not empty, skipping")
        return

    logger.info(f"Downloading AudioSet clips ({n_hours} hours)...")
    ds = datasets.load_dataset(AUDIOSET_REPO, split="train", streaming=True)
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))

    n_clips = n_hours * 3600 // 10
    for i, row in enumerate(tqdm(ds, total=n_clips)):
        name = f"audioset_{i:06d}.wav"
        audio = (row["audio"]["array"] * 32767).astype(np.int16)
        scipy.io.wavfile.write(str(audioset_dir / name), 16000, audio)
        if i + 1 >= n_clips:
            break


def download_fma(output_dir: Path, n_hours: int):
    fma_dir = output_dir / "fma"
    fma_dir.mkdir(parents=True, exist_ok=True)

    if any(fma_dir.iterdir()):
        logger.info("FMA directory is not empty, skipping")
        return

    logger.info(f"Downloading FMA clips ({n_hours} hours)...")
    ds = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))

    n_clips = n_hours * 3600 // 30
    for i, row in enumerate(tqdm(ds, total=n_clips)):
        name = Path(row["audio"]["path"]).stem + ".wav"
        audio = (row["audio"]["array"] * 32767).astype(np.int16)
        scipy.io.wavfile.write(str(fma_dir / name), 16000, audio)
        if i + 1 >= n_clips:
            break


def download_precomputed_features(output_dir: Path):
    features_path = output_dir / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    validation_path = output_dir / "validation_set_features.npy"

    for path, url, label in [
        (features_path, FEATURES_URL, "ACAV100M features (~2000 hrs)"),
        (validation_path, VALIDATION_URL, "validation set features (~11 hrs)"),
    ]:
        if path.exists():
            logger.info(f"{label} already exists, skipping")
        else:
            logger.info(f"Downloading {label}...")
            subprocess.run(["wget", "-O", str(path), url], check=True)


def main():
    parser = argparse.ArgumentParser(description="Download training data for openwakeword")
    parser.add_argument("--output-dir", type=Path, default=Path("./training_data"))
    parser.add_argument("--audioset-hours", type=int, default=1, help="Hours of AudioSet clips to download (default: 1)")
    parser.add_argument("--fma-hours", type=int, default=1, help="Hours of FMA clips to download (default: 1)")
    parser.add_argument("--skip", nargs="+", choices=ALL_STEPS, default=[], help="Steps to skip")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir.resolve()}")

    steps = {
        "piper": lambda: download_piper(args.output_dir),
        "rir": lambda: download_mit_rirs(args.output_dir),
        "audioset": lambda: download_audioset(args.output_dir, args.audioset_hours),
        "fma": lambda: download_fma(args.output_dir, args.fma_hours),
        "features": lambda: download_precomputed_features(args.output_dir),
    }

    for name, fn in steps.items():
        if name in args.skip:
            logger.info(f"Skipping {name}")
            continue
        fn()

    logger.info("All downloads complete!")


if __name__ == "__main__":
    main()
