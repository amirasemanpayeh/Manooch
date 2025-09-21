"""
Audio Separator Utility

This module provides an atomic, self-contained utility class for audio stem
separation built around Demucs (compatible with the python-audio-separator
approach). It mirrors the style of other utils (e.g., video_stitcher) with
clear logging, internal helpers prefixed by `_`, and storage conventions:

- Temporary files: `storage/audio_dump`
- Model cache: `storage/ai_models/audio_separator/...`

Two main APIs are provided:
- `separate_extract(audio, filters)`: returns separate files for requested filters
- `separate_keep(audio, filters)`: returns a single mixed file keeping only filters

Filters supported (case-insensitive):
- vocals (aliases: vocal, voice, singer)
- drums
- bass
- other (aliases: rest)
- instrumental (aliases: no_vocals, accompaniment, karaoke, music, instruments)

Implementation Notes:
- Uses Demucs via the module CLI (`python -m demucs.separate`) to simplify
  dependency handling. If Demucs is not available, raises a helpful error.
- Ensures model downloads (when first run) are cached under
  `storage/ai_models/audio_separator` by setting common cache env vars.
"""

from __future__ import annotations

import os
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import logging

try:
    # Soft dependency: pydub for mixing/format IO
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = False

# Note: Avoid importing AudioTools here to prevent circular imports.


def get_audio_separator_logger(name: str = "AudioSeparator") -> logging.Logger:
    """Create/fetch a configured logger for audio separation components."""
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


class AudioSeparator:
    """Demucs-based audio stem separation utility.

    Public API:
    - separate_extract(audio_input, filters): return per-filter isolated files
    - separate_keep(audio_input, filters): return single file with only kept stems
    """

    # Demucs model to use; compatible with common 4-stem output
    DEFAULT_MODEL = "htdemucs"

    # Canonical stems Demucs produces
    DEMUCS_STEMS = ["vocals", "drums", "bass", "other"]

    # Filter alias map to canonical tokens
    FILTER_ALIASES = {
        "vocals": "vocals",
        "vocal": "vocals",
        "voice": "vocals",
        "singer": "vocals",
        "drums": "drums",
        "bass": "bass",
        "other": "other",
        "piano": "other",
        "rest": "other",
        # Instrumental is a virtual stem = mix of all non-vocals
        "instrumental": "instrumental",
        "no_vocals": "instrumental",
        "accompaniment": "instrumental",
        "karaoke": "instrumental",
        "music": "instrumental",
        "instruments": "instrumental",
    }

    def __init__(self,
                 model_name: str = None,
                 temp_dir: Optional[str] = None,
                 models_dir: Optional[str] = None):
        self.logger = get_audio_separator_logger(f"AudioSeparator.{id(self)}")
        self.model_name = model_name or self.DEFAULT_MODEL

        # Setup storage directories
        self.audio_dump_path = self._get_audio_dump_path() if temp_dir is None else Path(temp_dir)
        self.audio_dump_path.mkdir(parents=True, exist_ok=True)

        self.models_dir = self._get_models_dir() if models_dir is None else Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Route model caches to project storage
        # Covers common locations used by Demucs/Torch/HF
        os.environ.setdefault("TORCH_HOME", str(self.models_dir / "torch"))
        os.environ.setdefault("HF_HOME", str(self.models_dir / "hf"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.models_dir / "hf_cache"))
        os.environ.setdefault("DEMUCS_CACHE", str(self.models_dir / "demucs_cache"))

        self.logger.info(f"Audio models directory: {self.models_dir}")
        self.logger.info(f"Audio temp directory: {self.audio_dump_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────
    def separate_extract(self, audio_input: str, filters: List[str]) -> Dict[str, str]:
        """Separate audio and return isolated files for each requested filter.

        Args:
            audio_input: Local path or URL to the audio file
            filters: List of strings defining which components to extract

        Returns:
            Dict mapping canonical filter name -> file path in `storage/audio_dump`

        Raises:
            RuntimeError on Demucs failure or if no valid filters
        """
        if not filters:
            raise ValueError("filters must be a non-empty list")

        local_audio = self._ensure_local_audio(audio_input)
        requested = self._normalize_filters(filters)
        if not requested:
            raise ValueError("No valid filters provided after normalization")

        # Run separation (idempotent per input path + model) into a unique folder
        stems = self._run_demucs(local_audio)

        # Build results per requested filter, supporting the special 'instrumental'
        results: Dict[str, str] = {}
        for token in requested:
            if token == "instrumental":
                # Mix all stems except vocals
                path = self._mix_stems(
                    [stems[s] for s in self.DEMUCS_STEMS if s != "vocals"],
                    label_prefix="instrumental"
                )
                results[token] = path
            elif token in stems:
                # Copy stem to a stable name under audio_dump
                out_path = self._copy_to_dump(stems[token], prefix=f"extract_{token}")
                results[token] = out_path
            else:
                self.logger.warning(f"Requested filter '{token}' not available in stems")

        if not results:
            raise RuntimeError("No stems produced for requested filters")

        return results

    def separate_keep(self, audio_input: str, filters: List[str], output_basename: Optional[str] = None) -> str:
        """Separate audio and return a single file keeping only requested filters.

        Args:
            audio_input: Local path or URL to the audio file
            filters: List of strings defining which components to keep
            output_basename: Optional base filename for the output

        Returns:
            Path to the mixed audio file in `storage/audio_dump`
        """
        if not filters:
            raise ValueError("filters must be a non-empty list")

        local_audio = self._ensure_local_audio(audio_input)
        requested = self._normalize_filters(filters)
        if not requested:
            raise ValueError("No valid filters provided after normalization")

        stems = self._run_demucs(local_audio)

        # Resolve which stems to include
        include_paths: List[str] = []
        if "instrumental" in requested:
            # Special case: keep everything except vocals
            include_paths = [stems[s] for s in self.DEMUCS_STEMS if s != "vocals"]
        else:
            for token in requested:
                if token in stems:
                    include_paths.append(stems[token])
                else:
                    self.logger.warning(f"Requested keep filter '{token}' not available in stems")

        if not include_paths:
            raise RuntimeError("No stems available to keep for requested filters")

        # Mix to single output
        label = output_basename or f"kept_{'_'.join(sorted(requested))}"
        mixed_path = self._mix_stems(include_paths, label_prefix=label)
        return mixed_path

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _normalize_filters(self, filters: List[str]) -> List[str]:
        """Normalize filters to canonical tokens used internally."""
        normalized = []
        for f in filters:
            if not f:
                continue
            key = str(f).strip().lower()
            token = self.FILTER_ALIASES.get(key)
            if token:
                normalized.append(token)
            else:
                # Allow direct canonical stems if provided
                if key in self.DEMUCS_STEMS or key == "instrumental":
                    normalized.append(key)
                else:
                    self.logger.warning(f"Unknown filter '{f}' ignored")
        # Deduplicate but order-preserving
        seen = set()
        result = []
        for t in normalized:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def _get_audio_dump_path(self) -> Path:
        """Resolve the storage/audio_dump path without importing AudioTools."""
        # utils/audio_tools/audio_seperator.py -> repo root -> storage/audio_dump
        return Path(__file__).resolve().parents[2] / "storage" / "audio_dump"

    def _get_models_dir(self) -> Path:
        """Resolve the storage/ai_models/audio_separator parent path."""
        return Path(__file__).resolve().parents[2] / "storage" / "ai_models" / "audio_separator"

    def _ensure_local_audio(self, audio_input: str) -> str:
        """Ensure input is available locally; download if URL.

        Returns a local file path.
        """
        if audio_input.startswith(("http://", "https://")):
            # Lazy import to avoid global dependency
            import requests
            suffix = Path(audio_input).suffix or ".wav"
            temp_path = self.audio_dump_path / f"input_{uuid.uuid4().hex}{suffix}"
            self.logger.info(f"Downloading audio: {audio_input}")
            resp = requests.get(audio_input, timeout=60)
            resp.raise_for_status()
            with open(temp_path, "wb") as f:
                f.write(resp.content)
            return str(temp_path)
        else:
            if not Path(audio_input).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            return str(Path(audio_input).resolve())

    def _run_demucs(self, local_audio_path: str) -> Dict[str, str]:
        """Run Demucs separation and return dict of stem -> file path.

        Uses the Demucs CLI to avoid deep Python API integration complexities.
        Ensures outputs land under `storage/audio_dump`.
        """
        # Check availability of demucs
        if not self._demucs_available():
            raise RuntimeError(
                "Demucs is not available. Please install it (see requirements.txt) "
                "or run: python3 -m pip install --break-system-packages demucs"
            )

        # Unique output folder under audio_dump
        batch_id = uuid.uuid4().hex[:8]
        out_dir = self.audio_dump_path / f"separated_{batch_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            self._python_executable(), "-m", "demucs.separate",
            "-n", self.model_name,
            "-o", str(out_dir),
            "-d", "cpu",  # default to CPU for portability
            str(local_audio_path),
        ]

        self.logger.info(f"Running Demucs: {' '.join(cmd)}")
        env = os.environ.copy()

        # Route caches to model dir (already set in __init__), keep explicit as well
        env.setdefault("TORCH_HOME", os.environ.get("TORCH_HOME", str(self.models_dir / "torch")))
        env.setdefault("HF_HOME", os.environ.get("HF_HOME", str(self.models_dir / "hf")))
        env.setdefault("TRANSFORMERS_CACHE", os.environ.get("TRANSFORMERS_CACHE", str(self.models_dir / "hf_cache")))
        env.setdefault("DEMUCS_CACHE", os.environ.get("DEMUCS_CACHE", str(self.models_dir / "demucs_cache")))

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        if proc.returncode != 0:
            self.logger.error(proc.stderr.decode(errors="ignore"))
            raise RuntimeError("Demucs separation failed")

        # Demucs outputs: out_dir/<model_name>/<track_name>/{drums,bass,other,vocals}.wav
        model_dir = out_dir / self.model_name
        # Find the sole track folder
        track_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
        if not track_dirs:
            raise RuntimeError("Demucs produced no output folders")
        track_dir = track_dirs[0]

        stems = {}
        for stem in self.DEMUCS_STEMS:
            stem_path = track_dir / f"{stem}.wav"
            if stem_path.exists():
                stems[stem] = str(stem_path)
            else:
                self.logger.warning(f"Missing expected stem output: {stem_path}")

        if not stems:
            raise RuntimeError("No stems found after Demucs run")

        return stems

    def _mix_stems(self, stem_paths: List[str], label_prefix: str = "mix") -> str:
        """Mix multiple stem WAV files into a single output.

        Uses pydub for simple summing with a small headroom to avoid clipping.
        Returns the output file path under audio_dump.
        """
        if not PYDUB_AVAILABLE:
            raise RuntimeError("pydub is required for stem mixing. Please install pydub.")

        if not stem_paths:
            raise ValueError("stem_paths must be non-empty")

        # Load stems and mix down
        mixed = None
        for i, path in enumerate(stem_paths):
            seg = AudioSegment.from_file(path)
            # Provide mild headroom per-add to avoid clipping (approx -3 dB per addition)
            seg = seg - 1.0
            mixed = seg if mixed is None else mixed.overlay(seg)

        # Gentle trim back to reasonable loudness
        mixed = mixed - 1.0

        # Write output WAV
        out_name = f"{label_prefix}_{uuid.uuid4().hex[:8]}.wav"
        out_path = self.audio_dump_path / out_name
        mixed.export(str(out_path), format="wav")
        return str(out_path)

    def _copy_to_dump(self, src_path: str, prefix: str) -> str:
        """Copy a file to audio_dump with a unique prefixed name and return path."""
        src = Path(src_path)
        ext = src.suffix or ".wav"
        out = self.audio_dump_path / f"{prefix}_{uuid.uuid4().hex[:8]}{ext}"
        shutil.copy2(src, out)
        return str(out)

    def _demucs_available(self) -> bool:
        """Check if demucs CLI is available in current environment."""
        try:
            proc = subprocess.run([self._python_executable(), "-m", "demucs", "--help"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return proc.returncode == 0
        except Exception:
            return False

    def _python_executable(self) -> str:
        """Resolve the current Python executable for CLI module invocations."""
        import sys
        return sys.executable or "python3"
