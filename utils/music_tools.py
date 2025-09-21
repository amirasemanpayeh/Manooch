"""
Music Tools - Professional Audio Post-Processing for AI-Generated Music

This module provides comprehensive audio post-processing tools to polish AI-generated music
(especially ACE-Step outputs) and remove "robotic" artifacts. The processing chain includes:

1. LUFS normalization for proper loudness staging
2. High-pass filtering to remove sub-bass rumble
3. De-essing to tame harsh sibilance and fizz
4. Gentle EQ to reduce harshness and metallic artifacts
5. Soft compression for glue and cohesion
6. Harmonic saturation for warmth
7. Subtle reverb to add natural space
8. Final limiting and loudness normalization

Key Features:
- Full mix processing with professional mastering chain
- Stem separation support using Demucs for advanced processing
- Configurable loudness targets for streaming vs. club scenarios
- Robust error handling and audio format support
"""

import os
import tempfile
import traceback
import logging
import glob
import copy
from pathlib import Path
from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from enum import Enum

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pedalboard import (
    Pedalboard, HighpassFilter, LowShelfFilter, PeakFilter, 
    HighShelfFilter, Compressor, Reverb, Limiter, Gain
)

# Note: Deesser not available in this pedalboard version, using alternative approach

# Optional imports for stem separation and advanced processing
try:
    from demucs import demucs
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

try:
    from scipy.signal import resample_poly, firwin, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class LoudnessTarget(Enum):
    """Predefined loudness targets for different use cases"""
    STREAMING = -14.0      # Spotify/Apple Music standard
    PODCAST = -16.0        # Podcast/spoken word
    RADIO = -12.0          # Radio broadcast
    CLUB = -10.0           # Club/dance music
    DEMO = -9.0            # Demo/preview (loudest)


@dataclass
class ProcessingSettings:
    """Configuration for audio processing chain"""
    
    # Loudness settings
    target_lufs: float = -14.0
    pre_normalize_lufs: float = -20.0
    
    # High-pass filter
    dc_blocker_freq: float = 20.0      # DC/infra-sonic blocker
    highpass_freq: float = 80.0
    
    # De-esser settings (pre-compression)
    deesser_threshold: float = -24.0
    deesser_ratio: float = 4.0
    deesser_frequency: float = 7000.0
    
    # Post-compression de-esser (lighter)
    post_deesser_enabled: bool = True
    post_deesser_threshold_offset: float = 2.0  # Added to main threshold
    post_deesser_ratio_reduction: float = 1.0   # Subtracted from main ratio
    
    # EQ settings
    harsh_freq_cut: float = 3500.0
    harsh_cut_db: float = -1.5
    high_shelf_freq: float = 9000.0
    high_shelf_db: float = -0.5
    reverb_high_cut_freq: float = 7000.0       # High-cut after reverb
    reverb_high_cut_db: float = -3.0
    
    # Compressor settings
    comp_threshold: float = -20.0
    comp_ratio: float = 1.8
    comp_attack_ms: float = 25.0
    comp_release_ms: float = 120.0
    
    # Saturation
    saturation_drive: float = 3.0
    
    # Reverb settings
    reverb_room_size: float = 0.12
    reverb_wet_level: float = 0.02
    reverb_dry_level: float = 0.98
    reverb_width: float = 0.9
    reverb_damping: float = 0.35
    
    # Limiter settings
    limiter_threshold: float = -1.0
    limiter_release_ms: float = 100.0
    
    # Safety settings
    true_peak_ceiling: float = 0.89  # ~-1 dBFS true peak
    stem_mix_headroom: float = 0.8   # Pre-limiter bus trim for stems
    
    @classmethod
    def pop_preset(cls):
        """Preset optimized for pop music"""
        settings = cls()
        settings.deesser_frequency = 6500.0
        settings.comp_attack_ms = 10.0
        settings.reverb_wet_level = 0.08
        return settings
    
    @classmethod 
    def rap_preset(cls):
        """Preset optimized for rap/hip-hop"""
        settings = cls()
        settings.deesser_threshold = -28.0  # More aggressive
        settings.harsh_cut_db = -2.5
        settings.comp_ratio = 2.5
        settings.reverb_wet_level = 0.04    # Drier
        return settings
    
    @classmethod
    def ballad_preset(cls):
        """Preset optimized for ballads"""
        settings = cls()
        settings.comp_attack_ms = 25.0      # Slower, more natural
        settings.reverb_wet_level = 0.10    # More spacious
        settings.saturation_drive = 2.0     # Gentler saturation
        return settings


def linear_phase_two_band_deesser(audio: np.ndarray, sr: int,
                                  split_hz: float = 6500.0,
                                  thr_db: float = -28.0,
                                  ratio: float = 3.5,
                                  att_ms: float = 2.0,
                                  rel_ms: float = 60.0) -> np.ndarray:
    """
    Split full mix into <split and >split with linear-phase FIR, fast-compress highs, recombine.
    Acts like a wideband de-esser on the 6.5k+ area without phase/comb artifacts.
    """
    if audio.size == 0:
        return audio

    # linear-phase FIR HP/LP
    numtaps = max(513, int(sr * 0.004) | 1)  # ~4ms, odd taps
    lp = firwin(numtaps, split_hz, fs=sr)              # lowpass
    hp = -lp; hp[numtaps // 2] += 1.0                  # spectral complement = highpass

    low = np.vstack([filtfilt(lp, [1.0], ch) for ch in audio])
    high = np.vstack([filtfilt(hp, [1.0], ch) for ch in audio])

    # fast envelope follower on highs
    atk = np.exp(-1.0 / (att_ms * 0.001 * sr))
    rel = np.exp(-1.0 / (rel_ms * 0.001 * sr))

    env = np.zeros_like(high)
    for ch in range(high.shape[0]):
        peak = 0.0
        h = high[ch]
        for i in range(h.size):
            x = abs(h[i])
            peak = max(x, peak * (atk if x < peak else rel))
            env[ch, i] = peak

    # gain computer (simple soft-knee above threshold)
    eps = 1e-9
    env_db = 20.0 * np.log10(env + eps)
    over = env_db - thr_db
    over = np.maximum(0.0, over)
    gr_db = -over * (1.0 - 1.0/ratio)
    gr = 10.0 ** (gr_db / 20.0)

    high_tamed = high * gr
    mixed = low + high_tamed
    return mixed.astype(np.float32)


def split_bands(audio: np.ndarray, sr: int,
                xover_low: float = 200.0,
                xover_high: float = 5000.0) -> Dict[str, np.ndarray]:
    """Split into lows (<xover_low), mids (xover_low–xover_high), highs (>xover_high)."""
    
    # Work on copies
    full = audio.copy()

    # Low = LP @ xover_low (approximated with extreme high shelf)
    low = Pedalboard([HighShelfFilter(cutoff_frequency_hz=xover_low, gain_db=-60.0, q=0.7)])(full, sr)

    # High = HP @ xover_high
    high = Pedalboard([HighpassFilter(cutoff_frequency_hz=xover_high)])(full, sr)

    # Mid ≈ full - low - high
    mid = full - low - high

    return {"low": low, "mid": mid, "high": high}


def multiband_glue(audio: np.ndarray, sr: int) -> np.ndarray:
    """Gentle 3-band processing with faster, deeper action on highs."""
    bands = split_bands(audio, sr)

    low_proc = Pedalboard([
        Compressor(threshold_db=-22.0, ratio=1.6, attack_ms=25.0, release_ms=160.0)
    ])(bands["low"], sr)

    mid_proc = Pedalboard([
        Compressor(threshold_db=-20.0, ratio=1.8, attack_ms=20.0, release_ms=140.0),
        PeakFilter(cutoff_frequency_hz=3500.0, gain_db=-1.0, q=1.0),  # mild mid harshness control
    ])(bands["mid"], sr)

    high_proc = Pedalboard([
        Compressor(threshold_db=-30.0, ratio=3.0, attack_ms=2.0, release_ms=80.0),  # acts like de-esser
        HighShelfFilter(cutoff_frequency_hz=9000.0, gain_db=-1.0, q=0.7),           # tiny tilt down
    ])(bands["high"], sr)

    mixed = low_proc + mid_proc + high_proc
    # small safety trim to avoid overs from summing
    return mixed * 0.97


def softclip_np(audio: np.ndarray, drive: float = 1.5) -> np.ndarray:
    """Symmetric tanh soft clip"""
    return np.tanh(drive * audio) / np.tanh(drive)


class AudioProcessor:
    """Professional audio processor for AI-generated music"""
    
    def __init__(self):
        """Initialize the audio processor"""
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup basic logging"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def read_audio(self, path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Read audio file and ensure stereo format
        
        Args:
            path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
            Audio data shape: (channels, samples)
        """
        try:
            audio, sr = sf.read(str(path), always_2d=False)
            
            # Convert to stereo if mono
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)  # mono -> stereo
            else:
                audio = audio.T  # (frames, channels) -> (channels, frames)
                
            return audio.astype(np.float32), sr
            
        except Exception as e:
            self.logger.error(f"Failed to read audio file {path}: {e}")
            raise
    
    def write_audio(self, path: Union[str, Path], audio: np.ndarray, sr: int, bit_depth: int = 24):
        """
        Write audio to file with proper dithering for 16-bit
        
        Args:
            path: Output file path
            audio: Audio data with shape (channels, samples)
            sr: Sample rate
            bit_depth: Bit depth (16 or 24, defaults to 24)
        """
        try:
            # Ensure output directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Transpose back to (samples, channels) for soundfile
            audio_out = audio.T
            
            # Apply dithering for 16-bit export
            if bit_depth == 16:
                audio_out = self._tpdf_dither(audio_out)
                subtype = "PCM_16"
            elif bit_depth == 24:
                subtype = "PCM_24"
            else:
                # Default to 24-bit for unknown bit depths
                subtype = "PCM_24"
                bit_depth = 24
            
            sf.write(str(path), audio_out, sr, subtype=subtype)
            
            self.logger.info(f"✅ Audio written to {path} ({bit_depth}-bit, {sr}Hz)")
            
        except Exception as e:
            self.logger.error(f"Failed to write audio file {path}: {e}")
            raise
    
    def _tpdf_dither(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply triangular probability density function (TPDF) dither for 16-bit export
        
        Args:
            audio: Audio data (samples, channels)
            
        Returns:
            Dithered audio
        """
        # TPDF dither: sum of two uniform random distributions
        dither_amp = 1.0 / (2**16)  # ~1 LSB for 16-bit
        noise1 = np.random.rand(*audio.shape) * dither_amp
        noise2 = np.random.rand(*audio.shape) * dither_amp
        dither = noise1 - noise2  # TPDF distribution
        
        dithered = audio + dither
        return np.clip(dithered, -1.0, 1.0)
    
    def _true_peak_guard(self, audio: np.ndarray, sr: int, ceiling_lin: float = 0.89) -> np.ndarray:
        """
        Guard against inter-sample peaks using 4x oversampling estimation
        
        Args:
            audio: Audio data (channels, samples)
            sr: Sample rate
            ceiling_lin: Linear ceiling (0.89 ≈ -1 dBFS)
            
        Returns:
            Peak-limited audio
        """
        if not SCIPY_AVAILABLE:
            # Fallback to simple sample peak limiting
            peak = np.max(np.abs(audio))
            if peak > ceiling_lin:
                scale = ceiling_lin / peak
                self.logger.info(f"🛡️ Sample peak limiting: {20*np.log10(scale):.1f} dB")
                return audio * scale
            return audio
        
        try:
            # 4x oversample to estimate inter-sample peaks
            upsampled = resample_poly(audio, 4, 1, axis=1)  # (channels, samples*4)
            true_peak = np.max(np.abs(upsampled))
            
            if true_peak > ceiling_lin:
                scale = ceiling_lin / true_peak
                self.logger.info(f"🛡️ True-peak safety scaling: {20*np.log10(scale):.1f} dB")
                return audio * scale
            
            return audio
            
        except Exception as e:
            self.logger.warning(f"True-peak guard failed, using sample peak: {e}")
            # Fallback to sample peak
            peak = np.max(np.abs(audio))
            if peak > ceiling_lin:
                scale = ceiling_lin / peak
                return audio * scale
            return audio
    
    def lufs_normalize(self, audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
        """
        Normalize audio to target LUFS loudness
        
        Args:
            audio: Audio data (channels, samples)
            sr: Sample rate
            target_lufs: Target loudness in LUFS
            
        Returns:
            Normalized audio
        """
        try:
            meter = pyln.Meter(sr)
            
            # pyloudnorm expects (samples, channels)
            audio_for_meter = audio.T
            
            # Measure current loudness
            loudness = meter.integrated_loudness(audio_for_meter)
            
            if np.isinf(loudness) or np.isnan(loudness):
                self.logger.warning("Could not measure loudness, skipping normalization")
                return audio
            
            # Calculate gain adjustment
            gain_db = target_lufs - loudness
            gain_linear = 10 ** (gain_db / 20.0)
            
            self.logger.info(f"LUFS: {loudness:.1f} → {target_lufs:.1f} (gain: {gain_db:+.1f} dB)")
            
            return (audio * gain_linear).astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"LUFS normalization failed: {e}")
            return audio
    
    def create_processing_chain(self, settings: ProcessingSettings) -> Pedalboard:
        """
        Create the professional mastering processing chain (without final limiter)
        
        Args:
            settings: Processing configuration
            
        Returns:
            Configured Pedalboard (limiter applied separately)
        """
        chain = [
            # 1. DC blocker - remove DC offset and infra-sonic content
            HighpassFilter(cutoff_frequency_hz=settings.dc_blocker_freq),
            
            # 2. Main high-pass filter - remove sub-bass rumble  
            HighpassFilter(cutoff_frequency_hz=settings.highpass_freq),
            
            # 3. Pre-compression EQ - gentle harsh frequency reduction
            PeakFilter(
                cutoff_frequency_hz=settings.harsh_freq_cut,
                gain_db=settings.harsh_cut_db,
                q=1.0
            ),
            
            # 4. Compressor - add glue and cohesion (slower attack for presence)
            Compressor(
                threshold_db=settings.comp_threshold,
                ratio=settings.comp_ratio,
                attack_ms=settings.comp_attack_ms,
                release_ms=settings.comp_release_ms
            ),
            
            # 5. Presence bump (subtle) - restore consonant clarity
            PeakFilter(
                cutoff_frequency_hz=2200.0,
                gain_db=0.8,
                q=0.9
            ),
            
            # 6. Post-compression high-frequency control - tame compression-enhanced sibilance
            # Using PeakFilter to reduce harsh frequencies instead of Deesser
            PeakFilter(
                cutoff_frequency_hz=settings.deesser_frequency,
                gain_db=-1.2,  # Gentle reduction
                q=2.0
            ) if settings.post_deesser_enabled else None,
            
            # 7. Reverb - tiny, dark reverb to "de-robotize"
            Reverb(
                room_size=settings.reverb_room_size,
                wet_level=settings.reverb_wet_level,
                dry_level=settings.reverb_dry_level,
                width=settings.reverb_width,
                damping=settings.reverb_damping
            ),
            
            # 8. Post-reverb high-cut - avoid shiny reverb tails
            HighShelfFilter(
                cutoff_frequency_hz=settings.reverb_high_cut_freq,
                gain_db=settings.reverb_high_cut_db,
                q=0.7
            ),
            
            # 9. Final high shelf - gentle high frequency reduction if needed
            HighShelfFilter(
                cutoff_frequency_hz=settings.high_shelf_freq,
                gain_db=settings.high_shelf_db,
                q=0.7
            ),
        ]
        
        # Filter out None elements (disabled processors)
        chain = [processor for processor in chain if processor is not None]
        
        return Pedalboard(chain)
    
    def polish_full_mix(self, 
                       audio: np.ndarray, 
                       sr: int, 
                       target_lufs: float = -14.0,
                       settings: Optional[ProcessingSettings] = None) -> np.ndarray:
        """
        Apply professional mastering chain to full mix with proper gain staging
        
        Args:
            audio: Input audio (channels, samples)
            sr: Sample rate
            target_lufs: Final loudness target
            settings: Processing settings (uses defaults if None)
            
        Returns:
            Processed audio
        """
        if settings is None:
            settings = ProcessingSettings()
            settings.target_lufs = target_lufs
        
        try:
            self.logger.info("🎛️ Starting full mix processing...")
            
            # Step 1: Pre-normalize to give processors headroom
            self.logger.info(f"📊 Pre-normalizing to {settings.pre_normalize_lufs} LUFS...")
            audio = self.lufs_normalize(audio, sr, settings.pre_normalize_lufs)
            
            # Step 2: Apply processing chain (WITHOUT final limiter)
            self.logger.info("🔧 Applying processing chain...")
            board = self.create_processing_chain(settings)
            processed = board(audio, sr)

            # Linear-phase two-band de-esser (if SciPy present)
            if SCIPY_AVAILABLE:
                self.logger.info("🌈 Linear-phase two-band de-esser...")
                processed = linear_phase_two_band_deesser(
                    processed, sr, split_hz=settings.deesser_frequency,
                    thr_db=-28.0, ratio=3.5, att_ms=2.0, rel_ms=60.0
                )
            
            # Step 3: Final loudness normalization (BEFORE limiter)
            self.logger.info(f"📈 Final normalization to {settings.target_lufs} LUFS...")
            processed = self.lufs_normalize(processed, sr, settings.target_lufs)

            # Step 4: Pre-limiter soft clip (gentle)
            self.logger.info("✨ Pre-limiter soft clip...")
            processed = softclip_np(processed, drive=1.15)
            
            # Step 5: Final limiter (LAST in chain to avoid normalizing after limiting)
            self.logger.info("🛡️ Applying final limiter...")
            final_limiter = Pedalboard([
                Limiter(
                    threshold_db=settings.limiter_threshold,
                    release_ms=settings.limiter_release_ms
                )
            ])
            processed = final_limiter(processed, sr)
            
            # Step 6: True-peak safety guard
            processed = self._true_peak_guard(processed, sr, settings.true_peak_ceiling)
            
            self.logger.info("✅ Full mix processing complete!")
            return processed.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Full mix processing failed: {e}")
            self.logger.error(traceback.format_exc())
            return audio
    
    def process_audio_file(self,
                          input_path: Union[str, Path],
                          output_path: Union[str, Path],
                          loudness_target: Union[float, LoudnessTarget] = LoudnessTarget.STREAMING,
                          settings: Optional[ProcessingSettings] = None) -> bool:
        """
        Process an audio file with the full mastering chain
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            loudness_target: Target loudness (float or LoudnessTarget enum)
            settings: Processing settings
            
        Returns:
            Success status
        """
        try:
            # Handle loudness target
            if isinstance(loudness_target, LoudnessTarget):
                target_lufs = loudness_target.value
            else:
                target_lufs = float(loudness_target)
            
            self.logger.info(f"🎵 Processing: {input_path} → {output_path}")
            self.logger.info(f"🎯 Target loudness: {target_lufs} LUFS")
            
            # Read input audio
            audio, sr = self.read_audio(input_path)
            self.logger.info(f"📊 Input: {audio.shape[1]/sr:.1f}s, {sr}Hz, {audio.shape[0]}ch")
            
            # Process audio
            processed = self.polish_full_mix(audio, sr, target_lufs, settings)
            
            # Write output
            self.write_audio(output_path, processed, sr)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process audio file: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def process_folder(self,
                      input_dir: Union[str, Path],
                      output_dir: Union[str, Path],
                      file_pattern: str = "*.wav",
                      loudness_target: Union[float, LoudnessTarget] = LoudnessTarget.STREAMING,
                      settings: Optional[ProcessingSettings] = None,
                      use_stems: bool = False) -> Dict[str, bool]:
        """
        Batch process all audio files in a folder
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path  
            file_pattern: File pattern to match (default: "*.wav")
            loudness_target: Target loudness
            settings: Processing settings
            use_stems: Whether to use stem separation
            
        Returns:
            Dictionary mapping filenames to success status
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all matching files
        pattern = str(input_path / file_pattern)
        files = glob.glob(pattern)
        
        if not files:
            self.logger.warning(f"No files found matching pattern: {pattern}")
            return {}
        
        self.logger.info(f"🎵 Processing {len(files)} files from {input_dir} → {output_dir}")
        
        results = {}
        for i, input_file in enumerate(files, 1):
            try:
                input_file_path = Path(input_file)
                output_file_path = output_path / f"polished_{input_file_path.name}"
                
                self.logger.info(f"📁 [{i}/{len(files)}] Processing: {input_file_path.name}")
                
                if use_stems:
                    # Not supported in simplified interface
                    success = self.process_audio_file(
                        input_file_path, output_file_path, loudness_target, settings
                    )
                else:
                    success = self.process_audio_file(
                        input_file_path, output_file_path, loudness_target, settings
                    )
                
                results[input_file_path.name] = success
                
                if success:
                    self.logger.info(f"✅ [{i}/{len(files)}] Completed: {input_file_path.name}")
                else:
                    self.logger.error(f"❌ [{i}/{len(files)}] Failed: {input_file_path.name}")
                    
            except Exception as e:
                self.logger.error(f"❌ [{i}/{len(files)}] Error processing {input_file}: {e}")
                results[Path(input_file).name] = False
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        self.logger.info(f"🎉 Batch processing complete: {successful}/{total} files processed successfully")
        
        return results


# Adaptive processing functions
def analyze_spectral_characteristics(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Analyze spectral characteristics to guide adaptive processing
    
    Args:
        audio: Audio data (channels, samples)
        sr: Sample rate
        
    Returns:
        Dictionary of spectral analysis results
    """
    try:
        # Convert to mono for analysis
        if audio.shape[0] > 1:
            mono = np.mean(audio, axis=0)
        else:
            mono = audio[0]
        
        # Simple spectral centroid estimation using FFT
        # Take middle section to avoid silence/fades
        start_idx = len(mono) // 4
        end_idx = 3 * len(mono) // 4
        segment = mono[start_idx:end_idx]
        
        # FFT analysis
        fft = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(len(segment), 1/sr)
        magnitude = np.abs(fft)
        
        # Calculate spectral centroid (weighted average frequency)
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            spectral_centroid = sr / 4  # Default fallback
        
        # High frequency energy ratio (above 8kHz)
        high_freq_mask = freqs > 8000
        high_freq_energy = np.sum(magnitude[high_freq_mask])
        total_energy = np.sum(magnitude)
        hf_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Brightness indicator (normalized spectral centroid)
        brightness = spectral_centroid / (sr / 2)  # Normalize to 0-1
        
        return {
            "spectral_centroid": spectral_centroid,
            "brightness": brightness,
            "high_freq_ratio": hf_ratio,
            "is_dark": brightness < 0.3,  # Threshold for "dark" content
            "is_bright": brightness > 0.7   # Threshold for "bright" content
        }
        
    except Exception:
        # Return neutral defaults if analysis fails
        return {
            "spectral_centroid": sr / 4,
            "brightness": 0.5,
            "high_freq_ratio": 0.1,
            "is_dark": False,
            "is_bright": False
        }


def create_adaptive_settings(spectral_analysis: Dict[str, float], 
                           base_settings: Optional[ProcessingSettings] = None) -> ProcessingSettings:
    """
    Create adaptive processing settings based on spectral analysis
    
    Args:
        spectral_analysis: Results from analyze_spectral_characteristics
        base_settings: Base settings to modify (uses defaults if None)
        
    Returns:
        Adapted processing settings
    """
    if base_settings is None:
        settings = ProcessingSettings()
    else:
        # Copy settings to avoid modifying original
        settings = copy.deepcopy(base_settings)
    
    brightness = spectral_analysis["brightness"]
    is_dark = spectral_analysis["is_dark"]
    is_bright = spectral_analysis["is_bright"]
    hf_ratio = spectral_analysis["high_freq_ratio"]
    
    # Adaptive adjustments
    if is_dark:
        # Content is already dark - reduce high frequency cuts
        settings.high_shelf_db = max(settings.high_shelf_db * 0.5, -0.5)  # Reduce cut
        settings.reverb_high_cut_db = max(settings.reverb_high_cut_db * 0.5, -1.0)
        settings.harsh_cut_db = max(settings.harsh_cut_db * 0.7, -1.0)  # Less harsh cut
        # Lighter de-essing since highs are already reduced
        settings.deesser_threshold += 2.0
        
    elif is_bright and hf_ratio > 0.15:
        # Content is very bright/harsh - increase processing
        settings.high_shelf_db = min(settings.high_shelf_db * 1.5, -3.0)  # More cut
        settings.harsh_cut_db = min(settings.harsh_cut_db * 1.3, -3.0)  # More harsh cut
        settings.deesser_threshold -= 2.0  # More aggressive de-essing
        settings.reverb_high_cut_db = min(settings.reverb_high_cut_db * 1.3, -3.0)
    
    # Adaptive reverb based on brightness
    if brightness > 0.6:
        # Bright content can handle a bit more reverb
        settings.reverb_wet_level = min(settings.reverb_wet_level * 1.2, 0.04)
    elif brightness < 0.4:
        # Dark content should stay completely dry to avoid "big room" effect
        settings.reverb_wet_level = 0.0
    
    return settings


# Stem separation functionality (requires Demucs)
class StemProcessor:
    """Advanced processor using stem separation for better results"""
    
    def __init__(self):
        """Initialize stem processor"""
        self.logger = self._setup_logging()
        
        if not DEMUCS_AVAILABLE:
            self.logger.warning("⚠️ Demucs not available. Install with: pip install demucs")
    
    def _setup_logging(self):
        """Setup basic logging"""
        logger = logging.getLogger(__name__ + ".stems")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def separate_stems(self, audio_path: Union[str, Path]) -> Optional[Dict[str, np.ndarray]]:
        """
        Separate audio into stems using Demucs
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Dictionary of stem names to audio arrays, or None if failed
        """
        if not DEMUCS_AVAILABLE:
            self.logger.error("Demucs not available for stem separation")
            return None
        
        try:
            self.logger.info(f"🎼 Separating stems from: {audio_path}")
            
            # Use Demucs API to separate stems
            separator = demucs.api.Separator(model="mdx_extra")
            origin, separated = separator.separate_audio_file(str(audio_path))
            
            stems = {}
            for stem_name, stem_audio in separated.items():
                # Convert to our format (channels, samples)
                if stem_audio.ndim == 2:
                    stems[stem_name] = stem_audio.T
                else:
                    stems[stem_name] = np.stack([stem_audio, stem_audio], axis=0)
            
            self.logger.info(f"✅ Separated {len(stems)} stems: {list(stems.keys())}")
            return stems
            
        except Exception as e:
            self.logger.error(f"Stem separation failed: {e}")
            return None
    
    def process_stems_individually(self,
                                  stems: Dict[str, np.ndarray],
                                  sr: int,
                                  processor: AudioProcessor) -> Dict[str, np.ndarray]:
        """
        Process each stem with customized settings and RMS balancing
        
        Args:
            stems: Dictionary of stem audio data
            sr: Sample rate
            processor: AudioProcessor instance
            
        Returns:
            Dictionary of processed stems
        """
        def rms(x):
            """Calculate RMS level"""
            return np.sqrt(np.mean(x**2))
        
        processed_stems = {}
        target_rms = 0.1  # Target RMS for loudness matching
        
        for stem_name, stem_audio in stems.items():
            try:
                self.logger.info(f"🔧 Processing {stem_name} stem...")
                
                # RMS balance before processing
                current_rms = rms(stem_audio)
                if current_rms > 1e-6:  # Avoid division by zero
                    rms_scale = min(1.0, target_rms / current_rms)
                    stem_audio = stem_audio * rms_scale
                    self.logger.info(f"  📊 RMS balanced: {current_rms:.4f} → {rms(stem_audio):.4f}")
                
                # Customize processing per stem type
                if stem_name == "vocals":
                    # Heavier de-essing and more reverb for vocals
                    settings = ProcessingSettings()
                    settings.deesser_threshold = -28.0  # More aggressive
                    settings.deesser_ratio = 5.0
                    settings.reverb_wet_level = 0.10   # More reverb
                    settings.harsh_cut_db = -2.5       # More harsh frequency cut
                    settings.post_deesser_enabled = True
                    
                elif stem_name == "drums":
                    # Preserve punch, minimal processing
                    settings = ProcessingSettings()
                    settings.comp_attack_ms = 5.0      # Faster attack
                    settings.comp_ratio = 1.5          # Lighter compression
                    settings.reverb_wet_level = 0.03   # Minimal reverb
                    settings.post_deesser_enabled = False  # No de-essing for drums
                    
                elif stem_name in ["bass", "other"]:
                    # Conservative processing for bass and other
                    settings = ProcessingSettings()
                    settings.highpass_freq = 40.0      # Keep more low end
                    settings.reverb_wet_level = 0.04   # Light reverb
                    settings.deesser_threshold = -20.0  # Lighter de-essing
                    
                else:
                    # Default settings for unknown stems
                    settings = ProcessingSettings()
                
                # Process the stem with pre-normalize to -20 LUFS for headroom
                processed = processor.polish_full_mix(stem_audio, sr, -20.0, settings)
                processed_stems[stem_name] = processed
                
            except Exception as e:
                self.logger.error(f"Failed to process {stem_name} stem: {e}")
                # Use original if processing fails
                processed_stems[stem_name] = stem_audio
        
        return processed_stems
    
    def mix_stems(self, stems: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Mix processed stems back together with proper headroom management
        
        Args:
            stems: Dictionary of processed stem audio
            
        Returns:
            Mixed audio with proper headroom
        """
        def rms(x):
            """Calculate RMS level"""
            return np.sqrt(np.mean(x**2))
        
        try:
            # RMS balance and mix stems
            target_rms = 0.1
            mixed = None
            
            for stem_name, stem_audio in stems.items():
                # RMS balance each stem before mixing
                current_rms = rms(stem_audio)
                if current_rms > 1e-6:
                    scale = min(1.0, target_rms / current_rms)
                    stem_audio_scaled = stem_audio * scale
                else:
                    stem_audio_scaled = stem_audio
                    scale = 1.0
                
                # Add to mix
                if mixed is None:
                    mixed = stem_audio_scaled.copy()
                else:
                    mixed += stem_audio_scaled
                
                self.logger.info(f"➕ {stem_name}: RMS {current_rms:.4f} → scaled {scale:.2f}")
            
            if mixed is None:
                raise ValueError("No stems to mix")
            
            # Apply bus headroom trim before final limiting
            mixed *= 0.8  # -1.94 dB headroom
            self.logger.info("🔊 Applied bus headroom trim (-1.94 dB)")
            
            self.logger.info("🎵 Stems mixed successfully with headroom management")
            return mixed
            
        except Exception as e:
            self.logger.error(f"Failed to mix stems: {e}")
            # Return first stem as fallback
            return next(iter(stems.values()))


# Convenience functions for automated processing
def polish_audio_bytes(audio_bytes: bytes, 
                      target_lufs: float = -14.0) -> Optional[bytes]:
    """
    Simple automated function to polish audio bytes with professional mastering
    
    Args:
        audio_bytes: Input audio data as bytes
        target_lufs: Target loudness (default: -14.0 for streaming)
        
    Returns:
        Polished audio bytes or None if processing failed
    """
    processor = AudioProcessor()
    
    try:
        # Write input bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
            temp_input.write(audio_bytes)
            temp_input_path = temp_input.name
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        try:
            # Read audio
            audio, sr = processor.read_audio(temp_input_path)
            
            # Analyze spectral characteristics for adaptive processing
            spectral_analysis = analyze_spectral_characteristics(audio, sr)
            
            # Create adaptive settings automatically
            settings = create_adaptive_settings(spectral_analysis)
            settings.target_lufs = target_lufs
            
            # Process audio with adaptive settings
            processed = processor.polish_full_mix(audio, sr, target_lufs, settings)
            
            # Write to temporary output
            processor.write_audio(temp_output_path, processed, sr, bit_depth=24)
            
            # Read processed audio as bytes
            with open(temp_output_path, 'rb') as f:
                result_bytes = f.read()
            
            return result_bytes
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
            except:
                pass
                
    except Exception as e:
        processor.logger.error(f"Audio processing failed: {e}")
        return None
