import os
import tempfile
import requests
import shutil
from pathlib import Path
from typing import List, Optional, Union
import logging
import warnings
import io
from urllib.parse import urlparse
from contextlib import redirect_stderr

# Filter MoviePy frame corruption warnings - they're handled gracefully anyway
warnings.filterwarnings("ignore", message=".*bytes wanted but 0 bytes read.*")
warnings.filterwarnings("ignore", message=".*Using the last valid frame instead.*")

try:
    from moviepy import (
        VideoFileClip, concatenate_videoclips, 
        CompositeVideoClip, ColorClip, vfx, afx, CompositeAudioClip
    )
    import numpy as np
    MOVIEPY_AVAILABLE = True
except ImportError as e:
    print(f"❌ MoviePy import error: {e}")
    print("Please install it with: python3 -m pip install --break-system-packages moviepy")
    MOVIEPY_AVAILABLE = False


def get_video_logger(name: str = "VideoStitcher") -> logging.Logger:
    """Create or fetch a configured logger for video processing components.

    - Ensures a single stream handler with a simple format.
    - Defaults to INFO level.
    """
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
        # Avoid duplicate logs if root logger has handlers
        log.propagate = False

    return log


class VideoStitcher:
    """
    A class to handle downloading and stitching videos together.
    """
    
    # Audio mixing constants for consistent crossfades
    CROSSFADE_GAIN = 0.707  # 1/sqrt(2) - preserves power during overlap
    
    def __init__(self, output_dir: str = "output", temp_dir: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Use consistent storage mechanism - prefer video_dump over random temp
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            self.temp_dir = self._get_video_dump_path()
        self.temp_dir.mkdir(exist_ok=True)
        
        # Setup logging using the project's logging pattern
        self.logger = get_video_logger(f"VideoStitcher.{id(self)}")
        
        # Session for downloads with proper cleanup
        self._session = None
    
    @property
    def session(self):
        """Get or create requests session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        return self._session
    
    def download_video(self, url: str, filename: str) -> Optional[str]:
        """
        Download a video from URL to local file.
        
        Args:
            url: The video URL to download
            filename: Local filename to save as
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            self.logger.info(f"🔽 Downloading video from: {url}")
            
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Intelligent content type validation
            content_type = response.headers.get('content-type', '').lower()
            file_extension = Path(filename).suffix.lower()
            
            # Check if this looks like a video file based on extension
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
            is_video_extension = file_extension in video_extensions
            
            if content_type and not content_type.startswith('video/') and not content_type.startswith('application/octet-stream'):
                if is_video_extension:
                    # File has video extension but wrong content type - likely server misconfiguration
                    self.logger.debug(f"🔧 Server reports '{content_type}' but filename suggests video - proceeding")
                else:
                    # Both content type and extension suggest this is not a video
                    self.logger.warning(f"⚠️ Unexpected content type '{content_type}' for non-video file")
            elif not content_type and is_video_extension:
                self.logger.debug(f"📁 No content type provided but filename suggests video - proceeding")
            
            file_path = self.temp_dir / filename
            total_size = int(response.headers.get('content-length', 0))
            
            # Add file size guard (default 500MB max)
            max_size = 500 * 1024 * 1024
            if total_size > max_size:
                raise ValueError(f"File too large: {total_size / (1024*1024):.1f}MB > {max_size / (1024*1024):.1f}MB")
            
            # Quick content validation to catch error pages
            first_chunk = None
            content_is_valid = True
            
            with open(file_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        # Check first chunk for obvious non-video content
                        if first_chunk is None:
                            first_chunk = chunk
                            content_preview = chunk[:100].decode('utf-8', errors='ignore').strip().lower()
                            if any(marker in content_preview for marker in ['<html', '<!doctype', '<?xml', 'error:', 'not found', 'access denied']):
                                content_is_valid = False
                                self.logger.error(f"❌ Downloaded content appears to be HTML/text, not video: {content_preview[:50]}...")
                                break
                        
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Show progress for large files
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # Every MB
                                self.logger.info(f"📥 Progress: {percent:.1f}%")
            
            # If content validation failed, clean up and return None
            if not content_is_valid:
                if file_path.exists():
                    file_path.unlink()
                return None
            
            # Validate downloaded file is actually a video and attempt repair if needed
            if MOVIEPY_AVAILABLE:
                try:
                    # Test load the video with error handling
                    test_clip = self._load_video_robustly(str(file_path))
                    if test_clip:
                        test_clip.close()
                        self.logger.debug(f"✓ Video validation passed for {filename}")
                    else:
                        self.logger.error(f"❌ Could not load video file {filename}")
                        if file_path.exists():
                            file_path.unlink()
                        return None
                except Exception as e:
                    self.logger.error(f"❌ Invalid video file {filename}: {e}")
                    if file_path.exists():
                        file_path.unlink()
                    return None
            
            self.logger.info(f"✅ Downloaded: {filename} ({downloaded / (1024*1024):.1f} MB)")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to download {url}: {e}")
            return None
    
    def stitch_videos(self, video_urls: List[str], output_filename: str = "stitched_video.mp4", midtrimm: int = 0) -> str:
        """
        Download and stitch multiple videos together.
        
        Args:
            video_urls: List of video URLs to stitch together
            output_filename: Name of the output file
            midtrimm: Milliseconds to trim from the end of each video (except the last one).
                     This creates smoother transitions between videos.
            
        Returns:
            Path to the stitched video file
        """
        if not video_urls:
            raise ValueError("No video URLs provided")
        
        self.logger.info(f"🎬 Starting video stitching process for {len(video_urls)} videos")
        if midtrimm > 0:
            self.logger.info(f"✂️ Mid-trim enabled: {midtrimm}ms will be trimmed from end of each video (except last)")
        
        # Download all videos
        downloaded_files = []
        for i, url in enumerate(video_urls):
            # Extract filename from URL or generate one
            parsed_url = urlparse(url)
            original_name = Path(parsed_url.path).name
            if not original_name or not original_name.endswith('.mp4'):
                original_name = f"video_{i+1}.mp4"
            
            downloaded_file = self.download_video(url, original_name)
            if downloaded_file:
                downloaded_files.append(downloaded_file)
            else:
                self.logger.warning(f"⚠️ Skipping failed download: {url}")
        
        if not downloaded_files:
            raise Exception("No videos were successfully downloaded")
        
        self.logger.info(f"📹 Successfully downloaded {len(downloaded_files)} videos, starting stitching...")
        
        # Load video clips with robust error handling
        video_clips = []
        try:
            for i, file_path in enumerate(downloaded_files):
                self.logger.info(f"🎞️ Loading video: {Path(file_path).name}")
                clip = self._load_video_robustly(file_path)
                
                if clip is None:
                    self.logger.error(f"❌ Failed to load video: {Path(file_path).name}")
                    continue
                
                # Apply mid-trim to all videos except the last one
                if midtrimm > 0 and i < len(downloaded_files) - 1:
                    trim_seconds = midtrimm / 1000.0  # Convert ms to seconds
                    if clip.duration > trim_seconds:
                        original_duration = clip.duration
                        clip = clip.subclipped(0, clip.duration - trim_seconds)
                        self.logger.info(f"   ✂️ Trimmed {midtrimm}ms from end: {original_duration:.2f}s → {clip.duration:.2f}s")
                    else:
                        self.logger.warning(f"   ⚠️ Cannot trim {midtrimm}ms from {clip.duration:.2f}s video - too short!")
                
                video_clips.append(clip)
                self.logger.info(f"   Duration: {clip.duration:.2f}s, Resolution: {clip.w}x{clip.h}")
            
            # Concatenate videos
            self.logger.info("🔗 Stitching videos together...")
            final_video = concatenate_videoclips(video_clips, method='compose')
            
            # Save final video
            output_path = self.output_dir / output_filename
            self.logger.info(f"💾 Saving stitched video to: {output_path}")
            
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                logger=None  # Suppress moviepy logging
            )
            
            # Clean up
            for clip in video_clips:
                clip.close()
            final_video.close()
            
            # Calculate final stats
            final_duration = sum(clip.duration for clip in video_clips)
            self.logger.info(f"✅ Video stitching complete!")
            self.logger.info(f"   Final video: {output_path}")
            self.logger.info(f"   Total duration: {final_duration:.2f}s")
            if midtrimm > 0:
                total_trimmed = (len(downloaded_files) - 1) * (midtrimm / 1000.0)
                self.logger.info(f"   Total time trimmed: {total_trimmed:.2f}s ({midtrimm}ms × {len(downloaded_files) - 1} videos)")
            self.logger.info(f"   File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
            
            return str(output_path)
            
        finally:
            # Clean up temporary files
            self.cleanup_temp_files(downloaded_files)
    
    def _load_video_robustly(self, file_path: str, max_retries: int = 3) -> Optional[VideoFileClip]:
        """
        Load video with robust error handling for corrupted files.
        
        Args:
            file_path: Path to video file
            max_retries: Maximum number of loading attempts
            
        Returns:
            VideoFileClip or None if loading fails
        """
        if not MOVIEPY_AVAILABLE:
            raise Exception("MoviePy is required for video processing")
        

        
        for attempt in range(max_retries):
            try:
                # Capture stderr to detect and handle frame reading warnings
                stderr_buffer = io.StringIO()
                
                with warnings.catch_warnings():
                    # Filter out known MoviePy warnings about corrupted frames
                    warnings.filterwarnings("ignore", message=".*bytes wanted but 0 bytes read.*")
                    warnings.filterwarnings("ignore", message=".*Using the last valid frame instead.*")
                    
                    with redirect_stderr(stderr_buffer):
                        # Load video (removed verbose parameter for MoviePy 2.x compatibility)
                        clip = VideoFileClip(file_path)
                        
                        # Test basic properties to ensure the clip is usable
                        duration = clip.duration
                        fps = clip.fps
                        size = clip.size
                        
                        # Test reading frames to catch corruption early - focus on end areas
                        try:
                            # Test beginning frame
                            if duration > 0:
                                frame_1 = clip.get_frame(0.1)
                            
                            # Test middle frame if video is long enough
                            if duration > 1.0:
                                frame_mid = clip.get_frame(duration / 2)
                            
                            # Test multiple points near the end where corruption typically occurs
                            if duration > 0.5:
                                # Calculate test points based on frame count to target specific corruption areas
                                # We know corruption often happens around frame 53-54 out of 57 total
                                test_points = [
                                    max(0.1, duration - 0.5),    # 500ms before end
                                    max(0.1, duration - 0.3),    # 300ms before end  
                                    max(0.1, duration - 0.21),   # 210ms before end (~3.31s for 3.52s video)
                                    max(0.1, duration - 0.14),   # 140ms before end (~3.38s for 3.52s video)
                                    max(0.1, duration - 0.1),    # 100ms before end
                                    max(0.1, duration - 0.05),   # 50ms before end
                                ]
                                
                                corruption_detected = False
                                for test_time in test_points:
                                    try:
                                        frame_test = clip.get_frame(test_time)
                                    except Exception as frame_err:
                                        self.logger.debug(f"🚨 Frame corruption detected at {test_time:.2f}s: {frame_err}")
                                        corruption_detected = True
                                        break
                                
                                # If corruption detected, close current clip and create safe version
                                if corruption_detected:
                                    clip.close()
                                    self.logger.warning(f"⚠️ Frame corruption detected, creating safe clip for {file_path}")
                                    return self._create_safe_clip(file_path)
                            
                            self.logger.debug(f"✓ Video loaded successfully: {duration:.2f}s, {fps:.1f}fps, {size[0]}x{size[1]}")
                            return clip
                            
                        except Exception as frame_error:
                            self.logger.warning(f"⚠️ Frame reading issues in {file_path} (attempt {attempt + 1}): {frame_error}")
                            clip.close()
                            
                            # If this is not the last attempt, try again
                            if attempt < max_retries - 1:
                                continue
                            else:
                                # Last attempt - try to create a more conservative clip
                                return self._create_safe_clip(file_path)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Video loading attempt {attempt + 1} failed for {file_path}: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    self.logger.error(f"❌ Failed to load video after {max_retries} attempts: {file_path}")
                    return None
        
        return None
    
    def _create_safe_clip(self, file_path: str) -> Optional[VideoFileClip]:
        """
        Create a safe video clip that handles corruption by detecting and avoiding corrupted frames.
        
        Args:
            file_path: Path to video file
            
        Returns:
            VideoFileClip or None if creation fails
        """
        try:

            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*bytes wanted but 0 bytes read.*")
                warnings.filterwarnings("ignore", message=".*Using the last valid frame instead.*")
                
                # Load the clip with basic parameters (removed verbose for MoviePy 2.x compatibility)
                full_clip = VideoFileClip(file_path)
                original_duration = full_clip.duration
                
                # Detect corrupted frames by testing read operations
                corrupted_time = None
                test_increment = 0.1  # Test every 100ms from the end backwards
                
                # Start from 90% and work backwards to find corruption point
                for test_time in [original_duration * 0.9, original_duration * 0.85, original_duration * 0.8, 
                                 original_duration * 0.75, original_duration * 0.7]:
                    try:
                        if test_time > 0:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", message=".*bytes wanted but 0 bytes read.*")
                                frame = full_clip.get_frame(test_time)
                            # If we get here, this time point is safe
                            safe_duration = test_time
                            break
                    except Exception as e:
                        self.logger.debug(f"🚨 Corruption detected at {test_time:.2f}s")
                        continue
                else:
                    # If all test points failed, use 50% of the video
                    safe_duration = original_duration * 0.5
                    self.logger.warning(f"⚠️ Extensive corruption detected, using 50% of video")
                
                # Create safe clip
                safe_clip = full_clip.subclipped(0, safe_duration)
                
                # Close the original clip to free memory
                full_clip.close()
                
                self.logger.info(f"🛠️ Created safe clip: trimmed from {original_duration:.2f}s to {safe_duration:.2f}s")
                return safe_clip
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create safe clip for {file_path}: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # TRANSITION METHODS - Advanced video stitching with cinematic transitions
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def stitch_with_transition(self, video_urls: List[str], transition_types: List[str], return_bytes: bool = False) -> Union[bytes, str]:
        """
        Main entry point for stitching videos with transitions.
        
        Args:
            video_urls: List of video URLs to stitch
            transition_types: List of transition types (enum values) - should be len(video_urls) - 1
            return_bytes: If True, return video as bytes; if False, return file path string
            
        Returns:
            Final stitched video as bytes or file path string
        """
        if not video_urls:
            raise ValueError("No video URLs provided")
        
        # Validate transition count
        expected_transitions = len(video_urls) - 1
        if len(transition_types) != expected_transitions:
            raise ValueError(f"Expected {expected_transitions} transitions for {len(video_urls)} videos, got {len(transition_types)}")
        
        if not MOVIEPY_AVAILABLE:
            raise Exception("MoviePy is required for advanced video transitions")
        
        self.logger.info(f"🎬 Starting video stitching with transitions for {len(video_urls)} videos")
        
        # Download all videos
        downloaded_files = []
        for i, url in enumerate(video_urls):
            parsed_url = urlparse(url)
            original_name = Path(parsed_url.path).name
            if not original_name or not original_name.endswith('.mp4'):
                original_name = f"video_{i+1}.mp4"
            
            downloaded_file = self.download_video(url, original_name)
            if downloaded_file:
                downloaded_files.append(downloaded_file)
            else:
                self.logger.warning(f"⚠️ Skipping failed download: {url}")
        
        if not downloaded_files:
            raise Exception("No videos were successfully downloaded")
        
        # Load video clips with robust error handling
        video_clips = []
        intermediate_clips = []  # Track intermediate composites for cleanup
        try:
            for file_path in downloaded_files:
                clip = self._load_video_robustly(file_path)
                if clip is None:
                    self.logger.error(f"❌ Failed to load video: {Path(file_path).name}")
                    continue
                video_clips.append(clip)
                self.logger.info(f"🎞️ Loaded: {Path(file_path).name} ({clip.duration:.2f}s)")
            
            # Process videos with transitions
            result_clips = []
            
            for i, clip in enumerate(video_clips):
                if i == 0:
                    # First video - no transition applied
                    result_clips.append(clip)
                else:
                    # Apply transition between previous and current video
                    transition_type = transition_types[i - 1]
                    prev_clip = result_clips[-1]
                    
                    self.logger.info(f"🔄 Applying [TRANSITION:{transition_type}] between video {i} and {i+1}")
                    
                    # Call appropriate transition function with default values
                    if transition_type == "Cut" or transition_type == "Continuous":
                        combined_clip = self._apply_cut_transition([prev_clip, clip], midtrimm=100)
                    elif transition_type == "Dissolve":
                        combined_clip = self._apply_dissolve_transition([prev_clip, clip], duration=0.5)
                    elif transition_type == "FadeBlack":
                        combined_clip = self._apply_fade_to_black_transition([prev_clip, clip], duration=0.5)
                    elif transition_type == "Slide":
                        combined_clip = self._apply_slide_transition([prev_clip, clip], duration=0.5, direction="left")
                    elif transition_type == "WhipPan":
                        combined_clip = self._apply_whip_pan_transition([prev_clip, clip], blur_intensity=15.0)
                    elif transition_type == "BlurDissolve":
                        combined_clip = self._apply_blur_dissolve_transition([prev_clip, clip], duration=0.5, blur_radius=10.0)
                    elif transition_type == "Glitch":
                        combined_clip = self._apply_glitch_transition([prev_clip, clip], duration=0.2)
                    else:
                        self.logger.warning(f"Unknown transition type: {transition_type}, using Cut")
                        combined_clip = self._apply_cut_transition([prev_clip, clip], midtrimm=100)
                    
                    # Track old clip for cleanup and replace with combined result
                    if result_clips and (result_clips[-1] not in intermediate_clips):
                        intermediate_clips.append(result_clips[-1])
                    result_clips[-1] = combined_clip
            
            # Final concatenation if we have multiple result clips
            if len(result_clips) > 1:
                final_video = concatenate_videoclips(result_clips, method='compose')
            else:
                final_video = result_clips[0]
            
            # Save to temporary file and return bytes or path
            output_path = self.output_dir / "transition_output.mp4"
            self.logger.info(f"💾 Saving final video to: {output_path}")
            
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                logger='bar'
            )
            
            if return_bytes:
                # Read as bytes if explicitly requested
                with open(output_path, 'rb') as f:
                    result_bytes = f.read()
                
                # Memory management warning for large files
                if len(result_bytes) > 100 * 1024 * 1024:  # 100MB threshold
                    self.logger.warning(f"Large file ({len(result_bytes) / (1024*1024):.1f} MB) returned as bytes")
                
                self.logger.info(f"✅ Video stitching with transitions complete!")
                self.logger.info(f"   Output size: {len(result_bytes)} bytes ({len(result_bytes) / (1024*1024):.1f} MB)")
                
                return result_bytes
            else:
                # Return path for large files (safer default)
                file_size = output_path.stat().st_size
                self.logger.info(f"✅ Video stitching with transitions complete!")
                self.logger.info(f"   Output file: {output_path}")
                self.logger.info(f"   File size: {file_size / (1024*1024):.1f} MB")
                
                return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error during video processing: {e}")
            raise
        finally:
            # Clean up video clips - only close original clips, not composites
            self._safe_cleanup_clips(video_clips)
            # Clean up intermediate composites
            self._safe_cleanup_clips(intermediate_clips)
            # Clean up final video only after writing
            if 'final_video' in locals():
                self._safe_cleanup_clips([final_video])
            # Clean up temp files
            self.cleanup_temp_files(downloaded_files)
    
    def _normalize_clip(self, clip: VideoFileClip, target_size=None, target_fps=None, audio_rate=None) -> VideoFileClip:
        """
        Normalize clip to consistent geometry/timing for cleaner processing.
        
        Args:
            clip: Input video clip
            target_size: Target resolution as (width, height) tuple
            target_fps: Target frame rate  
            audio_rate: Target audio sample rate
            
        Returns:
            Normalized video clip
        """
        normalized = clip
        
        if target_size:
            normalized = normalized.resize(target_size)
            
        if target_fps:
            normalized = normalized.set_fps(target_fps)
            
        if audio_rate and normalized.audio:
            normalized = normalized.set_audio(normalized.audio.set_fps(audio_rate))
            
        return normalized
    
    def _apply_cut_transition(self, clips: List[VideoFileClip], midtrimm: int = 100) -> VideoFileClip:
        """
        CUT: Simple concatenation with optional midtrimm (similar to stitch_videos).
        
        Args:
            clips: List of video clips to concatenate
            midtrimm: Milliseconds to trim from end of each video (except last)
        """
        if len(clips) == 1:
            return clips[0]
        
        trimmed_clips = []
        for i, clip in enumerate(clips):
            if midtrimm > 0 and i < len(clips) - 1:
                # Apply mid-trim to all clips except the last one
                trim_seconds = midtrimm / 1000.0
                if clip.duration > trim_seconds:
                    trimmed_clip = clip.subclipped(0, clip.duration - trim_seconds)
                    self.logger.info(f"   [CUT] ✂️ Trimmed {midtrimm}ms from clip {i+1}")
                else:
                    trimmed_clip = clip
                    self.logger.warning(f"   [CUT] ⚠️ Cannot trim {midtrimm}ms from {clip.duration:.2f}s clip {i+1}")
            else:
                trimmed_clip = clip
            
            trimmed_clips.append(trimmed_clip)
        
        return concatenate_videoclips(trimmed_clips, method='compose')
    
    def _apply_dissolve_transition(self, clips: List[VideoFileClip], duration: float = 0.5) -> VideoFileClip:
        """DISSOLVE: Crossfade between clips with proper audio handling."""
        if len(clips) == 1:
            return clips[0]
        
        result_clips = [clips[0]]
        
        for i in range(1, len(clips)):
            prev_clip = result_clips[-1]
            current_clip = clips[i]
            
            # Validate and clamp duration
            effective_duration = min(
                duration, 
                prev_clip.duration * 0.5, 
                current_clip.duration * 0.5,
                prev_clip.duration,
                current_clip.duration
            )
            
            if effective_duration != duration:
                self.logger.warning(f"[DISSOLVE] Duration clamped from {duration:.2f}s to {effective_duration:.2f}s")
            
            overlap_start = prev_clip.duration - effective_duration
            if overlap_start < 0:
                self.logger.warning(f"[DISSOLVE] Negative overlap detected, adjusting duration")
                effective_duration = prev_clip.duration
                overlap_start = 0
            
            # Create video crossfade using fx for compatibility
            fade_out = prev_clip.fx(vfx.fadeout, effective_duration)
            fade_in = current_clip.fx(vfx.fadein, effective_duration).set_start(overlap_start)
            
            # Create audio crossfade with sequential timing to avoid phase interference
            audio_parts = []
            if prev_clip.audio:
                # Previous audio: fade out and end before the second audio starts
                fade_duration = min(effective_duration * 0.5, 0.2)  # Shorter fade
                audio_end = overlap_start + (effective_duration * 0.5)  # End halfway through transition
                prev_audio = (prev_clip.audio
                            .set_start(0)
                            .set_end(audio_end)
                            .fx(afx.audio_fadeout, fade_duration))
                audio_parts.append(prev_audio)
            
            if current_clip.audio:
                # Current audio: start after previous audio ends to avoid overlap
                fade_duration = min(effective_duration * 0.5, 0.2)  # Shorter fade
                audio_start = overlap_start + (effective_duration * 0.5)  # Start halfway through transition
                current_audio = (current_clip.audio
                               .set_start(audio_start)
                               .fx(afx.audio_fadein, fade_duration))
                audio_parts.append(current_audio)
            
            # Composite video and audio
            composite_video = CompositeVideoClip([fade_out, fade_in])
            if audio_parts:
                composite_audio = CompositeAudioClip(audio_parts)
                composite_video = composite_video.set_audio(composite_audio)
            
            result_clips[-1] = composite_video
        
        return concatenate_videoclips(result_clips, method='compose')
    
    def _apply_fade_to_black_transition(self, clips: List[VideoFileClip], duration: float = 0.5) -> VideoFileClip:
        """FADE_TO_BLACK: Fade to black between clips with audio handling."""
        if len(clips) == 1:
            return clips[0]
        
        result_clips = []
        
        for i, clip in enumerate(clips):
            # Clamp duration to clip length
            effective_duration = min(duration, clip.duration * 0.3)
            
            if i == 0:
                # First clip: fade out video and audio using fx for compatibility
                video_part = clip.fx(vfx.fadeout, effective_duration)
                if clip.audio:
                    video_part = video_part.set_audio(clip.audio.fx(afx.audio_fadeout, effective_duration))
                result_clips.append(video_part)
            elif i == len(clips) - 1:
                # Last clip: fade in video and audio using fx for compatibility
                video_part = clip.fx(vfx.fadein, effective_duration)
                if clip.audio:
                    video_part = video_part.set_audio(clip.audio.fx(afx.audio_fadein, effective_duration))
                result_clips.append(video_part)
            else:
                # Middle clips: fade in and out using fx for compatibility
                video_part = clip.fx(vfx.fadein, effective_duration).fx(vfx.fadeout, effective_duration)
                if clip.audio:
                    audio_part = clip.audio.fx(afx.audio_fadein, effective_duration).fx(afx.audio_fadeout, effective_duration)
                    video_part = video_part.set_audio(audio_part)
                result_clips.append(video_part)
            
            # Add short black frame between clips (except after last)
            if i < len(clips) - 1:
                # Short black clip to reduce dead air
                black_duration = min(0.1, effective_duration * 0.2)
                black_clip = ColorClip(size=clip.size, color=(0, 0, 0), duration=black_duration)
                result_clips.append(black_clip)
        
        return concatenate_videoclips(result_clips, method='compose')
    
    
    
    def _apply_slide_transition(self, clips: List[VideoFileClip], duration: float = 0.5, direction: str = "left") -> VideoFileClip:
        """SLIDE: Smooth directional movement with proper timing and audio sync."""
        if len(clips) == 1:
            return clips[0]
        
        # Simple two-clip slide transition
        first_clip = clips[0]
        second_clip = clips[1]
        
        # Conservative transition duration
        transition_duration = min(
            duration,
            first_clip.duration * 0.25,  # Max 25% of first clip
            second_clip.duration * 0.25,  # Max 25% of second clip
            0.8  # Max 0.8 seconds
        )
        
        if transition_duration != duration:
            self.logger.warning(f"[SLIDE] Duration clamped from {duration:.2f}s to {transition_duration:.2f}s")
        
        # Calculate timing
        transition_start = first_clip.duration - transition_duration
        total_duration = first_clip.duration + second_clip.duration - transition_duration
        
        # Create video segments
        # First clip plays normally until transition
        first_main = first_clip.subclipped(0, transition_start) if transition_start > 0 else None
        
        # During transition: both clips visible with slide effect
        first_ending = first_clip.subclipped(transition_start) if transition_start >= 0 else first_clip
        second_beginning = second_clip.subclipped(0, transition_duration)
        second_remaining = second_clip.subclipped(transition_duration) if transition_duration < second_clip.duration else None
        
        w, h = first_clip.size
        
        # Simple slide position for second clip during transition
        def slide_in_position(t):
            # t is relative to the transition start
            progress = min(t / transition_duration, 1.0) if transition_duration > 0 else 1.0
            # Smooth easing
            progress = progress * progress * (3.0 - 2.0 * progress)
            
            if direction == "left":
                x = -w + (w * progress)  # Slide in from left
            elif direction == "right":  
                x = w - (w * progress)   # Slide in from right
            elif direction == "up":
                return (0, -h + (h * progress))  # Slide in from top
            elif direction == "down":
                return (0, h - (h * progress))   # Slide in from bottom
            else:
                x = -w + (w * progress)  # Default to left
                
            return (x, 0)
        
        # Position the sliding clip
        second_sliding = second_beginning.set_position(slide_in_position)
        
        # Create transition segment with both clips
        transition_segment = CompositeVideoClip(
            [first_ending, second_sliding],
            size=first_clip.size
        ).set_duration(transition_duration)
        
        # Audio handling - sequential, no overlap to avoid blips
        audio_parts = []
        current_time = 0
        
        # First clip audio (full duration until transition end)
        if first_clip.audio:
            # Audio plays normally, then fades out during transition
            first_audio_duration = transition_start + transition_duration
            first_audio = (first_clip.audio
                          .subclipped(0, first_audio_duration)
                          .fx(afx.audio_fadeout, transition_duration * 0.5)  # Gentle fadeout
                          .set_start(0))
            audio_parts.append(first_audio)
        
        # Second clip audio starts after transition begins with overlap
        if second_clip.audio:
            # Start audio halfway through transition to avoid blip
            audio_start_time = transition_start + (transition_duration * 0.5)
            second_audio = (second_clip.audio
                           .fx(afx.audio_fadein, transition_duration * 0.5)  # Gentle fadein
                           .set_start(audio_start_time))
            audio_parts.append(second_audio)
        
        # Combine all segments
        video_segments = []
        if first_main:
            video_segments.append(first_main)
        video_segments.append(transition_segment)
        if second_remaining:
            video_segments.append(second_remaining)
        
        # Create final video
        final_video = concatenate_videoclips(video_segments, method='compose')
        
        # Apply audio
        if audio_parts:
            final_audio = CompositeAudioClip(audio_parts)
            final_video = final_video.set_audio(final_audio)
        
        return final_video
    
    def _apply_whip_pan_transition(self, clips: List[VideoFileClip], blur_intensity: float = 15.0) -> VideoFileClip:
        """WHIP_PAN: Fast motion blur with directional slide for authentic whip pan effect."""
        if len(clips) == 1:
            return clips[0]
        
        # Simple two-clip whip pan transition
        first_clip = clips[0]
        second_clip = clips[1]
        
        # Very short, intense whip duration
        whip_duration = min(0.1, first_clip.duration * 0.1, second_clip.duration * 0.1)
        
        if whip_duration != 0.1:
            self.logger.warning(f"[WHIP_PAN] Duration clamped to {whip_duration:.2f}s")
        
        # Calculate segments
        transition_start = first_clip.duration - whip_duration
        w, h = first_clip.size
        
        # Create segments  
        first_main = first_clip.subclipped(0, transition_start) if transition_start > 0 else None
        first_whip = first_clip.subclipped(transition_start) if transition_start >= 0 else first_clip
        second_whip = second_clip.subclipped(0, whip_duration)
        second_main = second_clip.subclipped(whip_duration) if whip_duration < second_clip.duration else None
        
        # Simple whip effect: fast horizontal motion with slight darkening
        def whip_out_position(t):
            progress = min(t / whip_duration, 1.0) if whip_duration > 0 else 1.0
            # Fast motion to the right
            x_offset = w * progress * 0.5  # Move 50% of width
            return (x_offset, 0)
        
        def whip_in_position(t):
            progress = min(t / whip_duration, 1.0) if whip_duration > 0 else 1.0
            # Fast motion from left to center
            x_offset = -w * (1 - progress) * 0.5  # Start 50% left, move to center
            return (x_offset, 0)
        
        # Apply whip effects with simple darkening (no complex speed effects)
        first_whip_effect = (first_whip
                            .set_position(whip_out_position)
                            .fx(vfx.colorx, 0.6))  # Darken for motion blur effect
        
        second_whip_effect = (second_whip
                             .set_position(whip_in_position)
                             .fx(vfx.colorx, 0.6))  # Darken for motion blur effect
        
        # Create transition segment with overlapping whip effects
        transition_segment = CompositeVideoClip(
            [first_whip_effect, second_whip_effect],
            size=(w, h)
        ).set_duration(whip_duration)
        
        # Audio handling - sequential to avoid blips
        audio_parts = []
        
        if first_clip.audio:
            # First audio fades out quickly during whip
            first_audio_duration = transition_start + (whip_duration * 0.5)
            first_audio = (first_clip.audio
                          .subclipped(0, first_audio_duration)
                          .fx(afx.audio_fadeout, whip_duration * 0.5)
                          .set_start(0))
            audio_parts.append(first_audio)
        
        if second_clip.audio:
            # Second audio starts halfway through whip with quick fade in
            second_audio_start = transition_start + (whip_duration * 0.5)
            second_audio = (second_clip.audio
                           .fx(afx.audio_fadein, whip_duration * 0.5)
                           .set_start(second_audio_start))
            audio_parts.append(second_audio)
        
        # Combine all segments
        video_segments = []
        if first_main:
            video_segments.append(first_main)
        video_segments.append(transition_segment)
        if second_main:
            video_segments.append(second_main)
        
        # Create final video
        final_video = concatenate_videoclips(video_segments, method='compose')
        
        # Apply audio
        if audio_parts:
            final_audio = CompositeAudioClip(audio_parts)
            final_video = final_video.set_audio(final_audio)
        
        return final_video
    
    def _apply_blur_dissolve_transition(self, clips: List[VideoFileClip], duration: float = 0.5, blur_radius: float = 10.0) -> VideoFileClip:
        """BLUR_DISSOLVE: Both clips blur and crossfade with audio handling."""
        if len(clips) == 1:
            return clips[0]
        
        result_clips = [clips[0]]
        
        for i in range(1, len(clips)):
            prev_clip = result_clips[-1]
            current_clip = clips[i]
            
            # Validate and clamp duration
            effective_duration = min(
                duration, 
                prev_clip.duration * 0.5, 
                current_clip.duration * 0.5,
                prev_clip.duration,
                current_clip.duration
            )
            
            if effective_duration != duration:
                self.logger.warning(f"[BLUR-DISSOLVE] Duration clamped from {duration:.2f}s to {effective_duration:.2f}s")
            
            overlap_start = prev_clip.duration - effective_duration
            if overlap_start < 0:
                effective_duration = prev_clip.duration
                overlap_start = 0
            
            # Scale blur radius by resolution for consistency
            w, h = current_clip.size
            scaled_blur = max(1, min(blur_radius, min(w, h) / 50))
            
            # Create gentle blur-like effect with minimal color impact
            # Use very light color softening instead of heavy reduction
            prev_blur = (prev_clip
                        .fx(vfx.colorx, 0.95)  # Very light color reduction
                        .fx(vfx.fadeout, effective_duration))
            current_blur = (current_clip
                           .fx(vfx.colorx, 0.95)  # Very light color reduction  
                           .fx(vfx.fadein, effective_duration))
            
            # Sequential audio timing to prevent blips
            audio_parts = []
            total_audio_duration = overlap_start + effective_duration + (current_clip.duration - effective_duration)
            
            if prev_clip.audio:
                # Previous audio: full duration until overlap end, with fadeout
                prev_audio = (prev_clip.audio
                            .set_start(0)
                            .set_end(overlap_start + effective_duration)
                            .fx(afx.audio_fadeout, effective_duration)
                            .fx(afx.volumex, self.CROSSFADE_GAIN))
                audio_parts.append(prev_audio)
            
            if current_clip.audio:
                # Current audio: start after previous ends, with sequential timing
                current_audio_start = overlap_start + effective_duration * 0.5  # Offset to prevent phase conflict
                current_audio = (current_clip.audio
                               .set_start(current_audio_start)
                               .fx(afx.audio_fadein, effective_duration * 0.5)
                               .fx(afx.volumex, self.CROSSFADE_GAIN))
                audio_parts.append(current_audio)
            
            # Composite with blur
            current_blur = current_blur.set_start(overlap_start)
            composite_video = CompositeVideoClip([prev_blur, current_blur])
            
            if audio_parts:
                composite_audio = CompositeAudioClip(audio_parts)
                composite_video = composite_video.set_audio(composite_audio)
            
            result_clips[-1] = composite_video
        
        return concatenate_videoclips(result_clips, method='compose')
    
    def _apply_glitch_transition(self, clips: List[VideoFileClip], duration: float = 0.2) -> VideoFileClip:
        """GLITCH: Digital distortion effect with proper guards and audio effects."""
        if len(clips) == 1:
            return clips[0]
        
        result_clips = []
        
        for i, clip in enumerate(clips):
            result_clips.append(clip)
            
            # Add simple glitch effect between clips (except after last)
            if i < len(clips) - 1:
                next_clip = clips[i + 1]
                
                # Simple glitch: rapid cuts with color distortion
                try:
                    # Take short segments from end of current and start of next
                    glitch_duration = min(duration, 0.1)  # Max 0.1s for stability
                    available_end = min(0.05, clip.duration * 0.05)  # 5% of clip or 50ms
                    available_start = min(0.05, next_clip.duration * 0.05)
                    
                    if available_end > 0 and available_start > 0:
                        # Create alternating segments with color effects
                        current_segment = (clip
                                         .subclipped(clip.duration - available_end)
                                         .fx(vfx.colorx, 0.5)  # Darken significantly
                                         .fx(vfx.invert_colors))  # Invert colors for glitch
                        
                        next_segment = (next_clip
                                      .subclipped(0, available_start)
                                      .fx(vfx.colorx, 1.5)  # Brighten
                                      .fx(vfx.gamma_corr, 2.0))  # Alter gamma for digital look
                        
                        # Simple alternation
                        glitch_sequence = concatenate_videoclips([current_segment, next_segment])
                        result_clips.append(glitch_sequence)
                    
                except Exception as e:
                    self.logger.warning(f"[GLITCH] Effect failed, using simple cut: {e}")
                    # Skip glitch effect and continue with normal concatenation
                    pass
        
        return concatenate_videoclips(result_clips, method='compose')

    def _get_video_dump_path(self) -> Path:
        """Get the path to the video_dump folder for consistent storage."""
        current_dir = Path(__file__).parent.parent.parent  # Go up to project root
        video_dump_path = current_dir / "storage" / "video_dump"
        video_dump_path.mkdir(parents=True, exist_ok=True)
        return video_dump_path

    def cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary video files."""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.debug(f"🗑️  Cleaned up: {file_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to cleanup {file_path}: {e}")
    
    def _safe_cleanup_clips(self, clips: List):
        """Safely close video clips without raising exceptions."""
        for clip in clips:
            try:
                if hasattr(clip, 'close') and callable(clip.close):
                    clip.close()
            except Exception as e:
                self.logger.debug(f"⚠️ Error closing clip: {e}")
    
    def cleanup_temp_directory(self):
        """Clean up the entire temporary directory."""
        try:

            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.logger.debug(f"🗑️  Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cleanup temp directory: {e}")
    
    def close(self):
        """Explicitly close session and clean up resources."""
        if self._session:
            self._session.close()
            self._session = None
        self.cleanup_temp_directory()
    
    def __del__(self):
        """Safety net cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass  # Avoid exceptions during cleanup
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

