import os
import math
import tempfile
import shutil
import uuid
import json
import subprocess
from pathlib import Path
from pydub import AudioSegment


class AudioVideoMixer:
    """Utility functions for audio and video mixing"""

    @staticmethod
    def mix_audio_to_video(video_path: str, audio_files: list[str], audio_volumes: list[float], 
                          output_video_path: str = None) -> str:
        """
        Mix multiple audio files with specified volumes onto a video.
        
        Args:
            video_path: Path to the input video file (local path or URL)
            audio_files: List of audio file paths (local paths or URLs)
            audio_volumes: List of volume levels for each audio file (0.0 to 1.0)
            output_video_path: Optional output path, if None uses storage system
            
        Returns:
            Path to the final mixed video file
            
        Raises:
            ValueError: If audio_files and audio_volumes lists have different lengths
            FileNotFoundError: If video_path or any audio file doesn't exist
        """
        if len(audio_files) != len(audio_volumes):
            raise ValueError("audio_files and audio_volumes must have the same length")
        
        if not audio_files:
            raise ValueError("At least one audio file is required")
        
        # Validate volume ranges
        for i, volume in enumerate(audio_volumes):
            if not 0.0 <= volume <= 1.0:
                raise ValueError(f"Volume {i} must be between 0.0 and 1.0, got {volume}")
        
        video_dump_path = AudioVideoMixer._get_video_dump_path()
        batch_id = str(uuid.uuid4())
        
        try:
            # Step 1: Download/prepare video file
            if video_path.startswith(('http://', 'https://')):
                print("📥 Downloading video file...")
                video_local_path = AudioVideoMixer._download_file(video_path, video_dump_path, f"input_video_{batch_id}")
            else:
                video_local_path = video_path
            
            # Step 2: Download/prepare audio files and process volumes
            processed_audio_files = []

            # Detect target video duration to sync audio lengths
            target_duration = None
            try:
                probe = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(video_local_path)
                ], capture_output=True, text=True)
                if probe.returncode == 0:
                    info = json.loads(probe.stdout)
                    target_duration = float(info['format']['duration'])
            except Exception:
                target_duration = None

            for i, (audio_file, volume) in enumerate(zip(audio_files, audio_volumes)):
                print(f"🎵 Processing audio file {i+1}/{len(audio_files)} (volume: {volume})...")
                # Download if URL
                if audio_file.startswith(('http://', 'https://')):
                    audio_local_path = AudioVideoMixer._download_file(audio_file, video_dump_path, f"input_audio_{batch_id}_{i}")
                else:
                    audio_local_path = audio_file

                # Apply volume adjustment using pydub (correct dB conversion)
                audio_segment = AudioSegment.from_file(audio_local_path)

                if volume <= 0.0:
                    volume_db = -60.0
                else:
                    # Convert linear gain to dB accurately: gain_db = 20*log10(gain)
                    try:
                        volume_db = 20.0 * math.log10(volume)
                    except ValueError:
                        volume_db = 0.0

                adjusted_audio = audio_segment + volume_db

                # Save adjusted audio to temp
                adjusted_audio_path = video_dump_path / f"adjusted_audio_{batch_id}_{i}.wav"
                adjusted_audio.export(str(adjusted_audio_path), format="wav")

                # If possible, time-stretch to match target video duration
                if target_duration is not None:
                    # Probe audio duration
                    try:
                        a_probe = subprocess.run([
                            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(adjusted_audio_path)
                        ], capture_output=True, text=True)
                        if a_probe.returncode == 0:
                            a_info = json.loads(a_probe.stdout)
                            a_duration = float(a_info['format']['duration'])
                        else:
                            a_duration = None
                    except Exception:
                        a_duration = None

                    if a_duration and abs(a_duration - target_duration) > 0.05:
                        speed_factor = a_duration / target_duration
                        # Build atempo chain to cover wide factors
                        filters = []
                        current = speed_factor
                        if current > 0:
                            while current > 2.0:
                                filters.append('atempo=2.0')
                                current /= 2.0
                            while current < 0.5:
                                filters.append('atempo=0.5')
                                current /= 0.5
                            if abs(current - 1.0) > 0.01:
                                filters.append(f'atempo={current:.6f}')

                        synced_path = video_dump_path / f"synced_audio_{batch_id}_{i}.wav"
                        cmd_sync = [
                            'ffmpeg', '-y', '-i', str(adjusted_audio_path)
                        ]
                        if filters:
                            cmd_sync += ['-af', ','.join(filters)]
                        cmd_sync += [str(synced_path)]
                        try:
                            res = subprocess.run(cmd_sync, capture_output=True, text=True)
                            if res.returncode == 0:
                                # Replace adjusted path with synced path
                                try:
                                    adjusted_audio_path.unlink(missing_ok=True)
                                except Exception:
                                    pass
                                adjusted_audio_path = synced_path
                            else:
                                print(f"⚠️ Audio sync (speed adjust) failed for track {i}: {res.stderr}")
                        except Exception as e:
                            print(f"⚠️ Audio sync exception for track {i}: {e}")

                processed_audio_files.append(str(adjusted_audio_path))
            
            # Step 3: Use FFmpeg to mix all audio tracks with the video
            if output_video_path is None:
                # Create output path in storage system with unique identifier
                final_identifier = f"mixed_video_{batch_id}"
                output_video_path = video_dump_path / f"{final_identifier}.mp4"
            
            print("🎬 Mixing audio tracks with video using FFmpeg...")
            success = AudioVideoMixer._mix_with_ffmpeg(
                video_local_path,
                processed_audio_files,
                str(output_video_path)
            )
            
            if not success:
                raise RuntimeError("FFmpeg mixing failed")
            
            # The file is already in the right location, so we don't need to store/copy it again
            print(f"✅ Mixed video created at: {output_video_path}")
            
            # Step 5: Clean up temporary files
            for temp_audio in processed_audio_files:
                Path(temp_audio).unlink(missing_ok=True)
            
            # Clean up downloaded files if they were URLs
            if video_path.startswith(('http://', 'https://')):
                Path(video_local_path).unlink(missing_ok=True)
            
            for i, audio_file in enumerate(audio_files):
                if audio_file.startswith(('http://', 'https://')):
                    temp_audio_path = video_dump_path / f"input_audio_{batch_id}_{i}.*"
                    for temp_file in video_dump_path.glob(f"input_audio_{batch_id}_{i}.*"):
                        temp_file.unlink(missing_ok=True)
            
            print(f"✅ Audio-video mixing completed successfully!")
            return str(output_video_path)
            
        except Exception as e:
            print(f"❌ Error during audio-video mixing: {e}")
            # Clean up any temporary files
            for temp_file in video_dump_path.glob(f"*{batch_id}*"):
                temp_file.unlink(missing_ok=True)
            raise e

    @staticmethod
    def sync_blend_audio_video(video_data: bytes, 
                              audio_data: bytes, 
                              output_filename: str = "synced_video.mp4",
                              audio_volume: float = 1.0,
                              output_dir: str = None) -> bytes:
        """
        Intelligently sync and blend audio with video by auto-adjusting audio duration.
        
        This method automatically detects duration mismatches and adjusts audio speed
        to perfectly match video length while preserving pitch quality.
        
        Args:
            video_data: Video file data as bytes
            audio_data: Audio file data as bytes  
            output_filename: Name of the output file
            audio_volume: Volume multiplier for the audio (1.0 = original volume)
            output_dir: Output directory path (uses video_dump if None)
            
        Returns:
            Synced video file as bytes
            
        Raises:
            Exception: If ffmpeg processing fails
        """
        
        print(f"🎵 Starting intelligent audio/video sync-blend process")
        print(f"📹 Video data size: {len(video_data)} bytes")
        print(f"🔊 Audio data size: {len(audio_data)} bytes")
        
        # Get output directory
        if output_dir is None:
            video_dump_path = AudioVideoMixer._get_video_dump_path()
        else:
            video_dump_path = Path(output_dir)
            video_dump_path.mkdir(parents=True, exist_ok=True)
        
        # Create temporary files with unique names
        batch_id = str(uuid.uuid4())[:8]
        video_temp_path = video_dump_path / f"temp_video_{batch_id}.mp4"
        audio_temp_path = video_dump_path / f"temp_audio_{batch_id}.wav"
        
        # Save data to temporary files
        with open(video_temp_path, 'wb') as f:
            f.write(video_data)
        with open(audio_temp_path, 'wb') as f:
            f.write(audio_data)
        
        try:
            # Prepare output path
            output_path = video_dump_path / output_filename
            
            # Get video duration first
            duration_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(video_temp_path)
            ]
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
            
            if duration_result.returncode == 0:
                info = json.loads(duration_result.stdout)
                video_duration = float(info['format']['duration'])
                print(f"📏 Video duration detected: {video_duration:.3f}s")
            else:
                video_duration = None
                print("⚠️ Could not detect video duration")
            
            # Get audio duration for speed adjustment calculation
            audio_duration_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(audio_temp_path)
            ]
            audio_duration_result = subprocess.run(audio_duration_cmd, capture_output=True, text=True)
            
            if audio_duration_result.returncode == 0:
                audio_info = json.loads(audio_duration_result.stdout)
                audio_duration = float(audio_info['format']['duration'])
                print(f"🎵 Audio duration detected: {audio_duration:.3f}s")
            else:
                audio_duration = None
                print("⚠️ Could not detect audio duration")

            # Build ffmpeg command with high-quality settings
            cmd = [
                'ffmpeg',
                '-i', str(video_temp_path),  # Input video
                '-i', str(audio_temp_path),  # Input audio
                '-c:v', 'copy',              # Copy video stream (no re-encoding)
                '-c:a', 'aac',               # Encode audio as AAC
                '-b:a', '192k',              # High-quality audio bitrate
                '-ar', '48000',              # Standard sample rate for video
                '-ac', '2',                  # Stereo audio channels
                '-map', '0:v:0',             # Map video from first input
                '-map', '1:a:0',             # Map audio from second input
                '-y',                        # Overwrite output file
            ]
            
            # Build audio filter chain for intelligent sync and processing
            audio_filters = []
            
            # Handle duration mismatch with smart speed adjustment
            if video_duration and audio_duration and abs(video_duration - audio_duration) > 0.1:
                speed_factor = audio_duration / video_duration
                print(f"⚠️ Duration mismatch detected:")
                print(f"   Video: {video_duration:.2f}s, Audio: {audio_duration:.2f}s")
                print(f"🎵 Adjusting audio speed by {speed_factor:.3f}x to match video duration")
                
                # Use atempo filter for speed adjustment (supports 0.5x to 2.0x)
                if 0.5 <= speed_factor <= 2.0:
                    audio_filters.append(f'atempo={speed_factor}')
                else:
                    # For extreme speed changes, chain multiple atempo filters
                    current_factor = speed_factor
                    while current_factor > 2.0:
                        audio_filters.append('atempo=2.0')
                        current_factor /= 2.0
                    while current_factor < 0.5:
                        audio_filters.append('atempo=0.5')
                        current_factor /= 0.5
                    if abs(current_factor - 1.0) > 0.01:  # Only add if meaningful difference
                        audio_filters.append(f'atempo={current_factor}')
            elif video_duration and not audio_duration:
                # Fallback: pad audio to match video duration if we couldn't detect audio duration
                audio_filters.append(f'apad=whole_dur={video_duration}')
                print(f"🎬 Padding audio with silence to match video duration: {video_duration:.3f}s")
            
            # Add volume adjustment if needed
            if audio_volume != 1.0:
                audio_filters.append(f'volume={audio_volume}')
                print(f"🔊 Applying audio volume adjustment: {audio_volume}x")
            
            # Apply audio filters if any
            if audio_filters:
                cmd.insert(-1, '-af')
                cmd.insert(-1, ','.join(audio_filters))
            
            cmd.append(str(output_path))
            
            print("🔗 Sync-blending audio and video with ffmpeg...")
            print(f"💾 Saving synced video to: {output_path}")
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"ffmpeg sync-blend failed: {result.stderr}")
            
            # Read the output file back as bytes
            with open(output_path, 'rb') as f:
                result_bytes = f.read()
            
            print(f"✅ Audio/video sync-blend complete!")
            print(f"   Final video: {output_path}")
            print(f"   Output size: {len(result_bytes)} bytes ({len(result_bytes) / (1024*1024):.1f} MB)")
            
            return result_bytes
            
        finally:
            # Clean up temporary files
            try:
                video_temp_path.unlink(missing_ok=True)
                audio_temp_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"⚠️ Warning: Could not clean up temp files: {e}")

    @staticmethod
    def _mix_with_ffmpeg(video_path: str, audio_files: list[str], output_path: str) -> bool:
        """Use FFmpeg to mix multiple audio tracks with video"""
        try:
            # First, check if video has audio using ffprobe
            has_video_audio = AudioVideoMixer._check_video_has_audio(video_path)
            
            # Build FFmpeg command
            cmd = ["ffmpeg", "-y"]  # -y to overwrite output file
            
            # Add video input
            cmd.extend(["-i", video_path])
            
            # Add audio inputs
            for audio_file in audio_files:
                cmd.extend(["-i", audio_file])
            
            # Build filter complex based on whether video has audio
            if has_video_audio:
                # Video has audio - mix with input audio files
                if len(audio_files) == 1:
                    # Simple case: one audio track + video audio
                    filter_complex = (
                        f"[0:a][1:a]amix=inputs=2:duration=longest:normalize=1,"
                        f"alimiter=limit=0.95[a]"
                    )
                else:
                    # Multiple audio tracks: mix them all together with video audio
                    audio_mix_inputs = "".join([f"[{i+1}:a]" for i in range(len(audio_files))])
                    filter_complex = (
                        f"[0:a]{audio_mix_inputs}amix=inputs={len(audio_files)+1}:duration=longest:normalize=1,"
                        f"alimiter=limit=0.95[a]"
                    )
                
                cmd.extend([
                    "-filter_complex", filter_complex,
                    "-map", "0:v",  # Use video from first input
                    "-map", "[a]",  # Use mixed audio
                ])
            else:
                # Video has no audio - just add the input audio files
                if len(audio_files) == 1:
                    # Simple case: just use the one audio file
                    cmd.extend([
                        "-map", "0:v",  # Use video from first input
                        "-map", "1:a",  # Use audio from first audio input
                    ])
                else:
                    # Multiple audio files: mix them together
                    audio_mix_inputs = "".join([f"[{i+1}:a]" for i in range(len(audio_files))])
                    filter_complex = (
                        f"{audio_mix_inputs}amix=inputs={len(audio_files)}:duration=longest:normalize=1,"
                        f"alimiter=limit=0.95[a]"
                    )
                    cmd.extend([
                        "-filter_complex", filter_complex,
                        "-map", "0:v",  # Use video from first input
                        "-map", "[a]",  # Use mixed audio
                    ])
            
            # Add encoding options
            cmd.extend([
                "-c:v", "copy",  # Copy video without re-encoding
                "-c:a", "aac",   # Encode audio as AAC
                "-b:a", "192k",  # Audio bitrate
                # No -shortest: let video stream define duration; audios are time-adjusted to match
                output_path
            ])
            
            print(f"🔧 FFmpeg command: {' '.join(cmd[:8])}...")  # Show first part of command
            
            # Run FFmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ FFmpeg mixing successful")
                return True
            else:
                print(f"❌ FFmpeg error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ FFmpeg execution failed: {e}")
            return False

    @staticmethod
    def _check_video_has_audio(video_path: str) -> bool:
        """Check if video file has an audio stream using ffprobe"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                
                # Check if any stream is audio
                for stream in info.get("streams", []):
                    if stream.get("codec_type") == "audio":
                        print("🎵 Video has existing audio track")
                        return True
                
                print("🔇 Video has no audio track")
                return False
            else:
                print(f"⚠️ Could not check video audio: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"⚠️ Error checking video audio: {e}")
            return False

    @staticmethod
    def _download_file(url: str, download_path: Path, filename_prefix: str) -> str:
        """Download a file from URL to local storage"""
        import requests
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Try to get file extension from URL or Content-Type
            content_type = response.headers.get('content-type', '')
            if 'video' in content_type:
                extension = '.mp4'
            elif 'audio' in content_type:
                extension = '.wav'
            else:
                # Try to get from URL
                url_path = Path(url)
                extension = url_path.suffix if url_path.suffix else '.tmp'
            
            local_filename = download_path / f"{filename_prefix}{extension}"
            
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return str(local_filename)
            
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            raise e

    # Storage Management Functions
    # ============================
    
    @staticmethod
    def _get_video_dump_path() -> Path:
        """Get the path to the video_dump folder."""
        current_dir = Path(__file__).parent.parent  # Go up to project root
        video_dump_path = current_dir / "storage" / "video_dump"
        video_dump_path.mkdir(parents=True, exist_ok=True)
        return video_dump_path
    
    @staticmethod
    def store_video(video_path: str, identifier: str = None) -> str:
        """Store video file in video_dump folder and return stored file identifier.
        
        Args:
            video_path: Path to the video file to store
            identifier: Optional custom identifier, otherwise generates UUID
            
        Returns:
            String identifier that can be used to restore the file later
        """
        if identifier is None:
            identifier = str(uuid.uuid4())
        
        video_dump_path = AudioVideoMixer._get_video_dump_path()
        
        # Get file extension from original file
        original_extension = Path(video_path).suffix
        if not original_extension:
            original_extension = ".mp4"  # Default to mp4
        
        stored_filename = f"{identifier}{original_extension}"
        stored_path = video_dump_path / stored_filename
        
        # Copy the file to storage
        shutil.copy2(video_path, stored_path)
        
        return identifier
    
    @staticmethod
    def restore_video(identifier: str) -> str:
        """Restore video file from video_dump folder.
        
        Args:
            identifier: The identifier returned by store_video
            
        Returns:
            Path to the restored video file in video_dump folder
            
        Raises:
            FileNotFoundError: If the video file with given identifier doesn't exist
        """
        video_dump_path = AudioVideoMixer._get_video_dump_path()
        
        # Look for file with this identifier (any extension)
        matching_files = list(video_dump_path.glob(f"{identifier}.*"))
        
        if not matching_files:
            raise FileNotFoundError(f"No video file found with identifier: {identifier}")
        
        if len(matching_files) > 1:
            # If multiple files, prefer .mp4, then first alphabetically
            mp4_files = [f for f in matching_files if f.suffix == ".mp4"]
            if mp4_files:
                return str(mp4_files[0])
            matching_files.sort()
        
        return str(matching_files[0])
    
    @staticmethod
    def list_stored_videos() -> list[str]:
        """List all stored video file identifiers.
        
        Returns:
            List of identifiers for all stored video files
        """
        video_dump_path = AudioVideoMixer._get_video_dump_path()
        
        if not video_dump_path.exists():
            return []
        
        identifiers = []
        for video_file in video_dump_path.iterdir():
            if video_file.is_file() and video_file.suffix in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                identifier = video_file.stem  # Filename without extension
                identifiers.append(identifier)
        
        return identifiers
    
    @staticmethod
    def delete_stored_video(identifier: str) -> bool:
        """Delete stored video file from video_dump folder.
        
        Args:
            identifier: The identifier of the video file to delete
            
        Returns:
            True if file was deleted, False if file didn't exist
        """
        video_dump_path = AudioVideoMixer._get_video_dump_path()
        
        # Look for file with this identifier (any extension)
        matching_files = list(video_dump_path.glob(f"{identifier}.*"))
        
        if not matching_files:
            return False
        
        # Delete all matching files (in case there are multiple extensions)
        for file_path in matching_files:
            file_path.unlink()
        
        return True
    
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get video information using FFprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video information (duration, resolution, etc.)
        """
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                
                # Extract useful information
                video_info = {
                    "duration": 0,
                    "width": 0,
                    "height": 0,
                    "fps": 0,
                    "has_audio": False
                }
                
                # Get format information
                if "format" in info:
                    video_info["duration"] = float(info["format"].get("duration", 0))
                
                # Get stream information
                for stream in info.get("streams", []):
                    if stream.get("codec_type") == "video":
                        video_info["width"] = stream.get("width", 0)
                        video_info["height"] = stream.get("height", 0)
                        
                        # Calculate FPS
                        fps_str = stream.get("r_frame_rate", "0/1")
                        if "/" in fps_str:
                            num, den = fps_str.split("/")
                            video_info["fps"] = float(num) / float(den) if float(den) > 0 else 0
                    elif stream.get("codec_type") == "audio":
                        video_info["has_audio"] = True
                
                return video_info
            else:
                print(f"❌ FFprobe error: {result.stderr}")
                return {}
                
        except Exception as e:
            print(f"❌ Failed to get video info: {e}")
            return {}
