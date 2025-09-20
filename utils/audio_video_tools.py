import os
import tempfile
import shutil
import uuid
from pathlib import Path
from pydub import AudioSegment
import subprocess
import json


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
            for i, (audio_file, volume) in enumerate(zip(audio_files, audio_volumes)):
                print(f"🎵 Processing audio file {i+1}/{len(audio_files)} (volume: {volume})...")
                
                # Download if URL
                if audio_file.startswith(('http://', 'https://')):
                    audio_local_path = AudioVideoMixer._download_file(audio_file, video_dump_path, f"input_audio_{batch_id}_{i}")
                else:
                    audio_local_path = audio_file
                
                # Apply volume adjustment using pydub
                audio_segment = AudioSegment.from_file(audio_local_path)
                
                # Convert volume (0.0-1.0) to dB change
                if volume == 0.0:
                    # Mute the audio
                    volume_db = -60  # Very quiet
                else:
                    # Convert linear volume to dB (0.5 = -6dB, 1.0 = 0dB)
                    volume_db = 20 * (volume - 1)  # logarithmic scale
                
                # Apply volume adjustment
                adjusted_audio = audio_segment + volume_db
                
                # Save adjusted audio
                adjusted_audio_path = video_dump_path / f"adjusted_audio_{batch_id}_{i}.wav"
                adjusted_audio.export(str(adjusted_audio_path), format="wav")
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
                    filter_complex = f"[0:a][1:a]amix=inputs=2:duration=longest[a]"
                else:
                    # Multiple audio tracks: mix them all together with video audio
                    audio_mix_inputs = "".join([f"[{i+1}:a]" for i in range(len(audio_files))])
                    filter_complex = f"[0:a]{audio_mix_inputs}amix=inputs={len(audio_files)+1}:duration=longest[a]"
                
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
                    filter_complex = f"{audio_mix_inputs}amix=inputs={len(audio_files)}:duration=longest[a]"
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
                "-shortest",     # Make output duration match shortest input
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
