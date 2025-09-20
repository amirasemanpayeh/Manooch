import os
import tempfile
import wave
import shutil
import uuid
from pathlib import Path
from pydub import AudioSegment


class AudioTools:
    """Utility functions for audio processing"""

    @staticmethod
    def convert_to_wav(input_audio_path: str, output_audio_path: str) -> None:
        """Convert audio file to WAV format. Handles both local paths and URLs."""
        import requests
        
        # Check if it's a URL
        if input_audio_path.startswith(('http://', 'https://')):
            # Download the file first
            audio_dump_path = AudioTools._get_audio_dump_path()
            temp_download = audio_dump_path / f"temp_download_{uuid.uuid4()}.tmp"
            try:
                response = requests.get(input_audio_path)
                response.raise_for_status()
                with open(temp_download, 'wb') as f:
                    f.write(response.content)
                
                # Load from downloaded file
                audio = AudioSegment.from_file(str(temp_download))
                audio.export(output_audio_path, format="wav")
                
                # Clean up temp file
                temp_download.unlink()
            except Exception as e:
                # Clean up temp file if it exists
                if temp_download.exists():
                    temp_download.unlink()
                raise e
        else:
            # Handle local file path
            audio = AudioSegment.from_file(input_audio_path)
            audio.export(output_audio_path, format="wav")

    @staticmethod
    def get_audio_duration(audio_path: str) -> float:
        """Get the duration of an audio file in seconds."""
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
        return duration


    @staticmethod
    def add_silence_padding(input_audio_path: str, output_audio_path: str, target_duration_seconds: float, padding_side: str) -> None:
        """Add silence padding to reach the target duration."""
        audio = AudioSegment.from_file(input_audio_path)
        current_duration = len(audio) / 1000.0  # Convert to seconds
        
        # If audio is already equal or longer than target, just copy it
        if current_duration >= target_duration_seconds:
            audio.export(output_audio_path, format="wav")
            return
        
        # Calculate how much silence we need to add
        silence_needed = target_duration_seconds - current_duration
        silence_ms = silence_needed * 1000  # Convert to milliseconds
        
        if padding_side == "both":
            # Split silence symmetrically between left and right
            half_silence_ms = silence_ms / 2
            left_silence = AudioSegment.silent(duration=half_silence_ms)
            right_silence = AudioSegment.silent(duration=half_silence_ms)
            padded_audio = left_silence + audio + right_silence
        elif padding_side == "left":
            silence = AudioSegment.silent(duration=silence_ms)
            padded_audio = silence + audio
        elif padding_side == "right":
            silence = AudioSegment.silent(duration=silence_ms)
            padded_audio = audio + silence
        else:
            raise ValueError("Invalid padding_side. Choose from 'both', 'left', or 'right'.")
        
        padded_audio.export(output_audio_path, format="wav")


    @staticmethod
    def sync_multi_talk_audios(audio_files: list[str]) -> list[str]:
        """Process multiple audio files for sequential multi-speaker conversation."""
        
        # Generate batch identifier for this processing session
        batch_id = str(uuid.uuid4())
        converted_identifiers = []
        cumulative_time = 0.0
        
        # Step 1: Convert all audio files to wav and get durations
        audio_dump_path = AudioTools._get_audio_dump_path()
        
        for i, audio_file in enumerate(audio_files):
            # Convert to WAV and store
            temp_wav_path = audio_dump_path / f"temp_convert_{batch_id}_{i}.wav"
            AudioTools.convert_to_wav(audio_file, str(temp_wav_path))
            
            # Store converted file
            converted_id = f"{batch_id}_converted_{i:03d}"
            AudioTools.store_audio(temp_wav_path, converted_id)
            converted_identifiers.append(converted_id)
            
            # Clean up temp file
            temp_wav_path.unlink()
        
        # Step 2: Process each file with left padding
        processed_identifiers = []
        for i, converted_id in enumerate(converted_identifiers):
            # Restore converted file
            converted_path = AudioTools.restore_audio(converted_id)
            
            # Get duration
            duration = AudioTools.get_audio_duration(converted_path)
            
            # Create processed file with left padding
            temp_processed_path = audio_dump_path / f"temp_processed_{batch_id}_{i}.wav"
            
            if i == 0:
                # Primary speaker - no left padding
                AudioTools.convert_to_wav(converted_path, str(temp_processed_path))
            else:
                # Other speakers - add left padding to reach cumulative_time total duration
                AudioTools.add_silence_padding(converted_path, str(temp_processed_path), cumulative_time + duration, "left")
            
            # Store processed file
            processed_id = f"{batch_id}_processed_{i:03d}"
            AudioTools.store_audio(str(temp_processed_path), processed_id)
            processed_identifiers.append(processed_id)
            
            # Clean up temp file
            temp_processed_path.unlink()
            
            cumulative_time += duration
        
        # Step 3: Add right padding to all files to reach total duration
        total_duration = cumulative_time
        final_identifiers = []
        
        for i, processed_id in enumerate(processed_identifiers):
            # Restore processed file
            processed_path = AudioTools.restore_audio(processed_id)
            
            # Create final file with right padding
            temp_final_path = audio_dump_path / f"temp_final_{batch_id}_{i}.wav"
            AudioTools.add_silence_padding(processed_path, str(temp_final_path), total_duration, "right")
            
            # Store final file
            final_id = f"{batch_id}_final_{i:03d}"
            AudioTools.store_audio(str(temp_final_path), final_id)
            final_identifiers.append(final_id)
            
            # Clean up temp file
            temp_final_path.unlink()
        
        # Clean up intermediate files
        for converted_id in converted_identifiers:
            AudioTools.delete_stored_audio(converted_id)
        for processed_id in processed_identifiers:
            AudioTools.delete_stored_audio(processed_id)
        
        # Return paths to final stored files
        final_paths = []
        for final_id in final_identifiers:
            final_path = AudioTools.restore_audio(final_id)
            final_paths.append(final_path)
        
        return final_paths

    # Storage Management Functions
    # ============================
    
    @staticmethod
    def _get_audio_dump_path() -> Path:
        """Get the path to the audio_dump folder."""
        current_dir = Path(__file__).parent.parent  # Go up to project root
        audio_dump_path = current_dir / "storage" / "audio_dump"
        audio_dump_path.mkdir(parents=True, exist_ok=True)
        return audio_dump_path
    
    @staticmethod
    def store_audio(audio_path: str, identifier: str = None) -> str:
        """Store audio file in audio_dump folder and return stored file identifier.
        
        Args:
            audio_path: Path to the audio file to store
            identifier: Optional custom identifier, otherwise generates UUID
            
        Returns:
            String identifier that can be used to restore the file later
        """
        if identifier is None:
            identifier = str(uuid.uuid4())
        
        audio_dump_path = AudioTools._get_audio_dump_path()
        
        # Get file extension from original file
        original_extension = Path(audio_path).suffix
        if not original_extension:
            original_extension = ".wav"  # Default to wav
        
        stored_filename = f"{identifier}{original_extension}"
        stored_path = audio_dump_path / stored_filename
        
        # Copy the file to storage
        shutil.copy2(audio_path, stored_path)
        
        return identifier
    
    @staticmethod
    def restore_audio(identifier: str) -> str:
        """Restore audio file from audio_dump folder.
        
        Args:
            identifier: The identifier returned by store_audio
            
        Returns:
            Path to the restored audio file in audio_dump folder
            
        Raises:
            FileNotFoundError: If the audio file with given identifier doesn't exist
        """
        audio_dump_path = AudioTools._get_audio_dump_path()
        
        # Look for file with this identifier (any extension)
        matching_files = list(audio_dump_path.glob(f"{identifier}.*"))
        
        if not matching_files:
            raise FileNotFoundError(f"No audio file found with identifier: {identifier}")
        
        if len(matching_files) > 1:
            # If multiple files, prefer .wav, then first alphabetically
            wav_files = [f for f in matching_files if f.suffix == ".wav"]
            if wav_files:
                return str(wav_files[0])
            matching_files.sort()
        
        return str(matching_files[0])
    
    @staticmethod
    def list_stored_audio() -> list[str]:
        """List all stored audio file identifiers.
        
        Returns:
            List of identifiers for all stored audio files
        """
        audio_dump_path = AudioTools._get_audio_dump_path()
        
        if not audio_dump_path.exists():
            return []
        
        identifiers = []
        for audio_file in audio_dump_path.iterdir():
            if audio_file.is_file() and audio_file.suffix in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
                identifier = audio_file.stem  # Filename without extension
                identifiers.append(identifier)
        
        return identifiers
    
    @staticmethod
    def delete_stored_audio(identifier: str) -> bool:
        """Delete stored audio file from audio_dump folder.
        
        Args:
            identifier: The identifier of the audio file to delete
            
        Returns:
            True if file was deleted, False if file didn't exist
        """
        audio_dump_path = AudioTools._get_audio_dump_path()
        
        # Look for file with this identifier (any extension)
        matching_files = list(audio_dump_path.glob(f"{identifier}.*"))
        
        if not matching_files:
            return False
        
        # Delete all matching files (in case there are multiple extensions)
        for file_path in matching_files:
            file_path.unlink()
        
        return True
    
    @staticmethod
    def store_processed_audio_batch(audio_paths: list[str], batch_identifier: str = None) -> str:
        """Store a batch of processed audio files with a common batch identifier.
        
        Args:
            audio_paths: List of audio file paths to store
            batch_identifier: Optional batch identifier, otherwise generates UUID
            
        Returns:
            Batch identifier that can be used to restore all files later
        """
        if batch_identifier is None:
            batch_identifier = str(uuid.uuid4())
        
        for i, audio_path in enumerate(audio_paths):
            file_identifier = f"{batch_identifier}_part_{i:03d}"
            AudioTools.store_audio(audio_path, file_identifier)
        
        return batch_identifier
    
    @staticmethod
    def restore_processed_audio_batch(batch_identifier: str) -> list[str]:
        """Restore a batch of processed audio files.
        
        Args:
            batch_identifier: The batch identifier returned by store_processed_audio_batch
            
        Returns:
            List of paths to the restored audio files in correct order
        """
        audio_dump_path = AudioTools._get_audio_dump_path()
        
        # Find all files with this batch identifier
        pattern = f"{batch_identifier}_part_*"
        matching_files = list(audio_dump_path.glob(pattern))
        
        if not matching_files:
            return []
        
        # Sort by part number to maintain order
        matching_files.sort(key=lambda x: x.stem)
        
        return [str(f) for f in matching_files]