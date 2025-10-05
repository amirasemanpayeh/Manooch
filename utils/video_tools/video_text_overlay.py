"""
Video Text Overlay Manager

This module provides functionality to add text overlays to videos using HTML/CSS rendering.
Supports multiple presentation types: static, audio-synced, duration-based with animation, and progressive reveal.
"""

from typing import List, Tuple
import os
import logging
import requests
import shutil
import uuid
from pathlib import Path
from html2image import Html2Image
import numpy as np
import ffmpeg

# Import video generator models (corrected path)
from models.video_engine_models import TextOverlay, TextPresentationType, TextOverlayPosition


def get_text_overlay_logger(name: str = "VideoTextOverlay") -> logging.Logger:
    """Create or fetch a configured logger for video text overlay components.

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


class VideoTextOverlayManager:
    """
    Manages the addition of text overlays to videos using HTML/CSS rendering approach.
    """
    
    def __init__(self):
        """Initialize the VideoTextOverlayManager"""
        self.html2img = Html2Image()
        # Use consistent storage mechanism - video_dump folder
        self.temp_dir = str(self._get_video_dump_path())
        # Setup logging using the project's logging pattern
        self.logger = get_text_overlay_logger(f"VideoTextOverlay.{id(self)}")
        
    def add_text_overlays_to_video(self, video_url: str, overlays: List[TextOverlay]) -> bytes:
        """
        Main API function to add text overlays to a video.
        
        Args:
            video_url (str): URL or path to the input video
            overlays (List[TextOverlay]): List of text overlays to add
            
        Returns:
            bytes: The processed video with text overlays as bytes
        """
        self.logger.info(f"🎬 Adding {len(overlays)} text overlays to video")
        
        if not overlays:
            # No overlays to add, return original video
            self.logger.info("No overlays to add, returning original video")
            return self._download_video_as_bytes(video_url)
        
        # Download the original video
        input_video_path = self._download_video_to_temp(video_url)
        
        try:
            # Get video properties
            video_info = self._get_video_info(input_video_path)
            width, height, fps, duration = video_info
            
            # Process each overlay based on its presentation type
            processed_video_path = input_video_path
            
            for i, overlay in enumerate(overlays):
                self.logger.info(f"📝 Processing overlay {i+1}/{len(overlays)}: {overlay.properties.presentation_type.value}")
                
                # Route to appropriate handler based on presentation type
                if overlay.properties.presentation_type == TextPresentationType.STATIC_OVERLAY:
                    processed_video_path = self._handle_static_overlay(
                        processed_video_path, overlay, width, height, fps, duration
                    )
                #elif overlay.properties.presentation_type == TextPresentationType.AUDIO_SYNCHED:
                #    processed_video_path = self._handle_audio_synced_overlay(
                #        processed_video_path, overlay, width, height, fps, duration
                #    )
                elif overlay.properties.presentation_type == TextPresentationType.DURATION_BASED_WITH_ANIMATION:
                    processed_video_path = self._handle_duration_based_overlay(
                        processed_video_path, overlay, width, height, fps, duration
                    )
                elif overlay.properties.presentation_type == TextPresentationType.PROGRESSIVE_REVEAL:
                    processed_video_path = self._handle_progressive_reveal_overlay(
                        processed_video_path, overlay, width, height, fps, duration
                    )
                elif overlay.properties.presentation_type == TextPresentationType.SLIDING_WINDOW:
                    # Use window_size from overlay properties
                    window_size = getattr(overlay.properties, 'window_size', 3)  # Default to 3 if not specified
                    processed_video_path = self._handle_sliding_window_overlay(
                        processed_video_path, overlay, width, height, fps, duration, window_size=window_size
                    )
                else:
                    self.logger.warning(f"⚠️ Unknown presentation type {overlay.properties.presentation_type}")
            
            # Read the final processed video as bytes
            with open(processed_video_path, 'rb') as f:
                result_bytes = f.read()
            
            self.logger.info("✅ Text overlay processing complete")
            return result_bytes
            
        finally:
            # Clean up temporary files
            self._cleanup_temp_files()
    
    def _handle_static_overlay(self, video_path: str, overlay: TextOverlay, 
                              width: int, height: int, fps: float, duration: float) -> str:
        """
        Handle STATIC_OVERLAY presentation type.
        Text appears and stays visible for the entire video duration.
        """
        self.logger.info(f"📌 Adding static overlay: '{overlay.text[:30]}...'")
        
        # Generate HTML/CSS for the text overlay
        overlay_html = self._generate_overlay_html(overlay, width, height)
        
        # Create overlay image
        overlay_image_path = self._render_html_to_image(overlay_html, width, height)
        
        # Apply overlay to entire video
        output_path = self._get_temp_video_path()
        
        try:
            # Use simpler ffmpeg command with filter_complex
            start_time = overlay.properties.timing.start_time_seconds
            end_time = start_time + (overlay.properties.timing.duration_seconds or duration)
            
            # Ensure end_time doesn't exceed video duration
            end_time = min(end_time, duration)
            
            # Create inputs
            video_input = ffmpeg.input(video_path)
            overlay_input = ffmpeg.input(overlay_image_path)
            
            # Apply overlay with filter_complex
            video_with_overlay = ffmpeg.filter(
                [video_input, overlay_input], 
                'overlay', 
                x=0, y=0,
                enable=f'between(t,{start_time},{end_time})'
            )
            
            # Generate final output - check if input has audio
            has_audio = self._has_audio_stream(video_path)
            
            if has_audio:
                # Output with both video and audio streams
                ffmpeg.output(
                    video_with_overlay, video_input['a'], output_path,
                    vcodec='libx264', 
                    acodec='copy'
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            else:
                # Output with video only
                ffmpeg.output(
                    video_with_overlay, output_path,
                    vcodec='libx264'
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            
        except ffmpeg.Error as e:
            self.logger.error(f"❌ FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            raise
        
        return output_path
    
    def _handle_audio_synced_overlay(self, video_path: str, overlay: TextOverlay,
                                   width: int, height: int, fps: float, duration: float) -> str:
        """
        Handle AUDIO_SYNCHED presentation type.
        Text appears synchronized with audio cues - displays complete text when speech starts.
        """
        self.logger.info(f"🎵 Adding audio-synced overlay: '{overlay.text[:30]}...'")
        
        timing = overlay.properties.timing
        start_time = timing.start_time_seconds
        overlay_duration = timing.duration_seconds or 3.0
        end_time = start_time + overlay_duration
        
        # Generate HTML/CSS for the complete text overlay
        overlay_html = self._generate_overlay_html(overlay, width, height)
        
        # Create overlay image for the complete text
        overlay_image_path = self._render_html_to_image(overlay_html, width, height)
        
        # Apply overlay with precise audio timing
        output_path = self._get_temp_video_path()
        
        try:
            video_input = ffmpeg.input(video_path)
            overlay_input = ffmpeg.input(overlay_image_path)
            
            # Apply overlay filter - shows complete text during specified time range
            video_with_overlay = ffmpeg.filter(
                [video_input, overlay_input], 
                'overlay', 
                x=0, y=0,
                enable=f'between(t,{start_time},{end_time})'
            )
            
            # Output - check if input has audio
            has_audio = self._has_audio_stream(video_path)
            
            if has_audio:
                # Output with both video and audio streams
                ffmpeg.output(
                    video_with_overlay, video_input['a'], output_path,
                    vcodec='libx264', 
                    acodec='copy'
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            else:
                # Output with video only
                ffmpeg.output(
                    video_with_overlay, output_path,
                    vcodec='libx264'
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            
        except ffmpeg.Error as e:
            self.logger.error(f"❌ FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            raise
        
        return output_path
    
    
    def _handle_duration_based_overlay(self, video_path: str, overlay: TextOverlay,
                                     width: int, height: int, fps: float, duration: float) -> str:
        """
        Handle DURATION_BASED_WITH_ANIMATION presentation type.
        Text appears and disappears with animations at specific times.
        """
        self.logger.info(f"⏱️ Adding duration-based animated overlay: '{overlay.text[:30]}...'")
        
        timing = overlay.properties.timing
        start_time = timing.start_time_seconds
        overlay_duration = timing.duration_seconds or 3.0
        end_time = start_time + overlay_duration
        
        # Generate HTML/CSS with animations
        overlay_html = self._generate_animated_overlay_html(overlay, width, height)
        
        # Create overlay image
        overlay_image_path = self._render_html_to_image(overlay_html, width, height)
        
        # Apply overlay with timing and animations
        output_path = self._get_temp_video_path()
        
        # Create fade in/out filter based on animation type
        if overlay.properties.presentation_animation_type.value == "Fade In/Out":
            fade_in_duration = timing.animation_duration_in
            fade_out_duration = timing.animation_duration_out
            # Apply fade filters separately to avoid complex filter parsing
            overlay_input = ffmpeg.input(overlay_image_path)
            overlay_input = overlay_input.filter('scale', width, height)
            overlay_input = overlay_input.filter('fade', t='in', st=start_time, d=fade_in_duration)
            overlay_input = overlay_input.filter('fade', t='out', st=end_time-fade_out_duration, d=fade_out_duration)
        else:
            overlay_input = ffmpeg.input(overlay_image_path)
            overlay_input = overlay_input.filter('scale', width, height)
        
        try:
            video_input = ffmpeg.input(video_path)
            
            # Apply overlay filter
            video_with_overlay = ffmpeg.filter(
                [video_input, overlay_input], 
                'overlay', 
                x=0, y=0,
                enable=f'between(t,{start_time},{end_time})'
            )
            
            # Output - check if input has audio
            has_audio = self._has_audio_stream(video_path)
            
            if has_audio:
                # Output with both video and audio streams
                ffmpeg.output(
                    video_with_overlay, video_input['a'], output_path,
                    vcodec='libx264', 
                    acodec='copy'
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            else:
                # Output with video only
                ffmpeg.output(
                    video_with_overlay, output_path,
                    vcodec='libx264'
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            
        except ffmpeg.Error as e:
            self.logger.error(f"❌ FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            raise
        
        return output_path
    
    def _handle_progressive_reveal_overlay(self, video_path: str, overlay: TextOverlay,
                                         width: int, height: int, fps: float, duration: float) -> str:
        """
        Handle PROGRESSIVE_REVEAL presentation type.
        Text builds up progressively (word by word, character by character).
        """
        self.logger.info(f"📝 Adding progressive reveal overlay: '{overlay.text[:30]}...'")
        
        timing = overlay.properties.timing
        start_time = timing.start_time_seconds
        reveal_speed = timing.reveal_speed
        reveal_unit = timing.reveal_unit
        
        # Calculate available time for the text reveal
        # Always respect video boundaries - use the minimum of remaining video time and specified duration
        remaining_video_time = max(duration - start_time, 0.5)  # Minimum 0.5 second
        
        if timing.duration_seconds and timing.duration_seconds > 0:
            # Use the smaller of specified duration or remaining video time
            available_time = min(timing.duration_seconds, remaining_video_time)
        else:
            # Use remaining video time
            available_time = remaining_video_time
        
        self.logger.info(f"📊 Progressive reveal timing: start={start_time}s, remaining_video_time={remaining_video_time:.2f}s, available_time={available_time:.2f}s, video_duration={duration}s")
        
        # Split text based on reveal unit
        if reveal_unit == "character":
            text_units = list(overlay.text)
        elif reveal_unit == "word":
            text_units = overlay.text.split()
        elif reveal_unit == "line":
            text_units = overlay.text.split('\n')
        else:
            text_units = overlay.text.split()  # Default to words
        
        # Calculate optimal timing to fit all text within available time
        num_units = len(text_units)
        if num_units > 0:
            # Calculate time per unit to fit all text in available time
            # Reserve small buffer (0.2s) for the final text to be visible
            reveal_time = max(available_time - 0.2, 0.5)  # Minimum 0.5s for reveal
            
            # Calculate minimum time per frame based on fps
            min_time_per_frame = 1.0 / fps
            max_frames_available = int(reveal_time / min_time_per_frame)
            
            # Determine how many units to show per frame to fit everything
            if num_units <= max_frames_available:
                # We can show one unit per frame
                units_per_frame = 1
                num_steps = num_units
                time_per_step = reveal_time / num_steps
            else:
                # We need to show multiple units per frame to fit everything
                units_per_frame = int(np.ceil(num_units / max_frames_available))
                num_steps = int(np.ceil(num_units / units_per_frame))
                time_per_step = reveal_time / num_steps
                self.logger.info(f"🔧 Adaptive grouping: Showing {units_per_frame} {reveal_unit}s per frame to fit {num_units} {reveal_unit}s in {num_steps} steps")
            
            total_reveal_time = reveal_time
            
            self.logger.info(f"📝 Adaptive reveal: {num_units} {reveal_unit}s, {num_steps} steps, {units_per_frame} {reveal_unit}s/step, {time_per_step:.3f}s per step, total={total_reveal_time:.2f}s (fits in {available_time:.2f}s)")
        else:
            units_per_frame = 1
            num_steps = 1
            time_per_step = 1.0
            total_reveal_time = 1.0
        
        # Create multiple overlay images for progressive reveal with dynamic grouping
        overlay_images = []
        for step in range(num_steps + 1):
            # Calculate how many units to show in this step
            units_to_show = min(step * units_per_frame, num_units)
            
            if reveal_unit == "character":
                partial_text = ''.join(text_units[:units_to_show])
            elif reveal_unit == "word":
                partial_text = ' '.join(text_units[:units_to_show])
            elif reveal_unit == "line":
                partial_text = '\n'.join(text_units[:units_to_show])
            else:
                partial_text = ' '.join(text_units[:units_to_show])
            
            # Create overlay with partial text
            partial_overlay = TextOverlay(
                text=partial_text,
                position=overlay.position,
                properties=overlay.properties
            )
            
            if partial_text:  # Only create image if there's text
                overlay_html = self._generate_overlay_html(partial_overlay, width, height)
                image_path = self._render_html_to_image(overlay_html, width, height, suffix=f"_reveal_{step}")
                show_time = start_time + step * time_per_step
                overlay_images.append((image_path, show_time))
        
        # Apply progressive overlays
        output_path = self._get_temp_video_path()
        
        try:
            video_input = ffmpeg.input(video_path)
            current_video = video_input
            
            # Add each progressive overlay
            for i, (image_path, show_time) in enumerate(overlay_images):
                # Calculate when this overlay should end
                if i + 1 < len(overlay_images):
                    # Next overlay starts, so this one ends
                    next_show_time = overlay_images[i + 1][1]
                else:
                    # This is the final overlay - keep it visible until end of available time
                    next_show_time = start_time + available_time
                
                overlay_input = ffmpeg.input(image_path)
                current_video = ffmpeg.filter(
                    [current_video, overlay_input], 
                    'overlay', 
                    x=0, y=0,
                    enable=f'between(t,{show_time},{next_show_time})'
                )
            
            # Output - check if input has audio
            has_audio = self._has_audio_stream(video_path)
            
            if has_audio:
                # Output with both video and audio streams
                ffmpeg.output(
                    current_video, video_input['a'], output_path,
                    vcodec='libx264', 
                    acodec='copy'
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            else:
                # Output with video only
                ffmpeg.output(
                    current_video, output_path,
                    vcodec='libx264'
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            
        except ffmpeg.Error as e:
            self.logger.error(f"❌ FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            raise

        return output_path

    def _handle_sliding_window_overlay(
        self,
        video_path: str,
        overlay: TextOverlay,
        width: int,
        height: int,
        fps: float,
        duration: float,
        window_size: int,
    ) -> str:
        """Render an overlay that shows a rolling window of text during the reveal."""
        self.logger.info(
            f"📝 Adding sliding window overlay (size={window_size}): '{overlay.text[:30]}...'"
        )

        timing = overlay.properties.timing
        start_time = timing.start_time_seconds
        reveal_unit = timing.reveal_unit

        window_size = max(int(window_size or 1), 1)

        remaining_video_time = max(duration - start_time, 0.5)
        if timing.duration_seconds and timing.duration_seconds > 0:
            available_time = min(timing.duration_seconds, remaining_video_time)
        else:
            available_time = remaining_video_time

        if reveal_unit == "character":
            text_units = list(overlay.text)
        elif reveal_unit == "line":
            text_units = overlay.text.split('\n')
        else:
            text_units = overlay.text.split()

        num_units = len(text_units)
        if num_units == 0:
            return video_path

        reveal_time = max(available_time - 0.2, 0.5)
        min_time_per_frame = 1.0 / fps
        max_steps_available = max(int(reveal_time / min_time_per_frame), 1)

        if num_units <= max_steps_available:
            units_per_step = 1
            num_steps = num_units
        else:
            units_per_step = int(np.ceil(num_units / max_steps_available))
            num_steps = int(np.ceil(num_units / units_per_step))
            self.logger.info(
                f"🔧 Adaptive window grouping: advancing {units_per_step} {reveal_unit}(s) per step"
            )

        time_per_step = max(reveal_time / num_steps, min_time_per_frame)

        overlay_images = []
        for step in range(num_steps):
            units_to_include = min((step + 1) * units_per_step, num_units)
            window_start = max(units_to_include - window_size, 0)
            window_units = text_units[window_start:units_to_include]

            if reveal_unit == "character":
                partial_text = ''.join(window_units)
            elif reveal_unit == "line":
                partial_text = '\n'.join(window_units)
            else:
                partial_text = ' '.join(window_units)

            if not partial_text:
                continue

            partial_overlay = TextOverlay(
                text=partial_text,
                position=overlay.position,
                properties=overlay.properties
            )

            # Check if we should emphasize the latest word (defaults to False if not specified)
            emphasize_latest = getattr(overlay.properties, 'emphasize_latest_word', False)
            if emphasize_latest and reveal_unit == "word" and len(window_units) > 1:
                # Generate special HTML with emphasized latest word
                overlay_html = self._generate_sliding_window_html_with_emphasis(
                    partial_overlay, window_units, width, height
                )
            else:
                # Use standard HTML generation
                overlay_html = self._generate_overlay_html(partial_overlay, width, height)
            image_path = self._render_html_to_image(
                overlay_html, width, height, suffix=f"_window_{step}"
            )
            show_time = start_time + step * time_per_step
            overlay_images.append((image_path, show_time))

        if not overlay_images:
            return video_path

        output_path = self._get_temp_video_path()

        try:
            video_input = ffmpeg.input(video_path)
            current_video = video_input

            for i, (image_path, show_time) in enumerate(overlay_images):
                if i + 1 < len(overlay_images):
                    next_show_time = overlay_images[i + 1][1]
                else:
                    next_show_time = start_time + available_time

                overlay_input = ffmpeg.input(image_path)
                current_video = ffmpeg.filter(
                    [current_video, overlay_input],
                    'overlay',
                    x=0, y=0,
                    enable=f'between(t,{show_time},{next_show_time})'
                )

            has_audio = self._has_audio_stream(video_path)

            if has_audio:
                ffmpeg.output(
                    current_video, video_input['a'], output_path,
                    vcodec='libx264',
                    acodec='copy'
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            else:
                ffmpeg.output(
                    current_video, output_path,
                    vcodec='libx264'
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)

        except ffmpeg.Error as e:
            self.logger.error(f"❌ FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            raise

        return output_path

    def _generate_overlay_html(self, overlay: TextOverlay, video_width: int, video_height: int) -> str:
        """
        Generate HTML/CSS for a text overlay.
        """
        # Get position coordinates
        x, y = self._get_position_coordinates(overlay.position, video_width, video_height)
        
        # Build CSS styles
        styles = f"""
            font-family: '{overlay.properties.font_family}', sans-serif;
            font-size: {overlay.properties.font_size}px;
            color: {overlay.properties.color};
            background-color: {overlay.properties.background_color};
            padding: {overlay.properties.padding}px;
            border-radius: {overlay.properties.border_radius}px;
            position: absolute;
            left: {x}px;
            top: {y}px;
            transform: translate(-50%, -50%);
            white-space: pre-wrap;
            text-align: center;
            max-width: {video_width * 0.8}px;
            word-wrap: break-word;
        """
        
        body_height = int(video_height)
        body_width = int(video_width)
        overlay_x = int(round(x))
        overlay_y = int(round(y))
        line_height = max(overlay.properties.font_size * 1.2, overlay.properties.font_size + 2)
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link href="https://fonts.googleapis.com/css2?family={overlay.properties.font_family.replace(' ', '+')}:wght@300;400;700&display=swap" rel="stylesheet">
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    width: {body_width}px;
                    height: {body_height}px;
                    background: transparent;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .canvas {{
                    position: relative;
                    width: {video_width}px;
                    height: {video_height}px;
                }}
                .overlay {{
                    {styles}
                    overflow-wrap: anywhere;
                    display: inline-block;
                    padding: {overlay.properties.padding}px;
                    border-radius: {overlay.properties.border_radius}px;
                    background-color: {overlay.properties.background_color};
                    position: absolute;
                    left: {overlay_x}px;
                    top: {overlay_y}px;
                    transform: translate(-50%, -50%);
                    line-height: {line_height}px;
                }}
            </style>
        </head>
        <body>
            <div class="canvas">
                <div class="overlay">{overlay.text}</div>
            </div>
        </body>
        </html>
        """

        return html
    
    def _generate_animated_overlay_html(self, overlay: TextOverlay, video_width: int, video_height: int) -> str:
        """
        Generate HTML/CSS for animated text overlay.
        """
        # Get basic HTML and add CSS animations
        html = self._generate_overlay_html(overlay, video_width, video_height)
        
        # Add animation CSS based on animation type
        timing = overlay.properties.timing
        animation_css = ""
        
        if overlay.properties.presentation_animation_type.value == "Slide In/Out":
            # Add slide animations
            direction_in = timing.direction_in
            direction_out = timing.direction_out
            
            if direction_in == "bottom":
                transform_in = "translateY(100vh)"
            elif direction_in == "top":
                transform_in = "translateY(-100vh)"
            elif direction_in == "left":
                transform_in = "translateX(-100vw)"
            elif direction_in == "right":
                transform_in = "translateX(100vw)"
            else:
                transform_in = "translateY(100vh)"
            
            animation_css = f"""
                .overlay {{
                    animation: slideIn {timing.animation_duration_in}s {timing.animation_easing};
                }}
                
                @keyframes slideIn {{
                    from {{ transform: {transform_in}; }}
                    to {{ transform: translate(-50%, -50%); }}
                }}
            """
        
        # Insert animation CSS into HTML
        if animation_css:
            html = html.replace("</style>", f"{animation_css}</style>")
        
        return html
    
    def _generate_sliding_window_html_with_emphasis(self, overlay: TextOverlay, window_units: list, video_width: int, video_height: int) -> str:
        """
        Generate HTML/CSS for sliding window overlay with the latest word emphasized.
        """
        # Get position coordinates
        x, y = self._get_position_coordinates(overlay.position, video_width, video_height)
        
        # Create emphasized text with the last word larger
        normal_words = window_units[:-1]  # All words except the last
        emphasized_word = window_units[-1]  # The last word
        
        # Calculate font sizes
        normal_font_size = overlay.properties.font_size
        emphasized_font_size = normal_font_size + 6  # 6 points larger
        
        # Build the mixed content - ensure proper spacing
        normal_text = ' '.join(normal_words) if normal_words else ""
        
        # Build CSS styles for container
        container_styles = f"""
            font-family: '{overlay.properties.font_family}', sans-serif;
            color: {overlay.properties.color};
            background-color: {overlay.properties.background_color};
            padding: {overlay.properties.padding}px;
            border-radius: {overlay.properties.border_radius}px;
            position: absolute;
            left: {x}px;
            top: {y}px;
            transform: translate(-50%, -50%);
            white-space: pre-wrap;
            text-align: center;
            max-width: {video_width * 0.8}px;
            word-wrap: break-word;
            display: inline-block;
        """
        
        body_height = int(video_height)
        body_width = int(video_width)
        overlay_x = int(round(x))
        overlay_y = int(round(y))
        line_height = max(emphasized_font_size * 1.2, emphasized_font_size + 2)
        
        # Build complete text with proper spacing - simpler approach
        if normal_text and emphasized_word:
            complete_text = f"{normal_text} {emphasized_word}"
        elif normal_text:
            complete_text = normal_text  
        elif emphasized_word:
            complete_text = emphasized_word
        else:
            complete_text = ""
        
        # Split into words to rebuild with spans and proper spacing
        all_words = complete_text.split()
        total_words = len(all_words)
        normal_word_count = len(normal_words) if normal_words else 0
        
        # ROBUST APPROACH: Use explicit spacing with CSS margins and padding
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link href="https://fonts.googleapis.com/css2?family={overlay.properties.font_family.replace(' ', '+')}:wght@300;400;700&display=swap" rel="stylesheet">
            <style>
                * {{
                    box-sizing: border-box;
                }}
                body {{
                    margin: 0;
                    padding: 0;
                    width: {body_width}px;
                    height: {body_height}px;
                    background: transparent;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-family: '{overlay.properties.font_family}', sans-serif;
                }}
                .canvas {{
                    position: relative;
                    width: {video_width}px;
                    height: {video_height}px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .overlay {{
                    background-color: {overlay.properties.background_color};
                    padding: {overlay.properties.padding}px;
                    border-radius: {overlay.properties.border_radius}px;
                    text-align: center;
                    max-width: {video_width * 0.8}px;
                    line-height: {line_height}px;
                    display: inline-flex;
                    flex-wrap: wrap;
                    align-items: baseline;
                    justify-content: center;
                    gap: 0.35em;
                }}
                .word-span {{
                    display: inline-block;
                    white-space: nowrap;
                }}
                .normal-word {{
                    font-size: {normal_font_size}px;
                    font-weight: 400;
                    color: {overlay.properties.color};
                }}
                .emphasized-word {{
                    font-size: {emphasized_font_size}px;
                    font-weight: 700;
                    color: {overlay.properties.color};
                }}
            </style>
        </head>
        <body>
            <div class="canvas">
                <div class="overlay">"""
        
        # Build content word by word with explicit spacing using flexbox gap
        for i, word in enumerate(all_words):
            if i < normal_word_count:
                # This is a normal word - use word-span wrapper for proper spacing
                html += f'<span class="word-span normal-word">{word}</span>'
            else:
                # This is the emphasized word - use word-span wrapper for proper spacing
                html += f'<span class="word-span emphasized-word">{word}</span>'
        
        html += """
                </div>
            </div>
        </body>
        </html>
        """

        return html
    
    def _get_position_coordinates(self, position: TextOverlayPosition, width: int, height: int) -> Tuple[int, int]:
        """
        Convert TextOverlayPosition enum to x, y coordinates.
        """
        position_map = {
            TextOverlayPosition.TOP: (width // 2, height * 0.1),
            TextOverlayPosition.TOP_LEFT: (width * 0.1, height * 0.1),
            TextOverlayPosition.TOP_RIGHT: (width * 0.9, height * 0.1),
            TextOverlayPosition.CENTER: (width // 2, height // 2),
            TextOverlayPosition.CENTER_LEFT: (width * 0.1, height // 2),
            TextOverlayPosition.CENTER_RIGHT: (width * 0.9, height // 2),
            TextOverlayPosition.BOTTOM: (width // 2, height * 0.9),
            TextOverlayPosition.BOTTOM_LEFT: (width * 0.1, height * 0.9),
            TextOverlayPosition.BOTTOM_RIGHT: (width * 0.9, height * 0.9),
        }
        
        return position_map.get(position, (width // 2, height // 2))
    
    def _render_html_to_image(self, html: str, width: int, height: int, suffix: str = "") -> str:
        """
        Render HTML to PNG image using html2image.
        """
        filename = f"overlay{suffix}.png"
        output_path = os.path.join(self.temp_dir, filename)
        
        # Set the output directory for html2image
        self.html2img.output_path = self.temp_dir
        
        self.html2img.screenshot(
            html_str=html,
            save_as=filename,
            size=(width, height)
        )
        
        return output_path
    
    def _download_video_to_temp(self, video_url: str) -> str:
        """
        Download video from URL to temporary file, or copy local file to temp.
        """

        
        # Check if it's a local file path
        if os.path.exists(video_url) or video_url.startswith('/'):
            # It's a local file path, copy to temp directory
            temp_path = os.path.join(self.temp_dir, "input_video.mp4")
            shutil.copy2(video_url, temp_path)
            return temp_path
        else:
            # It's a URL, download it
            temp_path = os.path.join(self.temp_dir, "input_video.mp4")
            
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return temp_path
    
    def _download_video_as_bytes(self, video_url: str) -> bytes:
        """
        Download video from URL and return as bytes.
        """

        
        response = requests.get(video_url)
        response.raise_for_status()
        
        return response.content
    
    def _get_video_info(self, video_path: str) -> Tuple[int, int, float, float]:
        """
        Get video properties: width, height, fps, duration.
        """
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = float(video_stream['r_frame_rate'].split('/')[0]) / float(video_stream['r_frame_rate'].split('/')[1])
        duration = float(video_stream['duration'])
        
        return width, height, fps, duration
    
    def _has_audio_stream(self, video_path: str) -> bool:
        """
        Check if video has audio streams.
        """
        try:
            probe = ffmpeg.probe(video_path)
            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
            return len(audio_streams) > 0
        except Exception:
            return False
    
    def _get_temp_video_path(self) -> str:
        """
        Generate temporary video file path.
        """

        return os.path.join(self.temp_dir, f"output_{uuid.uuid4().hex[:8]}.mp4")
    
    def _get_video_dump_path(self) -> Path:
        """Get the path to the video_dump folder for consistent storage."""
        current_dir = Path(__file__).parent.parent.parent  # Go up to project root
        video_dump_path = current_dir / "storage" / "video_dump"
        video_dump_path.mkdir(parents=True, exist_ok=True)
        return video_dump_path
    
    def _cleanup_temp_files(self):
        """
        Clean up temporary files in the video_dump directory.
        """
        try:
            # Clean up only files we created, not the entire directory
            temp_path = Path(self.temp_dir)
            if temp_path.exists():
                # Remove overlay images and temporary helper videos (keep final outputs for inspection)
                for file_pattern in ["overlay*.png", "temp_*.mp4"]:
                    for temp_file in temp_path.glob(file_pattern):
                        try:
                            temp_file.unlink()
                            self.logger.debug(f"🗑️ Cleaned up: {temp_file.name}")
                        except Exception as file_error:
                            self.logger.warning(f"⚠️ Could not clean up {temp_file}: {file_error}")
        except Exception as e:
            self.logger.warning(f"⚠️ Warning: Could not clean up temp files: {e}")
        
        # Keep using the same video_dump directory - no need to recreate
        self.temp_dir = str(self._get_video_dump_path())
    
    def close(self):
        """Explicitly close and clean up resources."""
        self._cleanup_temp_files()
    
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
