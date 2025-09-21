#!/usr/bin/env python3
"""
Independent Video Stitching Tool
================================

A standalone video processing tool that can download and stitch together multiple MP4 videos.

Usage:
    python video_tools.py
    
Features:
- Download videos from URLs
- Stitch multiple videos together
- Command-line interface
- Progress tracking
- Error handling with retries

Dependencies:
    pip install requests moviepy
"""

import sys
import logging
import tempfile
import shutil
import random
import requests
import traceback
from pathlib import Path
from typing import List, Optional, Union
import argparse

from utils.audio_video_tools import AudioVideoMixer
from utils.video_tools.video_stitcher import MOVIEPY_AVAILABLE, VideoStitcher
from utils.video_tools.video_text_overlay import VideoTextOverlayManager
from models.video_engine_models import TextOverlay, TextOverlayProperties, TextOverlayTiming, TextPresentationType, TextOverlayPosition


def get_video_dump_path() -> Path:
    """Get the path to the video_dump folder for consistent storage across all video tools."""
    current_dir = Path(__file__).parent.parent  # Go up to project root
    video_dump_path = current_dir / "storage" / "video_dump"
    video_dump_path.mkdir(parents=True, exist_ok=True)
    return video_dump_path


def get_video_tools_logger(name: str = "VideoTools") -> logging.Logger:
    """Create or fetch a configured logger for video tools components.

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


class VideoTools:
    """
    Container for video processing tools.
    
    Provides access to VideoStitcher and AudioVideoMixer instances with shared configuration.
    Use the individual tools directly through .stitcher and .mixer attributes.
    
    Example:
        tools = VideoTools()
        
        # Use stitcher directly
        result = tools.stitcher.stitch_videos(urls, "output.mp4")
        
        # Use mixer directly  
        mixed = tools.mixer.mix_audio_video_ffmpeg(video_bytes, audio_bytes, "mixed.mp4")
    """
    
    def __init__(self, output_dir: str = "output", temp_dir: Optional[str] = None):
        """
        Initialize VideoTools with shared configuration.
        
        Args:
            output_dir: Directory for output files
            temp_dir: Directory for temporary files (auto-generated if None)
        """
        self.logger = get_video_tools_logger(f"VideoTools.{id(self)}")
        self.stitcher = VideoStitcher(output_dir=output_dir, temp_dir=temp_dir)
        self.mixer = AudioVideoMixer(output_dir=output_dir, temp_dir=temp_dir)
        self.text_overlay_manager = VideoTextOverlayManager()


def run_all_video_transitions():
    """Test all available video transitions on two sample videos."""
    logger = get_video_tools_logger("run_all_video_transitions")
    
    # Here we have the link to two separate videos
    video_links = [
        "https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/57b6e8eb-5c8f-4691-82ce-b33bde620e00.mp4",
        "https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/eb7355a5-01bd-4363-bde9-747ba7c049b1.mp4"
    ]

    # Define all available transitions with descriptive names
    transitions = [
        ("Cut", "basic_cut"),
        ("Dissolve", "crossfade_dissolve"),
        ("FadeBlack", "fade_to_black"),
        ("Slide", "slide_left"),
        ("WhipPan", "whip_pan_blur"),
        ("BlurDissolve", "blur_crossfade"),
        ("Glitch", "digital_glitch")
    ]

    logger.info(f"🎬 Testing {len(transitions)} transition effects on sample videos...")
    logger.info(f"📹 Video 1: {video_links[0].split('/')[-1]}")
    logger.info(f"📹 Video 2: {video_links[1].split('/')[-1]}")
    logger.info("")

    successful_transitions = 0
    failed_transitions = 0
    output_files = []

    for i, (transition_type, filename_suffix) in enumerate(transitions, 1):
        tools = None
        try:
            logger.info(f"[{i:2d}/{len(transitions)}] 🔄 Testing {transition_type} transition...")
            
            # Create output filename
            output_filename = f"transition_test_{filename_suffix}.mp4"
            final_output_path = Path("output") / output_filename
            
            # Initialize VideoTools with consistent storage directory
            temp_dir = get_video_dump_path()
            tools = VideoTools(output_dir="output", temp_dir=str(temp_dir))
            
            # Apply the transition - result is path to transition_output.mp4
            result_path = tools.stitcher.stitch_with_transition(
                video_urls=video_links,
                transition_types=[transition_type],  # Single transition between two videos
                return_bytes=False
            )
            
            # Immediately move the result file to prevent overwriting
            source_path = Path(result_path)
            
            if source_path.exists():
                # Ensure output directory exists
                final_output_path.parent.mkdir(exist_ok=True)
                
                # Remove existing file if present
                if final_output_path.exists():
                    final_output_path.unlink()
                
                # Move to final location with descriptive name
                shutil.move(str(source_path), str(final_output_path))
                output_files.append(str(final_output_path))
                
                logger.info(f"            ✅ Success! Saved as: {output_filename}")
                successful_transitions += 1
            else:
                logger.warning(f"            ❌ Failed: Output file not found at {result_path}")
                failed_transitions += 1
            
        except Exception as e:
            logger.error(f"            ❌ Failed: {str(e)}")
            failed_transitions += 1
        finally:
            # Cleanup tools if created
            if tools:
                try:
                    tools.stitcher.close()
                except:
                    pass
        
        logger.info("")  # Add spacing between tests

    # Final summary
    logger.info("=" * 60)
    logger.info(f"🎯 Transition Testing Complete!")
    logger.info(f"   ✅ Successful: {successful_transitions}/{len(transitions)}")
    logger.info(f"   ❌ Failed: {failed_transitions}/{len(transitions)}")
    
    if successful_transitions > 0:
        logger.info(f"   📁 Output files saved in: output/")
        logger.info("   📋 Generated files:")
        for filepath in output_files:
            if Path(filepath).exists():
                filename = Path(filepath).name
                filesize = Path(filepath).stat().st_size / (1024*1024)
                logger.info(f"      - {filename} ({filesize:.1f} MB)")
    
    # List all files in output directory to verify
    output_dir = Path("output")
    if output_dir.exists():
        all_files = list(output_dir.glob("transition_test_*.mp4"))
        if all_files:
            logger.info(f"\n   🔍 Total transition test files in output/: {len(all_files)}")
            for file_path in sorted(all_files):
                filesize = file_path.stat().st_size / (1024*1024)
                logger.info(f"      📄 {file_path.name} ({filesize:.1f} MB)")
    
    return successful_transitions, failed_transitions


def run_individual_transition(transition_name: str):
    """Test a specific transition type on two sample videos."""
    logger = get_video_tools_logger("run_individual_transition")
    
    video_links = [
        "https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/57b6e8eb-5c8f-4691-82ce-b33bde620e00.mp4",
        "https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/eb7355a5-01bd-4363-bde9-747ba7c049b1.mp4"
    ]
    
    # Map transition names to their types
    transition_map = {
        "cut": "Cut",
        "dissolve": "Dissolve",
        "fadeblack": "FadeBlack",
        "slide": "Slide",
        "whippan": "WhipPan",
        "blurdissolve": "BlurDissolve",
        "glitch": "Glitch"
    }
    
    transition_type = transition_map.get(transition_name.lower())
    if not transition_type:
        logger.error(f"❌ Unknown transition: {transition_name}")
        logger.error(f"Available transitions: {', '.join(transition_map.keys())}")
        return
    
    logger.info(f"🎬 Testing {transition_type} transition...")
    logger.info(f"📹 Video 1: {video_links[0].split('/')[-1]}")
    logger.info(f"📹 Video 2: {video_links[1].split('/')[-1]}")
    logger.info("")
    
    try:
        # Use consistent storage directory
        temp_dir = get_video_dump_path()
        tools = VideoTools(output_dir="output", temp_dir=str(temp_dir))
        
        # Create output filename
        output_filename = f"single_transition_{transition_name}.mp4"
        final_output_path = Path("output") / output_filename
        
        # Apply the transition
        result_path = tools.stitcher.stitch_with_transition(
            video_urls=video_links,
            transition_types=[transition_type],
            return_bytes=False
        )
        
        # Move result to final location
        source_path = Path(result_path)
        if source_path.exists():
            # Ensure output directory exists
            final_output_path.parent.mkdir(exist_ok=True)
            
            # Remove existing file if present
            if final_output_path.exists():
                final_output_path.unlink()
            
            # Move to final location
            shutil.move(str(source_path), str(final_output_path))
            
            # Check file size
            size_mb = final_output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Success! Saved as: {output_filename} ({size_mb:.1f} MB)")
            
        else:
            logger.warning(f"❌ Failed: Output file not found at {result_path}")
            
    except Exception as e:
        logger.error(f"❌ Failed: {str(e)}")
    finally:
        # Cleanup
        try:
            if 'tools' in locals():
                tools.stitcher.close()
        except:
            pass

def test_overlays():
    """Test text overlay functionality with a sample video."""
    logger = get_video_tools_logger("test_overlays")
    logger.info("🎬 Testing Text Overlay Functionality - Comprehensive Test")
    logger.info("=" * 60)
    
    try:
        # Use consistent storage directory
        temp_dir = get_video_dump_path()
        tools = VideoTools(output_dir="output", temp_dir=str(temp_dir))

        video_url = "https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/57b6e8eb-5c8f-4691-82ce-b33bde620e00.mp4"
        
        logger.info(f"📹 Testing video: {video_url.split('/')[-1]}")
        logger.info("📥 Downloading video once for reuse...")
        
        # Download the video once and save to temp directory
        try:
            response = requests.get(video_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save video to temp file
            local_video_path = Path(temp_dir) / "test_video.mp4"
            with open(local_video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            video_size_mb = local_video_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Video downloaded successfully ({video_size_mb:.1f} MB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to download video: {e}")
            return False
        
        logger.info("🎯 Generating 10 test videos with 3 text overlays each...")
        logger.info("")
        
        # Define different fonts, colors, and texts for variety
        fonts = ["Roboto", "Montserrat", "Open Sans", "Lato", "Oswald", "Playfair Display"]
        colors = ["#ffffff", "#ff6b35", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7", "#fd79a8", "#e17055"]
        bg_colors = ["rgba(0, 0, 0, 0.7)", "rgba(255, 255, 255, 0.9)", "rgba(52, 73, 94, 0.8)", "rgba(231, 76, 60, 0.7)"]
        
        static_texts = [
            "Brand Name", "Premium Quality", "Now Available", "Limited Time", "Special Offer",
            "New Collection", "Best Seller", "Trending Now", "Exclusive Deal", "Top Rated"
        ]
        
        animated_texts = [
            "Call to Action!", "Subscribe Now!", "Learn More", "Get Started", "Join Today!",
            "Download App", "Book Now", "Try Free", "Order Today", "Contact Us"
        ]
        
        progressive_texts = [
            "Amazing Features Await", "Discover Something New", "Quality You Can Trust", 
            "Innovation Meets Design", "Your Journey Starts Here", "Excellence in Every Detail",
            "Transform Your Experience", "Unleash Your Potential", "Create Something Beautiful", "Build Your Dreams"
        ]
        
        successful_videos = 0
        failed_videos = 0
        
        for video_num in range(1, 11):
            logger.info(f"🎬 [{video_num:2d}/10] Creating test video {video_num}...")
            
            try:
                # Create 3 different overlay types for this video
                overlays = [
                    # 1. Static overlay - top
                    TextOverlay(
                        text=random.choice(static_texts),
                        position=TextOverlayPosition.TOP,
                        properties=TextOverlayProperties(
                            font_family=random.choice(fonts),
                            font_size=random.randint(28, 36),
                            color=random.choice(colors),
                            background_color=random.choice(bg_colors),
                            padding=random.randint(8, 15),
                            border_radius=random.randint(3, 8),
                            presentation_type=TextPresentationType.STATIC_OVERLAY,
                            timing=TextOverlayTiming(
                                start_time_seconds=0,
                                duration_seconds=random.uniform(3.0, 5.0)
                            )
                        )
                    ),
                    
                    # 2. Duration-based animated overlay - center
                    TextOverlay(
                        text=random.choice(animated_texts),
                        position=TextOverlayPosition.CENTER,
                        properties=TextOverlayProperties(
                            font_family=random.choice(fonts),
                            font_size=random.randint(24, 32),
                            color=random.choice(colors),
                            background_color=random.choice(bg_colors),
                            padding=random.randint(10, 18),
                            border_radius=random.randint(5, 12),
                            presentation_type=TextPresentationType.DURATION_BASED_WITH_ANIMATION,
                            timing=TextOverlayTiming(
                                start_time_seconds=random.uniform(1.0, 2.5),
                                duration_seconds=random.uniform(2.5, 4.0),
                                animation_duration_in=random.uniform(0.3, 0.8),
                                animation_duration_out=random.uniform(0.3, 0.8)
                            )
                        )
                    ),
                    
                    # 3. Progressive reveal overlay - bottom
                    TextOverlay(
                        text=random.choice(progressive_texts),
                        position=TextOverlayPosition.BOTTOM,
                        properties=TextOverlayProperties(
                            font_family=random.choice(fonts),
                            font_size=random.randint(20, 28),
                            color=random.choice(colors),
                            background_color=random.choice(bg_colors),
                            padding=random.randint(8, 12),
                            border_radius=random.randint(4, 10),
                            presentation_type=TextPresentationType.PROGRESSIVE_REVEAL,
                            timing=TextOverlayTiming(
                                start_time_seconds=random.uniform(0.5, 1.5),
                                reveal_speed=random.uniform(1.5, 3.0),  # words per second
                                reveal_unit="word",
                                duration_seconds=random.uniform(3.0, 5.0)
                            )
                        )
                    )
                ]
                
                logger.info(f"      � Static: '{overlays[0].text}' ({overlays[0].properties.font_family})")
                logger.info(f"      🎬 Animated: '{overlays[1].text}' ({overlays[1].properties.font_family})")
                logger.info(f"      ⚡ Progressive: '{overlays[2].text}' ({overlays[2].properties.font_family})")
                
                # Process the video with overlays (using local file path)
                logger.info(f"      🔄 Processing overlays...")
                result = tools.text_overlay_manager.add_text_overlays_to_video(str(local_video_path), overlays)
                
                if result is not None and len(result) > 0:
                    # Save the result to output directory
                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)
                    
                    output_path = output_dir / f"test_overlays_video_{video_num:02d}.mp4"
                    with open(output_path, 'wb') as f:
                        f.write(result)
                    
                    # Check file size
                    size_mb = len(result) / (1024 * 1024)
                    logger.info(f"      ✅ Success! Saved as: {output_path.name} ({size_mb:.1f} MB)")
                    successful_videos += 1
                else:
                    logger.warning(f"      ❌ Failed: No result returned")
                    failed_videos += 1
                    
            except Exception as e:
                logger.error(f"      ❌ Failed: {str(e)}")
                failed_videos += 1
            
            logger.info("")  # Add spacing between videos
        
        # Final summary
        logger.info("=" * 60)
        logger.info(f"🎯 Comprehensive Text Overlay Test Complete!")
        logger.info(f"   ✅ Successful videos: {successful_videos}/10")
        logger.info(f"   ❌ Failed videos: {failed_videos}/10")
        logger.info(f"   📊 Total overlays tested: {successful_videos * 3} overlays")
        
        if successful_videos > 0:
            logger.info(f"   📁 Output files saved in: output/")
            logger.info("   📋 Generated test videos:")
            
            # List all generated files
            output_dir = Path("output")
            test_files = list(output_dir.glob("test_overlays_video_*.mp4"))
            total_size = 0
            
            for file_path in sorted(test_files):
                filesize = file_path.stat().st_size / (1024*1024)
                total_size += filesize
                logger.info(f"      - {file_path.name} ({filesize:.1f} MB)")
            
            logger.info(f"   💾 Total size: {total_size:.1f} MB")
            logger.info(f"   🎬 Each video contains 3 text overlay types:")
            logger.info(f"      • Static Overlay (TOP position)")
            logger.info(f"      • Duration-based Animation (CENTER position)")  
            logger.info(f"      • Progressive Reveal (BOTTOM position)")
        
        return successful_videos > 0
            
    except Exception as e:
        logger.error(f"   ❌ Failed: {str(e)}")
        logger.error(f"   🐛 Debug info: {traceback.format_exc()}")
        return False
    finally:
        # Cleanup
        try:
            if 'tools' in locals():
                tools.text_overlay_manager._cleanup_temp_files()
        except Exception as e:
            logger.warning(f"   ⚠️ Warning: Cleanup failed: {e}")
    
def test_audio_synced_overlays():
    """Test audio-synced text overlay functionality with a sample video."""
    logger = get_video_tools_logger("test_audio_synced_overlays")
    logger.info("🎵 Testing Audio-Synced Text Overlay Functionality")
    logger.info("=" * 55)
    
    try:
        # Use consistent storage directory
        temp_dir = get_video_dump_path()
        tools = VideoTools(output_dir="output", temp_dir=str(temp_dir))

        video_url = "https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/57b6e8eb-5c8f-4691-82ce-b33bde620e00.mp4"
        
        logger.info(f"📹 Testing video: {video_url.split('/')[-1]}")
        logger.info("📥 Downloading video once for reuse...")
        
        # Download the video once and save to temp directory
        try:
            response = requests.get(video_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save video to temp file
            local_video_path = Path(temp_dir) / "test_video.mp4"
            with open(local_video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            video_size_mb = local_video_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Video downloaded successfully ({video_size_mb:.1f} MB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to download video: {e}")
            return False
        
        logger.info("🎯 Generating 1 test video with audio-synced overlay...")
        logger.info("")
        
        # Define different fonts, colors, and texts for variety
        fonts = ["Roboto", "Montserrat", "Open Sans", "Lato", "Oswald", "Playfair Display"]
        colors = ["#ffffff", "#ff6b35", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7", "#fd79a8", "#e17055"]
        bg_colors = ["rgba(0, 0, 0, 0.7)", "rgba(255, 255, 255, 0.9)", "rgba(52, 73, 94, 0.8)", "rgba(231, 76, 60, 0.7)"]
        
        # Simple audio-synced test - complete text appears when speech starts
        audio_synced_test = {
            "text": "Did you know the sky is actually blue because it's constantly embarrassed?",
            "start_time": 0.5,  # Appears when speech might start
            "duration": 4.0,    # Shows for 4 seconds
            "font": "Montserrat",
            "size": 32,
            "position": TextOverlayPosition.CENTER
        }
        
        successful_videos = 0
        failed_videos = 0
        
        logger.info(f"🎵 [1/1] Creating audio-synced test video...")
        
        try:
            # Create audio-synced overlay
            overlay = TextOverlay(
                text=audio_synced_test["text"],
                position=audio_synced_test["position"],
                properties=TextOverlayProperties(
                    font_family=audio_synced_test["font"],
                    font_size=audio_synced_test["size"],
                    color=random.choice(colors),
                    background_color=random.choice(bg_colors),
                    padding=15,
                    border_radius=8,
                    presentation_type=TextPresentationType.PROGRESSIVE_REVEAL,
                    timing=TextOverlayTiming(
                        start_time_seconds=audio_synced_test["start_time"],
                        duration_seconds=audio_synced_test["duration"]
                    )
                )
            )
            
            logger.info(f"      🎵 Audio Sync: '{overlay.text}' ({overlay.properties.font_family})")
            logger.info(f"      ⏰ Timing: {audio_synced_test['start_time']}s → {audio_synced_test['start_time'] + audio_synced_test['duration']}s")
            
            # Process the video with overlay (using local file path)
            logger.info(f"      🔄 Processing audio-synced overlay...")
            result = tools.text_overlay_manager.add_text_overlays_to_video(str(local_video_path), [overlay])
            
            if result is not None and len(result) > 0:
                # Save the result to output directory
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                
                output_path = output_dir / "test_audio_synced_simple.mp4"
                with open(output_path, 'wb') as f:
                    f.write(result)
                
                # Check file size
                size_mb = len(result) / (1024 * 1024)
                logger.info(f"      ✅ Success! Saved as: {output_path.name} ({size_mb:.1f} MB)")
                successful_videos += 1
            else:
                logger.warning(f"      ❌ Failed: No result returned")
                failed_videos += 1
                
        except Exception as e:
            logger.error(f"      ❌ Failed: {str(e)}")
            failed_videos += 1
        
        logger.info("")  # Add spacing
        
        # Final summary
        logger.info("=" * 55)
        logger.info(f"🎯 Audio-Synced Text Overlay Test Complete!")
        logger.info(f"   ✅ Successful videos: {successful_videos}/1")
        logger.info(f"   ❌ Failed videos: {failed_videos}/1")
        logger.info(f"   📊 Total audio-synced overlays tested: {successful_videos} overlay")
        
        if successful_videos > 0:
            logger.info(f"   📁 Output files saved in: output/")
            logger.info("   📋 Generated test video:")
            
            # List the generated file
            output_dir = Path("output")
            test_file = output_dir / "test_audio_synced_simple.mp4"
            
            if test_file.exists():
                filesize = test_file.stat().st_size / (1024*1024)
                logger.info(f"      - {test_file.name} ({filesize:.1f} MB)")
                logger.info(f"   💾 Total size: {filesize:.1f} MB")
            
            logger.info(f"   🎵 Audio-synced overlay shows complete text when speech cue starts")
            logger.info(f"   ⏰ Simple timing: appears at 0.5s, shows for 4.0s")
        
        return successful_videos > 0
            
    except Exception as e:
        logger.error(f"   ❌ Failed: {str(e)}")
        logger.error(f"   🐛 Debug info: {traceback.format_exc()}")
        return False
    finally:
        # Cleanup
        try:
            if 'tools' in locals():
                tools.text_overlay_manager._cleanup_temp_files()
        except Exception as e:
            logger.warning(f"   ⚠️ Warning: Cleanup failed: {e}")
    
    logger.info("=" * 55)

def main():
    """
    Command-line interface for video stitching.
    """
    logger = get_video_tools_logger("main")
    logger.info("🎬 Video Tools Test Suite")
    logger.info("========================")
    logger.info("")
    logger.info("Available tests:")
    logger.info("  [TRANSITIONS]")
    transitions = ["cut", "dissolve", "fadeblack", "slide", "whippan", "blurdissolve", "glitch"]
    
    for i, trans in enumerate(transitions, 1):
        logger.info(f"  {i:2d}. {trans}")
    
    logger.info("")
    logger.info("  [TEXT OVERLAY TOOLS]")
    logger.info("   8. Test text overlays (comprehensive)")
    logger.info("  15. Test audio-synced overlays")
    logger.info("")
    logger.info("  [BATCH OPERATIONS]")
    logger.info("   0. Run all transitions")
    logger.info("")
    
    user_input = input("Enter your choice: ").strip()
    
    if user_input == "0":
        logger.info("Running all transitions...")
        run_all_video_transitions()
    elif user_input == "8":
        logger.info("Testing text overlays...")
        test_overlays()
    elif user_input == "15":
        logger.info("Testing audio-synced overlays...")
        test_audio_synced_overlays()
    elif user_input.isdigit() and 1 <= int(user_input) <= 7:
        transition_name = transitions[int(user_input) - 1]
        run_individual_transition(transition_name)
    elif user_input.lower() in transitions:
        run_individual_transition(user_input.lower())
    else:
        logger.error(f"❌ Invalid choice: {user_input}")
        logger.error("Please run again and choose a valid option.")


if __name__ == "__main__":
    main()
