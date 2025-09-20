"""
Modal Manager for Image and Video Generation

This class handles various AI-powered image and video generation tasks including:
- Image generation from text prompts
- Image editing and manipulation
- Video generation from single images
- Video generation from first and last frame images
"""

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ssl
import json
import time
import threading
import sys
import os
import random
import base64
from datetime import datetime
from typing import Optional, Dict, Any


class ProgressMonitor:
    """Live progress monitor with animated indicators"""
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.start_time = None
        
    def start(self, message="Processing"):
        """Start the progress animation"""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._animate, args=(message,))
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the progress animation"""
        self.running = False
        if self.thread:
            self.thread.join()
        print()  # New line after animation
        
    def _animate(self, message):
        """Animated progress indicator"""
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        frame_idx = 0
        
        while self.running:
            elapsed = time.time() - self.start_time
            elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
            
            # Clear line and show progress
            sys.stdout.write(f"\r{frames[frame_idx]} {message}... [{elapsed_str}] ")
            sys.stdout.flush()
            
            frame_idx = (frame_idx + 1) % len(frames)
            time.sleep(0.1)


class ModalManager:
    """Manager for AI-powered image and video generation operations"""
    
    # API endpoint URLs - Updated to use combined Qwen app
    QWEN_SUBMIT_API = "https://amirasemanpayeh--comfyapp-qwen-full-v0-1-comfyui-submit--12715b.modal.run"
    QWEN_STATUS_API = "https://amirasemanpayeh--comfyapp-qwen-full-v0-1-comfyui-get-job-status.modal.run"
    VIDEO_SUBMIT_API = "https://amirasemanpayeh--comfyapp-wan2-2-flftv-v0-1-comfyui-subm-f33226.modal.run"
    VIDEO_STATUS_API = "https://amirasemanpayeh--comfyapp-wan2-2-flftv-v0-1-comfyui-get--dfbcf3.modal.run"
    MUSIC_SUBMIT_API = "https://amirasemanpayeh--audio-vision-tools-v0-1-audiovisiontool-5c9fce.modal.run"
    MUSIC_STATUS_API = "https://amirasemanpayeh--audio-vision-tools-v0-1-audiovisiontool-e6421c.modal.run"
    
    def __init__(self):
        """Initialize the ModalManager"""
        print("ModalManager initialized")
        print(f"Qwen Combined Submit API: {self.QWEN_SUBMIT_API}")
        print(f"Qwen Combined Status API: {self.QWEN_STATUS_API}")
        print(f"Video Submit API: {self.VIDEO_SUBMIT_API}")
        print(f"Video Status API: {self.VIDEO_STATUS_API}")
        print(f"Music Submit API: {self.MUSIC_SUBMIT_API}")
        print(f"Music Status API: {self.MUSIC_STATUS_API}")
        
        # Create a robust session with SSL configuration and retries
        self.session = self._create_robust_session()

    def _create_robust_session(self) -> requests.Session:
        """Create a requests session with SSL configuration and retry strategy"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'User-Agent': 'AdVoyage-Backend/1.0',
            'Connection': 'keep-alive'
        })
        
        # Handle SSL verification issues in development
        try:
            # Try to verify SSL normally first
            test_response = session.get("https://httpbin.org/get", timeout=5)
            test_response.raise_for_status()
        except (requests.exceptions.SSLError, ssl.SSLError):
            print("⚠️ SSL verification issues detected - using fallback configuration")
            session.verify = False
            # Disable SSL warnings when verification is disabled
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            # Other connection issues - keep defaults
            pass
            
        return session
    
    def generate_image_from_prompt(self, 
                                 prompt: str, 
                                 width: int = 1024, 
                                 height: int = 1024, 
                                 batch_size: int = 1) -> Optional[bytes]:
        """Generate an image from a text prompt using async job submission
        
        Args:
            prompt: Text description of the image to generate
            width: Image width in pixels (default: 1024)
            height: Image height in pixels (default: 1024)
            batch_size: Number of images to generate (default: 1)
            
        Returns:
            Image data as bytes or None if failed
        """
        
        # Submit the job
        job_id = self._submit_image_job(prompt, width, height, batch_size)
        
        if not job_id:
            print("❌ Failed to submit image generation job")
            return None
        
        # Wait for completion and get image data
        image_data = self._wait_for_image_completion(job_id)
        
        return image_data
    
    def _submit_image_job(self, prompt: str, width: int, height: int, batch_size: int) -> Optional[str]:
        """Submit a TTI (text-to-image) generation job and return job ID"""
        
        print(f"🚀 Submitting TTI (text-to-image) generation job...")
        print(f"📝 Prompt: {prompt}")
        print(f"📊 Parameters: {width}x{height}, batch: {batch_size}")
        
        # Prepare TTI parameters for the combined Qwen app
        tti_params = {
            "workflow_type": "tti",
            "prompt": prompt,
            "width": width,
            "height": height,
            "batch_size": batch_size
        }
        
        try:
            response = self.session.post(self.QWEN_SUBMIT_API, json=tti_params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            job_id = result["job_id"]
            workflow_type = result.get("workflow_type", "tti")
            print(f"✅ TTI job submitted successfully!")
            print(f"📋 Job ID: {job_id}")
            print(f"🔧 Workflow: {workflow_type}")
            return job_id
            
        except Exception as e:
            print(f"❌ Failed to submit TTI job: {e}")
            return None

    def _wait_for_image_completion(self, job_id: str) -> Optional[bytes]:
        """Wait for TTI image generation completion and return image data"""
        
        print(f"\n🔄 Waiting for TTI image generation to complete...")
        print(f"📋 Job ID: {job_id}")
        
        max_attempts = 60  # 10 minutes at 10-second intervals for high-quality generation
        monitor = ProgressMonitor()
        monitor.start("🎨 Generating image")
        
        try:
            for attempt in range(max_attempts):
                try:
                    status_response = self.session.post(
                        self.QWEN_STATUS_API,
                        json={"job_id": job_id},
                        timeout=30
                    )
                    
                    # Success case - check for completion
                    if status_response.status_code == 200:
                        try:
                            status_data = status_response.json()
                            status = status_data.get("status")
                            
                            if status == "completed":
                                # Extract base64 image data from the new format
                                result = status_data.get("result", {})
                                image_base64 = result.get("image_base64")
                                
                                if image_base64:
                                    # Decode base64 to bytes
                                    image_bytes = base64.b64decode(image_base64)
                                    monitor.stop()
                                    print("🎉 TTI image generation completed!")
                                    
                                    image_size = len(image_bytes)
                                    print(f"📏 Image size: {image_size / 1024:.1f} KB")
                                    
                                    return image_bytes
                                else:
                                    monitor.stop()
                                    print(f"❌ No image data in response")
                                    return None
                            
                            elif status == "error":
                                error_msg = status_data.get("error", "Unknown error")
                                monitor.stop()
                                print(f"❌ Job failed: {error_msg}")
                                return None
                            
                            elif status in ["submitted", "running"]:
                                print(f"⏳ Status: {status} - waiting 10 seconds...")
                                time.sleep(10)
                                continue
                            
                            else:
                                print(f"⚠️ Unknown status: {status}")
                                time.sleep(10)
                                continue
                                
                        except Exception as json_error:
                            print(f"⚠️ Error parsing response: {json_error}")
                            if attempt < 50:  # Allow more time for complex generations
                                time.sleep(10)
                                continue
                            else:
                                monitor.stop()
                                return None
                    
                    # Handle processing states and temporary errors
                    elif status_response.status_code == 202:
                        print(f"⏳ Status: processing - waiting 10 seconds...")
                        time.sleep(10)
                        continue
                    
                    else:
                        print(f"⚠️ Status code: {status_response.status_code}")
                        if attempt < 50:
                            print(f"⏳ Retrying in 10 seconds...")
                            time.sleep(10)
                            continue
                        else:
                            monitor.stop()
                            print(f"❌ Job failed after multiple retries")
                            return None
                            
                except Exception as e:
                    if attempt < 50:
                        print(f"⏳ Retrying in 10 seconds...")
                        time.sleep(10)
                    else:
                        monitor.stop()
                        print(f"❌ Giving up after {max_attempts} attempts: {e}")
                        return None
            
            monitor.stop()
            print("⏰ Timeout: TTI image generation took longer than 10 minutes")
            return None
            
        except Exception as e:
            monitor.stop()
            print(f"❌ Failed to wait for TTI image completion: {e}")
            return None

    def _submit_image_edit_job(self, input_image_url: str, prompt: str) -> Optional[str]:
        """Submit an ITI (image-to-image) editing job and return job ID"""
        
        print(f"🚀 Submitting ITI (image-to-image) editing job...")
        print(f"📸 Input Image: {input_image_url}")
        print(f"📝 Edit Prompt: {prompt}")
        
        # Prepare ITI parameters for the combined Qwen app
        iti_params = {
            "workflow_type": "iti",
            "prompt": prompt,
            "input_image_url": input_image_url
        }
        
        try:
            response = self.session.post(self.QWEN_SUBMIT_API, json=iti_params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            job_id = result["job_id"]
            workflow_type = result.get("workflow_type", "iti")
            print(f"✅ ITI job submitted successfully!")
            print(f"📋 Job ID: {job_id}")
            print(f"🔧 Workflow: {workflow_type}")
            return job_id
            
        except Exception as e:
            print(f"❌ Failed to submit ITI job: {e}")
            return None

    def _wait_for_image_edit_completion(self, job_id: str) -> Optional[bytes]:
        """Wait for ITI image editing completion and return image data"""
        
        print(f"\n🔄 Waiting for ITI image editing to complete...")
        print(f"📋 Job ID: {job_id}")
        
        max_attempts = 60  # 10 minutes at 10-second intervals for high-quality editing
        monitor = ProgressMonitor()
        monitor.start("🎨 Editing image")
        
        try:
            for attempt in range(max_attempts):
                try:
                    status_response = self.session.post(
                        self.QWEN_STATUS_API,
                        json={"job_id": job_id},
                        timeout=30
                    )
                    
                    # Success case - check for completion
                    if status_response.status_code == 200:
                        try:
                            status_data = status_response.json()
                            status = status_data.get("status")
                            
                            if status == "completed":
                                # Extract base64 image data from the new format
                                result = status_data.get("result", {})
                                image_base64 = result.get("image_base64")
                                
                                if image_base64:
                                    # Decode base64 to bytes
                                    image_bytes = base64.b64decode(image_base64)
                                    monitor.stop()
                                    print("🎉 ITI image editing completed!")
                                    
                                    image_size = len(image_bytes)
                                    print(f"📏 Image size: {image_size / 1024:.1f} KB")
                                    
                                    return image_bytes
                                else:
                                    monitor.stop()
                                    print(f"❌ No image data in response")
                                    return None
                            
                            elif status == "error":
                                error_msg = status_data.get("error", "Unknown error")
                                monitor.stop()
                                print(f"❌ Job failed: {error_msg}")
                                return None
                            
                            elif status in ["submitted", "running"]:
                                print(f"⏳ Status: {status} - waiting 10 seconds...")
                                time.sleep(10)
                                continue
                            
                            else:
                                print(f"⚠️ Unknown status: {status}")
                                time.sleep(10)
                                continue
                                
                        except Exception as json_error:
                            print(f"⚠️ Error parsing response: {json_error}")
                            if attempt < 50:  # Allow more time for complex edits
                                time.sleep(10)
                                continue
                            else:
                                monitor.stop()
                                return None
                    
                    # Handle processing states and temporary errors
                    elif status_response.status_code == 202:
                        print(f"⏳ Status: processing - waiting 10 seconds...")
                        time.sleep(10)
                        continue
                    
                    else:
                        print(f"⚠️ Status code: {status_response.status_code}")
                        if attempt < 50:
                            print(f"⏳ Retrying in 10 seconds...")
                            time.sleep(10)
                            continue
                        else:
                            monitor.stop()
                            print(f"❌ Job failed after multiple retries")
                            return None
                            
                except Exception as e:
                    if attempt < 50:
                        print(f"⏳ Retrying in 10 seconds...")
                        time.sleep(10)
                    else:
                        monitor.stop()
                        print(f"❌ Giving up after {max_attempts} attempts: {e}")
                        return None
            
            monitor.stop()
            print("⏰ Timeout: ITI image editing took longer than 10 minutes")
            return None
            
        except Exception as e:
            monitor.stop()
            print(f"❌ Failed to wait for ITI image edit completion: {e}")
            return None

    def _submit_video_job(self, workflow_type: str, prompt: str, width: int, height: int, 
                         frames: int, fps: int, image_url: Optional[str] = None,
                         first_frame_url: Optional[str] = None, last_frame_url: Optional[str] = None,
                         audio_url: Optional[str] = None, audio1_url: Optional[str] = None, 
                         audio2_url: Optional[str] = None) -> Optional[str]:
        """Submit a video generation job and return job ID"""
        
        print(f"🚀 Submitting {workflow_type.upper()} video generation job...")
        print(f"📊 Parameters: {width}x{height}, {frames} frames @ {fps}fps")
        
        # Prepare video parameters
        video_params = {
            "workflow_type": workflow_type.lower(),
            "prompt": prompt,
            "width": width,
            "height": height,
            "frames": frames,
            "fps": fps,
            "format": "mp4"
        }
        
        # Add workflow-specific parameters
        if workflow_type.lower() == "i2v":
            if not image_url:
                raise ValueError("I2V workflow requires image_url")
            video_params["image_url"] = image_url
            print(f"📸 Image: {image_url}")
            
        elif workflow_type.lower() == "flf2v":
            if not first_frame_url or not last_frame_url:
                raise ValueError("FLF2V workflow requires both first_frame_url and last_frame_url")
            video_params["first_frame_url"] = first_frame_url
            video_params["last_frame_url"] = last_frame_url
            print(f"📸 First Frame: {first_frame_url}")
            print(f"📸 Last Frame: {last_frame_url}")
            
        elif workflow_type.lower() == "inf_talk_single":
            if not image_url or not audio_url:
                raise ValueError("inf_talk_single workflow requires both image_url and audio_url")
            video_params["image_url"] = image_url
            video_params["audio_url"] = audio_url
            print(f"📸 Image: {image_url}")
            print(f"🎵 Audio: {audio_url}")
            
        elif workflow_type.lower() == "inf_talk_multi":
            if not image_url or not audio1_url or not audio2_url:
                raise ValueError("inf_talk_multi workflow requires image_url, audio1_url, and audio2_url")
            video_params["image_url"] = image_url
            video_params["audio1_url"] = audio1_url
            video_params["audio2_url"] = audio2_url
            print(f"📸 Image: {image_url}")
            print(f"🎵 Audio 1: {audio1_url}")
            print(f"🎵 Audio 2: {audio2_url}")
            
        else:
            raise ValueError(f"Unsupported workflow type: {workflow_type}")
        
        print(f"📝 Prompt: {prompt}")
        
        try:
            response = self.session.post(self.VIDEO_SUBMIT_API, json=video_params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            job_id = result["job_id"]
            print(f"✅ Job submitted successfully!")
            print(f"📋 Job ID: {job_id}")
            return job_id
            
        except Exception as e:
            print(f"❌ Failed to submit job: {e}")
            return None
    
    def _wait_and_get_video(self, job_id: str) -> Optional[bytes]:
        """Wait for video completion and return video bytes"""
        
        print(f"\n🔄 Waiting for video generation to complete...")
        print(f"📋 Job ID: {job_id}")
        print(f"⏰ Max wait time: 45 minutes, checking every 45 seconds")
        
        # Start progress monitor
        monitor = ProgressMonitor()
        monitor.start("🎬 Generating video")
        
        max_wait_time = 2700  # 45 minutes max wait (increased for InfiniteTalk workflows)
        check_interval = 45   # Check every 45 seconds (less frequent to reduce load)
        start_time = time.time()
        
        try:
            while True:
                try:
                    # Check job status - new API returns video directly in response
                    status_response = self.session.post(
                        self.VIDEO_STATUS_API,
                        json={"job_id": job_id},
                        timeout=30
                    )
                    
                    if status_response.status_code == 200:
                        # Check if this is a direct video response (video bytes)
                        content_type = status_response.headers.get('content-type', '').lower()
                        
                        if 'video' in content_type or 'application/octet-stream' in content_type:
                            monitor.stop()
                            video_size = len(status_response.content)
                            print(f"🎉 Video generation completed!")
                            print(f"📏 Video size: {video_size / 1024:.1f} KB")
                            return status_response.content
                        
                        # Otherwise, it's a JSON status response
                        try:
                            status_data = status_response.json()
                            status = status_data.get('status', 'unknown')
                            
                            if status == 'error':
                                error_msg = status_data.get('error', 'Unknown error')
                                monitor.stop()
                                print(f"❌ Video generation failed: {error_msg}")
                                return None
                            elif status in ['submitted', 'running', 'processing']:
                                elapsed = time.time() - start_time
                                remaining = max_wait_time - elapsed
                                elapsed_min = elapsed / 60
                                remaining_min = remaining / 60
                                print(f"⏳ Video still processing... ({elapsed_min:.1f}min elapsed, {remaining_min:.1f}min remaining)")
                                
                                if elapsed > max_wait_time:
                                    monitor.stop()
                                    print(f"⏰ Timeout after {max_wait_time/60:.1f} minutes")
                                    return None
                                    
                                time.sleep(check_interval)
                            else:
                                print(f"❓ Unknown status: {status}")
                                time.sleep(check_interval)
                        except:
                            print(f"❌ Invalid JSON response from status endpoint")
                            return None
                            
                    elif status_response.status_code == 202:
                        # Still processing - this is normal, continue waiting
                        elapsed = time.time() - start_time
                        remaining = max_wait_time - elapsed
                        elapsed_min = elapsed / 60
                        remaining_min = remaining / 60
                        print(f"⏳ Video still processing (202)... ({elapsed_min:.1f}min elapsed, {remaining_min:.1f}min remaining)")
                        
                        if elapsed > max_wait_time:
                            monitor.stop()
                            print(f"⏰ Timeout after {max_wait_time/60:.1f} minutes")
                            return None
                            
                        time.sleep(check_interval)
                        
                    elif status_response.status_code == 500:
                        # Could be an error or still processing - check the response
                        try:
                            error_data = status_response.json()
                            error_msg = error_data.get("error", "")
                            
                            if "No video files generated" in error_msg or "still processing" in error_msg.lower():
                                # This might be a timing issue - video may still be processing
                                elapsed = time.time() - start_time
                                if elapsed < max_wait_time * 0.8:  # Allow 80% of max time for processing
                                    elapsed_min = elapsed / 60
                                    print(f"⏳ Video still processing (volume not ready) - waiting ({elapsed_min:.1f}min elapsed)...")
                                    time.sleep(check_interval)
                                    continue
                                else:
                                    monitor.stop()
                                    print(f"❌ Video generation failed after extended wait: {error_msg}")
                                    return None
                            else:
                                monitor.stop()
                                print(f"❌ Video generation failed: {error_msg}")
                                return None
                        except:
                            # If we can't parse the error, treat it as a temporary issue
                            elapsed = time.time() - start_time
                            if elapsed < max_wait_time * 0.5:  # Allow 50% of max time for temporary errors
                                elapsed_min = elapsed / 60
                                print(f"⏳ Temporary error (500) - retrying ({elapsed_min:.1f}min elapsed)...")
                                time.sleep(check_interval)
                                continue
                            else:
                                monitor.stop()
                                print(f"❌ Video generation failed with status 500 after extended wait")
                                return None
                                
                    else:
                        print(f"❌ Unexpected status code: {status_response.status_code}")
                        elapsed = time.time() - start_time
                        if elapsed < max_wait_time * 0.3:  # Allow 30% of max time for unexpected errors
                            elapsed_min = elapsed / 60
                            print(f"⏳ Retrying in {check_interval} seconds... ({elapsed_min:.1f}min elapsed)")
                            time.sleep(check_interval)
                            continue
                        else:
                            monitor.stop()
                            print(f"❌ Video generation failed after multiple unexpected errors")
                            return None
                    
                except KeyboardInterrupt:
                    monitor.stop()
                    print(f"\n⚠️ User interrupted while waiting for video")
                    return None
                except Exception as e:
                    print(f"❌ Exception while checking status: {e}")
                    time.sleep(check_interval)
                    
        except Exception as e:
            monitor.stop()
            print(f"❌ Video generation failed: {e}")
            return None
    
    def edit_image(self, 
                   input_image_url: str,
                   prompt: str) -> Optional[bytes]:
        """Edit an existing image using AI with async job submission
        
        Args:
            input_image_url: URL of the input image to edit
            prompt: Text description of the desired changes
            
        Returns:
            Edited image data as bytes or None if failed
        """
        
        # Submit the job
        job_id = self._submit_image_edit_job(input_image_url, prompt)
        
        if not job_id:
            print("❌ Failed to submit image editing job")
            return None
        
        # Wait for completion and get image data
        image_data = self._wait_for_image_edit_completion(job_id)
        
        return image_data
    
    def generate_video_from_image(self,
                                image_url: str,
                                prompt: str,
                                width: int = 640,
                                height: int = 640,
                                frames: int = 81,
                                fps: int = 16) -> Optional[bytes]:
        """Generate a video from a single image
        
        Args:
            image_url: URL of the input image
            prompt: Text description of the desired video motion
            width: Video width in pixels (default: 640)
            height: Video height in pixels (default: 640)
            frames: Number of frames to generate (default: 81)
            fps: Frames per second (default: 16)
            
        Returns:
            Video data as bytes or None if failed
        """
        
        # Submit the job
        job_id = self._submit_video_job(
            workflow_type="i2v",
            prompt=prompt,
            width=width,
            height=height,
            frames=frames,
            fps=fps,
            image_url=image_url
        )
        
        if not job_id:
            return None
        
        # Wait for completion and get video
        return self._wait_and_get_video(job_id)
    
    def generate_video_from_first_last_images(self,
                                            first_frame_url: str,
                                            last_frame_url: str,
                                            prompt: str,
                                            width: int = 640,
                                            height: int = 640,
                                            frames: int = 25,
                                            fps: int = 16) -> Optional[bytes]:
        """Generate a video from first and last frame images
        
        Args:
            first_frame_url: URL of the first frame image
            last_frame_url: URL of the last frame image
            prompt: Text description of the desired video transition
            width: Video width in pixels (default: 640)
            height: Video height in pixels (default: 640)
            frames: Number of frames to generate (default: 25)
            fps: Frames per second (default: 16)
            
        Returns:
            Video data as bytes or None if failed
        """
        
        # Submit the job
        job_id = self._submit_video_job(
            workflow_type="flf2v",
            prompt=prompt,
            width=width,
            height=height,
            frames=frames,
            fps=fps,
            first_frame_url=first_frame_url,
            last_frame_url=last_frame_url
        )
        
        if not job_id:
            return None
        
        # Wait for completion and get video
        return self._wait_and_get_video(job_id)
    
    def generate_infinite_talk_video(self,
                                   image_url: str,
                                   audio_files: list,
                                   prompt: str,
                                   width: int = 573,
                                   height: int = 806,
                                   frames: int = 200,
                                   fps: int = 25) -> Optional[bytes]:
        """Generate InfiniteTalk video with automatic workflow selection based on number of audio files
        
        Args:
            image_url: URL of the input image
            audio_files: List of audio file URLs (1 or 2 files supported)
            prompt: Text description of the desired video
            width: Video width in pixels (default: 573)
            height: Video height in pixels (default: 806) 
            frames: Number of frames to generate (default: 200)
            fps: Frames per second (default: 25)
            
        Returns:
            Video data as bytes or None if failed
        """
        
        # Validate audio files count
        if not audio_files or len(audio_files) < 1:
            print("❌ Error: At least 1 audio file is required")
            return None
        
        if len(audio_files) > 2:
            print("❌ Error: Maximum 2 audio files supported")
            return None
        
        # Determine workflow based on number of audio files
        if len(audio_files) == 1:
            print("🗣️ Using single-speaker InfiniteTalk workflow")
            workflow_type = "inf_talk_single"
            
            # Submit the job
            job_id = self._submit_video_job(
                workflow_type=workflow_type,
                prompt=prompt,
                width=width,
                height=height,
                frames=frames,
                fps=fps,
                image_url=image_url,
                audio_url=audio_files[0]
            )
            
        elif len(audio_files) == 2:
            print("🎤 Using multi-speaker InfiniteTalk workflow")
            workflow_type = "inf_talk_multi"
            
            # Submit the job
            job_id = self._submit_video_job(
                workflow_type=workflow_type,
                prompt=prompt,
                width=width,
                height=height,
                frames=frames,
                fps=fps,
                image_url=image_url,
                audio1_url=audio_files[0],
                audio2_url=audio_files[1]
            )
        
        if not job_id:
            return None
        
        # Wait for completion and get video
        return self._wait_and_get_video(job_id)
    
    def _submit_music_job(self, prompt: str, lyrics: str, duration: int) -> Optional[str]:
        """Submit a music generation job using ACE-Step tool and return job ID"""
        
        print(f"🚀 Submitting ACE-Step music generation job...")
        print(f"📝 Prompt: {prompt}")
        print(f"🎤 Lyrics: {lyrics}")
        print(f"⏱️ Duration: {duration}s")
        
        # Prepare music parameters for unified endpoint
        music_params = {
            "tool": "ace-step",
            "prompt": prompt,
            "lyrics": lyrics,
            "duration": float(duration)
        }
        
        try:
            response = self.session.post(self.MUSIC_SUBMIT_API, json=music_params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            job_id = result["job_id"]
            print(f"✅ ACE-Step job submitted successfully!")
            print(f"📋 Job ID: {job_id}")
            return job_id
            
        except Exception as e:
            print(f"❌ Failed to submit ACE-Step job: {e}")
            return None

    def _wait_for_music_completion(self, job_id: str) -> Optional[bytes]:
        """Wait for ACE-Step music generation completion and return audio data"""
        
        print(f"\n🔄 Waiting for ACE-Step music generation to complete...")
        print(f"📋 Job ID: {job_id}")
        
        max_wait_time = 600  # 10 minutes max wait for music generation
        check_interval = 10  # Check every 10 seconds
        start_time = time.time()
        
        monitor = ProgressMonitor()
        monitor.start("🎵 Generating music")
        
        try:
            while True:
                try:
                    status_response = self.session.post(
                        self.MUSIC_STATUS_API,
                        json={"job_id": job_id},
                        timeout=30
                    )
                    
                    # Success case - audio ready (200 with audio content)
                    if status_response.status_code == 200:
                        content_type = status_response.headers.get('content-type', '')
                        
                        # Check if we got an audio file back (direct download)
                        if content_type.startswith('audio/'):
                            monitor.stop()
                            print("🎉 ACE-Step music generation completed!")
                            
                            audio_size = len(status_response.content)
                            print(f"📏 Audio size: {audio_size / 1024:.1f} KB")
                            
                            return status_response.content
                        
                        elif 'application/json' in content_type:
                            # JSON response - might be status or error
                            try:
                                status_data = status_response.json()
                                status = status_data.get("status")
                                
                                if status == "error":
                                    error_msg = status_data.get("error", "Unknown error")
                                    monitor.stop()
                                    print(f"❌ ACE-Step job failed: {error_msg}")
                                    return None
                                else:
                                    # Continue waiting
                                    elapsed = time.time() - start_time
                                    if elapsed > max_wait_time:
                                        monitor.stop()
                                        print(f"⏰ Timeout after {max_wait_time/60:.1f} minutes")
                                        return None
                                    time.sleep(check_interval)
                            except:
                                # Invalid JSON, continue waiting
                                elapsed = time.time() - start_time
                                if elapsed > max_wait_time:
                                    monitor.stop()
                                    print(f"⏰ Timeout after {max_wait_time/60:.1f} minutes")
                                    return None
                                time.sleep(check_interval)
                        else:
                            # Unknown content type, continue waiting
                            elapsed = time.time() - start_time
                            if elapsed > max_wait_time:
                                monitor.stop()
                                print(f"❌ Unknown response format")
                                return None
                            time.sleep(check_interval)
                    
                    # Handle processing states (202)
                    elif status_response.status_code == 202:
                        # Still processing - normal state
                        elapsed = time.time() - start_time
                        elapsed_min = elapsed / 60
                        remaining_min = (max_wait_time - elapsed) / 60
                        
                        if elapsed > max_wait_time:
                            monitor.stop()
                            print(f"⏰ Timeout after {max_wait_time/60:.1f} minutes")
                            return None
                        
                        try:
                            status_data = status_response.json()
                            status = status_data.get("status", "processing")
                            print(f"⏳ ACE-Step status: {status} ({elapsed_min:.1f}min elapsed, {remaining_min:.1f}min remaining)")
                        except:
                            print(f"⏳ ACE-Step processing... ({elapsed_min:.1f}min elapsed, {remaining_min:.1f}min remaining)")
                        
                        time.sleep(check_interval)
                    
                    else:
                        # Other status codes - continue waiting with timeout
                        elapsed = time.time() - start_time
                        if elapsed > max_wait_time:
                            monitor.stop()
                            print(f"❌ ACE-Step job failed after timeout")
                            return None
                        
                        print(f"⚠️ Unexpected status code: {status_response.status_code}, continuing...")
                        time.sleep(check_interval)
                            
                except Exception as e:
                    elapsed = time.time() - start_time
                    if elapsed > max_wait_time:
                        monitor.stop()
                        print(f"❌ Giving up after {max_wait_time/60:.1f} minutes: {e}")
                        return None
                    
                    print(f"⚠️ Exception during status check: {e}, retrying...")
                    time.sleep(check_interval)
                    
        except Exception as e:
            monitor.stop()
            print(f"❌ Failed to wait for ACE-Step music completion: {e}")
            return None

    def generate_music_with_lyrics(self,
                                  prompt: str,
                                  lyrics: str,
                                  duration: int = 60) -> Optional[bytes]:
        """Generate music with lyrics using ACE-Step
        
        Args:
            prompt: Text description of the music style/genre (e.g., "sonata, piano, Violin, B Flat Major, allegro")
            lyrics: Lyrics for the song (use "[inst]" for instrumental)
            duration: Duration of the music in seconds (default: 60)
            
        Returns:
            Audio data as bytes or None if failed
        """
        
        # Submit the job
        job_id = self._submit_music_job(prompt, lyrics, duration)
        
        if not job_id:
            print("❌ Failed to submit music generation job")
            return None
        
        # Wait for completion and get audio data
        audio_data = self._wait_for_music_completion(job_id)
        
        return audio_data
    
    def _submit_audio_job(self, video_url: Optional[str], prompt: str) -> Optional[str]:
        """Submit an MMAudio audio effects generation job and return job ID"""
        
        print(f"🚀 Submitting MMAudio audio effects generation job...")
        print(f"📝 Prompt: {prompt}")
        if video_url:
            print(f"🎬 Video URL: {video_url}")
        else:
            print(f"🎵 Text-to-audio mode (no video)")
        
        # Prepare audio parameters for unified endpoint
        audio_params = {
            "tool": "mmaudio",
            "prompt": prompt
        }
        
        # Add video URL if provided
        if video_url:
            audio_params["video_url"] = video_url
        
        try:
            response = self.session.post(self.MUSIC_SUBMIT_API, json=audio_params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            job_id = result["job_id"]
            print(f"✅ MMAudio job submitted successfully!")
            print(f"📋 Job ID: {job_id}")
            return job_id
            
        except Exception as e:
            print(f"❌ Failed to submit MMAudio job: {e}")
            return None

    def _wait_for_audio_completion(self, job_id: str) -> Optional[bytes]:
        """Wait for MMAudio audio effects generation completion and return audio data"""
        
        print(f"\n🔄 Waiting for MMAudio audio generation to complete...")
        print(f"📋 Job ID: {job_id}")
        
        max_wait_time = 300  # 5 minutes max wait for audio effects
        check_interval = 10  # Check every 10 seconds
        start_time = time.time()
        
        monitor = ProgressMonitor()
        monitor.start("🎵 Generating audio effects")
        
        try:
            while True:
                try:
                    status_response = self.session.post(
                        self.MUSIC_STATUS_API,
                        json={"job_id": job_id},
                        timeout=30
                    )
                    
                    # Success case - audio ready (200 with audio content)
                    if status_response.status_code == 200:
                        content_type = status_response.headers.get('content-type', '')
                        
                        # Check if we got an audio file back (direct download)
                        if content_type.startswith('audio/'):
                            monitor.stop()
                            print("🎉 MMAudio generation completed!")
                            
                            audio_size = len(status_response.content)
                            print(f"📏 Audio size: {audio_size / 1024:.1f} KB")
                            
                            return status_response.content
                        
                        elif 'application/json' in content_type:
                            # JSON response - might be status or error
                            try:
                                status_data = status_response.json()
                                status = status_data.get("status")
                                
                                if status == "error":
                                    error_msg = status_data.get("error", "Unknown error")
                                    monitor.stop()
                                    print(f"❌ MMAudio job failed: {error_msg}")
                                    return None
                                else:
                                    # Continue waiting
                                    elapsed = time.time() - start_time
                                    if elapsed > max_wait_time:
                                        monitor.stop()
                                        print(f"⏰ Timeout after {max_wait_time/60:.1f} minutes")
                                        return None
                                    time.sleep(check_interval)
                            except:
                                # Invalid JSON, continue waiting
                                elapsed = time.time() - start_time
                                if elapsed > max_wait_time:
                                    monitor.stop()
                                    print(f"⏰ Timeout after {max_wait_time/60:.1f} minutes")
                                    return None
                                time.sleep(check_interval)
                        else:
                            # Unknown content type, continue waiting
                            elapsed = time.time() - start_time
                            if elapsed > max_wait_time:
                                monitor.stop()
                                print(f"❌ Unknown response format")
                                return None
                            time.sleep(check_interval)
                    
                    # Handle processing states (202)
                    elif status_response.status_code == 202:
                        # Still processing - normal state
                        elapsed = time.time() - start_time
                        elapsed_min = elapsed / 60
                        remaining_min = (max_wait_time - elapsed) / 60
                        
                        if elapsed > max_wait_time:
                            monitor.stop()
                            print(f"⏰ Timeout after {max_wait_time/60:.1f} minutes")
                            return None
                        
                        try:
                            status_data = status_response.json()
                            status = status_data.get("status", "processing")
                            print(f"⏳ MMAudio status: {status} ({elapsed_min:.1f}min elapsed, {remaining_min:.1f}min remaining)")
                        except:
                            print(f"⏳ MMAudio processing... ({elapsed_min:.1f}min elapsed, {remaining_min:.1f}min remaining)")
                        
                        time.sleep(check_interval)
                    
                    else:
                        # Other status codes - continue waiting with timeout
                        elapsed = time.time() - start_time
                        if elapsed > max_wait_time:
                            monitor.stop()
                            print(f"❌ MMAudio job failed after timeout")
                            return None
                        
                        print(f"⚠️ Unexpected status code: {status_response.status_code}, continuing...")
                        time.sleep(check_interval)
                            
                except Exception as e:
                    elapsed = time.time() - start_time
                    if elapsed > max_wait_time:
                        monitor.stop()
                        print(f"❌ Giving up after {max_wait_time/60:.1f} minutes: {e}")
                        return None
                    
                    print(f"⚠️ Exception during status check: {e}, retrying...")
                    time.sleep(check_interval)
                    
        except Exception as e:
            monitor.stop()
            print(f"❌ Failed to wait for MMAudio completion: {e}")
            return None

    def _submit_vision_job(self, image_url: str, detail_level: str = "detailed", reformat: bool = True) -> Optional[str]:
        """Submit a Florence2 vision analysis job and return job ID"""
        
        print(f"🚀 Submitting Florence2 vision analysis job...")
        print(f"📸 Image URL: {image_url}")
        print(f"📝 Detail level: {detail_level}")
        print(f"🔧 Reformat: {reformat}")
        
        # Prepare vision parameters for unified endpoint
        vision_params = {
            "tool": "florence2",
            "image_url": image_url,
            "detail_level": detail_level,
            "reformat": reformat
        }
        
        try:
            response = self.session.post(self.MUSIC_SUBMIT_API, json=vision_params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            job_id = result["job_id"]
            print(f"✅ Florence2 job submitted successfully!")
            print(f"📋 Job ID: {job_id}")
            return job_id
            
        except Exception as e:
            print(f"❌ Failed to submit Florence2 job: {e}")
            return None

    def _wait_for_vision_completion(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Wait for Florence2 vision analysis completion and return result data"""
        
        print(f"\n🔄 Waiting for Florence2 vision analysis to complete...")
        print(f"📋 Job ID: {job_id}")
        
        max_wait_time = 120  # 2 minutes max wait for vision analysis
        check_interval = 5   # Check every 5 seconds (vision is faster)
        start_time = time.time()
        
        monitor = ProgressMonitor()
        monitor.start("👁️ Analyzing image")
        
        try:
            while True:
                try:
                    status_response = self.session.post(
                        self.MUSIC_STATUS_API,
                        json={"job_id": job_id},
                        timeout=30
                    )
                    
                    # Success case - vision analysis ready (200 with JSON content)
                    if status_response.status_code == 200:
                        content_type = status_response.headers.get('content-type', '')
                        
                        # Florence2 should return JSON response
                        if 'application/json' in content_type:
                            try:
                                result_data = status_response.json()
                                
                                # Check if this is a completed result (has caption or result)
                                if 'caption' in result_data or 'result' in result_data:
                                    monitor.stop()
                                    print("🎉 Florence2 vision analysis completed!")
                                    
                                    # Extract caption from different possible formats
                                    caption = result_data.get('caption') or result_data.get('result', {}).get('caption', 'No caption found')
                                    print(f"📝 Generated caption: {caption}")
                                    
                                    return result_data
                                
                                # Check for error status
                                elif result_data.get("status") == "error":
                                    error_msg = result_data.get("error", "Unknown error")
                                    monitor.stop()
                                    print(f"❌ Florence2 job failed: {error_msg}")
                                    return None
                                
                                else:
                                    # Continue waiting
                                    elapsed = time.time() - start_time
                                    if elapsed > max_wait_time:
                                        monitor.stop()
                                        print(f"⏰ Timeout after {max_wait_time/60:.1f} minutes")
                                        return None
                                    time.sleep(check_interval)
                                    
                            except Exception as json_error:
                                print(f"⚠️ Error parsing JSON response: {json_error}")
                                elapsed = time.time() - start_time
                                if elapsed > max_wait_time:
                                    monitor.stop()
                                    print(f"⏰ Timeout after {max_wait_time/60:.1f} minutes")
                                    return None
                                time.sleep(check_interval)
                        else:
                            # Non-JSON response, continue waiting
                            elapsed = time.time() - start_time
                            if elapsed > max_wait_time:
                                monitor.stop()
                                print(f"❌ Unexpected response format")
                                return None
                            time.sleep(check_interval)
                    
                    # Handle processing states (202)
                    elif status_response.status_code == 202:
                        # Still processing - normal state
                        elapsed = time.time() - start_time
                        elapsed_sec = elapsed
                        remaining_sec = max_wait_time - elapsed
                        
                        if elapsed > max_wait_time:
                            monitor.stop()
                            print(f"⏰ Timeout after {max_wait_time} seconds")
                            return None
                        
                        try:
                            status_data = status_response.json()
                            status = status_data.get("status", "processing")
                            print(f"⏳ Florence2 status: {status} ({elapsed_sec:.0f}s elapsed, {remaining_sec:.0f}s remaining)")
                        except:
                            print(f"⏳ Florence2 processing... ({elapsed_sec:.0f}s elapsed, {remaining_sec:.0f}s remaining)")
                        
                        time.sleep(check_interval)
                    
                    else:
                        # Other status codes - continue waiting with timeout
                        elapsed = time.time() - start_time
                        if elapsed > max_wait_time:
                            monitor.stop()
                            print(f"❌ Florence2 job failed after timeout")
                            return None
                        
                        print(f"⚠️ Unexpected status code: {status_response.status_code}, continuing...")
                        time.sleep(check_interval)
                            
                except Exception as e:
                    elapsed = time.time() - start_time
                    if elapsed > max_wait_time:
                        monitor.stop()
                        print(f"❌ Giving up after {max_wait_time} seconds: {e}")
                        return None
                    
                    print(f"⚠️ Exception during status check: {e}, retrying...")
                    time.sleep(check_interval)
                    
        except Exception as e:
            monitor.stop()
            print(f"❌ Failed to wait for Florence2 completion: {e}")
            return None

    def generate_audio_effects(self,
                             prompt: str,
                             video_url: Optional[str] = None) -> Optional[bytes]:
        """Generate audio effects using MMAudio
        
        Args:
            prompt: Text description of the desired audio (e.g., "thunder and rain sounds")
            video_url: Optional video URL to generate audio for (if None, uses text-to-audio)
            
        Returns:
            Audio data as bytes or None if failed
        """
        
        # Submit the job
        job_id = self._submit_audio_job(video_url, prompt)
        
        if not job_id:
            print("❌ Failed to submit MMAudio job")
            return None
        
        # Wait for completion and get audio data
        audio_data = self._wait_for_audio_completion(job_id)
        
        return audio_data

    def describe_image(self,
                      image_url: str,
                      detail_level: str = "detailed",
                      reformat: bool = True) -> Optional[str]:
        """Generate detailed description of an image using Florence2
        
        Args:
            image_url: URL of the image to analyze
            detail_level: Level of detail ("basic", "detailed", "more_detailed")
            reformat: Whether to reformat the description for better readability
            
        Returns:
            Image description as string or None if failed
        """
        
        # Submit the job
        job_id = self._submit_vision_job(image_url, detail_level, reformat)
        
        if not job_id:
            print("❌ Failed to submit Florence2 job")
            return None
        
        # Wait for completion and get result
        result_data = self._wait_for_vision_completion(job_id)
        
        if result_data:
            # Extract caption from different possible formats
            caption = result_data.get('caption') or result_data.get('result', {}).get('caption')
            return caption
        
        return None
    
    def get_info(self):
        """Get system info and available capabilities"""
        return {
            "modal_manager_initialized": True,
            "qwen_combined_submit_api": self.QWEN_SUBMIT_API,
            "qwen_combined_status_api": self.QWEN_STATUS_API,
            "video_submit_api": self.VIDEO_SUBMIT_API,
            "video_status_api": self.VIDEO_STATUS_API,
            "audio_vision_submit_api": self.MUSIC_SUBMIT_API,
            "audio_vision_status_api": self.MUSIC_STATUS_API,
            "available_functions": [
                "generate_image_from_prompt",
                "edit_image", 
                "generate_video_from_image",
                "generate_video_from_first_last_images",
                "generate_infinite_talk_video",
                "generate_music_with_lyrics",
                "generate_audio_effects",
                "describe_image"
            ],
            "implemented_functions": [
                "generate_image_from_prompt",
                "edit_image",
                "generate_video_from_image",
                "generate_video_from_first_last_images", 
                "generate_infinite_talk_video",
                "generate_music_with_lyrics",
                "generate_audio_effects",
                "describe_image"
            ],
            "qwen_workflows": ["tti", "iti"],
            "video_workflows": ["i2v", "flf2v", "inf_talk_single", "inf_talk_multi"],
            "audio_vision_tools": ["ace-step", "mmaudio", "florence2"],
            "audio_vision_capabilities": {
                "ace-step": "Advanced music generation with lyrics",
                "mmaudio": "Audio effects generation from video or text prompts",
                "florence2": "Image captioning and visual understanding"
            },
            "response_format": "base64_encoded_images_video_bytes_and_json"
        }
