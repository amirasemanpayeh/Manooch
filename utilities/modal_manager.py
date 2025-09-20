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
                         first_frame_url: Optional[str] = None, last_frame_url: Optional[str] = None) -> Optional[str]:
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
        
        # Start progress monitor
        monitor = ProgressMonitor()
        monitor.start("🎬 Generating video")
        
        max_attempts = 45  # 7.5 minutes at 10-second intervals
        
        try:
            for attempt in range(max_attempts):
                try:
                    status_response = self.session.post(
                        self.VIDEO_STATUS_API,
                        json={"job_id": job_id},
                        timeout=30
                    )
                    
                    # Success case - video ready
                    if status_response.status_code == 200:
                        content_type = status_response.headers.get('content-type', '')
                        
                        if 'application/json' in content_type:
                            # JSON response - check status
                            status_data = status_response.json()
                            status = status_data.get("status")
                            
                            if status == "error":
                                monitor.stop()
                                error_msg = status_data.get("error", "Unknown error")
                                print(f"❌ Job failed: {error_msg}")
                                return None
                            elif status in ["submitted", "processing"]:
                                # Continue polling
                                time.sleep(10)
                                continue
                        else:
                            # Binary response - video completed!
                            monitor.stop()
                            video_size = len(status_response.content)
                            print(f"🎉 Video generation completed!")
                            print(f"📏 Video size: {video_size / 1024:.1f} KB")
                            return status_response.content
                    
                    # Handle processing states and temporary errors
                    elif status_response.status_code == 202:
                        time.sleep(10)
                        continue
                    
                    elif status_response.status_code == 500:
                        # Check if it's a "file not found" error (timing issue)
                        try:
                            error_data = status_response.json()
                            error_msg = error_data.get("error", "")
                            
                            if "Video file not found in volume" in error_msg:
                                if attempt < 40:
                                    time.sleep(15)
                                    continue
                                else:
                                    monitor.stop()
                                    print(f"❌ Job failed after extended wait: {error_msg}")
                                    return None
                            else:
                                monitor.stop()
                                print(f"❌ Job failed: {error_msg}")
                                return None
                        except:
                            if attempt < 40:
                                time.sleep(15)
                                continue
                            else:
                                monitor.stop()
                                print(f"❌ Job failed with status {status_response.status_code}")
                                return None
                    
                    else:
                        if attempt < 40:
                            time.sleep(10)
                            continue
                        else:
                            monitor.stop()
                            print(f"❌ Job failed after multiple retries")
                            return None
                            
                except Exception as e:
                    if attempt < 40:
                        time.sleep(10)
                    else:
                        monitor.stop()
                        print(f"❌ Giving up after {max_attempts} attempts: {e}")
                        return None
            
            monitor.stop()
            print("⏰ Timeout: Video generation took longer than 7.5 minutes")
            return None
            
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
                                width: int = 480,
                                height: int = 640,
                                frames: int = 81,
                                fps: int = 16) -> Optional[bytes]:
        """Generate a video from a single image
        
        Args:
            image_url: URL of the input image
            prompt: Text description of the desired video motion
            width: Video width in pixels (default: 480)
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
    
    def _submit_music_job(self, prompt: str, lyrics: str, duration: int) -> Optional[str]:
        """Submit a music generation job and return job ID"""
        
        print(f"🚀 Submitting music generation job...")
        print(f"📝 Prompt: {prompt}")
        print(f"🎤 Lyrics: {lyrics}")
        print(f"⏱️ Duration: {duration}s")
        
        # Prepare music parameters
        music_params = {
            "prompt": prompt,
            "lyrics": lyrics,
            "duration": duration
        }
        
        try:
            response = self.session.post(self.MUSIC_SUBMIT_API, json=music_params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            job_id = result["job_id"]
            print(f"✅ Job submitted successfully!")
            print(f"📋 Job ID: {job_id}")
            return job_id
            
        except Exception as e:
            print(f"❌ Failed to submit job: {e}")
            return None

    def _wait_for_music_completion(self, job_id: str) -> Optional[bytes]:
        """Wait for music generation completion and return audio data"""
        
        print(f"\n🔄 Waiting for music generation to complete...")
        print(f"📋 Job ID: {job_id}")
        
        max_attempts = 120  # 60 minutes at 30-second intervals (full songs take longer)
        monitor = ProgressMonitor()
        monitor.start("🎵 Generating music")
        
        try:
            for attempt in range(max_attempts):
                try:
                    status_response = self.session.post(
                        self.MUSIC_STATUS_API,
                        json={"job_id": job_id},
                        timeout=30
                    )
                    
                    # Success case - audio ready
                    if status_response.status_code == 200:
                        content_type = status_response.headers.get('content-type', '')
                        
                        # Check if we got an audio file back (direct download)
                        if content_type.startswith('audio/'):
                            monitor.stop()
                            print("🎉 Music generation completed!")
                            
                            audio_size = len(status_response.content)
                            print(f"📏 Audio size: {audio_size / 1024:.1f} KB")
                            
                            return status_response.content
                        
                        elif 'application/json' in content_type:
                            # JSON response - check status
                            status_data = status_response.json()
                            status = status_data.get("status")
                            
                            if status == "error":
                                error_msg = status_data.get("error", "Unknown error")
                                monitor.stop()
                                print(f"❌ Job failed: {error_msg}")
                                return None
                            elif status in ["submitted", "processing"]:
                                time.sleep(30)  # Wait 30 seconds before next check (music takes longer)
                                continue
                        else:
                            # Unknown content type
                            if attempt < 115:  # Allow more time for processing
                                time.sleep(30)
                                continue
                            else:
                                monitor.stop()
                                print(f"❌ Unknown response format")
                                return None
                    
                    # Handle processing states (202) and temporary errors (500)
                    elif status_response.status_code == 202:
                        time.sleep(30)
                        continue
                    
                    elif status_response.status_code == 500:
                        # Check if it's a "file not found" error (timing issue)
                        try:
                            error_data = status_response.json()
                            error_msg = error_data.get("error", "")
                            
                            if "Audio file not found" in error_msg or "file not found in volume" in error_msg:
                                # This is likely a timing issue - audio may still be processing
                                if attempt < 115:  # Allow more time for processing
                                    time.sleep(30)
                                    continue
                                else:
                                    monitor.stop()
                                    print(f"❌ Job failed after extended wait: {error_msg}")
                                    return None
                            else:
                                monitor.stop()
                                print(f"❌ Job failed: {error_msg}")
                                return None
                        except:
                            # If we can't parse the error, treat it as a temporary issue
                            if attempt < 115:
                                time.sleep(30)
                                continue
                            else:
                                monitor.stop()
                                print(f"❌ Job failed with status {status_response.status_code}")
                                return None
                    
                    else:
                        if attempt < 115:
                            time.sleep(30)
                            continue
                        else:
                            monitor.stop()
                            print(f"❌ Job failed after multiple retries")
                            return None
                            
                except Exception as e:
                    if attempt < 115:
                        time.sleep(30)
                    else:
                        monitor.stop()
                        print(f"❌ Giving up after {max_attempts} attempts: {e}")
                        return None
            
            monitor.stop()
            print("⏰ Timeout: Music generation took longer than 60 minutes")
            return None
            
        except Exception as e:
            monitor.stop()
            print(f"❌ Failed to wait for music completion: {e}")
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
    
    def get_info(self):
        """Get system info and available capabilities"""
        return {
            "modal_manager_initialized": True,
            "qwen_combined_submit_api": self.QWEN_SUBMIT_API,
            "qwen_combined_status_api": self.QWEN_STATUS_API,
            "video_submit_api": self.VIDEO_SUBMIT_API,
            "video_status_api": self.VIDEO_STATUS_API,
            "music_submit_api": self.MUSIC_SUBMIT_API,
            "music_status_api": self.MUSIC_STATUS_API,
            "available_functions": [
                "generate_image_from_prompt",
                "edit_image", 
                "generate_video_from_image",
                "generate_video_from_first_last_images",
                "generate_music_with_lyrics"
            ],
            "implemented_functions": [
                "generate_image_from_prompt",
                "edit_image",
                "generate_video_from_image",
                "generate_video_from_first_last_images",
                "generate_music_with_lyrics"
            ],
            "qwen_workflows": ["tti", "iti"],
            "response_format": "base64_encoded_images"
        }
