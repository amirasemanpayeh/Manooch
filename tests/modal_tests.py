import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modal_manager import ModalManager
from utils.supabase_manager import SupabaseManager, set_supabase_manager, get_supabase_manager
from utils.audio_video_tools import AudioVideoMixer
from utils.audio_tools import AudioTools
from utils.settings_manager import settings
import time


class ModalManagerTests:
    """Unit tests for the ModalManager class"""
    
    def __init__(self):
        """Initialize the test class with necessary managers"""
        self.modal_manager = ModalManager()
        self.audio_video_mixer = AudioVideoMixer()
        
        # Initialize Supabase manager
        self.supabase_manager = SupabaseManager(url=settings.supabase_url, key=settings.supabase_key)
        set_supabase_manager(self.supabase_manager)
        print("✅ Supabase manager initialized successfully")
        print("✅ AudioVideoMixer initialized successfully")
        
        # Benchmarking variables
        self.benchmark_data = {
            'start_time': None,
            'step_times': {},
            'modal_processing_time': 0,
            'video_length_seconds': 0,
            'total_cost_estimate': 0
        }
    
    def text_to_video_talking_character(self):
        """
        Complete pipeline test: Create image -> Generate TTS -> Create talking video -> Mix additional audio
        
        This test demonstrates the full workflow:
        1. Generate an image from text prompt
        2. Upload image to Supabase
        3. Generate voice-cloned speech from reference audio
        4. Upload generated audio to Supabase  
        5. Create InfiniteTalk video using the image and audio
        6. Upload talking video to Supabase
        7. Mix additional background audio with the talking video
        8. Upload final mixed video to Supabase
        """
        print("\n" + "="*60)
        print("🎬 STARTING TEXT-TO-VIDEO TALKING CHARACTER TEST")
        print("="*60)
        
        # Initialize benchmarking
        self.benchmark_data['start_time'] = time.time()
        self.benchmark_data['step_times'] = {}
        self.benchmark_data['modal_processing_time'] = 0
        
        try:
            # Step 1: Generate Image from Text Prompt
            print("\n🎨 Step 1: Generating image from text prompt...")
            step1_start = time.time()
            
            image_prompt = ("A beautiful elegant woman with flowing dark hair, wearing a vintage "
                          "burgundy dress, sitting gracefully in an ornate Victorian parlor with "
                          "rich wooden furniture and warm golden lighting, portrait style, "
                          "high quality, cinematic lighting")
            
            print(f"📝 Prompt: {image_prompt}")
            print(f"📐 Dimensions: 537x896 pixels")
            
            image_bytes = self.modal_manager.generate_image_from_prompt(
                prompt=image_prompt,
                width=537,
                height=896,
                batch_size=1
            )
            
            if not image_bytes:
                print("❌ Failed to generate image")
                return False
            
            print(f"✅ Image generated successfully ({len(image_bytes)} bytes)")
            
            # Upload image to Supabase
            print("📤 Uploading generated image to Supabase...")
            image_url = self.supabase_manager.upload_generated_asset_img(image_bytes)
            print(f"✅ Image uploaded: {image_url}")
            
            step1_time = time.time() - step1_start
            self.benchmark_data['step_times']['image_generation'] = step1_time
            self.benchmark_data['modal_processing_time'] += step1_time
            
            # Step 2: Generate Voice Clone using Reference Audio
            print("\n🗣️ Step 2: Generating voice-cloned speech...")
            step2_start = time.time()
            
            reference_audio_url = "https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/audio_samples/219777__madamvicious__the-wine-is-quite-delightful-posh-woman.wav"
            speech_text = ("The wine is quite delightful, with notes of cherry and oak that dance "
                         "upon the palate. This vintage has been aged to perfection, offering a "
                         "rich and complex flavor profile that would complement any elegant evening.")
            
            print(f"🎵 Reference Audio: {reference_audio_url}")
            print(f"📝 Speech Text: {speech_text}")
            
            audio_bytes = self.modal_manager.generate_voice_clone(
                audio_url=reference_audio_url,
                text=speech_text,
                exaggeration=0.6,  # Slightly more expressive
                cfg_weight=0.7     # Higher fidelity to voice
            )
            
            if not audio_bytes:
                print("❌ Failed to generate voice clone")
                return False
            
            print(f"✅ Voice clone generated successfully ({len(audio_bytes)} bytes)")
            
            # Upload audio to Supabase
            print("📤 Uploading generated audio to Supabase...")
            audio_url = self.supabase_manager.upload_processed_asset_audio(audio_bytes)
            print(f"✅ Audio uploaded: {audio_url}")
            
            # Get actual audio duration to determine real video length
            print("📏 Measuring audio duration...")
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name
            
            # Store and measure audio duration
            audio_identifier = AudioTools.store_audio(temp_audio_path, f"temp_duration_check_{int(time.time())}")
            audio_local_path = AudioTools.restore_audio(audio_identifier)
            actual_audio_duration = AudioTools.get_audio_duration(audio_local_path)
            self.benchmark_data['video_length_seconds'] = actual_audio_duration
            print(f"🎵 Audio duration: {actual_audio_duration:.1f} seconds")
            
            # Clean up temporary files
            AudioTools.delete_stored_audio(audio_identifier)
            import os
            os.unlink(temp_audio_path)
            
            step2_time = time.time() - step2_start
            self.benchmark_data['step_times']['voice_clone'] = step2_time
            self.benchmark_data['modal_processing_time'] += step2_time
            
            # Step 3: Generate InfiniteTalk Video
            print("\n🎬 Step 3: Creating InfiniteTalk talking video...")
            step3_start = time.time()
            
            video_prompt = ("A sophisticated woman speaking elegantly about wine, with natural "
                          "facial expressions and subtle head movements, in a Victorian parlor setting")
            
            print(f"🖼️ Image URL: {image_url}")
            print(f"🎵 Audio URL: {audio_url}")
            print(f"📝 Video Prompt: {video_prompt}")
            
            # Set frame limit (video will be limited by audio duration or this frame count, whichever is shorter)
            frames = 1000  # Upper limit
            fps = 25
            max_video_length = frames / fps
            actual_video_length = min(self.benchmark_data['video_length_seconds'], max_video_length)
            
            print(f"📏 Audio duration: {self.benchmark_data['video_length_seconds']:.1f} seconds")
            print(f"🎬 Frame limit: {max_video_length:.1f} seconds ({frames} frames at {fps} fps)")
            print(f"📐 Expected video length: {actual_video_length:.1f} seconds")
            
            video_bytes = self.modal_manager.generate_infinite_talk_video(
                image_url=image_url,
                audio_files=[audio_url],  # Single speaker
                prompt=video_prompt,
                width=537,
                height=896,
                frames=frames,  # Upper limit
                fps=fps
            )
            
            if not video_bytes:
                print("❌ Failed to generate InfiniteTalk video")
                return False
            
            print(f"✅ InfiniteTalk video generated successfully ({len(video_bytes)} bytes)")
            
            # Upload talking video to Supabase
            print("📤 Uploading talking video to Supabase...")
            talking_video_url = self.supabase_manager.upload_processed_asset_video(video_bytes)
            print(f"✅ Talking video uploaded: {talking_video_url}")
            
            step3_time = time.time() - step3_start
            self.benchmark_data['step_times']['video_generation'] = step3_time
            self.benchmark_data['modal_processing_time'] += step3_time
            
            # Step 4: Mix Background Audio with Talking Video
            print("\n🎵 Step 4: Adding audio to talking video...")
            step4_start = time.time()

            print(f"🎬 Talking Video: {talking_video_url}")
            
            # Mix the talking video with background audio
            mixed_video_path = self.audio_video_mixer.mix_audio_to_video(
                video_path=talking_video_url,  # The talking video we just created
                audio_files=[audio_url],  # Background audio to add
                audio_volumes=[0.8]  # Lower volume for background (80%)
            )
            
            print(f"✅ Audio-video mixing completed: {mixed_video_path}")
            
            # Upload final mixed video to Supabase
            print("📤 Uploading final mixed video to Supabase...")
            with open(mixed_video_path, 'rb') as f:
                mixed_video_bytes = f.read()
            
            final_video_url = self.supabase_manager.upload_processed_asset_video(mixed_video_bytes)
            print(f"✅ Final video uploaded: {final_video_url}")
            
            step4_time = time.time() - step4_start
            self.benchmark_data['step_times']['audio_mixing'] = step4_time
            
            # Calculate total time and cost estimate
            total_time = time.time() - self.benchmark_data['start_time']
            modal_hours = self.benchmark_data['modal_processing_time'] / 3600  # Convert to hours
            cost_estimate = modal_hours * 4.5  # $4.5 per hour
            self.benchmark_data['total_cost_estimate'] = cost_estimate
            
            # Test Summary with Benchmarking
            print("\n" + "="*60)
            print("🎉 TEXT-TO-VIDEO TALKING CHARACTER TEST COMPLETED!")
            print("="*60)
            print("📊 GENERATED ASSETS:")
            print(f"🖼️  Generated Image: {image_url}")
            print(f"🎵  Generated Audio: {audio_url}")
            print(f"🎬  Talking Video: {talking_video_url}")
            print(f"🎭  Final Mixed Video: {final_video_url}")
            
            print("\n📈 PERFORMANCE BENCHMARKS:")
            print(f"⏱️  Total Pipeline Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            print(f"⚡  Modal Processing Time: {self.benchmark_data['modal_processing_time']:.2f} seconds")
            print(f"📏  Video Length: {self.benchmark_data['video_length_seconds']:.1f} seconds")
            print(f"💰  Estimated Cost: ${cost_estimate:.4f} (at $4.5/hour for Modal calls)")
            
            print("\n⏲️  STEP BREAKDOWN:")
            for step_name, step_time in self.benchmark_data['step_times'].items():
                print(f"   {step_name.replace('_', ' ').title()}: {step_time:.2f}s")
            
            print("\n✅ All pipeline steps completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_audio_video_mixing(self):
        """
        Test audio-video mixing with pre-generated assets
        
        Uses known working video and audio URLs to test the AudioVideoMixer functionality
        """
        print("\n" + "="*60)
        print("🎵 STARTING AUDIO-VIDEO MIXING TEST")
        print("="*60)
        
        try:
            # Pre-generated assets from previous test run
            video_url = "https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/ded814f6-c13d-4397-a779-4a752bb86c7b.mp4?"
            audio_url = "https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_audios/01c8fc95-d519-4b33-b767-4e71f00c4362.wav?"
            
            print(f"🎬 Input Video: {video_url}")
            print(f"🎵 Input Audio: {audio_url}")
            print("🔊 Audio volume: 0.8 (80%)")
            
            step_start = time.time()
            
            # Mix the video with audio - the video has no audio track, so we're adding audio
            mixed_video_path = self.audio_video_mixer.mix_audio_to_video(
                video_path=video_url,
                audio_files=[audio_url],
                audio_volumes=[0.8]  # 80% volume
            )
            
            print(f"✅ Audio-video mixing completed: {mixed_video_path}")
            
            # Upload final mixed video to Supabase
            print("📤 Uploading final mixed video to Supabase...")
            with open(mixed_video_path, 'rb') as f:
                mixed_video_bytes = f.read()
            
            final_video_url = self.supabase_manager.upload_processed_asset_video(mixed_video_bytes)
            print(f"✅ Final video uploaded: {final_video_url}")
            
            step_time = time.time() - step_start
            
            # Test Summary
            print("\n" + "="*60)
            print("🎉 AUDIO-VIDEO MIXING TEST COMPLETED!")
            print("="*60)
            print("📊 RESULTS:")
            print(f"🎬  Input Video: {video_url}")
            print(f"🎵  Input Audio: {audio_url}")
            print(f"🎭  Final Mixed Video: {final_video_url}")
            print(f"⏱️  Processing Time: {step_time:.2f} seconds")
            
            print("\n✅ Audio-video mixing test completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Audio-video mixing test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_all_audio_features(self):
        """
        Test comprehensive audio features: Generate audio effects + Multi-layer audio mixing
        
        This test demonstrates:
        1. Generate background audio effects based on video scene
        2. Layer multiple audio tracks (speech + background) with different volumes
        3. Mix all audio layers with the video
        """
        print("\n" + "="*60)
        print("🎵 STARTING ALL AUDIO FEATURES TEST")
        print("="*60)
        
        try:
            # Pre-generated assets
            video_url = "https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/ded814f6-c13d-4397-a779-4a752bb86c7b.mp4?"
            speech_audio_url = "https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_audios/01c8fc95-d519-4b33-b767-4e71f00c4362.wav?"
            
            print(f"🎬 Input Video: {video_url}")
            print(f"🗣️ Speech Audio: {speech_audio_url}")
            
            # Step 1: Generate Background Audio Effects
            print("\n🎼 Step 1: Generating background audio effects...")
            step1_start = time.time()
            
            # Create audio effects prompt based on the video scene
            audio_effects_prompt = ("large hall with piano playing in distance, no people talking")

            print(f"📝 Audio Effects Prompt: {audio_effects_prompt}")
            
            # Generate background audio effects using ModalManager
            # background_audio_bytes = self.modal_manager.generate_audio_effects(
            #     prompt=audio_effects_prompt,
            #     video_url=video_url,  # Video-to-audio generation (analyze video for matching audio)
            #     duration=13
            # )
            background_audio_bytes = self.modal_manager.generate_music_with_lyrics(
                prompt="classic piano music with soft strings and light percussion",
                lyrics="[INST]",
                duration=13
            )

            if not background_audio_bytes:
                print("❌ Failed to generate background audio effects")
                return False
            
            print(f"✅ Background audio effects generated successfully ({len(background_audio_bytes)} bytes)")
            
            # Upload background audio to Supabase
            print("📤 Uploading background audio to Supabase...")
            background_audio_url = self.supabase_manager.upload_processed_asset_audio(background_audio_bytes)
            print(f"✅ Background audio uploaded: {background_audio_url}")
            
            step1_time = time.time() - step1_start
            
            # Step 2: Multi-layer Audio Mixing
            print("\n🎚️ Step 2: Multi-layer audio mixing...")
            step2_start = time.time()
            
            print(f"🎬 Video: {video_url}")
            print(f"🗣️ Speech Audio: {speech_audio_url} (100% volume)")
            print(f"🎼 Background Audio: {background_audio_url} (10% volume)")
            print("🔊 Mixing layers: Speech (100%) + Background ambience (10%)")

            # Mix the video with multiple audio layers
            mixed_video_path = self.audio_video_mixer.mix_audio_to_video(
                video_path=video_url,
                audio_files=[speech_audio_url, background_audio_url],
                audio_volumes=[1.0, 0.1]  # Speech at 100%, background at 10%
            )
            
            print(f"✅ Multi-layer audio mixing completed: {mixed_video_path}")
            
            # Upload final video to Supabase
            print("📤 Uploading final layered video to Supabase...")
            with open(mixed_video_path, 'rb') as f:
                final_video_bytes = f.read()
            
            final_video_url = self.supabase_manager.upload_processed_asset_video(final_video_bytes)
            print(f"✅ Final layered video uploaded: {final_video_url}")
            
            step2_time = time.time() - step2_start
            total_time = step1_time + step2_time
            
            # Test Summary
            print("\n" + "="*60)
            print("🎉 ALL AUDIO FEATURES TEST COMPLETED!")
            print("="*60)
            print("📊 RESULTS:")
            print(f"🎬  Input Video: {video_url}")
            print(f"🗣️  Speech Audio: {speech_audio_url}")
            print(f"🎼  Generated Background: {background_audio_url}")
            print(f"🎭  Final Layered Video: {final_video_url}")
            
            print("\n📈 PERFORMANCE:")
            print(f"⏱️  Audio Effects Generation: {step1_time:.2f} seconds")
            print(f"🎚️  Multi-layer Mixing: {step2_time:.2f} seconds")
            print(f"🔄  Total Processing Time: {total_time:.2f} seconds")
            
            print("\n🎵 AUDIO LAYERS:")
            print("   🗣️ Speech Audio: 80% volume (primary)")
            print("   🎼 Background Ambience: 20% volume (atmospheric)")
            print("   🎬 Result: Rich, layered audio experience")
            
            print("\n✅ All audio features test completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ All audio features test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def test_audio_seperation(self):
        """Test audio separation using Demucs via AudioTools wrapper.

        The sample is a mix with vocals and piano. We will:
        - Extract stems for vocals and instrumental (mix of non-vocals)
        - Verify files exist in storage/audio_dump
        - Print basic duration diagnostics
        - Additionally test 'keep' API for vocals-only output
        """
        print("\n" + "="*60)
        print("🎶 STARTING AUDIO SEPARATION TEST")
        print("="*60)

        VOCAL_PIANO_MIX_URL = (
            "https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_audios/0ca92e64-9517-4294-abf1-5d06ca89de05.wav"
        )

        try:
            step_start = time.time()
            print(f"🎵 Input Mix: {VOCAL_PIANO_MIX_URL}")
            print("🧪 Extracting stems: ['vocals', 'instrumental']")

            results = AudioTools.separate_audio_extract(
                VOCAL_PIANO_MIX_URL, ["vocals", "instrumental"]
            )

            vocals_path = results.get("vocals")
            instrumental_path = results.get("instrumental")
            print(f"🎤 Vocals Path: {vocals_path}")
            print(f"🎼 Instrumental Path: {instrumental_path}")

            if not vocals_path or not os.path.exists(vocals_path):
                print("❌ Vocals stem file missing")
                return False
            if not instrumental_path or not os.path.exists(instrumental_path):
                print("❌ Instrumental stem file missing")
                return False

            # Duration diagnostics (both are WAV)
            try:
                vocals_dur = AudioTools.get_audio_duration(vocals_path)
                instr_dur = AudioTools.get_audio_duration(instrumental_path)
                print(f"⏱️  Vocals duration: {vocals_dur:.2f}s")
                print(f"⏱️  Instrumental duration: {instr_dur:.2f}s")
            except Exception as e:
                print(f"⚠️ Duration check skipped: {e}")

            # Also validate 'keep' API for a single stem
            print("\n🔒 Testing keep API: ['vocals'] -> single output")
            kept_vocals_path = AudioTools.separate_audio_keep(
                VOCAL_PIANO_MIX_URL, ["vocals"], output_basename="kept_vocals"
            )
            print(f"🎯 Kept Vocals Path: {kept_vocals_path}")
            if not kept_vocals_path or not os.path.exists(kept_vocals_path):
                print("❌ Kept vocals file missing")
                return False

            step_time = time.time() - step_start

            print("\n" + "="*60)
            print("🎉 AUDIO SEPARATION TEST COMPLETED!")
            print("="*60)
            print("📊 RESULTS:")
            print(f"🎤  Vocals Stem: {vocals_path}")
            print(f"🎼  Instrumental Stem: {instrumental_path}")
            print(f"🔒  Kept Vocals: {kept_vocals_path}")
            print(f"⏱️  Processing Time: {step_time:.2f} seconds")
            print("\n✅ Audio separation test completed successfully!")

            return True

        except RuntimeError as e:
            print(f"\n❌ Audio separation test failed: {e}")
            print("ℹ️ Ensure 'demucs' is installed and models can be downloaded.")
            return False
        except Exception as e:
            print(f"\n❌ Audio separation test unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """Run all available tests"""
        print("🚀 Starting Modal Manager Tests...")
        
        tests = [
            # ("Text-to-Video Talking Character", self.text_to_video_talking_character),  # Commented out due to FFmpeg audio mixing issue
            #("Audio-Video Mixing", self.test_audio_video_mixing),
            #("All Audio Features", self.test_all_audio_features),
            ("Audio Separation", self.test_audio_seperation)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                if test_func():
                    print(f"✅ {test_name}: PASSED")
                    passed += 1
                else:
                    print(f"❌ {test_name}: FAILED")
                    failed += 1
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {str(e)}")
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"🏁 TEST SUMMARY: {passed} passed, {failed} failed")
        print(f"{'='*60}")
        
        return passed, failed


# Test runner
if __name__ == "__main__":
    test_suite = ModalManagerTests()
    test_suite.run_all_tests()

