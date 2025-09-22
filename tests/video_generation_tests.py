import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.video_engine_models import BackgroundAudioEffects, KeyFrame, KeyframeSource, Narration, RenderEngine, Set, SetVariant, TextOverlay, TextOverlayPosition, TextOverlayProperties, TextOverlayTiming, TextPresentationAnimationType, TextPresentationType, Video, VideoBlock
from utils.modal_manager import ModalManager
from utils.supabase_manager import SupabaseManager, set_supabase_manager
from utils.audio_video_tools import AudioVideoMixer
from utils.audio_tools import AudioTools
from utils.settings_manager import settings
from utils.video_generator_v2 import VideoGenerator


class VideoGeneratorTests:
    def __init__(self):
        self.modal = ModalManager()
        self.supabase = SupabaseManager(
            url=settings.supabase_url,
            key=settings.supabase_key,
        )
        set_supabase_manager(self.supabase)
        self.audio_tools = AudioTools()
        self.video_generator = VideoGenerator()

    def test_simple_talking_character_with_bg_audio_effects(self) -> bool:
        # This test is to create a talking character video without using the characters feature

        # Create video for scenario 1
        video = Video(
            id="scenario_1_balloon_adventure",
            title="Epic Balloon Journey to Space",
            description="Testing environment-only shot with magical balloon adventure",
            style="cinematic",
            cast=[],  # No characters for this scenario
            sets=[],
            shots=[VideoBlock(
                id="basicc_talking_character_with_bg_audio_effects",
                storyline_label="Balloon Adventure Intro",
                render_engine=RenderEngine.LIPSYNC_MOTION,  # Use LIPSYNC_MOTION with I2V
                duration_seconds=5,# Duration is ignored for LIPSYNC_MOTION
                fps=25,
                width=720,
                height=1280,
                first_keyframe=KeyFrame(
                    source=KeyframeSource.GENERATE,
                    linked_source_id=None,
                    supplied_image_url=None,
                    width=537,
                    height=896,
                    set=None,
                    characters=None,
                    basic_generation_prompt="A beautiful elegant woman with flowing blonde hair, wearing a vintage "
                                "burgundy dress, sitting gracefully in an ornate Victorian parlor, holding a glass of white wine,"
                                "with rich wooden furniture and warm golden lighting, portrait style, "
                                "high quality, cinematic lighting",
                    image_url=None,
                    rendered_frame_by_vid_gen_url=None
                ),
                last_keyframe=None,  # Not needed because render engine is LIPSYNC_MOTION
                video_prompt=("A sophisticated blonde woman, holding a glass of white wine, speaking elegantly about wine, with natural "
                            "facial expressions and subtle head movements, in a Victorian parlor setting"),
                style="cinematic, elegant",
                narration=Narration(
                    id="wine_tasting_narration",
                    exaggeration=0.5,
                    cfg_weight=0.5,
                    script=("The wine is quite delightful, with notes of cherry and oak that dance "
                                "upon the palate. This vintage has been aged to perfection, offering a "
                                "rich and complex flavor profile that would complement any elegant evening."),
                    voice_sample_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/audio_samples/219777__madamvicious__the-wine-is-quite-delightful-posh-woman.wav",
                    audio_url=None,
                ),
                overlays=None,
                bg_audio_effects=BackgroundAudioEffects(
                    is_enabled=True,
                    prompt=("large hall with piano playing in distance, no people talking")
                ),
                generated_video_clip_raw=None,
                generated_video_clip_with_overlays=None,
                generated_video_clip_with_audio_and_overlays=None,
                generated_video_clip_final=None,
                transition=None
            )],
            generated_video_url=None,
            background_music=None
        )
        

        processed_video = self.video_generator.process_video(video)

        print(f"Processed video: {processed_video}")

        return True
    
    def run_all_tests(self):
        """Run all available tests"""
        print("🚀 Starting Video Generator Tests...")

        tests = [
            ("Audio effect + speech video layering", self.test_simple_talking_character_with_bg_audio_effects)
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
    test_suite = VideoGeneratorTests()
    test_suite.run_all_tests()