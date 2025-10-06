import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.video_engine_models import BackgroundAudioEffects, KeyFrame, KeyframeSource, Narration, RenderEngine, Set, SetVariant, TextOverlay, TextOverlayPosition, TextOverlayProperties, TextOverlayTiming, TextPresentationAnimationType, TextPresentationType, Transition, Video, VideoBlock
from utils.modal_manager import ModalManager
from utils.supabase_manager import SupabaseManager, set_supabase_manager
from utils.audio_tools import AudioTools
from utils.settings_manager import settings
from utils.video_generator_v2 import VideoGenerator

from models.shorts_copy_models import (
    StrategyModel, load_strategy, ShortsVideoPlan, ShotData, ClipData, save_shots_plan, load_json_file, load_shots_plan
)


class StrategyManager:
    def __init__(self):
        self.modal = ModalManager()
        self.supabase = SupabaseManager(
            url=settings.supabase_url,
            key=settings.supabase_key,
        )
        set_supabase_manager(self.supabase)
        self.audio_tools = AudioTools()
        self.video_generator = VideoGenerator()


    def _generate_plan_from_prompt(self, prompt: str) -> ShortsVideoPlan:
        # TODO: Compelete this method
        pass

    def _generate_video_from_plan(self, plan: ShortsVideoPlan) -> ShortsVideoPlan:
        # TODO: Compelete this method
        pass
    
    # Public CLI methods
    def generate_prompt_from_story_file(self, story_json_path: str, strategy_json_path: str) -> str:
        """
        Generate LLM prompt from a story JSON file and strategy JSON file.
        
        Args:
            story_json_path: Path to JSON file containing source_title and source_text
            strategy_json_path: Path to strategy JSON file
            
        Returns:
            Generated prompt string
        """
        # Load the story data
        story_data = load_json_file(story_json_path)
        source_title = story_data.get("source_title", "")
        source_text = story_data.get("source_text", "")
        
        # Load the strategy
        strategy = load_strategy(strategy_json_path)
        
        # Generate and return the prompt - no extra context needed since we're using SOURCE_TITLE and SOURCE_TEXT
        prompt = strategy.create_prompt_from_strategy(source_title, source_text)
        return prompt

    def generate_video_from_plan_file(self, plan_json_path: str) -> str:
        """
        Load a plan JSON file and convert it into a ShortsVideoPlan model with properly ordered shots.
        
        Args:
            plan_json_path: Path to JSON file containing the shots plan
            
        Returns:
            ShortsVideoPlan with shots ordered by shot.order from 0 to end
        """
        # Load and deserialize the plan using the specialized function
        # This handles both {"shots": [...]} and {"shorts_video_plan": {"shots": [...]}} formats
        plan: ShortsVideoPlan = load_shots_plan(plan_json_path)
        
        # The load_shots_plan function already validates contiguous ordering (0, 1, 2, ...)
        # Get the ordered shots as a list
        ordered_shots = plan.ordered()

        video_blocks: list[VideoBlock] = []

        for shot in ordered_shots:
            if shot.clip:
                # its b roll
                video_block = VideoBlock(
                    id="short_video_shot_" + str(ordered_shots.index(shot)),
                    storyline_label="short_video_shot_" + str(ordered_shots.index(shot)),
                    render_engine=RenderEngine.GENERATIVE_MOTION,  # Use LIPSYNC_MOTION with I2V
                    duration_seconds=5,# Duration is ignored for LIPSYNC_MOTION
                    fps=25,
                    width=720,
                    height=1280,
                    first_keyframe=KeyFrame(
                        source=KeyframeSource.GENERATE,
                        linked_source_id=None,
                        width=720,
                        height=1280,
                        set=None,
                        characters=None,
                        basic_generation_prompt=shot.clip.keyframe_prompt,
                        supplied_image_url=None,
                        rendered_frame_by_vid_gen_url=None
                    ),
                    last_keyframe=None,  # Not needed because render engine is LIPSYNC_MOTION
                    video_prompt=shot.clip.motion_prompt,
                    style="cinematic",
                    narrations=[
                        Narration(
                            id="narration_001",
                            exaggeration=0.5,
                            cfg_weight=0.5,
                            script=shot.narration,
                            voice_sample_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/Temp%20Dump/JoseAudioSample.wav",
                            audio_url=None,
                            audio_padded_url=None
                        )
                    ],                
                    overlays = [
                        TextOverlay(
                            text=shot.narration,
                            position=TextOverlayPosition.CENTER,
                            properties=TextOverlayProperties(
                                font_size=42,
                                color="#C4F4E0",
                                background_color="rgba(0,0,0,0.8)",  # Better with transparency
                                font_family="Bebas Neue",  # Add missing font family
                                padding=10,           # Add missing padding
                                border_radius=5,      # Add missing border radius
                                presentation_type=TextPresentationType.SLIDING_WINDOW,
                                presentation_animation_type=TextPresentationAnimationType.NONE,  # Correct field name
                                window_size=2,        # Show 2 words at a time instead of default 3
                                emphasize_latest_word=True,  # Defaults to False if not specified
                                timing=TextOverlayTiming(
                                    start_time_seconds=0.0,     # Correct field name
                                    duration_seconds=None,       # Set to NULL to match the video length
                                    reveal_unit="word",         # Required for progressive reveal
                                    reveal_speed=1.0,           # Optional but good to specify
                                    animation_duration_in=0.5,  # For animations
                                    animation_duration_out=0.5  # For animations
                                )
                            )
                        )
                    ],
                    transition=Transition.CUT,
                    bg_audio_effects=None,
                    generated_video_clip_raw=None,
                    generated_video_clip_with_overlays=None,
                    generated_video_clip_with_audio_and_overlays=None,
                    generated_video_clip_final=None,
                )
                video_blocks.append(video_block)
            else:
                # its a roll
                print("  No clip data found for this shot.")
                video_block = VideoBlock(
                    id="short_video_shot_" + str(ordered_shots.index(shot)),
                    storyline_label="short_video_shot_" + str(ordered_shots.index(shot)),
                    render_engine=RenderEngine.LIPSYNC_MOTION,  # Use LIPSYNC_MOTION with I2V
                    duration_seconds=5,# Duration is ignored for LIPSYNC_MOTION
                    fps=25,
                    width=720,
                    height=1280,
                    first_keyframe=KeyFrame(
                        source=KeyframeSource.SUPPLIED,
                        linked_source_id=None,
                        width=720,
                        height=1280,
                        set=None,
                        characters=None,
                        basic_generation_prompt = None, # "Edit this image and make the man sit behind a desk in the same environment, same distance from the camera, cinematic lighting, high quality, photorealistic, detailed, 4k",
                        supplied_image_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/Temp%20Dump/tempImageq4QLM2.png",
                        rendered_frame_by_vid_gen_url=None
                    ),
                    last_keyframe=None,  # Not needed because render engine is LIPSYNC_MOTION
                    video_prompt=("A man talking passionately"),
                    style="cinematic",
                    narrations=[
                        Narration(
                            id="narration_001",
                            exaggeration=0.5,
                            cfg_weight=0.5,
                            script=shot.narration,
                            voice_sample_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/Temp%20Dump/JoseAudioSample.wav",
                            audio_url=None,
                            audio_padded_url=None
                        )
                    ],
                    overlays = [
                        TextOverlay(
                            text=shot.narration,
                            position=TextOverlayPosition.CENTER,
                            properties=TextOverlayProperties(
                                font_size=42,
                                color="#C4F4E0",
                                background_color="rgba(0,0,0,0.8)",  # Better with transparency
                                font_family="Bebas Neue",  # Add missing font family
                                padding=10,           # Add missing padding
                                border_radius=5,      # Add missing border radius
                                presentation_type=TextPresentationType.SLIDING_WINDOW,
                                presentation_animation_type=TextPresentationAnimationType.NONE,  # Correct field name
                                window_size=2,        # Show 2 words at a time instead of default 3
                                emphasize_latest_word=True,  # Defaults to False if not specified
                                timing=TextOverlayTiming(
                                    start_time_seconds=0.0,     # Correct field name
                                    duration_seconds=7.0,       # hard 7 seconds to match narration audio
                                    reveal_unit="word",         # Required for progressive reveal
                                    reveal_speed=1.0,           # Optional but good to specify
                                    animation_duration_in=0.5,  # For animations
                                    animation_duration_out=0.5  # For animations
                                )
                            )
                        )
                    ],
                    transition=Transition.CUT,
                    bg_audio_effects=None,
                    generated_video_clip_raw=None,
                    generated_video_clip_with_overlays=None,
                    generated_video_clip_with_audio_and_overlays=None,
                    generated_video_clip_final=None
                )
                video_blocks.append(video_block)
        
        # Now create a Video object to hold all the blocks
        # Create video for scenario 1
        video = Video(
            id="first_short_content_test_001",
            title="Rodrigo Mom Story Full Video",
            description="Full video from generated shots",
            style="cinematic",
            cast=[],  # No characters for this scenario
            sets=[],  # No sets for this scenario
            shots= video_blocks,
            generated_video_url=None,
            background_music=None
        )

        processed_video = self.video_generator.process_video(video)
                
        print(f"Processed video: {processed_video}")
        return processed_video.generated_video_url
