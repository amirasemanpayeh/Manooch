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
    StrategyModel, load_strategy, ShortsVideoPlan, ShotData, ClipData, save_shots_plan, load_json_file
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

    def _generate_prompt_from_story(self, source_title: str, source_text: str, strategy: StrategyModel) -> str:
        # TODO: Compelete this method
        pass

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



        #TODO : Create the methods for different scenarions:
        # Public method to generate shorts video, these need to only take in the json file paths and be accessible via CLI
        # or via .sh scripts

        # generate talk_broll_talk prompt from source_title and source_text, the strategy model is for now hardcoded to talk_broll_talk
        # generate talk_broll_talk video plan from prompt, this will be need to then store the generated plan to a json file
        # generate talk_broll_talk video from plan, this will be need to append the generated video urls to the json file