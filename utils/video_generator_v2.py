# ───────────────────────────────────────────────────────────────────────────────
# Video Scene Generation Model (Short-Form) — Implementation-Ready & Commented
# ───────────────────────────────────────────────────────────────────────────────
# Purpose:
#   - Serves developers (engine + validation), UI authors (field meanings), and AI agents
#     (what to populate) with one unified schema.
#
# Design headlines:
#   - Shots are 5s blocks (min/max 5s) inside a Scene.
#   - Start = link_frame_strategy; End = optional lock_end_keyframe.
#   - RenderEngine controls how frames are turned into motion.
#   - Characters: multiple allowed, but only one Main Character (MC) per scene may use a
#     reference image — and only when the shot is independent+generated.
#   - Supplied independent start ⇒ prompt-only (no reference anchoring) for that shot.
#   - Engine always caches first/last frames.
# ───────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import io
import os
import requests

from typing import List, Optional, Tuple
from datetime import datetime
from PIL import Image

# Import external dependencies
from prompts.video_engine_prompts import VideoEnginePromptGenerator
from models.video_engine_models import Character, CharacterVariant, KeyFrame, KeyframeSource, Music, Placement, RenderEngine, Set, SetVariant, Transition, Video, VideoBlock
from utils.audio_tools import AudioTools
from utils.modal_manager import ModalManager
from utils.openai_manager import OpenAIManager
from utils.audio_video_tools import AudioVideoMixer
from utils.video_tools.video_stitcher import VideoStitcher
import importlib.util as _importlib_util
from pathlib import Path as _Path

from utils.supabase_manager import SupabaseManager
from utils.settings_manager import Settings

# Note: There is both a module `utils/video_tools.py` and package `utils/video_tools/`.
# We instantiate stitcher/mixer directly to avoid name collisions.

# Background removal support
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("⚠️ rembg not available - background removal will be skipped")



# ───────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ───────────────────────────────────────────────────────────────────────────────

class VideoGenerator:
    """
    Orchestrates the entire text-to-video pipeline.
    Handles story segmentation into scenes and shots,
    generates prompts, images, video, audio, and final assembly.
    """

    def __init__(self):
        """Initialize the VideoGenerator with external dependencies"""
        # Load settings
        self.settings = Settings()
        
        # Initialize external dependencies
        self.modal_manager = ModalManager()
        # Instantiate OpenAI only if configured, to avoid optional dependency issues
        self.openai_manager = (
            OpenAIManager(api_key=self.settings.openai_api_key)
            if getattr(self.settings, "use_openai_for_image_editing", False)
            else None
        )
        
        # Initialize video processing tools individually to avoid circular imports
        self.video_stitcher: VideoStitcher = VideoStitcher()
        self.audio_video_mixer: AudioVideoMixer = AudioVideoMixer()

        # Dynamically import VideoTools class from utils/video_tools.py (name collides with package)
        try:
            _vt_path = _Path(__file__).parent / "video_tools.py"
            _spec = _importlib_util.spec_from_file_location("utils.video_tools_file", str(_vt_path))
            if _spec and _spec.loader:
                _vt_mod = _importlib_util.module_from_spec(_spec)
                _spec.loader.exec_module(_vt_mod)
                _VT = getattr(_vt_mod, "VideoTools", None)
                self.video_tools = _VT() if _VT else None
            else:
                self.video_tools = None
        except Exception as e:
            print(f"Warning: Could not load VideoTools: {e}")
            self.video_tools = None

        # Import here to avoid circular dependency
        try:
            from utils.video_tools.video_text_overlay import VideoTextOverlayManager
            self.text_overlay_manager = VideoTextOverlayManager()
        except ImportError as e:
            print(f"Warning: Could not import VideoTextOverlayManager: {e}")
            self.text_overlay_manager = None
        
        self.supabase_manager = SupabaseManager(
            url=self.settings.supabase_url, 
            key=self.settings.supabase_key
        )

        # Setup logging - use module logger with proper isolation to avoid duplicates
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Only add handler if none exists and prevent propagation to avoid duplicates
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            # Prevent propagation to parent loggers to avoid duplicate messages
            self.logger.propagate = False


    # Upload helpers using Supabase
    def _upload_video_asset_to_bucket(self, video_bytes: bytes) -> str:
        """
        Upload the generated video bytes to a storage bucket and return the URL.
        """
        return self.supabase_manager.upload_processed_asset_video(video_bytes)

    def _upload_audio_asset_to_bucket(self, audio_bytes: bytes) -> str:
        """
        Upload the generated audio bytes to a storage bucket and return the URL.
        """
        return self.supabase_manager.upload_processed_asset_audio(audio_bytes)

    def _upload_image_asset_to_bucket(self, image_bytes: bytes) -> str:
        """
        Upload the generated image bytes to a storage bucket and return the URL.
        """
        return self.supabase_manager.upload_processed_asset_img(image_bytes)

    # ==========================================================================
    # Universal Image Editing Method - Routes to OpenAI or Modal based on settings
    # ==========================================================================
    
    def _edit_image_with_provider(self, input_image_url: str, prompt: str, target_width: int = None, target_height: int = None) -> bytes:
        """
        Universal image editing method that routes to OpenAI or Modal based on settings.
        Handles size constraints and resizing automatically for OpenAI.
        """
        primary_is_openai = bool(getattr(self.settings, "use_openai_for_image_editing", False))

        def try_openai():
            if not self.openai_manager:
                return None
            self.logger.info("Using OpenAI GPT-image-1 for image editing")
            return self._edit_image_with_openai(input_image_url, prompt, target_width, target_height)

        def try_modal():
            self.logger.info("Using Qwen I2I (Modal) for image editing")
            try:
                return self.modal_manager.edit_image(input_image_url=input_image_url, prompt=prompt)
            except Exception as e:
                self.logger.error(f"Modal I2I failed: {e}")
                return None

        # Attempt primary provider, then fallback to the other if available
        result = None
        try:
            result = try_openai() if primary_is_openai else try_modal()
        except Exception as e:
            self.logger.error(f"Primary image editing provider error: {e}")
            result = None

        if result is None:
            self.logger.info("Falling back to alternate image editing provider")
            try:
                result = try_modal() if primary_is_openai else try_openai()
            except Exception as e:
                self.logger.error(f"Fallback image editing provider error: {e}")
                result = None

        return result

    def _edit_image_with_openai(self, input_image_url: str, prompt: str, target_width: int = None, target_height: int = None) -> bytes:
        """
        Edit image using OpenAI GPT-image-1 with proper sizing and resizing.
        """
        try:
            # Download the input image
            response = requests.get(input_image_url)
            if response.status_code != 200:
                self.logger.error(f"Failed to download input image: {response.status_code}")
                return None
            
            original_image_bytes = response.content
            
            # If no target dimensions specified, use original image dimensions
            if not target_width or not target_height:
                from PIL import Image
                img = Image.open(io.BytesIO(original_image_bytes))
                target_width = target_width or img.width
                target_height = target_height or img.height
            
            # Select appropriate OpenAI size
            openai_size = self._select_openai_image_size(target_width, target_height)
            
            # Generate with OpenAI
            result_bytes = self.openai_manager.edit_image_gpt_image_1(
                images=[original_image_bytes],
                prompt=prompt,
                size=openai_size,
                quality="medium"
            )
            
            if not result_bytes:
                return None
            
            # Resize to target dimensions if needed
            if openai_size != f"{target_width}x{target_height}":
                result_bytes = self._resize_image_to_target(result_bytes, target_width, target_height)
            
            return result_bytes
            
        except Exception as e:
            self.logger.error(f"OpenAI image editing failed: {e}")
            return None


    def _resolve_video_assets(self, video: Video) -> Video:
        """
        Main entry point for resolving all video-level assets.
        This should be called before processing any keyframes.
        """
        self.logger.info(f"Resolving assets for video '{video.title}'")
        
        # First resolve cast templates
        self._resolve_cast_for_video(video)
        
        # Then resolve set templates  
        self._resolve_sets_for_video(video)
        
        self.logger.info(f"Completed asset resolution for video '{video.title}'")
        return video

    def _resolve_cast_for_video(self, video: Video) -> None:
        """
        Resolve the cast for a given video by ensuring all characters are properly set up.
        For each character in the cast:
        - If no image but has prompt: generate character image
        - If has image but no prompt: generate description from image
        - If has both: validate consistency
        - If has neither: log warning
        """
        if not video.cast:
            self.logger.info("No cast defined for video")
            return

        self.logger.info(f"Resolving cast for video '{video.title}': {[char.name for char in video.cast]}")

        for char in video.cast:
            self.logger.info(f"Processing character '{char.name}' (ID: {char.id})")
            
            # Case 1: No image, but has prompt - generate image
            if not char.image_url and char.prompt:
                self.logger.info(f"Generating image for character '{char.name}' from prompt")
                char.image_url = self._generate_character_image(char)
                
            # Case 2: Has image, but no prompt - generate description
            elif char.image_url and not char.prompt:
                self.logger.info(f"Generating prompt for character '{char.name}' from image")
                char.prompt = self._generate_character_prompt_from_image(char)
                
            # Case 3: Has both - validate they're consistent (optional)
            elif char.image_url and char.prompt:
                self.logger.info(f"Character '{char.name}' has both image and prompt - using as-is")
                
            # Case 4: Has neither - warning
            else:
                self.logger.warning(f"Character '{char.name}' has neither image nor prompt - may cause issues")

    def _resolve_sets_for_video(self, video: Video) -> None:
        """
        Resolve the sets for a given video by ensuring all set templates are properly set up.
        For each set in the sets list:
        - If no image but has prompt: generate set image
        - If has image but no prompt: generate description from image  
        - If has both: validate consistency
        - If has neither: log warning
        """
        if not video.sets:
            self.logger.info("No sets defined for video")
            return

        self.logger.info(f"Resolving sets for video '{video.title}': {len(video.sets)} sets")

        for set_template in video.sets:
            self.logger.info(f"Processing set '{set_template.id}' with prompt: '{set_template.prompt[:50]}...'")
            
            # Case 1: No image, but has prompt - generate image
            if not set_template.image_url and set_template.prompt:
                self.logger.info(f"Generating image for set '{set_template.id}' from prompt")
                set_template.image_url = self._generate_set_image(set_template)
                
            # Case 2: Has image, but no prompt - generate description
            elif set_template.image_url and not set_template.prompt:
                self.logger.info(f"Generating prompt for set '{set_template.id}' from image")
                set_template.prompt = self._generate_set_prompt_from_image(set_template)
                
            # Case 3: Has both - validate they're consistent (optional)
            elif set_template.image_url and set_template.prompt:
                self.logger.info(f"Set '{set_template.id}' has both image and prompt - using as-is")
                
            # Case 4: Has neither - warning
            else:
                self.logger.warning(f"Set '{set_template.id}' has neither image nor prompt - may cause issues")    
    
    def _generate_character_variant_image(self, parent_character: Character, character_variant: CharacterVariant) -> str:
        """
        Generate a variant image for a character based on their parent image.
        Uses I2I to modify the parent character's appearance/pose/action.
        Includes automatic background removal for better compositing.
        """
        self.logger.info(f"Generating variant image for character variant based on '{parent_character.name}'")
        
        if not parent_character.image_url:
            self.logger.error(f"Parent character '{parent_character.name}' has no image_url to base variant on")
            return None
        
        try:
            # Build the I2I prompt for character variation
            variant_prompt = VideoEnginePromptGenerator.build_character_variant_prompt(character_variant.prompt, character_variant.style, character_variant.action_in_frame)
            
            # Use universal image editing method (routes to OpenAI or Modal)
            image_bytes = self._edit_image_with_provider(
                input_image_url=parent_character.image_url,
                prompt=variant_prompt,
                target_width=720,  # Standard character size
                target_height=1280
            )
            
            if not image_bytes:
                self.logger.error(f"Failed to generate variant image bytes for character variant")
                return None
            
            # Remove background for better board composition
            processed_bytes = self._remove_background(image_bytes)
            
            # Upload processed image to Supabase and get public URL
            image_url = self._upload_image_asset_to_bucket(processed_bytes)
            
            if not image_url:
                self.logger.error(f"Failed to upload character variant image")
                return None
            
            self.logger.info(f"Generated character variant image with background removal: {image_url}")
            return image_url
            
        except Exception as e:
            self.logger.error(f"Failed to generate variant image: {e}")
            return None

    def _select_openai_image_size(self, target_width: int, target_height: int) -> str:
        """
        Select the best OpenAI image size based on target dimensions.
        OpenAI supports: 1024x1024 (square), 1536x1024 (landscape), 1024x1536 (portrait)
        """
        aspect_ratio = target_width / target_height
        
        if aspect_ratio > 1.3:  # Landscape
            return "1536x1024"
        elif aspect_ratio < 0.8:  # Portrait
            return "1024x1536"
        else:  # Square or close to square
            return "1024x1024"

    def _resize_image_to_target(self, image_bytes: bytes, target_width: int, target_height: int) -> bytes:
        """
        Resize image bytes to target dimensions while maintaining quality.
        """
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Resize to target dimensions
            resized_image = image.resize((target_width, target_height), Image.LANCZOS)
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            resized_image.save(output_buffer, format='PNG', quality=95)
            return output_buffer.getvalue()
            
        except Exception as e:
            self.logger.error(f"Failed to resize image: {e}")
            return image_bytes  # Return original if resize fails

    def _generate_set_variant_image(self, parent_set: Set, set_variant: SetVariant) -> str:
        """
        Generate a variant image for a set based on the parent set image.
        Uses either OpenAI GPT-image-1 or Qwen I2I based on settings flag.
        """
        self.logger.info(f"Generating variant image for set variant based on parent set '{parent_set.id}'")
        
        if not parent_set.image_url:
            self.logger.error(f"Parent set '{parent_set.id}' has no image_url to base variant on")
            return None
        
        try:
            # Build the I2I prompt for set variation
            #variant_prompt = self._build_set_variant_prompt(set_variant, parent_set)
            variant_prompt = VideoEnginePromptGenerator.build_set_variant_prompt(set_variant.prompt, set_variant.style)
            
            # Choose provider based on settings flags (check both new universal flag and old specific flag)
            use_openai = self.settings.use_openai_for_image_editing or self.settings.use_openai_for_set_variants
            
            if use_openai:
                self.logger.info("Using OpenAI GPT-image-1 for set variant generation")
                # Get target dimensions
                target_width = getattr(set_variant, 'width', None) or parent_set.width
                target_height = getattr(set_variant, 'height', None) or parent_set.height
                
                image_bytes = self._edit_image_with_openai(
                    input_image_url=parent_set.image_url,
                    prompt=variant_prompt,
                    target_width=target_width,
                    target_height=target_height
                )
            else:
                self.logger.info("Using Qwen I2I for set variant generation")
                image_bytes = self.modal_manager.edit_image(
                    input_image_url=parent_set.image_url,
                    prompt=variant_prompt
                )
            
            if not image_bytes:
                self.logger.error(f"Failed to generate variant image bytes for set variant")
                return None
            
            # Upload to Supabase and get public URL
            image_url = self._upload_image_asset_to_bucket(image_bytes)

            if not image_url:
                self.logger.error(f"Failed to upload set variant image")
                return None
            
            self.logger.info(f"Generated set variant image: {image_url}")
            return image_url
            
        except Exception as e:
            self.logger.error(f"Failed to generate set variant image: {e}")
            return None
    # ==========================================================================
    # Character Image Generation Methods
    # ==========================================================================
    
    def _generate_character_image(self, character: Character) -> str:
        """
        Generate a reference image for a character based on their prompt.
        This uses text-to-image generation to create the base character appearance.
        Includes automatic background removal for better compositing.
        """
        self.logger.info(f"Generating reference image for character '{character.name}'")
        
        try:
            # Construct the full prompt for character generation
            full_prompt = VideoEnginePromptGenerator.build_character_generator_prompt(
                character.description,
                character.prompt,
                character.style,
                headshot_only=False  # Full body for character templates
                )
            
            # Use ModalManager for T2I generation
            image_bytes = self.modal_manager.generate_image_from_prompt(
                prompt=full_prompt,
                width=1024,  # Standard character image size
                height=1024,
                batch_size=1
            )
            
            if not image_bytes:
                self.logger.error(f"Failed to generate image bytes for character '{character.name}'")
                return None
            
            # Remove background for better board composition
            processed_bytes = self._remove_background(image_bytes)
            
            # Upload processed image to Supabase and get public URL
            image_url = self.supabase_manager.upload_generated_asset_img(processed_bytes)
            
            if not image_url:
                self.logger.error(f"Failed to upload character image for '{character.name}'")
                return None
            
            self.logger.info(f"Generated character image with background removal: {image_url}")
            return image_url
            
        except Exception as e:
            self.logger.error(f"Failed to generate image for character '{character.name}': {e}")
            return None

    def _generate_character_prompt_from_image(self, character: Character) -> str:
        """
        Generate a descriptive prompt for a character based on their existing image.
        This uses image-to-text analysis to describe what's in the character image.
        """
        self.logger.info(f"Generating prompt for character '{character.name}' from image")
        
        try:
            # Use ModalManager.describe_image for I2T analysis
            description = self.modal_manager.describe_image(
                image_url=character.image_url,
                detail_level="basic",
                reformat=True
            )
            
            if not description:
                self.logger.error("Failed to analyze character image")
                return f"Character {character.name} (image analysis failed)"
            
            # Enhance the description for character context
            enhanced_description = f"Character {character.name}: {description}"
            
            self.logger.info(f"Generated character description: {enhanced_description}")
            return enhanced_description
            
        except Exception as e:
            self.logger.error(f"Failed to generate prompt for character '{character.name}': {e}")
            return f"Character {character.name} (description generation failed)"


    # ==========================================================================
    # Set Image Generation Methods  
    # ==========================================================================
    
    def _generate_set_image(self, set_template: Set) -> str:
        """
        Generate an environment image for a set based on its prompt.
        This creates the base background/environment that characters will be composited into.
        """
        self.logger.info(f"Generating environment image for set '{set_template.id}'")
        
        try:
            # Construct the full prompt for set generation
            full_prompt = VideoEnginePromptGenerator.build_set_generation_prompt(
                prompt=set_template.prompt,
                style=set_template.style,
                width=set_template.width,
                height=set_template.height
            )

            # Use ModalManager for T2I generation
            image_bytes = self.modal_manager.generate_image_from_prompt(
                prompt=full_prompt,
                width=set_template.width,
                height=set_template.height,
                batch_size=1
            )
            
            if not image_bytes:
                self.logger.error(f"Failed to generate image bytes for set '{set_template.id}'")
                return None
            
            # Upload to Supabase and get public URL
            image_url = self.supabase_manager.upload_generated_asset_img(image_bytes)
            
            if not image_url:
                self.logger.error(f"Failed to upload set image for '{set_template.id}'")
                return None
            
            self.logger.info(f"Generated set image: {image_url}")
            return image_url
            
        except Exception as e:
            self.logger.error(f"Failed to generate image for set '{set_template.id}': {e}")
            return None

    def _generate_set_prompt_from_image(self, set_template: Set) -> str:
        """
        Generate a descriptive prompt for a set based on its existing image.
        This uses image-to-text analysis to describe the environment.
        """
        self.logger.info(f"Generating prompt for set '{set_template.id}' from image")
        
        try:
            # Use ModalManager.describe_image for I2T environment analysis
            description = self.modal_manager.describe_image(
                image_url=set_template.image_url,
                detail_level="more_detailed",
                reformat=True
            )
            
            if not description:
                self.logger.error("Failed to analyze set image")
                return f"Environment scene (image analysis failed)"
            
            # Enhance the description for environment context
            enhanced_description = f"Environment scene: {description}"
            
            self.logger.info(f"Generated set description: {enhanced_description}")
            return enhanced_description
            
        except Exception as e:
            self.logger.error(f"Failed to generate prompt for set '{set_template.id}': {e}")
            return f"Environment scene (description generation failed)"

    # ==========================================================================
    # Keyframe-level processing (placeholder for next phase)  
    # ==========================================================================
    def _resolve_keyframe(self, video: Video, keyframe: KeyFrame) -> KeyFrame:
        """
        Resolve a keyframe by ensuring it has all necessary data.
        """
        self.logger.info(f"Resolving keyframe '{keyframe.id}' with source: {keyframe.source}")
        
        if keyframe.source == KeyframeSource.SUPPLIED:
            keyframe.image_url = keyframe.supplied_image_url
            self.logger.warning(f"Keyframe '{keyframe.id}' is using supplied image URL - characters and scenes not considered")
            
        elif keyframe.source == KeyframeSource.GENERATE:
            keyframe.image_url = self._generate_keyframe_image(video, keyframe)
            
        elif keyframe.source == KeyframeSource.COPY_PREV_LAST:
            if keyframe.linked_source_id:
                linked_keyframe = self._find_keyframe_by_id(video, keyframe.linked_source_id)
                if linked_keyframe:
                    # Copy the entire keyframe except the ID for image generation phase
                    self.logger.info(f"Keyframe '{keyframe.id}' copying from linked keyframe: {linked_keyframe.id}")
                    
                    # Preserve the original ID and linked_source_id
                    original_id = keyframe.id
                    original_linked_source_id = keyframe.linked_source_id
                    original_source = keyframe.source

                    # Copy all other attributes from the linked keyframe
                    keyframe.supplied_image_url = linked_keyframe.supplied_image_url
                    keyframe.width = linked_keyframe.width
                    keyframe.height = linked_keyframe.height
                    keyframe.set = linked_keyframe.set
                    keyframe.characters = linked_keyframe.characters
                    keyframe.image_url = linked_keyframe.image_url
                    keyframe.rendered_frame_by_vid_gen_url = linked_keyframe.rendered_frame_by_vid_gen_url
                    
                    # Restore the original ID and linked_source_id
                    keyframe.id = original_id
                    keyframe.linked_source_id = original_linked_source_id
                    keyframe.source = original_source

                    self.logger.info(f"Keyframe '{keyframe.id}' successfully copied from linked keyframe")
                else:
                    self.logger.warning(f"Keyframe '{keyframe.id}' has invalid linked_source_id - falling back to GENERATE")
                    keyframe.image_url = self._generate_keyframe_image(video, keyframe)
            else:
                self.logger.warning(f"Keyframe '{keyframe.id}' has no linked_source_id - falling back to GENERATE")
                keyframe.image_url = self._generate_keyframe_image(video, keyframe)

        elif keyframe.source == KeyframeSource.NOT_USED:
            # This purely for the second key frame. It can be set to unused to generate the video with only the first frame
            self.logger.warning(f"Keyframe '{keyframe.id}' is marked as NOT_USED - skipping generation")
            keyframe.image_url = None

        return keyframe

    def _generate_keyframe_image(self, video: Video, keyframe: KeyFrame) -> str:
        """
        Generate a complete keyframe image by compositing set variant + character variants.
        Process: 1) Resolve set variant 2) Resolve character variants 3) Create board 4) Final edit
        """
        self.logger.info(f"Generating keyframe image for '{keyframe.id}'")
        
        try:
            # Step 1: Resolve set variant image
            if keyframe.set:
                keyframe.set = self._resolve_set_variant(video, keyframe.set)
                if not keyframe.set.image_url:
                    self.logger.error(f"Failed to resolve set variant for keyframe '{keyframe.id}'")
                    return None
            else:
                self.logger.warning(f"Keyframe '{keyframe.id}' has no set variant - using white background")
            
            # Step 2: Resolve character variant images
            resolved_character_variants = []
            for char_variant in keyframe.characters:
                resolved_variant = self._resolve_character_variant(video, char_variant)
                if resolved_variant and resolved_variant.image_url:
                    resolved_character_variants.append(resolved_variant)
                else:
                    self.logger.warning(f"Failed to resolve character variant '{char_variant.id}' - skipping")
            
            # Update keyframe with resolved variants
            keyframe.characters = resolved_character_variants
            
            # Check if there are no characters - if so, skip board creation and I2I editing
            if not keyframe.characters:
                self.logger.info(f"Keyframe '{keyframe.id}' has no characters - using set variant image directly")
                if keyframe.set and keyframe.set.image_url:
                    self.logger.info(f"Using set variant image as keyframe: {keyframe.set.image_url}")
                    return keyframe.set.image_url
                else:
                    self.logger.error(f"No set variant image available for keyframe '{keyframe.id}'")
                    return None
            
            # Step 3: Create composition board (only when there are characters)
            board_url = self._create_keyframe_board(
                set_variant=keyframe.set,
                character_variants=keyframe.characters,
                width=keyframe.width,
                height=keyframe.height,
                keyframe_id=keyframe.id
            )
            
            if not board_url:
                self.logger.error(f"Failed to create board for keyframe '{keyframe.id}'")
                return None
            
            # Step 4: Final image edit to polish the composition (only when there are characters)
            # get the final prompt
            chars_prompts = []
            char_variant_prompts = []
            chars_actions = []
            chars_placements = []

            for character in keyframe.characters: 
                # Get parent character information for better prompts
                parent_character = video.get_character_by_id(character.parent_id)
                chars_prompts.append(parent_character.prompt)
                char_variant_prompts.append(character.prompt)
                chars_actions.append(character.action_in_frame)
                chars_placements.append(character.placement_hint.prompt_label)

            #final_prompt = VideoEnginePromptGenerator.build_keyframe_composition_prompt(
            #    chars_prompts,
            #    chars_actions,
            #    chars_placements
            #)

            final_prompt = VideoEnginePromptGenerator.build_keyframe_composition_prompt_v2(
                chars_prompts,
                char_variant_prompts,
                chars_actions,
                chars_placements
            )

            #final_prompt = self._build_keyframe_composition_prompt(video, keyframe)
            final_image_bytes = self._edit_image_with_provider(
                input_image_url=board_url,
                prompt=final_prompt,
                target_width=keyframe.width,
                target_height=keyframe.height
            )
            
            if not final_image_bytes:
                self.logger.error(f"Failed to generate final edit for keyframe '{keyframe.id}'")
                return board_url  # Return board as fallback
            
            # Upload final image
            final_url = self._upload_image_asset_to_bucket(final_image_bytes)
            if not final_url:
                self.logger.error(f"Failed to upload final keyframe image")
                return board_url  # Return board as fallback
            
            self.logger.info(f"Generated keyframe image: {final_url}")
            return final_url
            
        except Exception as e:
            self.logger.error(f"Failed to generate keyframe image: {e}")
            return None

    def _resolve_set_variant(self, video: Video, set_variant: SetVariant) -> SetVariant:
        """
        Resolve a set variant image - either use existing or generate from parent set.
        Returns the updated SetVariant object.
        """
        if set_variant.image_url:
            self.logger.info(f"Set variant '{set_variant.id}' already has image")
            return set_variant
            
        # Find parent set
        parent_set = video.get_set_by_id(set_variant.parent_id)
        if not parent_set:
            self.logger.error(f"Set variant '{set_variant.id}' has invalid parent_id '{set_variant.parent_id}'")
            return set_variant
            
        # Check if variant has no prompt and no style - means user wants original
        if (not set_variant.prompt or set_variant.prompt.strip() == "") and (not set_variant.style or set_variant.style.strip() == ""):
            self.logger.info(f"Set variant '{set_variant.id}' has no prompt and no style - copying original image and marking as 'original'")
            set_variant.image_url = parent_set.image_url
            set_variant.prompt = "original"
            return set_variant
            
        # Generate variant image from parent
        self.logger.info(f"Generating set variant image from parent set '{parent_set.id}'")
        set_variant.image_url = self._generate_set_variant_image(parent_set, set_variant)
        return set_variant

    def _resolve_character_variant(self, video: Video, character_variant: CharacterVariant) -> CharacterVariant:
        """
        Resolve a character variant image - either use existing or generate from parent character.
        Returns the updated CharacterVariant object.
        """
        if character_variant.image_url:
            self.logger.info(f"Character variant '{character_variant.id}' already has image")
            return character_variant
            
        # Find parent character
        parent_character = video.get_character_by_id(character_variant.parent_id)
        if not parent_character:
            self.logger.error(f"Character variant '{character_variant.id}' has invalid parent_id '{character_variant.parent_id}'")
            return character_variant
            
        # Check if variant has no prompt and no style - means user wants original
        if (not character_variant.prompt or character_variant.prompt.strip() == "") and (not character_variant.style or character_variant.style.strip() == ""):
            self.logger.info(f"Character variant '{character_variant.id}' has no prompt and no style - copying original image and marking as 'original'")
            character_variant.image_url = parent_character.image_url
            character_variant.prompt = "original"
            return character_variant
            
        # Generate variant image from parent
        self.logger.info(f"Generating character variant image from parent character '{parent_character.name}'")
        character_variant.image_url = self._generate_character_variant_image(parent_character, character_variant)
        return character_variant

    def _calculate_optimal_character_size(self, char_img: Image.Image, target_height: int, frame_width: int, depth: str) -> Tuple[int, int]:
        """
        Calculate optimal character size based on film industry standards.
        Maintains aspect ratio while ensuring proper framing.
        
        Args:
            char_img: Character image to resize
            target_height: Desired height based on depth ratio
            frame_width: Width of the composition frame
            depth: Character depth (fg, mid, bg)
            
        Returns:
            Tuple of (optimal_width, optimal_height)
        """
        char_aspect_ratio = char_img.width / char_img.height
        
        # Calculate width based on target height and aspect ratio
        calculated_width = int(target_height * char_aspect_ratio)
        
        # Define maximum width constraints based on film industry standards
        width_constraints = {
            "fg": 0.9,   # Foreground can take up to 90% of frame width
            "mid": 0.75, # Midground limited to 75% for better composition
            "bg": 0.6    # Background limited to 60% to maintain scene focus
        }
        
        max_width = int(frame_width * width_constraints.get(depth, 0.8))
        
        if calculated_width > max_width:
            # If character is too wide, constrain by width and recalculate height
            optimal_width = max_width
            optimal_height = int(max_width / char_aspect_ratio)
            # Don't exceed the original target height
            optimal_height = min(optimal_height, target_height)
        else:
            optimal_width = calculated_width
            optimal_height = target_height
        
        return optimal_width, optimal_height

    def _apply_film_industry_framing(self, paste_y: int, char_height: int, frame_height: int, depth: str) -> int:
        """
        Apply film industry framing standards for character positioning.
        
        Args:
            paste_y: Original Y position
            char_height: Height of the character
            frame_height: Height of the composition frame
            depth: Character depth (fg, mid, bg)
            
        Returns:
            Adjusted Y position following film industry standards
        """
        # Define framing constraints based on cinematography best practices
        framing_rules = {
            "fg": {"top_margin": 0.08, "bottom_margin": 0.02},  # Foreground: proper headroom
            "mid": {"top_margin": 0.05, "bottom_margin": 0.05}, # Midground: balanced framing
            "bg": {"top_margin": 0.02, "bottom_margin": 0.02}   # Background: minimal constraints
        }
        
        rules = framing_rules.get(depth, framing_rules["mid"])
        min_y = int(frame_height * rules["top_margin"])
        max_y = int(frame_height * (1.0 - rules["bottom_margin"])) - char_height
        
        # Ensure character fits within frame boundaries
        adjusted_y = max(min_y, min(paste_y, max_y))
        
        return adjusted_y

    def _create_keyframe_board(self, set_variant: Optional[SetVariant], character_variants: List[CharacterVariant], width: int, height: int, keyframe_id: str = "unknown") -> str:
        """
        Create a composition board by layering character variants onto the set variant.
        Based on legacy _composite_cast_on_scene logic but adapted for new data structures.
        Saves boards locally for debugging and inspection.
        """
        self.logger.info(f"Creating keyframe board with {len(character_variants)} characters for keyframe {keyframe_id[:8]}")
        
        try:
            from PIL import Image
            import io
            
            # Start with set variant as base (or white background)
            if set_variant and set_variant.image_url:
                base_bytes = self._download_image_bytes(set_variant.image_url)
                if not base_bytes:
                    self.logger.error("Failed to download set variant image")
                    return None
                base = Image.open(io.BytesIO(base_bytes)).convert("RGBA")
            else:
                base = Image.new("RGBA", (width, height), (255, 255, 255, 255))
            
            # Resize base to target dimensions
            if (base.width, base.height) != (width, height):
                base = base.resize((width, height), Image.LANCZOS)
            
            # Create canvas for compositing
            canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            canvas.alpha_composite(base, (0, 0))
            
            if not character_variants:
                # No characters to composite, save and upload base
                buf = io.BytesIO()
                canvas.convert("RGB").save(buf, format="PNG")
                board_bytes = buf.getvalue()
                
                # Save locally for debugging
                self._save_board_locally(board_bytes, keyframe_id, "no_characters")

                return self._upload_image_asset_to_bucket(board_bytes)

            # Depth-based sizing with film industry standards
            # Based on cinematography best practices for character composition
            depth_ratios = {
                "fg": {"base": 0.75, "min": 0.60, "max": 0.90},   # Foreground: dominant presence
                "mid": {"base": 0.55, "min": 0.45, "max": 0.65},  # Midground: supporting role  
                "bg": {"base": 0.35, "min": 0.25, "max": 0.45}    # Background: environmental context
            }
            
            # Group by depth for proper layering
            grouped = {"bg": [], "mid": [], "fg": []}
            for char_variant in character_variants:
                if not char_variant.image_url:
                    self.logger.warning(f"Character variant '{char_variant.id}' has no image_url - skipping")
                    continue
                    
                placement = char_variant.placement_hint
                if isinstance(placement, str):
                    # Convert string to Placement enum if needed
                    try:
                        placement = Placement[placement]
                    except:
                        placement = Placement.CENTER_FOREGROUND
                grouped[placement.depth].append((char_variant, placement))
            
            # Composite in depth order: bg -> mid -> fg
            for depth_key in ["bg", "mid", "fg"]:
                chars_in_depth = grouped[depth_key]
                if not chars_in_depth:
                    continue
                    
                # Calculate optimal character size based on film industry standards
                depth_config = depth_ratios[depth_key]
                base_ratio = depth_config["base"]
                min_ratio = depth_config["min"] 
                max_ratio = depth_config["max"]
                
                # Adjust for character crowding (avoid overlapping)
                count = len(chars_in_depth)
                if count > 1:
                    # Reduce size when multiple characters share the same depth
                    crowd_factor = 1.0 / (1.0 + 0.12 * (count - 1))
                    char_ratio = max(min_ratio, base_ratio * crowd_factor)
                else:
                    char_ratio = base_ratio
                
                # Ensure we don't exceed the maximum ratio for this depth
                char_ratio = min(char_ratio, max_ratio)
                
                self.logger.info(f"Depth '{depth_key}': {count} character(s), ratio: {char_ratio:.2f}")
                
                for char_variant, placement in chars_in_depth:
                    # Download character image
                    char_bytes = self._download_image_bytes(char_variant.image_url)
                    if not char_bytes:
                        self.logger.warning(f"Failed to download character variant image: {char_variant.image_url}")
                        continue
                        
                    char_img = Image.open(io.BytesIO(char_bytes)).convert("RGBA")
                    
                    # Calculate size maintaining aspect ratio (film industry standard)
                    char_aspect_ratio = char_img.width / char_img.height
                    desired_height = int(char_ratio * height)
                    
                    # Ensure character fits within frame width while maintaining aspect ratio
                    max_width = int(width * 0.8)  # Leave 20% margin on sides
                    calculated_width = int(desired_height * char_aspect_ratio)
                    
                    if calculated_width > max_width:
                        # If too wide, constrain by width and adjust height accordingly
                        new_width = max_width
                        new_height = int(max_width / char_aspect_ratio)
                        # Don't let height exceed the original ratio-based calculation
                        new_height = min(new_height, desired_height)
                    else:
                        new_width = calculated_width
                        new_height = desired_height
                    
                    # Resize character maintaining aspect ratio
                    char_img = char_img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Position using placement anchor with proper framing
                    anchor_x, anchor_y = placement.anchor
                    center_x = int(anchor_x * width)
                    center_y = int(anchor_y * height)
                    
                    # Apply film industry framing standards (ensure proper headroom)
                    if depth_key == "fg":
                        # Foreground characters: ensure proper headroom (10% of frame)
                        min_y_position = int(0.1 * height)
                        max_y_position = height - int(0.05 * height)  # Small bottom margin
                    elif depth_key == "mid":
                        # Midground: more flexible positioning
                        min_y_position = int(0.05 * height)
                        max_y_position = height - int(0.05 * height)
                    else:  # bg
                        # Background: can use full frame
                        min_y_position = 0
                        max_y_position = height
                    
                    # Calculate paste position with framing constraints
                    paste_x = int(center_x - char_img.width / 2)
                    paste_y = int(center_y - char_img.height / 2)
                    
                    # Ensure character doesn't extend beyond frame
                    paste_x = max(0, min(paste_x, width - char_img.width))
                    paste_y = max(min_y_position, min(paste_y, max_y_position - char_img.height))
                    
                    self.logger.info(f"Character {char_variant.id[:8]} ({depth_key}): {new_width}x{new_height} at ({paste_x}, {paste_y})")
                    
                    # Paste with alpha blending
                    canvas.alpha_composite(char_img, (paste_x, paste_y))
            
            # Create final board bytes
            buf = io.BytesIO()
            canvas.convert("RGB").save(buf, format="PNG")
            board_bytes = buf.getvalue()
            
            # Save locally for debugging
            shot_info = f"{len(character_variants)}chars"
            self._save_board_locally(board_bytes, keyframe_id, shot_info)
            
            # Upload board
            board_url = self._upload_image_asset_to_bucket(board_bytes)

            if not board_url:
                self.logger.error("Failed to upload keyframe board")
                return None
                
            self.logger.info(f"Created keyframe board: {board_url}")
            return board_url
            
        except Exception as e:
            self.logger.error(f"Failed to create keyframe board: {e}")
            return None

    def _download_image_bytes(self, url: str) -> bytes:
        """Download an image from a public URL and return raw bytes."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            self.logger.error(f"Failed to download image from {url}: {e}")
            return None

    def _remove_background(self, image_bytes: bytes) -> bytes:
        """
        Remove background from image bytes using rembg AI model.
        Returns the processed image bytes with transparent background.
        Falls back to original image if background removal fails.
        """
        if not REMBG_AVAILABLE:
            self.logger.warning("rembg not available - skipping background removal")
            return image_bytes
            
        try:
            self.logger.info("Removing background from image using AI model...")
            
            # Process with rembg
            processed_bytes = remove(image_bytes)
            
            # Validate the result
            if processed_bytes and len(processed_bytes) > 0:
                self.logger.info("✅ Background removal successful")
                return processed_bytes
            else:
                self.logger.warning("Background removal returned empty result - using original")
                return image_bytes
                
        except Exception as e:
            self.logger.error(f"Background removal failed: {e} - using original image")
            return image_bytes
        
    def _extract_last_frame_from_video(self, video_bytes: bytes) -> str:
        """Extract the last frame from the video bytes and return its URL."""
        import tempfile
        import subprocess
        from PIL import Image
        import io
        
        if not video_bytes:
            self.logger.error("[extract_last_frame] video_bytes is empty")
            return ""
            
        try:
            # Create temporary files for input video and output frame
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video.write(video_bytes)
                temp_video_path = temp_video.name
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_frame:
                temp_frame_path = temp_frame.name
            
            try:
                # Use FFmpeg to extract the last frame
                # First get video duration, then seek to near the end
                cmd_info = [
                    'ffprobe', '-v', 'quiet',
                    '-show_entries', 'format=duration',
                    '-of', 'csv=p=0',
                    temp_video_path
                ]
                
                duration_result = subprocess.run(
                    cmd_info, 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                
                if duration_result.returncode == 0 and duration_result.stdout.strip():
                    try:
                        duration = float(duration_result.stdout.strip())
                        # Seek to 0.1 seconds before the end, or to 90% of duration if very short
                        seek_time = max(0, duration - 0.1) if duration > 0.2 else duration * 0.9
                    except ValueError:
                        seek_time = 0  # Fallback to beginning if duration parsing fails
                else:
                    seek_time = 0  # Fallback if duration detection fails
                
                # Extract frame at the calculated time
                cmd = [
                    'ffmpeg', '-y',  # -y to overwrite output file
                    '-ss', str(seek_time),  # Seek to near end
                    '-i', temp_video_path,
                    '-vframes', '1',  # Extract only 1 frame
                    '-q:v', '2',     # High quality
                    temp_frame_path
                ]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                
                if result.returncode != 0:
                    self.logger.error(f"[extract_last_frame] FFmpeg failed: {result.stderr}")
                    return ""
                
                # Read the extracted frame and upload it
                with open(temp_frame_path, 'rb') as frame_file:
                    frame_bytes = frame_file.read()
                
                if not frame_bytes:
                    self.logger.error("[extract_last_frame] extracted frame is empty")
                    return ""
                
                # Upload the frame and return URL
                frame_url = self._upload_image_asset_to_bucket(frame_bytes)
                if frame_url:
                    self.logger.info(f"[extract_last_frame] ✅ Successfully extracted and uploaded last frame")
                    return frame_url
                else:
                    self.logger.error("[extract_last_frame] failed to upload frame")
                    return ""
                    
            finally:
                # Clean up temporary files
                import os
                try:
                    os.unlink(temp_video_path)
                    os.unlink(temp_frame_path)
                except OSError:
                    pass  # Ignore cleanup errors
                    
        except subprocess.TimeoutExpired:
            self.logger.error("[extract_last_frame] FFmpeg timeout")
            return ""
        except Exception as e:
            self.logger.error(f"[extract_last_frame] unexpected error: {e}")
            return ""

    def _save_board_locally(self, board_bytes: bytes, keyframe_id: str, shot_info: str = "") -> str:
        """
        Save a board composition locally for debugging and inspection.
        Returns the local file path where the board was saved.
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = "output/boards"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_shot_info = "".join(c for c in shot_info if c.isalnum() or c in ('-', '_')).strip()
            filename = f"board_{timestamp}_{keyframe_id[:8]}_{safe_shot_info}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save the board
            with open(filepath, 'wb') as f:
                f.write(board_bytes)
            
            self.logger.info(f"📁 Board saved locally: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save board locally: {e}")
            return None

    def _find_keyframe_by_id(self, video: Video, keyframe_id: str) -> Optional[KeyFrame]:
        """
        Find a keyframe by ID across all shots in the video.
        """
        for shot in video.shots:
            if shot.first_keyframe and shot.first_keyframe.id == keyframe_id:
                return shot.first_keyframe
            if shot.last_keyframe and shot.last_keyframe.id == keyframe_id:
                return shot.last_keyframe
        return None

    def _render_video(self, shot: VideoBlock, video: Video) -> Optional[VideoBlock]:
        """
        Placeholder for video rendering logic.
        This would handle calling ffmpeg or other video generation tools.
        Returns the URL of the rendered video file.
        """
        video_bytes = None
        frame_count = shot.frame_count()

        self.logger.info(f"Rendering video for shot '{shot.id}'")
        if shot.render_engine == RenderEngine.STILL:
            # Create a still video from the first keyframe image for the shot duration
            if not shot.first_keyframe or not shot.first_keyframe.image_url:
                raise ValueError("STILL engine requires first_keyframe with image_url")

            duration_sec = shot.duration_seconds
            try:
                if self.video_tools and hasattr(self.video_tools, "image_to_still_video"):
                    video_bytes = self.video_tools.image_to_still_video(
                        image=shot.first_keyframe.image_url,
                        duration_seconds=duration_sec,
                        width=shot.width,
                        height=shot.height,
                        fps=shot.fps,
                    )
                else:
                    raise RuntimeError("VideoTools.image_to_still_video not available")
            except Exception as e:
                self.logger.error(f"Failed to create still video: {e}")
                video_bytes = None
        elif (shot.render_engine == RenderEngine.GENERATIVE_MOTION):
            if not shot.first_keyframe:
                raise ValueError("GENERATIVE_MOTION engine requires first_keyframe")

            if (shot.last_keyframe.source == KeyframeSource.NOT_USED):
                # Video is to be generated only with the first keyframe, use i2v
                video_bytes = self.modal_manager.generate_video_from_image(
                        image_url=shot.first_keyframe.image_url,
                        prompt=shot.video_prompt,
                        width=shot.width,
                        height=shot.height,
                        frames=frame_count,
                        fps=shot.fps
                )
            else:
                # Video is to be generated with both keyframes use fflf2v
                if not shot.last_keyframe:
                    raise ValueError("GENERATIVE_MOTION engine requires last_keyframe (or set it explicitly to NOT_USED)")
                
                start_frame = None
                if (shot.first_keyframe.source == KeyframeSource.COPY_PREV_LAST):
                    # if the first frames source is COPY_PREV_LAST, then use the use the rendered_frame_by_vid_gen_url of the linked frame
                    linked_frame = self._find_keyframe_by_id(video, shot.first_keyframe.linked_source_id)
                    if linked_frame:
                        start_frame = linked_frame.rendered_frame_by_vid_gen_url

                # if missing use the shot.first_keyframe.image_url which is populated as backup when we resolved frames
                if not start_frame:
                    start_frame = shot.first_keyframe.image_url

                video_bytes = self.modal_manager.generate_video_from_first_last_images(
                    first_frame_url=start_frame,
                    last_frame_url=shot.last_keyframe.image_url,
                    prompt=shot.video_prompt,
                    width=shot.width,
                    height=shot.height,
                    frames=frame_count,
                    fps=shot.fps
                )
        elif (shot.render_engine == RenderEngine.LIPSYNC_MOTION):
            # Audio-driven lip-sync using InfiniteTalk
            if not shot.first_keyframe or not shot.first_keyframe.image_url:
                raise ValueError("LIPSYNC_MOTION requires first_keyframe with image_url")

            # Collect up to 2 audio files: prefer character variants, fallback to narration
            audio_files = []
            try:
                for cv in (shot.first_keyframe.characters or []):
                    if getattr(cv, 'audio_padded_url', None):
                        audio_files.append(cv.audio_padded_url)
                        if len(audio_files) >= 2:
                            break
                if not audio_files and shot.narration and shot.narration.audio_url:
                    audio_files.append(shot.narration.audio_url)
            except Exception:
                pass

            if not audio_files:
                raise ValueError("LIPSYNC_MOTION requires at least one audio_url (character variant or narration)")

            video_bytes = self.modal_manager.generate_infinite_talk_video(
                image_url=shot.first_keyframe.image_url,
                audio_files=audio_files,
                prompt=shot.video_prompt,
                width=shot.width,
                height=shot.height,
                frames=shot.frame_count(),  # upper limit; audio dictates final length
                fps=shot.fps
            )

        # Check if video is generated okay, then extract its last frame from the video file and upload
        # Set the url to shot.last_keyframe.rendered_frame_by_vid_gen_url
        if video_bytes:
            # Upload the generated video
            video_url = self._upload_video_asset_to_bucket(video_bytes)
            shot.generated_video_clip = video_url
            shot.generated_video_clip_raw = video_url
            self.logger.info(f"Successfully generated and uploaded video: {video_url}")

            # Extract the last frame of the video and replace the shot's last_keyframe.rendered_frame_by_vid_gen_url
            try:
                last_frame_url = self._extract_last_frame_from_video(video_bytes)
                if shot.last_keyframe:
                    shot.last_keyframe.rendered_frame_by_vid_gen_url = last_frame_url
            except Exception as e:
                self.logger.warning(f"Failed to extract last frame: {e}")
        else:
            self.logger.error("⚠️ Warning: Video generation returned no data")
            raise ValueError("Failed to generate video from single image source")
        
        return shot
    
    def _apply_overlays(self, shot: VideoBlock) -> Optional[VideoBlock]:
        """Apply text overlays (static/duration/progressive)."""
        # Process text overlays if any are specified
        base_video_url = shot.generated_video_clip_raw or shot.generated_video_clip
        if shot.overlays and base_video_url:
            self.logger.info(f"🎨 Processing {len(shot.overlays)} text overlays for video block")

            try:
                # Use the text overlay manager to add overlays to the video
                if self.text_overlay_manager:
                    # Apply overlays to the generated video
                    video_with_overlays_bytes = self.text_overlay_manager.add_text_overlays_to_video(
                        video_url=base_video_url,
                        overlays=shot.overlays
                    )
                    
                    if video_with_overlays_bytes:
                        # Upload the video with overlays
                        overlayed_video_url = self._upload_video_asset_to_bucket(video_with_overlays_bytes)
                        shot.generated_video_clip_with_overlays = overlayed_video_url
                        shot.generated_video_clip = overlayed_video_url  # maintain backward compatibility
                        self.logger.info(f"✅ Successfully added text overlays and uploaded: {overlayed_video_url}")
                    else:
                        self.logger.warning("⚠️ Warning: Failed to generate video with overlays, keeping original video")
                else:
                    self.logger.warning("⚠️ Warning: VideoTextOverlayManager not available, skipping overlays")

            except Exception as e:
                self.logger.warning(f"⚠️ Warning: Failed to process text overlays: {e}")
                self.logger.info("Continuing with original video without overlays")

        elif shot.overlays and not base_video_url:
            self.logger.warning("⚠️ Warning: Text overlays specified but no video generated to apply them to")
            pass

        # If no overlays applied or none specified, carry forward raw video
        if not shot.generated_video_clip_with_overlays:
            if base_video_url:
                shot.generated_video_clip_with_overlays = base_video_url
            else:
                shot.generated_video_clip_with_overlays = shot.generated_video_clip

        return shot

    def _create_narration_audio(self, shot: VideoBlock) -> Optional[VideoBlock]:
        if (shot.narration and not shot.narration.audio_url) and shot.narration.script:
            # Generate the audio using chatterbox
            self.logger.info(f"🎤 Generating audio narration for script: {shot.narration.script[:50]}...")

            # TODO: Implement audio generation config class to define the audio generation parameters including sample, exaggeration, etc.
            # Use voice sample from narration if provided, otherwise use default parameters
            try:
                # Get voice sample URL from narration, fallback to None for default voice
                voice_sample_url = getattr(shot.narration, 'voice_sample_url', None)
                if voice_sample_url and voice_sample_url.strip():
                    self.logger.info(f"🎵 Using voice sample: {voice_sample_url}")
                else:
                    voice_sample_url = None
                    self.logger.info("🎵 Using default TTS voice")
                
                # Generate speech bytes using ModalManager's voice clone
                audio_bytes = self.modal_manager.generate_voice_clone(
                    audio_url=voice_sample_url if voice_sample_url else "",
                    text=shot.narration.script,
                    exaggeration=0.5,
                    cfg_weight=0.6
                )
                if audio_bytes:
                    audio_url = self._upload_audio_asset_to_bucket(audio_bytes)
                    shot.narration.audio_url = audio_url
                    self.logger.info(f"✅ Successfully generated and uploaded audio: {audio_url}")
                else:
                    self.logger.warning("⚠️ Warning: Failed to generate audio narration")
            except Exception as e:
                self.logger.warning(f"⚠️ Warning: Audio generation failed: {e}")
                # Continue processing without audio
        return shot
    
    def _create_bg_audio_effects(self, shot: VideoBlock) -> Optional[VideoBlock]:
        """Generate audio effects when present."""
        #TODO: Finalise this
        # Here we need to use modal mmaudio to pass the generated video, its best to do it before narration mixing
        # Then mmaudio will return a new audio track with effects
        # Sometimes it creates vocals too, se we need to remove the vocals from the generated audio
        
        if shot.bg_audio_effects and shot.bg_audio_effects.is_enabled:
            self.logger.info("🎬 Generating background audio effects for final assembly...")

            try:

                # Generate audio effects using mmaudio
                effects_audio_bytes = self.modal_manager.generate_audio_effects(
                    video_url=shot.generated_video_clip,
                    prompt=shot.bg_audio_effects.prompt,
                    duration=shot.duration_seconds
                )
                
                if effects_audio_bytes:
                    # Upload the generated audio effects
                    effects_audio_url = self._upload_audio_asset_to_bucket(effects_audio_bytes)

                    # Remove vocals while preserving all other stems (drums, bass, other)
                    try:
                        instrumental_path = AudioTools.separate_audio_keep(
                            effects_audio_url, ["instrumental"]
                        )

                        # Upload the filtered (no vocals) background SFX
                        with open(instrumental_path, 'rb') as f:
                            filtered_audio_bytes = f.read()
                        filtered_bg_effect_url = self._upload_audio_asset_to_bucket(filtered_audio_bytes)
                        shot.bg_audio_effects.audio_url = filtered_bg_effect_url
                        self.logger.info(f"✅ Generated background SFX (no vocals): {filtered_bg_effect_url}")
                    except Exception as sep_err:
                        # Fallback: use original effects track if separation fails
                        self.logger.warning(f"⚠️ Vocal removal failed; using original effects: {sep_err}")
                        shot.bg_audio_effects.audio_url = effects_audio_url
                else:
                    self.logger.warning("⚠️ Warning: Failed to generate audio effects")
            except Exception as e:
                self.logger.warning(f"⚠️ Warning: Audio effects generation failed: {e}")
                # Continue processing without audio effects
        return shot
    

    def _create_characters_speech_audio(self, video: Video, shot: VideoBlock) -> Optional[VideoBlock]:
        """Generate character speech audio per variant, then produce padded versions for multi-talk sync.

        - Uses parent character voice_sample_url (if available) + variant.script to synthesize speech.
        - Uploads raw speech to variant.audio_url.
        - Runs sync_multi_talk_audios on up to 2 characters (in order) and uploads padded files to variant.audio_padded_url.
        """
        try:
            # Only relevant for LipSync engine and when characters are present
            if shot.render_engine != RenderEngine.LIPSYNC_MOTION:
                return shot
            if not shot.first_keyframe or not shot.first_keyframe.characters:
                return shot

            variants = shot.first_keyframe.characters

            # Step 1: Generate raw speech for each variant that has a script
            for cv in variants:
                script = getattr(cv, 'script', None)
                if not script or not script.strip():
                    continue
                if getattr(cv, 'audio_url', None):
                    continue  # already has audio

                # Find parent character voice sample
                parent = video.get_character_by_id(cv.parent_id) if hasattr(video, 'get_character_by_id') else None
                voice_sample_url = getattr(parent, 'voice_sample_url', None) if parent else None

                try:
                    audio_bytes = self.modal_manager.generate_voice_clone(
                        audio_url=voice_sample_url or "",
                        text=script,
                        exaggeration=0.5,
                        cfg_weight=0.6,
                    )
                except Exception as e:
                    self.logger.warning(f"Voice clone failed for variant {cv.id}: {e}")
                    audio_bytes = None

                if audio_bytes:
                    url = self._upload_audio_asset_to_bucket(audio_bytes)
                    cv.audio_url = url
                    self.logger.info(f"Generated speech for variant {cv.id}: {url}")

            # Step 2: Prepare ordered list (max 2) for sync; prefer padded later
            ordered_with_audio = []
            for cv in variants:
                url = getattr(cv, 'audio_url', None)
                if url:
                    ordered_with_audio.append((cv, url))
                if len(ordered_with_audio) >= 2:
                    break

            if not ordered_with_audio:
                return shot

            if len(ordered_with_audio) == 1:
                # Single speaker: just set padded URL equal to raw audio for convenience
                cv, url = ordered_with_audio[0]
                cv.audio_padded_url = url
                return shot

            # Step 3: Multi-speaker (max 2) sync with left/right padding
            try:
                audio_urls = [url for (_, url) in ordered_with_audio]
                synced_paths = AudioTools.sync_multi_talk_audios(audio_urls)
                if len(synced_paths) != len(ordered_with_audio):
                    self.logger.warning("sync_multi_talk_audios returned unexpected count; skipping padding upload")
                    return shot

                for (cv, _), local_path in zip(ordered_with_audio, synced_paths):
                    try:
                        with open(local_path, 'rb') as f:
                            padded_bytes = f.read()
                        padded_url = self._upload_audio_asset_to_bucket(padded_bytes)
                        cv.audio_padded_url = padded_url
                        self.logger.info(f"Uploaded padded speech for variant {cv.id}: {padded_url}")
                    except Exception as e:
                        self.logger.warning(f"Failed to upload padded audio for variant {cv.id}: {e}")
            except Exception as e:
                self.logger.warning(f"Failed syncing multi-talk audios: {e}")
                # Fall back to unpadded
                for cv, url in ordered_with_audio:
                    cv.audio_padded_url = url

            return shot

        except Exception as e:
            self.logger.warning(f"Characters speech generation failed: {e}")
            return shot


    def _mix_audio_layers_into_video(self, shot: VideoBlock) -> Optional[VideoBlock]:
        """Blend audio layers (characters, narration, SFX) and add them to the video.

        Produces and uploads the final per-shot video with overlays and audio, setting
        `generated_video_clip_with_audio_and_overlays` and `generated_video_clip_final`.
        """
        try:
            # Determine base video to mix onto: overlays result preferred, else raw
            base_video = shot.generated_video_clip_with_overlays or shot.generated_video_clip_raw or shot.generated_video_clip
            if not base_video:
                return shot

            audio_files: list[str] = []
            audio_vols: list[float] = []

            # Character padded audios (in order, up to 2)
            try:
                if shot.first_keyframe and shot.first_keyframe.characters:
                    for cv in shot.first_keyframe.characters:
                        url = getattr(cv, 'audio_padded_url', None)
                        if url:
                            audio_files.append(url)
                            audio_vols.append(1.0)
                            if len(audio_files) >= 2:
                                break
            except Exception:
                pass

            # Narration (if present)
            if shot.narration and shot.narration.audio_url:
                audio_files.append(shot.narration.audio_url)
                audio_vols.append(1.0)

            # Background audio effects (SFX), quieter
            if shot.bg_audio_effects and shot.bg_audio_effects.audio_url:
                audio_files.append(shot.bg_audio_effects.audio_url)
                audio_vols.append(0.3)

            if not audio_files:
                # No audio to add; carry forward overlays video
                shot.generated_video_clip_with_audio_and_overlays = base_video
                shot.generated_video_clip_final = base_video
                return shot

            # Mix all audio layers onto the video
            mixed_path = self.audio_video_mixer.mix_audio_to_video(
                video_path=base_video,
                audio_files=audio_files,
                audio_volumes=audio_vols,
            )

            # Upload mixed video bytes
            with open(mixed_path, 'rb') as f:
                mixed_bytes = f.read()
            final_video_url = self._upload_video_asset_to_bucket(mixed_bytes)

            shot.generated_video_clip_with_audio_and_overlays = final_video_url
            shot.generated_video_clip_final = final_video_url
            self.logger.info(f"✅ Mixed audio (chars/narration/SFX) and uploaded: {final_video_url}")
        except Exception as e:
            self.logger.warning(f"⚠️ Audio blending failed: {e}")
            # Fallback: retain overlays-only video
            shot.generated_video_clip_with_audio_and_overlays = (
                shot.generated_video_clip_with_overlays or shot.generated_video_clip_raw or shot.generated_video_clip
            )
            shot.generated_video_clip_final = shot.generated_video_clip_with_audio_and_overlays
        return shot

    def _resolve_background_music(self, bg_music: Music, forced_duration: int) -> Music:
        """Resolve background music asset for a shot."""
        if bg_music:
            self.logger.info(f"🎵 Resolving background music: {bg_music.title}")
            prompt = bg_music.prompt if bg_music.prompt else "sonata, piano, Violin, B Flat Major, allegro"
            lyrics = bg_music.lyrics if bg_music.lyrics else "[inst]"
            # Pass duration as-is, -1 means follow lyrics to the end

            if forced_duration > 0:
                duration = forced_duration
            else:
                duration = bg_music.duration if bg_music.duration else 60

            # Generate or validate music asset
            music_bytes = self.modal_manager.generate_music_with_lyrics(
                prompt=prompt,
                lyrics=lyrics,
                duration=duration
            )

            if music_bytes:
                # Upload original music for debugging
                original_music_url = self._upload_audio_asset_to_bucket(music_bytes)
                print(f"🎵 Original generated music (debug): {original_music_url}")
                
                # Polish the music using professional mastering
                self.logger.info("🎭 Polishing generated music with professional mastering...")
                from utils.music_tools import polish_audio_bytes
                
                polished_music_bytes = polish_audio_bytes(
                    audio_bytes=music_bytes,
                    target_lufs=-14.0  # Streaming standard
                )
                
                if polished_music_bytes:
                    bg_music.audio_url = self._upload_audio_asset_to_bucket(polished_music_bytes)
                    self.logger.info(f"✅ Successfully polished and uploaded background music: {bg_music.audio_url}")
                else:
                    # Fallback to original if polishing fails
                    self.logger.warning("⚠️ Warning: Music polishing failed, using original")
                    bg_music.audio_url = original_music_url
            else:
                self.logger.warning("⚠️ Warning: Failed to generate background music")
        return bg_music

    def _validate_transition_sequence(self, shots: List[VideoBlock]) -> List[str]:
        """
        Validate and prepare transition sequence for a list of shots.
        Returns a list of transition types to be applied between consecutive shots.
        """
        if len(shots) <= 1:
            return []
        
        transitions = []
        for i in range(len(shots) - 1):
            shot = shots[i]
            # Use the transition from the current shot (applied after this shot)
            transition_type = shot.transition.value if shot.transition else "Cut"
            transitions.append(transition_type)
            
            print(f"🔀 Shot {i+1} → Shot {i+2}: {transition_type}")
        
        return transitions


    def process_video(self, video: Video) -> Video:
        """
        Main public entry point for processing a complete video.
        Handles the full pipeline: resolve assets -> process all keyframes for all shots.
        """
        self.logger.info(f"Starting video processing for '{video.title}'")
        
        try:
            # Step 1: Resolve all video-level assets (cast and sets)
            self.logger.info("Step 1: Resolving video-level assets")
            self._resolve_video_assets(video)
            
            # Step 2: Process all shots and their keyframes
            self.logger.info(f"Step 2: Processing {len(video.shots)} shots")
            for i, shot in enumerate(video.shots):
                self.logger.info(f"Processing shot {i+1}/{len(video.shots)} (ID: {shot.id})")
                
                # Process first keyframe
                if shot.first_keyframe:
                    self.logger.info(f"Processing first keyframe for shot {i+1}")
                    shot.first_keyframe = self._resolve_keyframe(video, shot.first_keyframe)
                else:
                    self.logger.warning(f"Shot {i+1} has no first keyframe")
                
                # Process last keyframe  
                if shot.last_keyframe:
                    self.logger.info(f"Processing last keyframe for shot {i+1}")
                    shot.last_keyframe = self._resolve_keyframe(video, shot.last_keyframe)
                else:
                    self.logger.warning(f"Shot {i+1} has no last keyframe")

            # Step 3: Process the shots for video generation
            for shot in video.shots:
                # For lip-sync motion, generate character speech and padded audios first because they are needed for video generation
                if shot.render_engine == RenderEngine.LIPSYNC_MOTION:
                    shot = self._create_characters_speech_audio(video, shot)

                # Generate the video for the shot based on the render engine and keyframes
                shot = self._render_video(shot, video)

                # Create background audio effect if needed, it needs to happen after rendering the video and before overlays are added
                shot = self._create_bg_audio_effects(shot)                
                # Create narration audio if needed
                shot = self._create_narration_audio(shot)

                # Apply text overlays if any
                shot = self._apply_overlays(shot)

                # Finally blend all audio layers for this shot (characters, narration, SFX)
                shot = self._mix_audio_layers_into_video(shot)

                video.shots[video.shots.index(shot)] = shot


            # Collect video URLs and transition types from processed shots
            video_urls = []

            for shot in video.shots:
                if shot.generated_video_clip_final:
                    video_urls.append(shot.generated_video_clip_final)
                    print(f"✅ Shot {shot.id[:8]}... has video: {shot.generated_video_clip_final}")
                else:
                    print(f"⚠️ Warning: Shot {shot.id[:8]}... has no generated video")
            
            # Validate and prepare transitions for stitching
            transition_types = self._validate_transition_sequence(video.shots)

            # Stitch videos together if we have multiple videos
            if len(video_urls) > 1:
                print(f"🔗 Stitching {len(video_urls)} videos with {len(transition_types)} transitions for video {video.id}")

                try:
                    # Use VideoStitcher with transition support
                    stitched_video_bytes = self.video_stitcher.stitch_with_transition(
                        video_urls=video_urls,
                        transition_types=transition_types,
                        return_bytes=True  # Return bytes directly for upload
                    )
                    
                    # Upload the stitched video
                    video_video_url = self._upload_video_asset_to_bucket(stitched_video_bytes)
                    video.add_generated_video(video_video_url)

                    print(f"✅ Video stitched with transitions and uploaded: {video_video_url}")

                except Exception as e:
                    raise Exception(f"Video stitching failed for video {video.id}: {e}. Check video URLs and transition types for compatibility issues.")
            elif len(video_urls) == 1:
                # Only one video, use it directly as the video
                video.add_generated_video(video_urls[0])
                print(f"📹 Single video scene: Using video directly")

            # Apply background music if available
            if video.background_music:
                video.background_music = self._resolve_background_music(bg_music=video.background_music, forced_duration=video.duration_seconds)

                # Now combine the music with the final video using available mixer
                if video.generated_video_url and video.background_music.audio_url:
                    self.logger.info("🎵 Combining final video with background music...")
                    
                    try:
                        mixed_path = self.audio_video_mixer.mix_audio_to_video(
                            video_path=video.generated_video_url,
                            audio_files=[video.background_music.audio_url],
                            audio_volumes=[0.1]
                        )
                        with open(mixed_path, 'rb') as f:
                            mixed_bytes = f.read()
                        final_video_with_music_url = self._upload_video_asset_to_bucket(mixed_bytes)
                        video.generated_video_url = final_video_with_music_url
                        self.logger.info(f"✅ Successfully mixed background music: {final_video_with_music_url}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Warning: Failed to combine video with background music: {e}")
                        self.logger.info("Continuing with video without background music")
                
                print(f"🎶 Background music processed for video: {video.background_music.title}")

            else:
                print(f"⚠️ Warning: No videos generated for scene {video.id}")
            return video
            
        except Exception as e:
            self.logger.error(f"Failed to process video '{video.title}': {e}")
            raise
    
