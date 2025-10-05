import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.video_engine_models import BackgroundAudioEffects, KeyFrame, KeyframeSource, Narration, RenderEngine, Set, SetVariant, TextOverlay, TextOverlayPosition, TextOverlayProperties, TextOverlayTiming, TextPresentationAnimationType, TextPresentationType, Transition, Video, VideoBlock
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

    def test_single_character_supplied_image_audio(self) -> bool:# This test is to create a talking character video without using the characters feature

        # Create video for scenario 1
        video = Video(
            id="single_character_supplied_image_audio_test_001",
            title="Single Character Supplied Image and Audio Test",
            description="Testing single character with supplied image and audio",
            style="cinematic",
            cast=[],  # No characters for this scenario
            sets=[],
            shots=[VideoBlock(
                id="single_character_supplied_image_audio_test_001_shot_001",
                storyline_label="Single Character Test Shot",
                render_engine=RenderEngine.LIPSYNC_MOTION,  # Use LIPSYNC_MOTION with I2V
                duration_seconds=5,# Duration is ignored for LIPSYNC_MOTION
                fps=25,
                width=720,
                height=1080,
                first_keyframe=KeyFrame(
                    source=KeyframeSource.SUPPLIED,
                    linked_source_id=None,
                    width=720,
                    height=1080,
                    set=None,
                    characters=None,
                    basic_generation_prompt=None,
                    supplied_image_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/Temp%20Dump/matt-silver--jqTm3EaGus-unsplash.jpg",
                    #"https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/Temp%20Dump/matt-silver--jqTm3EaGus-unsplash.jpg",
                    rendered_frame_by_vid_gen_url=None
                ),
                last_keyframe=KeyFrame( # OR NULL to indicate not used
                    id="last_keyframe_002",
                    source=KeyframeSource.NOT_USED,  # This tells the engine to use I2V instead of FLF2V
                    linked_source_id=None,
                    width=720,
                    height=1080,
                    set=None,
                    characters=None,
                    basic_generation_prompt=None,
                    supplied_image_url=None,
                    rendered_frame_by_vid_gen_url=None
                ),
                video_prompt=("A man talking passionately"),
                style="cinematic",
                narrations=[
                    Narration(
                        id="narration_001",
                        exaggeration=0.5,
                        cfg_weight=0.5,
                        script="Listen to this — this woman’s mom says she deserves to be her baby’s “first mom”… because she raised her.",
                        voice_sample_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/Temp%20Dump/JoseAudioSample.wav",
                        audio_url=None,
                        audio_padded_url=None
                        #"https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/Temp%20Dump/test.wav"
                        #"https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/audio_samples/please%20please%20pleasew%20short.wav"
                    )
                ],
                overlays=None,
                bg_audio_effects=None,
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
                    basic_generation_prompt="A beautiful elegant woman with flowing ginger hair, wearing a vintage "
                                "burgundy dress, sitting gracefully in an ornate Victorian parlor, holding a glass of white wine,"
                                "with rich wooden furniture and warm golden lighting, portrait style, "
                                "high quality, cinematic lighting",
                    image_url=None,
                    rendered_frame_by_vid_gen_url=None
                ),
                last_keyframe=None,  # Not needed because render engine is LIPSYNC_MOTION
                video_prompt=("A sophisticated ginger woman, holding a glass of white wine, speaking elegantly about wine, with natural "
                            "facial expressions and subtle head movements, in a Victorian parlor setting"),
                style="cinematic, elegant",
                narrations=[
                    Narration(
                        id="wine_tasting_narration",
                        exaggeration=0.5,
                        cfg_weight=0.5,
                        script=(
                            "The wine is quite delightful, with notes of cherry and oak that dance "
                            "upon the palate. This vintage has been aged to perfection, offering a "
                            "rich and complex flavor profile that would complement any elegant evening."
                        ),
                        voice_sample_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/audio_samples/219777__madamvicious__the-wine-is-quite-delightful-posh-woman.wav",
                        audio_url=None,
                    )
                ],
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
    
    def test_simple_two_characters_conversation_with_bg_effects(self) -> bool:
        # This test is to create two talking character video without using the characters feature
        # Create video for scenario 1
        video = Video(
            id="scenario_1_balloon_adventure",
            title="Epic Tale of Two Characters",
            description="Testing two characters talking with background audio effects",
            style="cinematic",
            cast=[],  # No characters for this scenario
            sets=[],
            shots=[VideoBlock(
                id="basic_two_talking_character_with_bg_audio_effects",
                storyline_label="Test two talking characters",
                render_engine=RenderEngine.LIPSYNC_MOTION,  # Use LIPSYNC_MOTION with I2V
                duration_seconds=5,# Duration is ignored for LIPSYNC_MOTION
                fps=25,
                width=537,
                height=896,
                first_keyframe=KeyFrame(
                    source=KeyframeSource.GENERATE,
                    linked_source_id=None,
                    supplied_image_url=None,
                    width=537,
                    height=896,
                    set=None,
                    characters=None,
                    basic_generation_prompt="A beautiful elegant woman with flowing ginger hair, wearing a vintage "
                                "burgundy dress, sitting gracefully in an ornate Victorian parlor, holding a glass of white wine,"
                                "with rich wooden furniture and warm golden lighting,"
                                "A monkey in a suit, standing next to her on the right side, fanning her with a large ornate fan, looking attentive and respectful,"
                                "portrait style, high quality, cinematic lighting",
                    image_url=None,
                    rendered_frame_by_vid_gen_url=None
                ),
                last_keyframe=None,  # Not needed because render engine is LIPSYNC_MOTION
                video_prompt=("A sophisticated ginger woman, holding a glass of white wine, speaking elegantly about wine, "
                            "A respectful monkey standing on her right hands side, speaking with passion, with natural "
                            "facial expressions and subtle head movements, in a Victorian parlor setting"),
                style="cinematic, elegant",
                narrations=[
                    Narration(
                        id="wine_tasting_narration",
                        exaggeration=0.5,
                        cfg_weight=0.5,
                        script=(
                            "The wine is quite delightful, with notes of cherry and oak that dance "
                            "upon the palate."
                        ),
                        voice_sample_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/audio_samples/219777__madamvicious__the-wine-is-quite-delightful-posh-woman.wav",
                        audio_url=None,
                    ),                    
                    Narration(
                        id="buttler_narration",
                        exaggeration=0.5,
                        cfg_weight=0.5,
                        script=(
                            "Fuck off you stupid bitch"
                        ),
                        voice_sample_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/audio_samples/what-can-i-do-for-you-npc-british-male-99751.mp3",
                        audio_url=None,
                    )
                ],
                overlays=None,
                bg_audio_effects=BackgroundAudioEffects(
                    is_enabled=False,
                    prompt=("Hall with piano playing in distance, no people talking")
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
    

    def test_sim_singing(self) -> bool:
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
                    source=KeyframeSource.SUPPLIED,
                    linked_source_id=None,
                    supplied_image_url=None,
                    width=581,
                    height=896,
                    set=None,
                    characters=None,
                    basic_generation_prompt=None,
                    image_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/audio_samples/IMG_9710.jpg",
                    rendered_frame_by_vid_gen_url=None
                ),
                last_keyframe=None,  # Not needed because render engine is LIPSYNC_MOTION
                video_prompt=("A woman singing passionately"),
                style="cinematic",
                narrations=[
                    Narration(
                        id="wine_tasting_narration",
                        exaggeration=0.5,
                        cfg_weight=0.5,
                        script=None,
                        voice_sample_url=None,
                        audio_url=None,
                        audio_padded_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/audio_samples/please%20please%20pleasew%20short.wav"
                    )
                ],
                overlays=None,
                bg_audio_effects=None,
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
    

    def first_short_content_test_shot_001(self) -> bool:# This test is to create a talking character video without using the characters feature

        # Create video for scenario 1
        video = Video(
            id="first_short_content_test_001",
            title="Rodrigo Intro",
            description="Intro to weird mother story",
            style="cinematic",
            cast=[],  # No characters for this scenario
            sets=[],
            shots=[VideoBlock(
                id="first_short_content_test_001_shot_001",
                storyline_label="First Short Content Test Shot 001",
                render_engine=RenderEngine.LIPSYNC_MOTION,  # Use LIPSYNC_MOTION with I2V
                duration_seconds=5,# Duration is ignored for LIPSYNC_MOTION
                fps=25,
                width=720,
                height=1080,
                first_keyframe=KeyFrame(
                    source=KeyframeSource.GENERATE,
                    linked_source_id=None,
                    width=720,
                    height=1080,
                    set=None,
                    characters=None,
                    basic_generation_prompt="Edit this image and make the man sit behind a desk in the same environment, same distance from the camera, cinematic lighting, high quality, photorealistic, detailed, 4k",
                    supplied_image_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/Temp%20Dump/matt-silver--jqTm3EaGus-unsplash.jpg",
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
                        script="Listen to this — this woman’s mom says she deserves to be her baby’s “first mom”… because she raised her.",
                        voice_sample_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/Temp%20Dump/JoseAudioSample.wav",
                        audio_url=None,
                        audio_padded_url=None
                    )
                ],
                overlays=None,
                bg_audio_effects=None,
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
    
    def first_short_content_test_shot_002(self) -> bool:

        # Create video for scenario 1
        video = Video(
            id="first_short_content_test_002",
            title="Setup | Generic Visual",
            description="Setup | Generic Visual",
            style="cinematic",
            cast=[],  # No characters for this scenario
            sets=[],
            shots=[VideoBlock(
                id="first_short_content_test_001_shot_002",
                storyline_label="First Short Content Test Shot 002",
                render_engine=RenderEngine.GENERATIVE_MOTION,  # Use LIPSYNC_MOTION with I2V
                duration_seconds=5,# Duration is ignored for LIPSYNC_MOTION
                fps=25,
                width=720,
                height=1080,
                first_keyframe=KeyFrame(
                    source=KeyframeSource.GENERATE,
                    linked_source_id=None,
                    width=720,
                    height=1080,
                    set=None,
                    characters=None,
                    basic_generation_prompt="Young woman holding newborn while her older mother hovers nearby helping, cozy home, natural daylight, cinematic tone.",
                    supplied_image_url=None,
                    rendered_frame_by_vid_gen_url=None
                ),
                last_keyframe=None,  # Not needed because render engine is LIPSYNC_MOTION
                video_prompt=("Slow pan around young mom and older mom with baby, gentle motion, warm domestic lighting, hint of tension."),
                style="cinematic",
                narrations=[
                    Narration(
                        id="narration_001",
                        exaggeration=0.5,
                        cfg_weight=0.5,
                        script="She’s twenty-six, just had her first baby — and her mom’s way too involved.",
                        voice_sample_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/Temp%20Dump/JoseAudioSample.wav",
                        audio_url=None,
                        audio_padded_url=None
                    )
                ],
                overlays=None,
                bg_audio_effects=None,
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

    def first_short_content_test_shot_00X(self, shotnum: int, first_image_prompt: str, video_prompt: str, script: str) -> bool:

        # Create video for scenario 1
        video = Video(
            id="first_short_content_test"+f"_{shotnum:03d}",
            title="Setup | Generic Visual",
            description="Setup | Generic Visual",
            style="cinematic",
            cast=[],  # No characters for this scenario
            sets=[],
            shots=[VideoBlock(
                id="first_short_content_test_001_shot_"+f"{shotnum:03d}",
                storyline_label="First Short Content Test Shot "+f"{shotnum:03d}",
                render_engine=RenderEngine.GENERATIVE_MOTION,  # Use LIPSYNC_MOTION with I2V
                duration_seconds=5,# Duration is ignored for LIPSYNC_MOTION
                fps=25,
                width=720,
                height=1080,
                first_keyframe=KeyFrame(
                    source=KeyframeSource.GENERATE,
                    linked_source_id=None,
                    width=720,
                    height=1080,
                    set=None,
                    characters=None,
                    basic_generation_prompt=first_image_prompt,
                    supplied_image_url=None,
                    rendered_frame_by_vid_gen_url=None
                ),
                last_keyframe=None,  # Not needed because render engine is LIPSYNC_MOTION
                video_prompt=video_prompt,
                style="cinematic",
                narrations=[
                    Narration(
                        id="narration_001",
                        exaggeration=0.5,
                        cfg_weight=0.5,
                        script=script,
                        voice_sample_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/Temp%20Dump/JoseAudioSample.wav",
                        audio_url=None,
                        audio_padded_url=None
                    )
                ],
                overlays=None,
                bg_audio_effects=None,
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
    
    def first_short_content_test_generic_shots(self) -> bool:
        first_image_prompts = [
            "Young woman holding newborn while her older mother hovers nearby helping, cozy home, natural daylight, cinematic tone.",
            "Older woman leaning over crib, young mother watching uneasily, subtle tension, warm light.",
            "Older woman introducing baby to a friend proudly, young mother surprised, emotional contrast, cinematic realism.",
            "Older mother correcting young mother on feeding a baby, kitchen background, visible frustration.",
            "Argument between young woman and older mother in living room, baby toys visible, emotional scene.",
            "Grandmother holding baby’s overnight bag by the door, young mom conflicted, night lighting.",
            "Young mother sitting alone on couch holding baby protectively, emotional mood, soft light.",
            "Close-up of young mother kissing baby’s forehead, teary-eyed, emotional soft background."
        ]

        video_prompts = [
            "Slow pan around young mom and older mom with baby, gentle motion, warm domestic lighting, hint of tension.",
            "Slow zoom showing older mom hovering over baby while younger mom looks uncomfortable, natural handheld motion.",
            "Short motion clip — older woman chatting happily while young mother reacts with shock, soft camera drift.",
            "Camera moves between both women — older one demonstrating, younger one hurt, realistic gestures.",
            "Dramatic short clip — two women arguing quietly in a home setting, handheld camera feel.",
            "Camera focuses on grandmother ready to leave with baby bag, mom uncertain, dim cinematic tone.",
            "Slow drift around lonely mother hugging baby, emotional lighting, cinematic mood.",
            "Cinematic pan showing tender kiss on baby’s forehead, emotional lighting, calm movement."
        ]

        scripts = [
            "She’s twenty-six, just had her first baby — and her mom’s way too involved.",
            "At first it was sweet — but lately her mom’s crossing lines.",
            "She even called the baby ‘our baby’ — and said, ‘I’m the first mom, I raised you.’",
            "Now she critiques everything — feeding, swaddling, even saying the baby likes her more.",
            "When called out, she snapped — said her daughter’s ungrateful and can’t parent alone.",
            "Now she wants a trial run — to take the baby overnight and ‘prove she can do it better.’",
            "She says it feels like her mom doesn’t see her as a parent — just an assistant.",
            "It’s heartbreaking — she loves her mom but feels completely undermined."
        ]

        for i in range(len(first_image_prompts)):
            self.first_short_content_test_shot_00X(i, first_image_prompts[i], video_prompts[i], scripts[i])

        return True
    
    def first_short_content_test_shot_00last(self) -> bool:# This test is to create a talking character video without using the characters feature

        # Create video for scenario 1
        video = Video(
            id="first_short_content_test_00last",
            title="Rodrigo Outro",
            description="Outro to weird mother story",
            style="cinematic",
            cast=[],  # No characters for this scenario
            sets=[],
            shots=[VideoBlock(
                id="first_short_content_test_00last_shot_00last",
                storyline_label="First Short Content Test Shot 00last",
                render_engine=RenderEngine.LIPSYNC_MOTION,  # Use LIPSYNC_MOTION with I2V
                duration_seconds=5,# Duration is ignored for LIPSYNC_MOTION
                fps=25,
                width=720,
                height=1080,
                first_keyframe=KeyFrame(
                    source=KeyframeSource.SUPPLIED,
                    linked_source_id=None,
                    width=720,
                    height=1080,
                    set=None,
                    characters=None,
                    basic_generation_prompt=None,
                    supplied_image_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/Temp%20Dump/matt-silver--jqTm3EaGus-unsplash.jpg",
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
                        script="If your own parent claimed they were the ‘first mom’ to your child, what would you do? Comment below — would you cut contact, or try to make peace?",
                        voice_sample_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/Temp%20Dump/JoseAudioSample.wav",
                        audio_url=None,
                        audio_padded_url=None
                    )
                ],
                overlays=None,
                bg_audio_effects=None,
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
    
    def first_short_content_test_make_full_video_from_generated_shot_videos(self) -> bool:# This test is to create a talking character video without using the characters feature

        # Create video for scenario 1
        video = Video(
            id="first_short_content_test_001",
            title="Rodrigo Mom Story Full Video",
            description="Full video from generated shots",
            style="cinematic",
            cast=[],  # No characters for this scenario
            sets=[],
            shots=[VideoBlock(
                id="first_short_content_test_001_shot_001",
                storyline_label="First Short Content Test Shot 001",
                fps=25,
                width=720,
                height=1080,
                supplied_video_with_audio_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/ac7ba161-1abb-41d4-b646-4959e8629500.mp4?",
                overlays = [
                    TextOverlay(
                        text="Listen to this — this woman’s mom says she deserves to be her baby’s ‘first mom’… because she raised her.",
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
                transition=Transition.CUT
            ),
            VideoBlock(
                id="first_short_content_test_001_shot_002",
                storyline_label="First Short Content Test Shot 002",
                fps=25,
                width=720,
                height=1080,
                supplied_video_with_audio_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/641c415d-38fc-404c-9f52-cec5ba58a4ac.mp4",
                overlays = [
                    TextOverlay(
                        text="She’s twenty-six, just had her first baby — and her mom’s way too involved.",
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
                transition=Transition.CUT
            ),
            VideoBlock(
                id="first_short_content_test_001_shot_003",
                storyline_label="First Short Content Test Shot 003",
                fps=25,
                width=720,
                height=1080,
                supplied_video_with_audio_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/0972e2d3-8582-4c4f-8903-4c11083dc7bb.mp4",
                overlays = [
                    TextOverlay(
                        text="At first it was sweet — but lately her mom’s crossing lines",
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
                transition=Transition.CUT
            ),
            VideoBlock(
                id="first_short_content_test_001_shot_004",
                storyline_label="First Short Content Test Shot 004",
                fps=25,
                width=720,
                height=1080,
                supplied_video_with_audio_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/22743f73-08cf-4023-9a84-d0b5f644cbba.mp4",
                overlays = [
                    TextOverlay(
                        text="She even called the baby ‘our baby’ — and said, ‘I’m the first mom, I raised you.’",
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
                transition=Transition.CUT
            ),
            VideoBlock(
                id="first_short_content_test_001_shot_005",
                storyline_label="First Short Content Test Shot 005",
                fps=25,
                width=720,
                height=1080,
                supplied_video_with_audio_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/9cc7d4b9-04c5-492b-a2fd-c676c2ba146c.mp4",
                overlays = [
                    TextOverlay(
                        text="Now she critiques everything — feeding, swaddling, even saying the baby likes her more.",
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
                transition=Transition.CUT
            ),
            VideoBlock(
                id="first_short_content_test_001_shot_006",
                storyline_label="First Short Content Test Shot 006",
                fps=25,
                width=720,
                height=1080,
                supplied_video_with_audio_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/57e57bbc-7bc5-476a-b1e3-f550ace48848.mp4",
                overlays = [
                    TextOverlay(
                        text="When called out, she snapped — said her daughter’s ungrateful and can’t parent alone.",
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
                transition=Transition.CUT
            ),
            VideoBlock(
                id="first_short_content_test_001_shot_007",
                storyline_label="First Short Content Test Shot 007",
                fps=25,
                width=720,
                height=1080,
                supplied_video_with_audio_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/87e8f332-a291-4318-ba05-9fdbcaec1b51.mp4",
                overlays = [
                    TextOverlay(
                        text="Now she wants a trial run — to take the baby overnight and ‘prove she can do it better.",
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
                transition=Transition.CUT
            ),
            VideoBlock(
                id="first_short_content_test_001_shot_008",
                storyline_label="First Short Content Test Shot 008",
                fps=25,
                width=720,
                height=1080,
                supplied_video_with_audio_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/a2d5ae01-5428-4ec7-abe5-c6782f13f3ce.mp4",
                overlays = [
                    TextOverlay(
                        text="She says it feels like her mom doesn’t see her as a parent, just an assistant.",
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
                transition=Transition.CUT
            ),
            VideoBlock(
                id="first_short_content_test_001_shot_009",
                storyline_label="First Short Content Test Shot 009",
                fps=25,
                width=720,
                height=1080,
                supplied_video_with_audio_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/ddcd9272-975a-457e-8ea5-b456c65860af.mp4",
                overlays = [
                    TextOverlay(
                        text="It’s heartbreaking, she loves her mom but feels completely undermined.",
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
                transition=Transition.CUT
            ),
            VideoBlock(
                id="first_short_content_test_001_shot_010",
                storyline_label="First Short Content Test Shot 010",
                fps=25,
                width=720,
                height=1080,
                supplied_video_with_audio_url="https://lmegqlleznqzhwxeyzdh.supabase.co/storage/v1/object/public/generated_videos/bb22621a-3f46-4213-8d2a-270cdde80001.mp4",
                overlays = [
                    TextOverlay(
                        text="If your own parent claimed they were the ‘first mom’ to your child, what would you do? Comment below. Would you cut contact, or try to make peace?",
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
                transition=Transition.CUT
            ),
            ],
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
            #("Audio effect + speech video layering", self.test_simple_talking_character_with_bg_audio_effects)
            #("Two characters conversation with background audio effects", self.test_simple_two_characters_conversation_with_bg_effects)
            #("Sim singing", self.test_sim_singing)
            ("Single character supplied image and audio", self.first_short_content_test_shot_001)
            #("First short content test shot 002", self.first_short_content_test_shot_002)
            #("First short content test generic shots", self.first_short_content_test_generic_shots)
            #("First short content test shot last", self.first_short_content_test_shot_00last)
            #("First short content test make full video from generated shot videos", self.first_short_content_test_make_full_video_from_generated_shot_videos)
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
