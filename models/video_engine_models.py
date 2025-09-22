from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
from math import ceil
import uuid
# Import external dependencies
import io
from typing import Tuple
import requests

# ───────────────────────────────────────────────────────────────────────────────
# Tunables / Engine constants
# ───────────────────────────────────────────────────────────────────────────────
MIN_SHOT_DURATION_SECONDS = 3   # Smallest achievable AI-generated shot length
MAX_SHOT_DURATION_SECONDS = 5  # Longest achievable AI-generated shot length (can be adjusted)
DEFAULT_SHOT_DURATION_SECONDS = 5  # Default average shot length for planning


# ───────────────────────────────────────────────────────────────────────────────
# Enums & simple value classes
# ───────────────────────────────────────────────────────────────────────────────

class LightSourceType(Enum):
    DAYLIGHT = "Daylight"
    ARTIFICIAL_LIGHT = "Artificial Light"
    MOONLIGHT = "Moonlight"
    PRACTICAL_LIGHT = "Practical Light"
    FIRELIGHT = "Firelight"
    FLUORESCENT_LIGHT = "Fluorescent Light"
    OVERCAST_LIGHT = "Overcast Light"
    MIXED_LIGHT = "Mixed Light"
    SUNLIGHT = "Sunlight"

class LightQuality(Enum):
    SOFT_LIGHT = "Soft Light"
    HARD_LIGHT = "Hard Light"
    TOP_LIGHT = "Top Light"
    SIDE_LIGHT = "Side Light"
    BACKLIGHT = "Backlight"
    BOTTOM_LIGHT = "Bottom Light"
    RIM_LIGHT = "Rim Light"
    SILHOUETTE = "Silhouette"
    LOW_CONTRAST = "Low Contrast"
    HIGH_CONTRAST = "High Contrast"

class TimeOfDay(Enum):
    DAYTIME = "Daytime"
    NIGHTTIME = "Nighttime"
    DUSK = "Dusk"
    SUNSET = "Sunset"
    DAWN = "Dawn"
    SUNRISE = "Sunrise"

class ShotSize(Enum):
    CLOSE_UP = "Close-up"
    MEDIUM_CLOSE_UP = "Medium Close-up"
    MEDIUM_SHOT = "Medium Shot"
    MEDIUM_LONG_SHOT = "Medium Long Shot"
    LONG_SHOT = "Long Shot"
    FULL_SHOT = "Full Shot"
    WIDE_ANGLE = "Wide Angle"

class Composition(Enum):
    CENTERED = "Centered"
    BALANCED = "Balanced"
    RIGHT_WEIGHTED = "Right Weighted"
    LEFT_WEIGHTED = "Left Weighted"
    SYMMETRICAL = "Symmetrical"
    SHORT_SIDING = "Short-siding"

class LensFocalLength(Enum):
    MEDIUM = "Medium"
    WIDE_ANGLE = "Wide Angle"
    LONG = "Long"
    TELEPHOTO = "Telephoto"
    FISHEYE = "Fisheye"

class LensAngle(Enum):
    OVER_THE_SHOULDER = "Over-the-shoulder"
    HIGH_ANGLE = "High Angle"
    LOW_ANGLE = "Low Angle"
    TILTED_ANGLE = "Tilted Angle"
    AERIAL_SHOT = "Aerial Shot"
    TOP_DOWN_VIEW = "Top-down View"

class ShotType(Enum):
    SINGLE_SHOT = "Single Shot"
    TWO_SHOT = "Two Shot"
    THREE_SHOT = "Three Shot"
    GROUP_SHOT = "Group Shot"
    ESTABLISHING_SHOT = "Establishing Shot"

class ColorTone(Enum):
    WARM_TONE = "Warm Tone"
    COOL_TONE = "Cool Tone"
    HIGH_SATURATION = "High Saturation"
    LOW_SATURATION = "Low Saturation"

# ─────────────────────────────
# 4. Dynamic Control
# ─────────────────────────────

class MotionType(Enum):
    RUNNING = "Running"
    SKATEBOARDING = "Skateboarding"
    SOCCER = "Soccer"
    TENNIS = "Tennis"
    PING_PONG = "Ping Pong"
    SKIING = "Skiing"
    BASKETBALL = "Basketball"
    FOOTBALL = "Football"
    BOWL_DANCE = "Bowl Dance"
    CARTWHEEL = "Cartwheel"

class Emotion(Enum):
    ANGER = "Anger"
    FEAR = "Fear"
    JOY = "Joy"
    SADNESS = "Sadness"
    SURPRISE = "Surprise"

# ─────────────────────────────
# 5. Camera Work
# ─────────────────────────────

class BasicCameraMovement(Enum):
    PUSH_IN = "Push-in"
    PULL_OUT = "Pull-out"
    PAN_RIGHT = "Pan Right"
    PAN_LEFT = "Pan Left"
    TILT_UP = "Tilt Up"

class AdvancedCameraMovement(Enum):
    HANDHELD = "Handheld"
    COMPOUND = "Compound"
    FOLLOWING = "Following"
    ORBIT = "Orbit"

# ─────────────────────────────
# 6. Stylization
# ─────────────────────────────

class VisualStyle(Enum):
    FELT_STYLE = "Felt Style"
    CARTOON_3D = "3D Cartoon"
    PIXEL_ART = "Pixel Art"
    PUPPET_ANIMATION = "Puppet Animation"
    GAME_3D = "3D Game"
    CLAYMATION = "Claymation"
    ANIME = "Anime"
    WATERCOLOR = "Watercolor"
    BW_ANIMATION = "B&W Animation"
    OIL_PAINTING = "Oil Painting"
    CINEMATIC = "Cinematic"

class SpecialEffect(Enum):
    TILT_SHIFT = "Tilt-shift"
    TIME_LAPSE = "Time-lapse"

@dataclass
class VideoPromptTTV:
    # Suitable for users with some AI video experience. On top of the basic formula, add richer and more detailed descriptions.
    # This effectively improves video texture, vividness, and storytelling.
    #	•	Subject Description: describes the detailed appearance of the subject, using adjectives or short phrases.
    # Example: “A young Miao ethnic girl with black hair, wearing traditional clothing”
    # Example: “A celestial fairy from another world, wearing old but ornate clothing, with a pair of wings made of ruins behind her.”
    #	•	Scene Description: details of the environment where the subject is located, using adjectives or short phrases.
    #	•	Motion Description: details of the movement, including amplitude, speed, and effect.
    # Example: “Swinging violently,” “Moving slowly,” “Shattering glass.”
    #	•	Aesthetic Control: includes light source, light quality, shot size, perspective, lens, camera movement, etc.
    #	•	Stylization: describes style language of the picture, such as “cyberpunk,” “line illustration,” “wasteland style.”
    subject: str
    scene: str
    motion: str
    aesthetics: str
    stylization: str

@dataclass
class VideoPromptITV:
    # Since an image already defines subject, scene, and style, 
    # the prompt mainly describes the dynamic process and camera movement.
    motion: str
    camera_movement: str

# ───────────────────────────────────────────────────────────────────────────────
# Rendering & linking enums
# ───────────────────────────────────────────────────────────────────────────────
class RenderEngine(Enum):
    """
    How frames are realized into motion for the shot.
    - GENERATIVE_MOTION selects I2V normally; if end-locked, engine uses FF+LF under the hood.
    - STILL holds the start frame for the duration (title cards, info slates).
    - LIPSYNC_MOTION uses a sound-to-video model (image+audio → talking).
    """
    STILL = "Still"
    GENERATIVE_MOTION = "GenerativeMotion"
    LIPSYNC_MOTION = "LipSyncMotion"

class Transition(Enum):
    """Editorial transition out of this shot into the next."""
    CUT = "Cut"
    DISSOLVE = "Dissolve"
    FADE_TO_BLACK = "FadeBlack"
    SLIDE = "Slide"
    WHIP_PAN = "WhipPan"
    BLUR_DISSOLVE = "BlurDissolve"
    GLITCH = "Glitch"
    CONTINUOUS = "Continuous"  # treat boundary as seamless (engine may overlap frames)

class KeyframeSource(Enum):
    """Where the keyframe(s) come from when needed by the chosen strategy."""
    NOT_USED = "NotUsed" # Used purely for the last frame if we want to skip it
    GENERATE = "Generate"  # Generate a new keyframe
    SUPPLIED = "Supplied"  # user/system-provided still
    COPY_PREV_LAST = "CopyPrevLast" # Linked: start from source last frame


# ───────────────────────────────────────────────────────────────────────────────
# Overlays (text UI)
# ───────────────────────────────────────────────────────────────────────────────
class TextPresentationType(Enum):
    """High-level behavior for how text appears over the video."""
    STATIC_OVERLAY = "Static Overlay"               # Always on screen for its duration
    DURATION_BASED_WITH_ANIMATION = "Duration Based" # Visible for a duration, animated in/out
    PROGRESSIVE_REVEAL = "Progressive Reveal"       # Reveals over time (by character/word/line)

class TextPresentationAnimationType(Enum):
    """Animation styles for entering/leaving."""
    NONE = "None"
    FADE_IN_OUT = "Fade In/Out"
    SLIDE_IN_OUT = "Slide In/Out"

class TextOverlayPosition(Enum):
    """Canonical anchor positions for overlays (used by UI layout)."""
    TOP = "Top"
    TOP_LEFT = "Top Left"
    TOP_RIGHT = "Top Right"
    BOTTOM = "Bottom"
    BOTTOM_LEFT = "Bottom Left"
    BOTTOM_RIGHT = "Bottom Right"
    CENTER = "Center"
    CENTER_LEFT = "Center Left"
    CENTER_RIGHT = "Center Right"

@dataclass
class TextOverlayTiming:
    """
    Timing/animation knobs for a single overlay.
    UI ties directly to these fields; engine reads and applies.
    """
    # Basic timing control within the shot
    start_time_seconds: float = 0.0
    duration_seconds: Optional[float] = None   # If None, lasts until end of shot
    end_time_seconds: Optional[float] = None   # Alternative to duration (UI may compute one or the other)

    # Animation timing/easing
    animation_duration_in: float = 0.5
    animation_duration_out: float = 0.5
    animation_easing: str = "ease-in-out"

    # Progressive reveal configuration
    reveal_speed: float = 1.0           # units per second
    reveal_unit: str = "word"           # "character" | "word" | "line"

    # Slide directions (for SLIDE_IN_OUT)
    direction_in: str = "bottom"        # "top" | "bottom" | "left" | "right"
    direction_out: str = "bottom"

    # Optional audio sync hints (for caption-like overlays)
    audio_cue_timestamp: Optional[float] = None
    audio_character_id: Optional[str] = None
    speech_rate: float = 2.5            # words/sec if doing basic auto-timing from script

@dataclass
class TextOverlayProperties:
    """
    Visual properties for a text overlay. Kept separate so UI can provide presets.
    """
    font_family: str
    font_size: int
    color: str
    background_color: str = "transparent"
    padding: int = 0
    border_radius: int = 0
    presentation_type: TextPresentationType = TextPresentationType.STATIC_OVERLAY
    presentation_animation_type: TextPresentationAnimationType = TextPresentationAnimationType.NONE
    timing: TextOverlayTiming = field(default_factory=TextOverlayTiming)

@dataclass
class TextOverlay:
    """
    One overlay item: content + placement + visual properties.
    """
    text: str
    position: TextOverlayPosition
    properties: TextOverlayProperties

# ───────────────────────────────────────────────────────────────────────────────
# Characters & narration
# ───────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Placement enum: doubles as geometry + prompt-friendly labels
# ─────────────────────────────────────────────────────────────────────────────
class Placement(Enum):
    LEFT_FOREGROUND   = ("left foreground",   (0.20, 0.78), "fg")
    CENTER_FOREGROUND = ("center foreground", (0.50, 0.80), "fg")
    RIGHT_FOREGROUND  = ("right foreground",  (0.80, 0.78), "fg")

    LEFT_MID          = ("left mid-ground",   (0.25, 0.60), "mid")
    CENTER_MID        = ("center mid-ground", (0.50, 0.60), "mid")
    RIGHT_MID         = ("right mid-ground",  (0.75, 0.60), "mid")

    LEFT_BACKGROUND   = ("left background",   (0.30, 0.42), "bg")
    CENTER_BACKGROUND = ("center background", (0.50, 0.40), "bg")
    RIGHT_BACKGROUND  = ("right background",  (0.70, 0.42), "bg")

    @property
    def prompt_label(self) -> str:
        return self.value[0]

    @property
    def anchor(self) -> Tuple[float, float]:
        # (x_frac, y_frac) of canvas where the character should be centered
        return self.value[1]

    @property
    def depth(self) -> str:
        # "fg" | "mid" | "bg"
        return self.value[2]


@dataclass
class Character:
    """
    Represents a character that can appear in a scene or shot.
    - Characters are listed directly in each shot (characters[0..n]).
    - characters[0] = primary character for that shot (eligible for anchoring).
    - characters[1..n] = additional characters, always prompt-only.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""                              # Human-readable label ("Sam", "Ava")
    description: str = ""                       # Optional longer description (for UI/UX, not engine)

    prompt: str = ""                            # Prompt for the character's appearance and behavior, used to generate if no image provided
    style: str = ""                             # Visual style for the character (e.g., "realistic photography", "animation", "cartoon", "cinematic")
    image_url: Optional[str] = None              # Optional URL for the character's image

    voice_sample_url: Optional[str] = None      # Optional voice sample for TTS / narration


@dataclass
class CharacterVariant:
    """
    Represents a variant of a character for a specific shot.
    - Variants are used to modify the appearance or behavior of a character in a shot.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str = "" # The ID of the parent character

    style: str = ""  # Visual style for the character (e.g., "realistic photography", "animation", "cartoon", "cinematic")
    prompt: str = "" # The prompt describing the variant, this is used to edit the parent character image
    action_in_frame: str = "" # What this character is doing in this frame, this is used to edit the parent character image
    image_url: Optional[str] = None # The URL of the generated image for the variant or the uploaded image by the user

    script: str = "" # Characters script to be spoken
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    audio_url: Optional[str] = None # The URL of the audio of the character spoken script
    audio_padded_url: Optional[str] = None # The URL of the padded audio for the character spoken script with other characters involved.

    placement_hint: Placement = Placement.CENTER_FOREGROUND


@dataclass
class Narration:
    """
    Scene/shot narration or voiceover metadata.
    - For S2V, audio_url is the required input to drive lip-sync.
    """
    id: str
    script: str
    voice_sample_url: str
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    audio_url: Optional[str] = None
    audio_padded_url: Optional[str] = None

@dataclass
class Music:
    """Defines the background music for a video block."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    prompt: str = ""
    lyrics: str = ""
    duration: int = 0 # set to -1 to indicate no specific duration, dictated by the lyrics
    audio_url: Optional[str] = None  # The URL of the audio file


@dataclass
class BackgroundAudioEffects:
    """Defines the background audio effects for a video block."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_enabled: bool = True
    prompt: str = ""
    audio_url: Optional[str] = None  # The URL of the audio file


# Set Model: Used to define the scene for the frames
@dataclass
class Set:
    """Defines the scene for the frames."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = "" # The prompt describing the scene, used to generate the scene and its elements
    style: str = "" # The style to apply to the scene, used to guide the image generation model
    width: int = 720 # output width
    height: int = 1280 # output height
    image_url: Optional[str] = None # The URL of the generated image for the set or the uploaded image by the user

# Set Variant: Used to define variations of a scene for different frames
@dataclass
class SetVariant:
    """Defines a variation of a set for a specific frame."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str = "" # The ID of the parent set
    prompt: str = "" # The prompt describing the variation
    style: str = "" # The style to apply to the variation
    image_url: Optional[str] = None # The URL of the generated image for the variant or the uploaded image by the user

@dataclass
class KeyFrame:
    """Represents a keyframe within a shot."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: KeyframeSource = KeyframeSource.SUPPLIED # The source of the keyframe, SUPPLIED means use supplied_image_url
    linked_source_id: Optional[str] = None  # The ID of the linked source keyframe

    supplied_image_url: Optional[str] = None  # The URL of the generated image for the keyframe

    width: int = 720  # The width of the keyframe image
    height: int = 1280  # The height of the keyframe image

    set: Optional[SetVariant] = None # The set for the keyframe

    characters: List[CharacterVariant] = field(default_factory=list)

    basic_generation_prompt: str = ""  # Simple prompt for the keyframe (for I2V / FF+LF

    image_url: Optional[str] = None  # The URL of the generated image for the keyframe
    rendered_frame_by_vid_gen_url: Optional[str] = None  # The URL of the rendered frame by the video generator


# ───────────────────────────────────────────────────────────────────────────────
# Core hierarchy: VideoBlock (Shot) → Scene → Sequence
# ───────────────────────────────────────────────────────────────────────────────
@dataclass
class VideoBlock:
    """
    Single 5s shot within a scene.
    - Start is determined by link_frame_strategy.
    - End can be locked with lock_end_keyframe.
    - RenderEngine controls how frames are realized into motion.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))    
    storyline_label: Optional[str] = None    # Optional UX/AI grouping label (parallel narratives etc.)

    render_engine: RenderEngine = RenderEngine.GENERATIVE_MOTION        # Rendering mode (how to realize frames)
    duration_seconds: int = DEFAULT_SHOT_DURATION_SECONDS    # Timing
    fps: int = 16
    width: int = 720  # The width of the keyframe image
    height: int = 1280  # The height of the keyframe image

    first_keyframe: Optional[KeyFrame] = None  # The first keyframe in the shot
    last_keyframe: Optional[KeyFrame] = None    # The last keyframe in the shot

    video_prompt: str = ""         # Motion / camera / action description (I2V / FF+LF hints)
    style: str = "default"

    narrations: Optional[List[Narration]] = None  # Optional narration (also provides audio for LIPSYNC_MOTION)
    bg_audio_effects: Optional[BackgroundAudioEffects] = None
    overlays: Optional[List[TextOverlay]] = None  # Overlays

    generated_video_clip_raw: Optional[str] = None    # Engine outputs / caches (populated by the engine)
    generated_video_clip_with_overlays: Optional[str] = None    # Engine outputs / caches (populated by the engine)
    generated_video_clip_with_audio_and_overlays: Optional[str] = None    # Engine outputs / caches (populated by the engine)

    generated_video_clip_final: Optional[str] = None    # Final output after music added (populated by the engine)

    transition: Transition = Transition.CUT     # Linking / continuity


    # Utility / validation helpers
    def validated_duration(self) -> int:
        """Clamp duration to the product's hard 5s shot design."""
        d = self.duration_seconds or DEFAULT_SHOT_DURATION_SECONDS
        return max(MIN_SHOT_DURATION_SECONDS, min(MAX_SHOT_DURATION_SECONDS, d))

    def frame_count(self) -> int:
        """Compute total frames (inclusive of last) for timeline math."""
        return int(ceil((self.validated_duration() * self.fps) + 1))

@dataclass
class Video:
    """
    A video is composed of multiple shots with shared cast and sets.
    - Cast: Base character templates that can be referenced by variants in shots
    - Sets: Base environment templates that can be referenced by variants in shots
    - All characters are treated equally - no primary character concept
    """
    id: str
    title: str
    description: str
    style: str

    # Base templates for the entire video
    cast: List[Character] = field(default_factory=list)      # Character templates
    sets: List[Set] = field(default_factory=list)           # Environment templates

    # The shot list
    shots: List[VideoBlock] = field(default_factory=list)

    # Optional compiled output
    generated_video_url: Optional[str] = None
    background_music: Optional[Music] = None

    @property
    def duration_seconds(self) -> int:
        return sum(blk.validated_duration() for blk in self.shots)

    def add_shot(self, video_block: VideoBlock) -> None:
        self.shots.append(video_block)

    def add_generated_video(self, video_url: str) -> None:
        self.generated_video_url = video_url

    # Cast management helpers
    def add_character(self, character: Character) -> None:
        """Add a character to the video's cast."""
        self.cast.append(character)
    
    def get_character_by_id(self, character_id: str) -> Optional[Character]:
        """Get a character from the cast by ID."""
        for character in self.cast:
            if character.id == character_id:
                return character
        return None
    
    def get_character_by_name(self, name: str) -> Optional[Character]:
        """Get a character from the cast by name."""
        for character in self.cast:
            if character.name.lower() == name.lower():
                return character
        return None

    # Set management helpers  
    def add_set(self, set_template: Set) -> None:
        """Add a set template to the video."""
        self.sets.append(set_template)
    
    def get_set_by_id(self, set_id: str) -> Optional[Set]:
        """Get a set template by ID."""
        for set_template in self.sets:
            if set_template.id == set_id:
                return set_template
        return None

    # Shot management
    def get_shot_by_id(self, shot_id: str) -> Optional[VideoBlock]:
        """Get a shot by ID."""
        for shot in self.shots:
            if shot.id == shot_id:
                return shot
        return None
    
    # Validation helpers
    def validate_character_variants(self) -> List[str]:
        """Validate that all character variants reference valid parent characters."""
        errors = []
        for shot in self.shots:
            if shot.first_keyframe:
                for variant in shot.first_keyframe.characters:
                    if not self.get_character_by_id(variant.parent_id):
                        errors.append(f"Character variant {variant.id} references invalid parent {variant.parent_id}")
            if shot.last_keyframe:
                for variant in shot.last_keyframe.characters:
                    if not self.get_character_by_id(variant.parent_id):
                        errors.append(f"Character variant {variant.id} references invalid parent {variant.parent_id}")
        return errors
    
    def validate_set_variants(self) -> List[str]:
        """Validate that all set variants reference valid parent sets."""
        errors = []
        for shot in self.shots:
            if shot.first_keyframe and shot.first_keyframe.set:
                if not self.get_set_by_id(shot.first_keyframe.set.parent_id):
                    errors.append(f"Set variant {shot.first_keyframe.set.id} references invalid parent {shot.first_keyframe.set.parent_id}")
            if shot.last_keyframe and shot.last_keyframe.set:
                if not self.get_set_by_id(shot.last_keyframe.set.parent_id):
                    errors.append(f"Set variant {shot.last_keyframe.set.id} references invalid parent {shot.last_keyframe.set.parent_id}")
        return errors
