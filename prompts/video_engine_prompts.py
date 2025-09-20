from typing import List


class VideoEnginePromptGenerator:
    # Unified prompt footer macro (kept T2I-safe and concise)
    PROMPT_FOOTER = "Stable composition. Preserve existing background content."

    @staticmethod
    def _merge_parent_and_variant(parent_prompt: str, variant_prompt: str) -> str:
        """
        Build a concise 'current appearance' descriptor.
        - Take base identity from the first clause of parent (split by ',' or '.').
        - If variant exists, append it as authoritative updates.
        - Else, return the parent as-is.
        """
        parent = (parent_prompt or "").strip()
        variant = (variant_prompt or "").strip()

        if not parent and not variant:
            return "the character"

        if not variant:
            return parent  # no changes; use parent

        # Extract base identity from parent (first clause)
        base = parent
        for sep in [",", "."]:
            if sep in base:
                base = base.split(sep, 1)[0].strip()
                break
        if not base:
            base = parent.strip()

        # Compose current appearance
        return f"Character: {base}; {variant}"

    # -----------------------------
    # Character: Text-to-Image
    # -----------------------------
    @staticmethod
    def build_character_generator_prompt(
        description: str,
        prompt: str,
        style: str,
        headshot_only: bool = False
    ):
        """
        Generic Qwen prompt builder for any 'character' (human/non-human).
        Plate-friendly: neutral background, centered subject, clean silhouette.
        Uses only description, prompt, style. (No name.)
        """

        # Normalize
        description = (description or "").strip()
        user_prompt = (prompt or "").strip()
        style_str = (style or "").strip()

        # Subject
        subject_bits = []
        if user_prompt:
            subject_bits.append(user_prompt)
        if description:
            subject_bits.append(description)

        framing = (
            "tight portrait, centered, head and shoulders fully visible, no cropping"
            if headshot_only else
            "entire subject fully visible, head-to-toe (or complete form), centered, ample margin, no cropping"
        )

        # Scene (flat neutral for easy keying)
        scene = (
            "plain white seamless studio background (flat, uniform; no gradient or pattern); "
            "clean floor with a soft contact shadow only; no on-image text or logos"
        )

        # Style
        style_block = style_str or "clean, contemporary rendering"

        # Composition / Lighting
        composition = "natural perspective at subject height; centered framing; whitespace for easy cropping"
        atmosphere  = "even soft key plus subtle rim light for edge separation; no fog/particles/motion blur"

        # Constraints and quality reinforcement
        constraints = (
            "clean, crisp silhouette; balanced colors; avoid complex shadows bleeding into background. "
            "Enhance micro-detail on the subject (hair, fabric, materials) while keeping tones natural."
        )

        lines = [
            "## Subject",
            f"{'; '.join(subject_bits)}; {framing}".strip("; "),
            "## Scene",
            scene,
            "## Style",
            style_block,
            "## Composition",
            composition,
            "## Atmosphere",
            atmosphere,
            "## Constraints",
            constraints
        ]

        return "\n".join(lines).strip()

    # -----------------------------
    # Set / Environment: Text-to-Image
    # -----------------------------
    @staticmethod
    def build_set_generation_prompt(
        prompt: str,
        style: str,
        width: int,
        height: int,
        character_anchor: str = "center"  # kept for compatibility; unused
    ):
        """
        Character-agnostic plate generator (no people).
        Neutral lighting, strong depth, sharp focus, minimal distortion.
        """
        base = (prompt or "").strip()
        style_str = (style or "").strip()

        # Aspect hint (concise)
        if width and height and height > 0:
            ar = width / float(height)
            if ar > 1.05:
                ar_hint = "horizontal composition with cinematic balance"
            elif ar < 0.95:
                ar_hint = "vertical composition"
            else:
                ar_hint = "square composition with centered balance"
        else:
            ar_hint = "balanced composition"

        parts = []
        if base:
            parts.append(base)

        # Scene / composition / cleanliness
        parts.extend([
            "detailed environment",
            "plate-ready background",
            "no people visible",
            ar_hint,
            "natural perspective with minimal distortion; straight verticals",
            "spacious layout; readable depth (foreground, midground, background)",
            f"Style: {style_str}" if style_str else "realistic architectural photography style",
            "even, professional lighting; neutral color grading",
            "sharp focus throughout",
            "clean lines and materials; subtle reflections only (avoid mirror-like reflections)",
            "no on-image text, logos, or signage",
            "no fog, no haze, no particles, no motion blur"
        ])

        return ". ".join(parts).strip()

    # -----------------------------
    # Set Variant: Image-to-Image
    # -----------------------------
    @staticmethod
    def build_set_variant_prompt(variant_prompt: str, variant_style: str) -> str:
        """
        I2I set enhancement. Edit-scope + preservation to avoid background drift.
        """
        prompt = (variant_prompt or "Enhance the environment").strip()
        style  = (variant_style or "photorealistic, professional photography").strip()

        after = (
            "Maintain the original layout, geometry, materials, and lighting. "
            "Improve local clarity and texture fidelity in soft or low-quality areas. "
            "Preserve all other regions identical to the input. "
            "Avoid global grading or stylization changes to untouched areas."
        )

        return f"{prompt}. Apply {style} style. {after}"

    # -----------------------------
    # Character Variant: Image-to-Image
    # -----------------------------
    @staticmethod
    def build_character_variant_prompt(variant_prompt: str, variant_style: str, variant_action: str) -> str:
        """
        I2I character edit: apply BOTH appearance/orientation updates (variant_prompt)
        AND pose/action (variant_action) when provided.
        Generic, model-friendly wording with identity lock, framing, white background,
        and preservation clauses to prevent non-edited degradation.
        """
        vprompt = (variant_prompt or "").strip()
        action  = (variant_action or "").strip()
        style   = (variant_style or "").strip()

        parts = []

        # Apply both updates if available; otherwise fall back gracefully
        if vprompt and action:
            parts.append("Apply both updates simultaneously")
            parts.append(f"Appearance & orientation update: {vprompt}")
            parts.append(f"Pose/action: {action}")
        elif vprompt:
            parts.append(f"Modify character appearance and orientation: {vprompt}")
        elif action:
            parts.append(f"Change pose/action: {action}")
        else:
            parts.append("Adjust character appearance")

        # Identity / framing / background
        parts.extend([
            "Keep identity: preserve face, hairstyle, clothing design, colors, and body proportions",
            "Framing: keep the subject fully visible within frame boundaries, centered composition, adequate headroom and footroom, no cropping",
            "Background: plain white seamless; isolated subject; no objects or environment"
        ])

        # Preservation & targeted enhancement
        parts.extend([
            "Edit scope: modify only the character; do not alter any other part of the image",
            "Preserve all pixels outside the character identical to the input",
            "Avoid global color grading, contrast, or sharpening on untouched areas",
            "Where the character appears soft or low-quality, enhance local detail and clarity without changing the approved design"
        ])

        # Optional style carry-over
        if style:
            parts.append(f"Maintain style: {style}")

        return ". ".join(parts)

    # -----------------------------
    # Keyframe Composition: Image-to-Image
    # -----------------------------
    @staticmethod
    def build_keyframe_composition_prompt(
        characters_prompts: List[str], 
        characters_actions: List[str], 
        characters_placements: List[str],
    ) -> str:
        """
        Compose multiple characters into a cohesive scene (edit on board/set).
        Uses plain language placements; adds preservation to protect the set.
        """
        n = len(characters_prompts)
        lines = []
        lines.append("Modify this image to create a cohesive scene. Only include the specified characters.")

        # Character directives
        for subj, act, pos in zip(characters_prompts, characters_actions, characters_placements):
            subj_txt = (subj or "the character").strip()
            act_txt  = (act or "appropriate action").strip()
            pos_txt  = (pos or "appropriate placement").strip()
            lines.append(f"{subj_txt}: {act_txt}; place at {pos_txt}.")

        # Global guidance + preservation + targeted enhancement
        lines.extend([
            f"Show exactly {n} character(s); no extras.",
            "Keep identities locked (face, hairstyle, skin tone, clothing).",
            "Maintain scene lighting, perspective, and geometry.",
            "Edit scope: modify only the characters and their immediate contact shadows.",
            "Preserve the background, furniture, and all other regions identical to the input.",
            "Avoid global grading or stylization on untouched areas.",
            "Where the characters appear soft or low-resolution, improve local detail and clarity without changing scene appearance."
        ])

        return " ".join(lines).strip()
    
    @staticmethod
    def build_keyframe_composition_prompt_v2(
        parent_character_prompts: List[str],
        variant_character_prompts: List[str],
        characters_actions: List[str],
        characters_placements: List[str],
    ) -> str:
        """
        Compose multiple characters into a cohesive scene using the *current* look.
        - Merges parent identity (first clause) + variant deltas to avoid regressions.
        - Applies action and placement per character.
        - Includes preservation language to protect the set.
        """
        n = len(parent_character_prompts)
        lines = []
        lines.append("Modify this image to create a cohesive scene. Only show the specified characters.")

        for parent_p, variant_p, act, pos in zip(
            parent_character_prompts,
            variant_character_prompts,
            characters_actions,
            characters_placements
        ):
            descriptor = VideoEnginePromptGenerator._merge_parent_and_variant(parent_p, variant_p)
            subj_txt = (descriptor or "the character").strip()
            act_txt  = (act or "appropriate action").strip()
            pos_txt  = (pos or "appropriate placement").strip()

            # Explicitly mark the variant as authoritative
            lines.append(f"{subj_txt}. Pose/action: {act_txt}. Place at {pos_txt}.")

        lines.extend([
            f"Show exactly {n} character(s); no extras.",
            "Keep identities locked (face, hairstyle, skin tone, clothing as described).",
            "Maintain scene lighting, perspective, and geometry.",
            "Edit scope: modify only the characters and their immediate contact shadows.",
            "Preserve the background, furniture, and all other regions identical to the input.",
            "Avoid global grading or stylization on untouched areas.",
            "Where the characters appear soft or low-resolution, improve local detail and clarity without changing the described appearance."
        ])

        return " ".join(lines).strip()