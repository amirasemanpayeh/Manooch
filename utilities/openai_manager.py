import base64
from io import BytesIO
import json
import re
from PIL import Image
import tempfile
import time  
import logging
import webcolors
from typing import Any, Dict, List, Optional, Union
from openai import OpenAI # type: ignore
from fastapi import APIRouter, HTTPException, Query
import requests
from app.utils.settings_manager import settings  # Adjust the import path as needed
from app.utils.logging_config import get_openai_manager_logger

# Use the emoji logger
logger = get_openai_manager_logger("OpenAIManager")

class OpenAIManager:
    _instance = None
    
    def __new__(cls, api_key: str = None):
        if cls._instance is None:
            cls._instance = super(OpenAIManager, cls).__new__(cls)
            cls._instance.logger = get_openai_manager_logger("OpenAIManager")
            if api_key:
                cls._instance.client = OpenAI(api_key=api_key)
            else:
                raise ValueError("API key must be provided for the first initialization")
        return cls._instance
    
    def set_instance(self, instance):
        if self._instance is None:  # Ensure it's only set once
            self._instance = instance

    @classmethod
    def get_instance(self):
        if self._instance is None:
            raise Exception("OpenAIManager has not been initialized")
        return self._instance
    
    def extract_json_from_response(content: str):
        # Use regular expression to find the JSON part
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return match.group(0)
        return None

    def generate_image_gpt_image_1(self, prompt: str, quality: str = "medium", size: str = "1024x1536") -> Optional[bytes]:
        try:
            response = self.client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size=size,
                # background="transparent",  # Uncomment if needed
                quality=quality,
            )

            # --- Validation ---
            if not response or not hasattr(response, "data"):
                logger.error("No data in image edit response.")
                return None

            if not isinstance(response.data, list) or len(response.data) == 0:
                logger.error("Empty data list in image edit response.")
                return None

            image_base64 = getattr(response.data[0], "b64_json", None)
            if not image_base64 or not isinstance(image_base64, str):
                logger.error("Error: Missing or invalid base64 image string in response.")
                return None

            # Decode base64 to raw bytes
            image_bytes = base64.b64decode(image_base64)

            return image_bytes

        except Exception as error:
            logger.error(f"Exception in generate_image_gpt_image_1: {error}")
            return None

    def edit_image_gpt_image_1(
            self,
            images: List[str],
            prompt: str,
            size: str = "1024x1536",
            quality: str = "medium",
            background: str = "auto",
        ) -> Optional[bytes]:
            """
            Uses GPT-image-1 to edit one or more PNG images.
            :param images: List of raw PNG bytes.
            :param prompt: Description of edits.
            :param size: "1024x1536", etc.
            :param quality: "high", "medium", "low".
            :param background: "transparent", "opaque", "auto".
            :returns: List of edited PNG bytes, or None on error.
            """
            temp_paths = []

            try:

                # Step 1: Write each image to disk
                for idx, img_bytes in enumerate(images):
                    path = f"/tmp/input_image_{idx}.png"
                    with open(path, "wb") as f:
                        f.write(img_bytes)
                    temp_paths.append(path)

                # Step 2: Open those files like OpenAI example
                open_files = [open(p, "rb") for p in temp_paths]


                response = self.client.images.edit(
                    model="gpt-image-1",
                    image=open_files,
                    prompt=prompt,
                    size=size,  # Size can be adjusted as needed
                    n=1,
                    quality=quality,
                    background=background,
                )

                # Validate
                if not response or not hasattr(response, "data"):
                    logger.error("No data in image edit response.")
                    return None
                if not isinstance(response.data, list) or not response.data:
                    logger.error("Empty data list in image edit response.")
                    return None

                image_base64 = getattr(response.data[0], "b64_json", None)
                if not image_base64 or not isinstance(image_base64, str):
                    logger.error("Error: Missing or invalid base64 image string in response.")
                    return None

                # Decode base64 to raw bytes
                image_bytes = base64.b64decode(image_base64)
                return image_bytes

            except Exception as e:
                logger.error(f"Exception in edit_image_gpt_image_1: {e}")
                return None

    def ask_chatgpt(self, system_message: str, assistant_message: str, user_prompt: str):
        # Here we create a request to OpenAI's API
        #   Chat models take a list of messages as input and return a model-generated message as output.
        #   Although the chat format is designed to make multi-turn conversations easy, it’s just as useful for single-turn tasks without any conversation.
        #   The main input is the messages parameter.
        #   Messages must be an array of message objects, where each object has a role (either "system", "user", or "assistant") and content.
        #   Conversations can be as short as one message or many back and forth turns.
        #   Typically, a conversation is formatted with a system message first, followed by alternating user and assistant messages.
        #
        #   "system" role : The system message helps set the behavior of the assistant. 
        #        For example, you can modify the personality of the assistant or provide specific instructions about how it should behave throughout the conversation
        #   "user" role: The user messages provide requests or comments for the assistant to respond to.
        #   "assistant" role: Assistant messages store previous assistant responses, but can also be written by you to give examples of desired behavior.
        #       Including conversation history is important when user instructions refer to prior messages.
        #       In the example below, the user’s final question of "Where was it played?"
        #       only makes sense in the context of the prior messages about the World Series of 2020. \
        #       Because the models have no memory of past requests, all relevant information must be supplied as part of the conversation history in each request.    
        #
        #           messages=[
        #                {"role": "system", "content": "You are a helpful assistant."},
        #                {"role": "user", "content": "Who won the world series in 2020?"},
        #                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        #                {"role": "user", "content": "Where was it played?"}
        #            ] 
        #

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ]
                
            )
            self.logger.debug("Response data from AI")
            self.logger.debug(f"AI Response: {response}")
            # Assuming `response` is the object printed out
            if hasattr(response, 'choices') and len(response.choices) > 0:
                # Access the content property of the ChatCompletionMessage object
                content = response.choices[0].message.content

                try:
                    # Extract the JSON part from the content
                    json_str = OpenAIManager.extract_json_from_response(content)
                    if json_str:
                        response_data = json.loads(json_str)
                        return response_data
                    else:
                        self.logger.warning("No JSON found in the content")
                        return "No JSON found in the content"
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse content as JSON")
                    return "Failed to parse content as JSON"
            else:
                return "No response choices found."
        except Exception as error:
            return f"Error processing the response: {str(error)}"

    
    def generate_system_message(input_data: str, brand_info):
        def decimal_to_hex_color(value) -> str:
            """Converts a decimal color value to a hex color string."""
            try:
                # Ensure value is a string and strip any spaces
                value = str(value).strip()

                # Convert to an integer (base 10)
                decimal_value = int(value, 10)

                # Convert to 8-character hex (includes alpha channel)
                hex_value = f"{decimal_value:08X}"

                # Drop the first two characters (alpha channel)
                rgb_hex = hex_value[2:]

                # Return formatted hex color
                return f"#{rgb_hex.lower()}"

            except (ValueError, TypeError):
                raise ValueError(f"Invalid input: '{value}' is not a valid decimal color.")

        def closest_css3_color(hex_color: str) -> str:
            """Find the closest named CSS3 color for a given hex code."""
            try:
                # First, try to find an exact match
                return webcolors.hex_to_name(hex_color, spec='css3')
            except ValueError:
                # Convert hex to RGB
                r, g, b = webcolors.hex_to_rgb(hex_color)

                # Find the closest CSS3 color by computing RGB distance
                min_distance = float('inf')
                closest_color = None

                # Use the new webcolors API to get CSS3 color names
                for name in webcolors.names(webcolors.CSS3):
                    try:
                        css_hex = webcolors.name_to_hex(name, spec='css3')
                        css_r, css_g, css_b = webcolors.hex_to_rgb(css_hex)
                        distance = ((r - css_r) ** 2) + ((g - css_g) ** 2) + ((b - css_b) ** 2)

                        if distance < min_distance:
                            min_distance = distance
                            closest_color = name
                    except ValueError:
                        # Skip if color name conversion fails
                        continue

                return closest_color        # Convert decimal colors to hex
        primary_color_hex = decimal_to_hex_color(brand_info.primary_color)
        secondary_color_hex = decimal_to_hex_color(brand_info.secondary_color)
        neutral_color_hex = decimal_to_hex_color(brand_info.neutral_color)
        accent_color_hex = decimal_to_hex_color(brand_info.accent_color)

        # Find closest CSS3 color names
        primary_color_name = closest_css3_color(primary_color_hex)
        secondary_color_name = closest_css3_color(secondary_color_hex)
        neutral_color_name = closest_css3_color(neutral_color_hex)
        accent_color_name = closest_css3_color(accent_color_hex)

        # print all colour names
        #print(primary_color_name)
        #print(secondary_color_name)
        #print(neutral_color_name)
        #print(accent_color_name) 

        return f"""
        You are an expert advertising content creator and copy writer, with a focus on social media advertising. 
        You consider brand information and the objectives of a marketing campaign to write ads and create content. 
        You are very creative. You currently work for a business with the following brand information:

        Name: {brand_info.name}
        Industry: {brand_info.industry}
        Slogan: {brand_info.slogan}
        Description: {brand_info.description}
        Unique Selling Proposition: {brand_info.unique_selling_proposition}
        Mission and Vision: {brand_info.mission_vision}
        Personality Trait: {brand_info.personality_trait}
        Voice Style: {brand_info.voice_style}
        Key Messages, Benefits, Features: {brand_info.key_messages_benefits_features}
        Logo URL: {brand_info.logo_url}
        Primary Font: {brand_info.primary_font}
        Secondary Font: {brand_info.secondary_font}
        Primary Color: {primary_color_hex}
        Secondary Color: {secondary_color_hex}
        Neutral Color: {neutral_color_hex}
        Accent Color: {accent_color_hex}
        Specific Ad Theme Guidelines: {brand_info.specific_ad_theme_guidelines}
        Address: {brand_info.address}
        Telephone: {brand_info.telephone}
        Email: {brand_info.email}
        Website URL: {brand_info.website_url}
        """

    def generate_brand_data_prompt(self, brand_info, traits_string: str, voice_styles_string: str):
        return f"""
        Based on the following brand information, fill in any missing details to create a complete brand profile:

        Name: {brand_info.name}
        Industry: {brand_info.industry}
        Slogan: {brand_info.slogan}
        Description: {brand_info.description}
        Unique Selling Proposition: {brand_info.unique_selling_proposition}
        Mission and Vision: {brand_info.mission_vision}
        Personality Trait: {brand_info.personality_trait}
        Voice Style: {brand_info.voice_style}
        Key Messages, Benefits, Features: {brand_info.key_messages_benefits_features}
        Logo URL: {brand_info.logo_url}
        Primary Font: {brand_info.primary_font}
        Secondary Font: {brand_info.secondary_font}
        Primary Color: {brand_info.primary_color}
        Secondary Color: {brand_info.secondary_color}
        Neutral Color: {brand_info.neutral_color}
        Accent Color: {brand_info.accent_color}
        Specific Ad Theme Guidelines: {brand_info.specific_ad_theme_guidelines}
        Address: {brand_info.address}
        Telephone: {brand_info.telephone}
        Email: {brand_info.email}
        Website URL: {brand_info.website_url}

        Please review, improve and complete the brand profile. 
        Add missing colours to be used for advertising visual content. 
        Use Google Fonts for the primary and secondary fonts.

        For personality_trait:
        Ensure to choose and copy the exact text that you think is most suitable for the brand from the following lists for industry, personality trait, and voice style. Do not create lists, just provide single strings:
        {traits_string}

        For voice_style:
        Ensure to choose and copy the exact text that you think is most suitable for the brand from the following lists for industry, personality trait, and voice style. Do not create lists, just provide single strings:
        {voice_styles_string}

        Return in the following JSON format:
        {{
            "name": "Brand Name",
            "industry": "Industry",
            "slogan": "Slogan",
            "description": "Description",
            "unique_selling_proposition": "Unique Selling Proposition",
            "mission_vision": "Mission and Vision",
            "personality_trait": "Personality Trait",
            "voice_style": "Voice Style",
            "key_messages_benefits_features": "Key Messages, Benefits, Features",
            "primary_font": "Primary Font",
            "secondary_font": "Secondary Font",
            "primary_color": "Primary Color",
            "secondary_color": "Secondary Color",
            "neutral_color": "Neutral Color",
            "accent_color": "Accent Color",
            "specific_ad_theme_guidelines": "Specific Ad Theme Guidelines",
        }}
        """
    
    
    def get_openai_response(self, system_message, user_message, expected_keys, max_retries=3, retry_delay=5):
        """
        Calls OpenAI API with retry logic and returns the response data with status.

        Args:
            openai_manager: The OpenAI manager instance to call the API.
            system_message (str): The system prompt/message for the OpenAI request.
            user_message (str): The user input for the OpenAI request.
            expected_keys (list): List of expected keys in the response.
            max_retries (int): Maximum number of retry attempts.
            retry_delay (int): Delay in seconds before retrying.

        Returns:
            tuple: (response_data, status)
                - response_data (dict or None): Parsed response data from OpenAI.
                - status (str): "success" if response is valid, else "failed".
        """

        # Ensure max_retries is an integer
        if not isinstance(max_retries, int):
            raise TypeError(f"max_retries should be an integer, but got {type(max_retries).__name__}: {max_retries}")

        # Ensure retry_delay is an integer or float (time.sleep requires a number)
        if not isinstance(retry_delay, (int, float)):
            raise TypeError(f"retry_delay should be a number, but got {type(retry_delay).__name__}: {retry_delay}")

        # Ensure expected_keys is a list of strings
        if not isinstance(expected_keys, list) or not all(isinstance(key, str) for key in expected_keys):
            raise TypeError(f"expected_keys should be a list of strings, but got {type(expected_keys).__name__}: {expected_keys}")

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Attempt {attempt + 1}: Sending request to OpenAI")

                # Call the OpenAI API
                response = self.ask_chatgpt(system_message, "", user_message)

                # Check if the response is empty or None
                if not response:
                    raise ValueError("Empty response received")

                # Parse response as a dictionary
                if isinstance(response, str):
                    response_data = json.loads(response)
                else:
                    response_data = response

                # Validate response contains all expected keys
                if all(key in response_data for key in expected_keys):
                    return response_data, "success"

                raise ValueError(f"Response missing expected keys: {expected_keys}")

            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Error parsing response (attempt {attempt + 1}): {e}")

                if attempt < max_retries - 1:
                    time.sleep(retry_delay)  # Wait before retrying
                else:
                    self.logger.error("Max retries reached. Unable to process response.")

        return None, "failed"

# Module-level instance initialization
# Initialize SupabaseManager
openai_manager = OpenAIManager(api_key=settings.openai_api_key)

def set_openai_manager(instance):
    openai_manager.set_instance(instance)

def get_openai_manager():
    return openai_manager.get_instance()