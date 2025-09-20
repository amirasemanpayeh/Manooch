from enum import Enum
import logging
import json
import random
import uuid
import asyncio
import websockets
from datetime import datetime, timezone

from supabase import create_client
from typing import Any, Dict, Iterable, Literal, Optional, List
from models.basic_models import (
    AdSpecModel,
    AppNotificationModel,
    BrandModel,
    CreditGrantBatchModel,
    CreditTransactionModel,
    InviteTokenModel,
    PostModel,
    PostQuestionnaireModel,
    UserProfileModel,
    WaitListUserModel,
)

from utils.settings_manager import settings  # Adjusted import path for this repo



def get_supabase_logger(name: str = "SupabaseManager") -> logging.Logger:
    """Create or fetch a configured logger for Supabase components.

    - Ensures a single stream handler with a simple format.
    - Respects an optional `settings.log_level` if present, defaults to INFO.
    """
    log = logging.getLogger(name)

    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)

        level_name = getattr(settings, "log_level", "INFO")
        level = getattr(logging, str(level_name).upper(), logging.INFO)
        log.setLevel(level)
        # Avoid duplicate logs if root logger has handlers
        log.propagate = False

    return log


# Initialize logger
logger = get_supabase_logger("SupabaseManager")

def log_realtime(message: str, level: str = 'info'):
    """Structured logging for realtime events"""
    if settings.realtime_enable_structured_logging:
        getattr(logger, level.lower())(message)
    else:
        logger.info(message)

def debug_print(message: str):
    """Gate verbose debug prints behind setting"""
    if settings.realtime_verbose_debug_prints:
        log_realtime(f"Debug: {message}", 'debug')



class SupabaseRealtimeClient:
    def __init__(self, url, api_key):
        self.url = url.replace('https://', 'wss://').replace('.supabase.co', '.supabase.co/realtime/v1/websocket')
        self.api_key = api_key
        self.websocket = None
        self.callbacks = {}
        self.channels = {}
        self.heartbeat_task = None  # 🆕 Track the heartbeat task
        self._connection_lock = asyncio.Lock()  # 🆕 Prevent concurrent connection attempts

    def _is_websocket_connected(self) -> bool:
        """
        Safely check if the websocket is connected.
        
        Returns:
            bool: True if websocket is connected and open, False otherwise
        """
        try:
            if not self.websocket:
                return False
            
            # Use state property for websockets library compatibility
            return self.websocket.state.name == 'OPEN'
        except AttributeError:
            # Fallback for different websockets library versions
            try:
                return hasattr(self.websocket, 'close') and not getattr(self.websocket, 'closed', True)
            except:
                return False
        except Exception:
            return False

    async def connect(self):
        """Connect to the Supabase Realtime server."""
        # Use lock to prevent concurrent connection attempts
        async with self._connection_lock:
            # Small delay before reconnection attempts for stability
            if self.websocket and self._is_websocket_connected():
                return  # Already connected
                
            await asyncio.sleep(settings.realtime_connection_retry_delay)
            
            params = f"apikey={self.api_key}&vsn=1.0.0"
            ws_url = f"{self.url}?{params}"

            try:
                self.websocket = await websockets.connect(ws_url)
                log_realtime("✅ Connected to server")

                # Start listening for messages
                asyncio.create_task(self._message_handler())

                # Cancel old heartbeat task if any
                if self.heartbeat_task and not self.heartbeat_task.done():
                    self.heartbeat_task.cancel()
                    try:
                        await self.heartbeat_task
                    except asyncio.CancelledError:
                        log_realtime("🛑 Old heartbeat task cancelled")

                # Start new heartbeat task
                self.heartbeat_task = asyncio.create_task(self._send_heartbeat())

                # Re-subscribe to all existing channels
                # Create a copy of channels to avoid "dictionary changed size during iteration" error
                channels_copy = dict(self.channels)
                if channels_copy:
                    log_realtime(f"🔄 Re-subscribing to {len(channels_copy)} existing channels...")
                    for channel_id, channel_info in channels_copy.items():
                        try:
                            callback_key = f"{channel_info['schema']}-{channel_info['table']}-{channel_info['event']}"
                            if callback_key in self.callbacks:
                                await self.subscribe(
                                    event=channel_info['event'],
                                    schema=channel_info['schema'],
                                    table=channel_info['table'],
                                    callback=self.callbacks[callback_key]
                                )
                                log_realtime(f"✅ Re-subscribed to {channel_info['schema']}.{channel_info['table']} ({channel_info['event']})")
                            else:
                                log_realtime(f"⚠️ Missing callback for {callback_key}", 'warning')
                        except Exception as e:
                            log_realtime(f"❌ Failed to re-subscribe to {channel_info['schema']}.{channel_info['table']}: {e}", 'error')
                else:
                    log_realtime("ℹ️ No existing channels to re-subscribe")

            except Exception as e:
                # Specifically handle the dictionary iteration error
                if "dictionary changed size during iteration" in str(e):
                    log_realtime(f"⚠️ Dictionary mutation error during reconnection (fixed): {e}", 'warning')
                else:
                    log_realtime(f"❌ Connection error: {e}", 'error')
                # Add backoff delay to prevent tight retry loops
                await asyncio.sleep(2)
                # Don't raise the exception to allow reconnection attempts to continue


    async def _send_heartbeat(self):
        """Send heartbeat messages to keep the connection alive."""
        try:
            while True:
                try:
                    heartbeat = {
                        "topic": "phoenix",
                        "event": "heartbeat",
                        "payload": {},
                        "ref": str(uuid.uuid4())
                    }
                    await self.websocket.send(json.dumps(heartbeat))
                    await asyncio.sleep(30)
                except websockets.exceptions.ConnectionClosed:
                    log_realtime("❌ WebSocket connection closed — stopping heartbeat", 'warning')
                    break
                except Exception as e:
                    log_realtime(f"❤️‍🔥 Heartbeat error: {e}", 'error')
                    await asyncio.sleep(5)
        except Exception as e:
            log_realtime(f"Heartbeat loop exited due to: {e}", 'error')
        log_realtime("💔 Heartbeat loop ended")


    async def _message_handler(self):
        """Handle incoming WebSocket messages."""
        while True:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)

                if data.get('event') == 'postgres_changes' and 'payload' in data:
                    outer_payload = data.get('payload', {})
                    inner_payload = outer_payload.get('data', {})

                    schema = inner_payload.get('schema')
                    table = inner_payload.get('table')
                    event_type = inner_payload.get('type')

                    channel_key = f"{schema}-{table}-{event_type}"

                    if channel_key in self.callbacks:
                        try:
                            callback = self.callbacks[channel_key]
                            if asyncio.iscoroutinefunction(callback):
                                await callback(inner_payload)
                            else:
                                callback(inner_payload)
                        except Exception as callback_error:
                            log_realtime(f"❌ Callback error for {channel_key}: {callback_error}", 'error')

            except websockets.exceptions.ConnectionClosed:
                log_realtime("🔌 Disconnected — reconnecting...", 'warning')
                await asyncio.sleep(2)  # avoid fast loop
                await self.connect()
                break  # exit this handler; a new one will be spawned on reconnect

            except Exception as e:
                log_realtime(f"❗ Message handling error: {e}", 'error')
                await asyncio.sleep(1)

    async def subscribe(self, event, schema, table, callback):
        """Subscribe to table changes."""
        if self.websocket is None:
            await self.connect()

        channel_key = f"{schema}-{table}-{event}"
        
        # Check if we're already subscribed to this channel
        # Avoid duplicate subscriptions during reconnection
        existing_channel_id = None
        for ch_id, ch_info in self.channels.items():
            if (ch_info['event'] == event and 
                ch_info['schema'] == schema and 
                ch_info['table'] == table):
                existing_channel_id = ch_id
                break
        
        # If already subscribed, just update the callback
        if existing_channel_id:
            self.callbacks[channel_key] = callback
            log_realtime(f"🔄 Updated callback for existing subscription: {schema}.{table} ({event})")
            return existing_channel_id

        channel_id = str(uuid.uuid4())
        self.callbacks[channel_key] = callback
        
        # Store channel info for reconnection
        self.channels[channel_id] = {
            'event': event,
            'schema': schema,
            'table': table
        }

        # The correct topic format for Supabase Realtime
        channel_topic = f"realtime:{schema}:{table}"

        # Step 1: Join the channel with the correct topic
        join_message = {
            "topic": channel_topic,
            "event": "phx_join",
            "payload": {
                "config": {
                    "postgres_changes": [
                        {
                            "event": event,
                            "schema": schema,
                            "table": table
                        }
                    ]
                }
            },
            "ref": str(uuid.uuid4())
        }
        
        await self.websocket.send(json.dumps(join_message))
        logger.info(f"Joined channel: {channel_topic}")

        return channel_id

    async def disconnect(self):
        """
        Gracefully disconnect from the Supabase Realtime server.
        This method should be called during application shutdown.
        """
        logger.info("Starting graceful disconnect...")
        
        try:
            # Cancel heartbeat task if running
            if self.heartbeat_task and not self.heartbeat_task.done():
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                    log_realtime("🛑 Heartbeat task cancelled")
                except asyncio.CancelledError:
                    pass  # Expected
                except Exception as e:
                    log_realtime(f"⚠️ Error cancelling heartbeat: {e}", 'warning')
            
            # Close WebSocket connection
            if self.websocket and self._is_websocket_connected():
                await self.websocket.close()
                log_realtime("🛑 WebSocket connection closed")
            
            # Clear state
            self.websocket = None
            self.heartbeat_task = None
            # Note: We keep callbacks and channels for potential reconnection
            
            log_realtime("✅ Graceful disconnect completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            return False

    def get_connection_status(self) -> dict:
        """
        Get current connection status and health information.
        Useful for monitoring and debugging.
        """
        return {
            'connected': self._is_websocket_connected(),
            'heartbeat_running': self.heartbeat_task is not None and not self.heartbeat_task.done(),
            'channels_count': len(self.channels),
            'callbacks_count': len(self.callbacks),
            'websocket_state': str(getattr(self.websocket, 'state', 'None')) if self.websocket else 'None'
        }


class SupabaseManager:

    _instance = None

    def __new__(cls, url: str = None, key: str = None):
        if cls._instance is None:
            cls._instance = super(SupabaseManager, cls).__new__(cls)
            # Initialize logger for the class
            cls._instance.logger = get_supabase_logger("SupabaseManager")
            if url and key:
                cls._instance.client = create_client(url, key)
                cls._instance.realtime_client = SupabaseRealtimeClient(url, key)
            else:
                raise ValueError("URL and Key must be provided for the first initialization")
        return cls._instance

    def set_instance(self, instance):
        if self._instance is None:  # Ensure it's only set once
            self._instance = instance

    def get_instance(self):
        if self._instance is None:
            raise Exception("SupabaseManager has not been initialized")
        return self._instance

    def login(self, email: str, password: str):
        response = self.client.auth.sign_in_with_password({"email": email, "password": password})
        return response

    # ==============================
    # Waitlist users table functions
    # ==============================
    def get_waitlist_users_with_status(self, status: str) -> List[dict]:
        try:
            response = self.client.table('waitlist_user').select("*").eq("status", status).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error fetching waitlist users with status '{status}': {e}")
            return []
        
    def update_waitlist_user_all_fields(self, waitListUser: WaitListUserModel) -> bool:
        try:
            update_fields = {
                k: v for k, v in waitListUser.to_dict().items()
                if v not in [None, ''] and k != 'id'
            }
            response = self.client.table('waitlist_user')\
                                .update(update_fields)\
                                .eq('id', str(waitListUser.id))\
                                .execute()

            # ✅ Correct way to handle result from supabase-py
            if hasattr(response, 'error') and response.error:
                self.logger.error(f"Supabase error updating waitlist user: {response.error}")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Error updating waitlist user: {e}")
            return False

    def get_waitlist_user_by_id(self, user_id: str) -> Optional[dict]:
        """
        Get a single waitlist user by their ID.
        
        Args:
            user_id (str): The ID of the waitlist user to fetch
            
        Returns:
            Optional[dict]: The waitlist user data if found, None otherwise
        """
        try:
            response = self.client.table('waitlist_user').select("*").eq("id", user_id).single().execute()
            if response.data:
                return response.data
            return None
        except Exception as e:
            self.logger.error(f"Error fetching waitlist user by ID '{user_id}': {e}")
            return None
        
    # ==============================
    #  Invite Token Table Functions
    # ==============================
    def create_invite_token(self, invitation_token: InviteTokenModel) -> Optional[str]:
        try:
            insert_fields = {
                k: v for k, v in invitation_token.to_dict().items()
                if v not in [None, ''] and k != 'id'
            }
            response = self.client.table('invite_token').insert(insert_fields).execute()
            if response.data and isinstance(response.data, list) and "id" in response.data[0]:
                return response.data[0]["id"]
            return None
        except Exception as e:
            self.logger.error(f"Error creating invite token: {e}")
            return None

    # ==============================
    #  Brand Table Functions
    # ==============================
    
    def get_brands(self):
        try:
            response = self.client.table('brand').select("*").execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error fetching brands: {e}")
            return None

    def get_brand_by_user_id(self, user_id: str):
        try:
            response = self.client.table('brand').select("*").eq('user_id', user_id).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error fetching brand for user_id {user_id}: {e}")
            return None

    def get_brand_by_id(self, brand_id: str) -> Optional[BrandModel]:
        """
        Get a single brand by its ID.
        
        Args:
            brand_id (str): The ID of the brand to fetch
            
        Returns:
            Optional[BrandModel]: The brand model if found, None otherwise
        """
        try:
            response = self.client.table('brand').select("*").eq('id', brand_id).single().execute()
            if response.data:
                return BrandModel.to_obj(response.data)
            return None
        except Exception as e:
            self.logger.error(f"Error fetching brand by ID '{brand_id}': {e}")
            return None
        

    def create_brand(self, brandData: BrandModel) -> Optional[str]:
        fields_to_insert = {
            k: v.value if isinstance(v, Enum) else v
            for k, v in brandData.to_dict().items()
            if v not in [None, ''] and k not in ['id', 'created_at', 'updated_at']
        }

        try:
            response = self.client.table("brand").insert(fields_to_insert).execute()
            self.logger.info(f"Supabase insert response: {response}")
            return response.data[0]["id"] if response.data else None
        except Exception as e:
            self.logger.error(f"Supabase insert failed with exception: {e}")
            # Log HTTP response from supabase-py (if available)
            if hasattr(e, 'response'):
                self.logger.error(f"Supabase response: {e.response.text}")
            raise

        
    def get_brands_list_with_status(self, status: str) -> List[BrandModel]:
        """
        Fetches a list of BrandModel instances from the 'brand' table filtered by the given main status.
        
        Args:
            status (str): The status to filter brands by (e.g. 'New', 'Ready').

        Returns:
            List[BrandModel]: A list of brand objects with the specified status.
        """
        try:
            response = self.client.table('brand').select("*").eq("status", status).execute()

            if response.data and isinstance(response.data, list):
                return [BrandModel.to_obj(item) for item in response.data]
            else:
                return []

        except Exception as e:
            self.logger.error(f"Error fetching brands with status '{status}': {e}")
            return []
    
    def update_brand_status(self, brand_id: str, status: str) -> bool:
        try:
            self.client.table('brand').update({'status': status}).eq('id', brand_id).execute()
            return True
        except Exception as e:
            self.logger.error(f"Error updating brand status: {e}")
            return False
        
    def update_brand_assistant_status(self, brand_id: str, assistant_status: str) -> bool:
        try:
            self.client.table('brand').update({'assistant_status': assistant_status}).eq('id', brand_id).execute()
            return True
        except Exception as e:
            self.logger.error(f"Error updating brand assistant status: {e}")
            return False


    def update_brand_all_fields(self, brandData: BrandModel) -> bool:
        try:
            update_fields = {
                k: v.value if isinstance(v, Enum) else v
                for k, v in brandData.to_dict().items()
                if v not in [None, ''] and k != 'id'
            }

            self.client.table('brand').update(update_fields).eq('id', str(brandData.id)).execute()
            return True
        except Exception as e:
            self.logger.error(f"Error updating brand: {e}")
            return False

    # ==============================
    #  onboarding_questionnaire Table Functions
    # ==============================

    def get_onboarding_questionnaire_for_user(self, user_id: str):
        """Fetch the onboarding questionnaire for a specific user with error handling."""
        try:
            if not user_id:
                raise ValueError("Invalid user_id: It cannot be empty.")

            response = self.client.table('onboarding_questionnaire').select("*").eq('user_id', user_id).execute()

            error = getattr(response, 'error', None)
            if error:
                self.logger.info(f"Error fetching onboarding questionnaire for user {user_id}: {error}")
                return {"error": str(error)}

            if response.data:
                return response.data
            else:
                self.logger.info(f"No onboarding questionnaire found for user ID: {user_id}")
                return None

        except Exception as e:
            self.logger.info(f"Exception in get_onboarding_questionnaire_for_user({user_id}): {e}")
            return None

    def get_onboarding_questionnaire_by_status(self, status: str):
        """Fetch onboarding questionnaires by status with error handling."""
        try:
            if not status:
                raise ValueError("Invalid status: It cannot be empty.")

            response = self.client.table('onboarding_questionnaire').select("*").eq('status', status).execute()

            if response.data:
                return response.data
            else:
                return None

        except Exception as e:
            self.logger.info(f"Exception in get_onboarding_questionnaire_by_status({status}): {e}")
            return None

    def update_onboarding_questionnaire_status(self, id: str, status: str):
        """Update the status of an onboarding questionnaire with error handling."""
        try:
            if not id or not status:
                raise ValueError("Invalid input: ID and status cannot be empty.")

            response = self.client.table('onboarding_questionnaire').update({'status': status}).eq('id', id).execute()

            if response.data:
                return response.data
            else:
                self.logger.info(f"Failed to update onboarding questionnaire status for ID: {id}")
                return None

        except Exception as e:
            self.logger.info(f"Exception in update_onboarding_questionnaire_status({id}, {status}): {e}")
            return None

    def get_onboarding_questionnaire_by_id(self, questionnaire_id: str) -> Optional[dict]:
        """
        Get a single onboarding questionnaire by its ID.
        
        Args:
            questionnaire_id (str): The ID of the questionnaire to fetch
            
        Returns:
            Optional[dict]: The questionnaire data if found, None otherwise
        """
        try:
            response = self.client.table('onboarding_questionnaire').select("*").eq('id', questionnaire_id).single().execute()
            if response.data:
                return response.data
            return None
        except Exception as e:
            self.logger.error(f"Error fetching onboarding questionnaire by ID '{questionnaire_id}': {e}")
            return None
        
    # ==============================
    # Target Audience Table Functions
    # ==============================
    def get_target_audience_by_id(self, id: str):
        try:
            response = self.client.table('target_audience').select("*").eq('id', id).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error fetching target audience by ID {id}: {e}")
            return None
        

    # ==============================
    #  Post Questionnaire Table Functions
    # ==============================

    def get_post_questionnaire_by_status(self, status: str):
        try:
            response = self.client.table('post_questionnaire').select("*").eq('status', status).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error fetching post questionnaires by status: {e}")
            return None

    def get_post_questionnaire_by_id(self, id: str):
        """Fetch a single post questionnaire by its ID."""
        try:
            response = self.client.table('post_questionnaire').select("*").eq('id', id).execute()
            if response.data:
                return PostQuestionnaireModel.to_obj(response.data[0])  # Convert to model object
            return None
        except Exception as e:
            self.logger.error(f"Error fetching post questionnaire by ID {id}: {e}")
            return None

    def update_post_questionnaire_status(self, id: str, status: str):
        """Update the status of a post questionnaire."""
        try:
            response = self.client.table('post_questionnaire').update({"status": status}).eq('id', id).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error updating post questionnaire status: {e}")
            return None

    def update_post_questionnaire_failure_reason(self, id: str, failure_reason: str):
        """Update the failure reason for a post questionnaire."""
        try:
            response = self.client.table('post_questionnaire').update({"failure_reason": failure_reason}).eq('id', id).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error updating failure reason for post questionnaire {id}: {e}")
            return None

    def increment_post_questionnaire_generation_counter(self, id: str):
        """Increments the generation counter for a post questionnaire."""
        try:
            response = self.client.rpc("increment_generation_counter", {"questionnaire_id": id}).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error incrementing generation counter for post questionnaire {id}: {e}")
            return None

    def update_post_questionnaire_linked_ad_spec(self, id: str, ad_spec_id: str):
        """Update the linked AdSpec ID for a post questionnaire."""
        try:
            response = self.client.table('post_questionnaire').update({"ad_spec_id": ad_spec_id}).eq('id', id).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error updating linked ad_spec_id: {e}")
            return None

    def update_post_questionnaire(self, id: str, fields_to_update: dict):
        """Update multiple fields in a post questionnaire."""
        try:
            # Ensure data integrity with PostQuestionnaireModel
            valid_fields = {
                "post_title",
                "objective",
                "call_to_action",
                "audience_emotion",
                "key_message",
                "ad_spec_id",
                "user_id",
                "status",
                "failure_reason",
                "generation_counter",
            }
            fields_to_update = {k: v for k, v in fields_to_update.items() if k in valid_fields and v is not None}

            response = self.client.table('post_questionnaire').update(fields_to_update).eq('id', id).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error updating post questionnaire {id}: {e}")
            return None
    
    # ==============================
    #  Ad Spec Table Functions
    # ==============================
    def create_new_ad_spec(self, adSpecData: AdSpecModel) -> str:
        fields_to_insert = {k: v for k, v in adSpecData.to_dict().items() if v not in [None, '']}
        response = self.client.table('ad_spec').insert(fields_to_insert).execute()
        return str(response.data[0]['id'])

    def update_ad_spec(self, id: str, adSpecData: AdSpecModel):
        # Convert AdSpec object to dictionary
        fields_to_update = adSpecData.to_dict()

        #   Debugging: Check the raw dictionary before filtering out None values
        self.logger.info(f"Raw fields before filtering: {fields_to_update}")

        # Remove fields that are None or empty to avoid accidental deletion in the DB
        fields_to_update = {k: v for k, v in fields_to_update.items() if v not in [None, '']}

        #   Debugging: Check what fields remain after filtering
        self.logger.debug(f"Fields after filtering: {fields_to_update}")

        # Proceed with the update only if there are fields to update
        if not fields_to_update:
            self.logger.info("⚠️ No valid fields to update in AdSpec, skipping database update.")
            return None

        # Perform the update operation
        response = self.client.table('ad_spec').update(fields_to_update).eq('id', id).execute()
        
        #   Debugging: Check response from the database
        self.logger.debug(f"Supabase response: {response}")
        
        return response.data
    
    def increment_ad_spec_generation_counter(self, ad_spec_id: str):
        """
        Fetches the current generation_counter for an AdSpec, increments it by 1, and updates it in Supabase.

        Args:
            ad_spec_id (str): The ID of the AdSpec whose counter should be incremented.

        Returns:
            int | None: The new generation counter value if successful, None if failed.
        """
        self.logger.info(f"🔄 Incrementing generation counter for AdSpec {ad_spec_id} in Supabase...")

        try:
            # ✅ Convert UUID to string if necessary
            if isinstance(ad_spec_id, uuid.UUID):
                ad_spec_id = str(ad_spec_id)

            # ✅ Step 1: Fetch the current counter
            response = self.client.table("ad_spec").select("generation_counter").eq("id", ad_spec_id).execute()

            if not response.data or len(response.data) == 0:
                self.logger.info(f"❌ No AdSpec found with ID {ad_spec_id}.")
                return None

            current_counter = response.data[0].get("generation_counter", 0)  # Default to 0 if None
            new_counter = current_counter + 1

            # ✅ Step 2: Update the counter
            update_response = self.client.table("ad_spec").update({"generation_counter": new_counter}).eq("id", ad_spec_id).execute()

            # Debugging: Check the update response
            self.logger.info(f"Update response: {update_response}")

            if not update_response.data or len(update_response.data) == 0:
                self.logger.info(f"❌ Failed to increment generation counter for {ad_spec_id}.")
                return None

            self.logger.info(f"Successfully incremented generation counter for {ad_spec_id}. New value: {new_counter}")
            return new_counter  # ✅ Return the updated counter value

        except Exception as e:
            self.logger.error(f"Exception while incrementing generation counter for {ad_spec_id}: {e}")
            return None

    def get_brands_by_status(self, status: str):
        response = self.client.table('brand').select("*").eq('status', status).execute()
        return response.data

    def get_ad_spec_by_status(self, status: str):
        response = self.client.table('ad_spec').select("*").eq('status', status).execute()
        return response.data

    def get_ad_spec_by_id(self, id: str):

        try:
            response = self.client.table('ad_spec').select("*").eq('id', id).execute()
            if response.data:
                return AdSpecModel.to_obj(response.data[0])  # Convert to model object
            return None
        except Exception as e:
            self.logger.error(f"Error fetching ad spec by ID {id}: {e}")
            return None
        
    def update_ad_spec_status(self, id: str, new_status: str):
        """Update the status of a ad spec."""
        try:
            response = self.client.table('ad_spec').update({"status": new_status}).eq('id', id).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error updating post questionnaire status: {e}")
            return None

    def update_ad_spec_failure_reason(self, id: str, failure_reason: str):
        """Update the failure reason for a ad spec."""
        try:
            response = self.client.table('ad_spec').update({"failure_reason": failure_reason}).eq('id', id).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error updating failure reason for ad spec {id}: {e}")
            return None

    # ==============================
    #  Post Table Functions
    # ==============================

    def get_post_by_assistant_status(self, status: str):
        response = self.client.table('post').select("*").eq('assistant_status', status).execute()
        return response.data

    def get_post_by_main_status(self, status: str):
        response = self.client.table('post').select("*").eq('main_status', status).execute()
        return response.data

    def create_new_post(self, postData: PostModel):
        fields_to_insert = {k: v for k, v in postData.to_dict().items() if v not in [None, '']}
        
        self.logger.info(f"Inserting post with fields = {fields_to_insert}")

        response = self.client.table('post').insert(fields_to_insert).execute()

        # Log the response for debugging
        self.logger.info(f"Supabase response = {response}")

        # Ensure response.data is valid before accessing it
        if not response.data or not isinstance(response.data, list) or len(response.data) == 0:
            self.logger.info("❌ Error: No data returned from Supabase for post insert.")
            return None  # Return None to indicate failure

        return response.data  # Return full response, keeping expected format

    def get_post_id_by_ad_spec_id(self, adSpecId: str):
        response = self.client.table('post').select("*").eq('ad_spec_id', adSpecId).execute()
        if response.data:
            return response.data  # ✅ Return the full post object instead of just the ID
        return None

    def get_post_by_id(self, post_id: str) -> Optional[dict]:
        """
        Get a single post by its ID.
        
        Args:
            post_id (str): The ID of the post to fetch
            
        Returns:
            Optional[dict]: The post data if found, None otherwise
        """
        try:
            response = self.client.table('post').select("*").eq('id', post_id).single().execute()
            if response.data:
                return response.data
            return None
        except Exception as e:
            self.logger.error(f"Error fetching post by ID '{post_id}': {e}")
            return None

    def update_post(self, postData: PostModel):
        """Update a post with the provided data while handling potential errors."""
        try:
            fields_to_update = {k: v for k, v in postData.to_dict().items() if v not in [None, '']}
            
            if 'id' not in fields_to_update:
                self.logger.info("❌ Error: 'id' field is missing from postData.")
                return None
            
            response = self.client.table('post').update(fields_to_update).eq('id', fields_to_update['id']).execute()
            
            if not response.data or len(response.data) == 0:
                self.logger.info(f"❌ Error updating post {fields_to_update['id']}: No data returned from Supabase.")
                return None

            self.logger.info(f"Successfully updated post {fields_to_update['id']}.")
            return response.data

        except Exception as e:
            self.logger.error(f"Exception while updating post {postData.id}: {e}")
            return None
    
    def update_post_main_status(self, id: str, new_status: str):
        """Update the status of a post."""
        try:
            response = self.client.table('post').update({"main_status": new_status}).eq('id', id).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error updating post questionnaire status: {e}")
            return None

    def update_post_assistant_status(self, id: str, new_status: str):
        """Update the status of a post."""
        try:
            response = self.client.table('post').update({"assistant_status": new_status}).eq('id', id).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error updating post questionnaire status: {e}")
            return None
        
    def update_post_failure_reason(self, id: str, failure_reason: str):
        """Update the failure reason for a post."""
        try:
            response = self.client.table('post').update({"failure_reason": failure_reason}).eq('id', id).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error updating failure reason for post {id}: {e}")
            return None
        
    def increment_post_generation_counter(self, post_id: str):
        """
        Fetches the current generation_counter for an post, increments it by 1, and updates it in Supabase.

        Args:
            post_id (str): The ID of the post whose counter should be incremented.

        Returns:
            int | None: The new generation counter value if successful, None if failed.
        """
        self.logger.info(f"🔄 Incrementing generation counter for post {post_id} in Supabase...")

        try:
            # ✅ Convert UUID to string if necessary
            if isinstance(post_id, uuid.UUID):
                post_id = str(post_id)

            # ✅ Step 1: Fetch the current counter
            response = self.client.table("post").select("generation_counter").eq("id", post_id).execute()

            if not response.data or len(response.data) == 0:
                self.logger.info(f"❌ No Post found with ID {post_id}.")
                return None

            current_counter = response.data[0].get("generation_counter", 0)  # Default to 0 if None
            new_counter = current_counter + 1

            # ✅ Step 2: Update the counter on the correct table ('post')
            update_response = self.client.table("post").update({"generation_counter": new_counter}).eq("id", post_id).execute()

            # Debugging: Check the update response
            self.logger.info(f"Update response: {update_response}")

            if not update_response.data or len(update_response.data) == 0:
                self.logger.info(f"❌ Failed to increment generation counter for {post_id}.")
                return None

            self.logger.info(f"Successfully incremented generation counter for {post_id}. New value: {new_counter}")
            return new_counter  # ✅ Return the updated counter value

        except Exception as e:
            self.logger.error(f"Exception while incrementing generation counter for {post_id}: {e}")
            return None

    # ============================
    #  POST AI ASSISTANCE LEVEL Table Functions
    # ============================    
    def get_post_ai_assistance_level_by_id(self, id: str):
        """Fetch a post_ai_assistance_level entry by ID with error handling."""
        try:
            if not id:
                raise ValueError("Invalid ID: It cannot be empty.")

            response = self.client.table('post_ai_assistance_level').select("*").eq('id', id).execute()

            if response.data:
                return response.data
            else:
                self.logger.info(f"No post_ai_assistance_level entry found for ID: {id}")
                return None

        except Exception as e:
            self.logger.info(f"Exception in get_post_ai_assistance_level_by_id({id}): {e}")
            return None
        
    # ============================
    # POST Generation style Table Functions
    # ============================
    def get_post_generation_style_by_id(self, id: str):
        """Fetch a post generation style by its ID with error handling."""
        try:
            if not id:
                raise ValueError("Invalid ID: ID cannot be empty.")

            response = self.client.table('post_generation_style').select("*").eq('id', id).execute()

            if response.data:
                return response.data
            else:
                return None

        except Exception as e:
            return None
        
    # ==============================
    #  Post Template Table Functions
    # ==============================

    def get_post_template_by_id(self, id: str):
        """Fetch a post template by its ID with error handling."""
        try:
            if not id:
                raise ValueError("Invalid ID: ID cannot be empty.")

            response = self.client.table('post_template').select("*").eq('id', id).execute()

            if response.data:
                return response.data
            else:
                return None

        except Exception as e:
            return None

    def get_post_template_random(self):
        """Fetch a random post template ID with error handling."""
        try:
            response = self.client.table('post_template').select("id").execute()

            if response.data and len(response.data) > 0:
                random_id = random.choice(response.data)['id']
                return random_id
            else:
                return None

        except Exception as e:
            return None
        
    
    # ==============================
    #  Asset Related  Functions
    # ==============================
    def upload_generated_asset_img(self, img_bytes: bytes):
        # Generate a unique filename for the image
        filename = f"{uuid.uuid4()}.jpg"

        # Upload the image to Supabase bucket
        bucket_name = "post-images"  # Replace with your actual bucket name
        res = self.client.storage.from_(bucket_name).upload(filename, img_bytes)

        # Get the URL of the uploaded image
        url = self.client.storage.from_(bucket_name).get_public_url(filename)

        return url
    

    def upload_processed_asset_img(self, img_bytes: bytes):
        # Generate a unique filename for the image
        filename = f"{uuid.uuid4()}.jpg"

        # Upload the image to Supabase bucket
        bucket_name = "post-processed-assets"
        res = self.client.storage.from_(bucket_name).upload(filename, img_bytes)

        # Get the URL of the uploaded image
        url = self.client.storage.from_(bucket_name).get_public_url(filename)

        return url
    
    def upload_processed_asset_video(self, video_bytes: bytes):
        # Generate a unique filename for the video
        filename = f"{uuid.uuid4()}.mp4"

        # Upload the video to Supabase bucket
        bucket_name = "generated_videos"
        res = self.client.storage.from_(bucket_name).upload(filename, video_bytes)

        # Get the URL of the uploaded video
        url = self.client.storage.from_(bucket_name).get_public_url(filename)

        return url
    
    def upload_processed_asset_audio(self, audio_bytes: bytes):
        # Generate a unique filename for the audio
        filename = f"{uuid.uuid4()}.wav"

        # Upload the audio to Supabase bucket
        bucket_name = "generated_audios"
        res = self.client.storage.from_(bucket_name).upload(filename, audio_bytes)

        # Get the URL of the uploaded audio
        url = self.client.storage.from_(bucket_name).get_public_url(filename)

        return url
    
    # ==============================
    #  Social Media Profile Table Functions
    # ==============================

    def insert_social_media_profile(self, platform: str, access_token: str,expires_in:datetime,user_id: str, status: str,
                                    profile_data: any):
        self.logger.info("Insert social profile started")

        # profile data has id,name
        # check if record exits
        profileID = profile_data['id']

        data,count = self.client.table('social_media_profile').select('id').eq('profile_data ->> id',profileID).execute()
        if(len(data[1]) != 0):
            # update the record
            # 'data' is a tuple
            self.logger.info("Social profile exists")

            rowID = data[1][0]['id']

            data,count = self.client.table('social_media_profile').update({'access_token':access_token,'expires_in':expires_in.isoformat(),'status': status}).eq('id',rowID).execute()
            
            self.logger.info("Social profile updated")
        else:
            data, count = self.client.table('social_media_profile').insert(
                {'platform': platform, 'access_token': access_token,'expires_in':expires_in.isoformat(),'user_id': user_id, 'status': status,'profile_data':profile_data}).execute()
            
            self.logger.info("New social profile created")
            self.logger.debug(f"Profile data: {data}")


    # ==============================
    # Notification System Related Functions
    # ==============================
    def create_app_notification(self, notification: AppNotificationModel) -> Optional[AppNotificationModel]:
        fields_to_insert = {
            k: v.value if isinstance(v, Enum) else v
            for k, v in notification.to_dict().items()
            if v not in [None, ''] and k not in ['created_at']
        }
        try:
            res = self.client.table("app_notification").insert(fields_to_insert).execute()
            data = getattr(res, 'data', None)
            
            # Handle different response formats
            if data is not None:
                if isinstance(data, list) and len(data) > 0:
                    return AppNotificationModel.to_obj(data[0])
                elif isinstance(data, dict):
                    return AppNotificationModel.to_obj(data)
            
            # Check for successful creation even without data
            if res and getattr(res, 'status_code', None) == 201:
                # HTTP 201 Created but no data returned (common for REST API)
                self.logger.info(f"[create_app_notification] Notification created successfully (status 201)")
                return notification
            else:
                self.logger.info(f"[create_app_notification] No data returned: {data}, status: {getattr(res, 'status_code', 'unknown')}")
        except Exception as e:
            self.logger.info(f"[create_app_notification] Error: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        return None

    def update_app_notification(self, notification: AppNotificationModel) -> Optional[AppNotificationModel]:
        if not notification.id:
            self.logger.info("[update_app_notification] Missing notification ID.")
            return None

        update_fields = {
            k: v.value if isinstance(v, Enum) else v
            for k, v in notification.to_dict().items()
            if v not in [None, ''] and k not in ['id', 'created_at']
        }
        try:
            res = self.client.table("app_notification").update(update_fields).eq("id", str(notification.id)).execute()
            if res.data and len(res.data) > 0:
                return AppNotificationModel.to_obj(res.data[0])
            return None
        except Exception as e:
            self.logger.info(f"[update_app_notification] Error: {e}")
            return None


    # ==============================
    # Credit System Related Functions
    # ==============================
    def create_credit_transaction(self, tx: CreditTransactionModel) -> Optional[CreditTransactionModel]:
        fields_to_insert = {
            k: v.value if isinstance(v, Enum) else v
            for k, v in tx.to_dict().items()
            if v not in [None, ''] and k not in ['id', 'created_at']
        }
        try:
            res = self.client.table("credit_transaction").insert(fields_to_insert).execute()
            if res.data and len(res.data) > 0:
                return CreditTransactionModel.to_obj(res.data[0])
        except Exception as e:
            self.logger.info(f"[create_credit_transaction] Error: {e}")
        return None

    def update_credit_transaction(self, tx: CreditTransactionModel) -> bool:
        if not tx.id:
            self.logger.info("[update_credit_transaction] Missing transaction ID.")
            return False

        update_fields = {
            k: v.value if isinstance(v, Enum) else v
            for k, v in tx.to_dict().items()
            if v not in [None, ''] and k not in ['id', 'created_at']
        }
        try:
            res = self.client.table("credit_transaction").update(update_fields).eq("id", str(tx.id)).execute()
            if res.data and len(res.data) > 0:
                return CreditTransactionModel.to_obj(res.data[0])
        except Exception as e:
            self.logger.info(f"[update_credit_transaction] Error: {e}")
            return False

    def get_credit_transaction_by_id(self, id: str) -> Optional[CreditTransactionModel]:
        try:
            res = self.client.table("credit_transaction").select("*").eq("id", id).single().execute()
            if res.data:
                return CreditTransactionModel.to_obj(res.data)
        except Exception as e:
            self.logger.info(f"[get_credit_transaction_by_id] Error: {e}")
        return None

    def get_credit_transaction_by_attachment_id(self, attachment_id: str) -> Optional[CreditTransactionModel]:
        try:
            res = self.client.table("credit_transaction").select("*").eq("attachment_id", attachment_id).single().execute()
            if res.data:
                return CreditTransactionModel.to_obj(res.data)
        except Exception as e:
            self.logger.info(f"[get_credit_transaction_by_attachment_id] Error: {e}")
        return None

    def get_credit_transactions_by_user_id(self, user_id: str) -> List[CreditTransactionModel]:
        try:
            res = self.client.table("credit_transaction").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
            return [CreditTransactionModel.to_obj(row) for row in res.data] if res.data else []
        except Exception as e:
            self.logger.info(f"[get_credit_transactions_by_user_id] Error: {e}")
            return []

    def get_credit_transactions_by_status(self, status: str) -> List[CreditTransactionModel]:
        """Get all credit transactions by status - useful for sweeping pending transactions."""
        try:
            res = self.client.table("credit_transaction").select("*").eq("status", status).order("created_at", desc=False).execute()
            return [CreditTransactionModel.to_obj(row) for row in res.data] if res.data else []
        except Exception as e:
            self.logger.info(f"[get_credit_transactions_by_status] Error: {e}")
            return []
        

    def create_credit_grant_batch(self, batch: CreditGrantBatchModel) -> Optional[CreditGrantBatchModel]:
        fields_to_insert = {
            k: v.value if isinstance(v, Enum) else v
            for k, v in batch.to_dict().items()
            if v not in [None, ''] and k not in ['created_at']
        }
        try:
            res = self.client.table("credit_grant_batch").insert(fields_to_insert).execute()
            if res.data and len(res.data) > 0:
                return CreditGrantBatchModel.to_obj(res.data[0])
        except Exception as e:
            self.logger.info(f"[create_credit_grant_batch] Error: {e}")
        return None

    def update_credit_grant_batch(self, batch: CreditGrantBatchModel) -> bool:
        if not batch.id:
            self.logger.info("[update_credit_grant_batch] Missing batch ID.")
            return False

        update_fields = {
            k: v.value if isinstance(v, Enum) else v
            for k, v in batch.to_dict().items()
            if v not in [None, ''] and k not in ['id', 'created_at']
        }
        try:
            res = self.client.table("credit_grant_batch").update(update_fields).eq("id", str(batch.id)).execute()
            if res.data and len(res.data) > 0:
                return CreditGrantBatchModel.to_obj(res.data[0])
        except Exception as e:
            self.logger.info(f"[update_credit_grant_batch] Error: {e}")
            return False

    def get_credit_grant_batch_by_id(self, id: str) -> Optional[CreditGrantBatchModel]:
        try:
            res = self.client.table("credit_grant_batch").select("*").eq("id", id).single().execute()
            if res.data:
                return CreditGrantBatchModel.to_obj(res.data)
        except Exception as e:
            self.logger.info(f"[get_credit_grant_batch_by_id] Error: {e}")
        return None

    def get_credit_grant_batches_by_user_id(self, user_id: str) -> List[CreditGrantBatchModel]:
        try:
            res = self.client.table("credit_grant_batch").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
            return [CreditGrantBatchModel.to_obj(row) for row in res.data] if res.data else []
        except Exception as e:
            self.logger.info(f"[get_credit_grant_batches_by_user_id] Error: {e}")
            return []

    # ==============================
    # Profile Table Functions
    # ==============================
    def get_user_profile_by_id(self, id: str) -> Optional[UserProfileModel]:
        try:
            res = self.client.table("profile").select("*").eq("id", id).single().execute()
            if res.data:
                return UserProfileModel.to_obj(res.data)
        except Exception as e:
            self.logger.info(f"[get_user_profile_by_id] Exception: {e}")
        return None

    def update_user_profile_credits(self, id: str, credits: int) -> bool:
        try:
            res = self.client.table("profile").update({"credits": credits}).eq("id", id).execute()
            return (res.data and len(res.data) > 0)

        except Exception as e:
            self.logger.info(f"[update_user_profile_credits] Exception: {e}")
            return False

    def increment_user_profile_credits(self, id: str, amount: int) -> Optional[int]:
        try:
            res = self.client.table("profile").select("credits").eq("id", id).single().execute()
            if res.error:
                self.logger.info(f"[increment_user_profile_credits] Supabase error on fetch: {res.error.message}")
                return None
            if not res.data:
                self.logger.info(f"[increment_user_profile_credits] No user profile found for ID {id}")
                return None

            current_credits = int(res.data.get("credits", 0))
            new_credits = current_credits + amount
            update_res = self.client.table("profile").update({"credits": new_credits}).eq("id", id).execute()

            if update_res.error:
                self.logger.info(f"[increment_user_profile_credits] Supabase error on update: {update_res.error.message}")
                return None
            if update_res.data and len(update_res.data) > 0:
                return new_credits
        except Exception as e:
            self.logger.info(f"[increment_user_profile_credits] Exception: {e}")
        return None
    
    
    #=============================
    # Tasks Table Functions
    #=============================
    def create_task_row(self, task_type: str, entity_id: str, context: dict) -> dict:
        """Create a new task row in the tasks table with consistent UTC timestamps."""
        try:
            # CRITICAL: Use timezone-aware UTC timestamps consistently
            now = datetime.now(timezone.utc).isoformat()
            
            fields = {
                "task_type": task_type,
                "entity_id": str(entity_id),
                "context": context or {},
                "status": "Pending",
                "progress": 0,  # Explicitly set as int
                "current_step": "",
                "worker_id": None,
                "error_message": None,
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "finished_at": None,
            }
            res = self.client.table("tasks").insert(fields).execute()
            if res.data and isinstance(res.data, list) and len(res.data) > 0:
                return res.data[0]
            return None
        except Exception as e:
            self.logger.info(f"[create_task_row] Error: {e}")
            return None

    def get_latest_task_by_entity(self, entity_id: str, task_type: Optional[str] = None) -> Optional[dict]:
        """Fetch the latest task for an entity (optionally filtered by type)."""
        try:
            query = self.client.table("tasks").select("*").eq("entity_id", str(entity_id))
            if task_type:
                query = query.eq("task_type", task_type)
            res = query.order("created_at", desc=True).limit(1).execute()
            if res.data and len(res.data) > 0:
                return res.data[0]
            return None
        except Exception as e:
            self.logger.info(f"[get_latest_task_by_entity] Error: {e}")
            return None

    def get_task_by_id(self, task_id: str) -> Optional[dict]:
        """Fetch a single task by its ID for full hydration support."""
        try:
            res = self.client.table("tasks").select("*").eq("id", str(task_id)).single().execute()
            if res.data:
                return res.data
            return None
        except Exception as e:
            self.logger.info(f"[get_task_by_id] Error: {e}")
            return None

    def update_task_mark_failed(self, task_id: str, error_message: str) -> Optional[dict]:
        """Mark a task as failed."""
        try:
            # Use timezone-aware UTC timestamp
            now = datetime.now(timezone.utc).isoformat()
            fields = {
                "status": "Failed",
                "error_message": error_message,
                "finished_at": now,
                "updated_at": now,
            }
            res = self.client.table("tasks").update(fields).eq("id", str(task_id)).execute()
            if res.data and len(res.data) > 0:
                return res.data[0]
            return None
        except Exception as e:
            self.logger.info(f"[update_task_mark_failed] Error: {e}")
            return None

    # Comprehensive update - syncs ALL task fields to database.
    def update_task_comprehensive(self, task_id: str, status: str, progress: int, current_step: str, worker_id: Optional[str], error_message: Optional[str], started_at: Optional[str], finished_at: Optional[str]) -> Optional[dict]:
        """
        Comprehensive task update - syncs ALL task fields to database.
        This ensures the database stays in sync with the in-memory task state.
        
        Args:
            task_id: The task ID to update
            status: Task status (Pending, Running, Succeeded, Failed, Cancelled)
            progress: Progress percentage (0-100)
            current_step: Current processing step description
            worker_id: ID of worker processing the task (None if no worker)
            error_message: Error message if task failed (None if no error)
            started_at: ISO timestamp when task started (None if not started)
            finished_at: ISO timestamp when task finished (None if not finished)
            
        Returns:
            Updated task row dict if successful, None if failed
        """
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            # Build comprehensive update with all fields
            fields = {
                "status": status,
                "progress": max(0, min(100, int(progress))),  # Ensure valid smallint range
                "current_step": current_step or "",
                "worker_id": worker_id,  # Can be None - that's fine
                "error_message": error_message,  # Can be None - that's fine
                "updated_at": now,
            }
            
            # Only set started_at if provided and not already set (avoid unnecessary updates)
            if started_at is not None:
                # Check if we need to fetch current state to avoid overwriting existing started_at
                current_task = self.get_task_by_id(task_id)
                if not current_task or not current_task.get('started_at'):
                    fields["started_at"] = started_at
            
            # Always set finished_at if provided (task completion)
            if finished_at is not None:
                fields["finished_at"] = finished_at
            
            res = self.client.table("tasks").update(fields).eq("id", str(task_id)).execute()
            if res.data and len(res.data) > 0:
                return res.data[0]
            return None
        except Exception as e:
            self.logger.info(f"[update_task_comprehensive] Error: {e}")
            return None

    def release_worker_from_task(self, task_id: str) -> bool:
        """
        Release the worker from a task by clearing the worker_id.
        Task remains in current status but available for other workers.
        
        Args:
            task_id: The task ID to release worker from
            
        Returns:
            True if worker was released successfully
        """
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            fields = {
                "worker_id": None,  # Clear the worker assignment
                "updated_at": now,
            }
            
            res = self.client.table("tasks").update(fields).eq("id", str(task_id)).execute()
            
            # Check for Supabase errors
            if hasattr(res, 'error') and res.error:
                self.logger.info(f"[release_worker_from_task] Supabase error: {res.error}")
                return False
            
            # Check if any rows were actually updated
            if not res.data or len(res.data) == 0:
                self.logger.info(f"[release_worker_from_task] No rows updated for task_id: {task_id}")
                return False
            
            return True
        except Exception as e:
            self.logger.info(f"[release_worker_from_task] Error: {e}")
            return False

    def get_active_tasks_for_startup(self) -> List[dict]:
        """
        Get all active (Pending/Running) tasks for application startup.
        Returns only the latest task per (task_type, entity_id) to avoid loading stale duplicates.
        Ordered by created_at DESC to prefer newest when deduplicating.
        CRITICAL: Only returns status IN ('Pending','Running') for correct hydration.
        """
        try:
            # Get all active tasks ordered by creation time (newest first) - CRITICAL for correct hydration
            active_statuses = ["Pending", "Running"]
            all_tasks = []
            
            for status in active_statuses:
                # Get tasks for this status, ordered by created_at DESC - CRITICAL ordering
                res = self.client.table("tasks").select("*").eq("status", status).order("created_at", desc=True).execute()
                if res.data:
                    all_tasks.extend(res.data)
            
            # Deduplicate to get only the LATEST task per (task_type, entity_id)
            # This prevents hydrating stale active tasks if duplicates exist
            latest_tasks = {}
            for task in all_tasks:  # Already sorted DESC, so first occurrence is newest
                key = f"{task.get('task_type')}:{task.get('entity_id')}"
                if key not in latest_tasks:
                    latest_tasks[key] = task
            
            deduped_tasks = list(latest_tasks.values())
            
            if len(deduped_tasks) != len(all_tasks):
                self.logger.warning(f" [get_active_tasks_for_startup] Deduped {len(all_tasks)} → {len(deduped_tasks)} active tasks (preferred newest)")
            
            return deduped_tasks
            
        except Exception as e:
            self.logger.info(f"[get_active_tasks_for_startup] Error: {e}")
            return []

    def set_task_context_by_entity(self, entity_id: str, context: dict, task_type: Optional[str] = None) -> bool:
        """
        Set the task context for an entity by finding its latest task.
        Used by TaskStore to sync context changes to database.
        
        Args:
            entity_id: The entity ID to update context for
            context: The new context dictionary
            task_type: Optional task type filter for targeting specific task
            
        Returns:
            True if context was updated successfully
        """
        try:
            # Find the latest task for this entity (optionally filtered by type)
            latest_task = self.get_latest_task_by_entity(entity_id, task_type)
            if not latest_task:
                self.logger.info(f"[set_task_context_by_entity] No task found for entity {entity_id}")
                return False
            
            # Single update operation for context and timestamp
            result = self.client.table("tasks").update({
                "context": context or {},
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", latest_task['id']).execute()
            
            # Check for Supabase errors in response
            if hasattr(result, 'error') and result.error:
                self.logger.info(f"[set_task_context_by_entity] Supabase error: {result.error}")
                return False
            
            # Check if any rows were actually updated
            if not result.data or len(result.data) == 0:
                self.logger.info(f"[set_task_context_by_entity] No rows updated for entity {entity_id}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.info(f"[set_task_context_by_entity] Error: {e}")
            return False

    def merge_task_context_by_entity(self, entity_id: str, updates: dict, task_type: Optional[str] = None) -> bool:
        """
        Merge updates into the existing task context for an entity.
        Used by TaskStore to update specific context keys without replacing everything.
        
        Args:
            entity_id: The entity ID to update context for
            updates: Dictionary of context updates to merge
            task_type: Optional task type filter for targeting specific task
            
        Returns:
            True if context was updated successfully
        """
        try:
            # Find the latest task for this entity (optionally filtered by type)
            latest_task = self.get_latest_task_by_entity(entity_id, task_type)
            if not latest_task:
                self.logger.info(f"[merge_task_context_by_entity] No task found for entity {entity_id}")
                return False
            
            # Get existing context and merge updates
            existing_context = latest_task.get('context', {})
            if isinstance(existing_context, dict):
                existing_context.update(updates or {})
            else:
                existing_context = updates or {}
            
            # Update the merged context
            result = self.client.table("tasks").update({
                "context": existing_context,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", latest_task['id']).execute()
            
            # Check for Supabase errors in response
            if hasattr(result, 'error') and result.error:
                self.logger.info(f"[merge_task_context_by_entity] Supabase error: {result.error}")
                return False
            
            # Check if any rows were actually updated
            if not result.data or len(result.data) == 0:
                self.logger.info(f"[merge_task_context_by_entity] No rows updated for entity {entity_id}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.info(f"[merge_task_context_by_entity] Error: {e}")
            return False
        

    def get_oldest_task_candidate(self, task_type: str, task_status: List[str]) -> Optional[dict]:
        """
        Returns a single 'tasks' row for the specified task type with status in ('Pending','Running'),
        ordered by created_at ASC. Prefer rows with worker_id IS NULL.
        """
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Prefer rows with no worker
                q = (self.client.table('tasks')
                     .select('*')
                     .eq('task_type', task_type)
                     .in_('status', task_status)
                     .is_('worker_id', 'null')
                     .order('created_at', desc=False)
                     .limit(1))
                resp = q.execute()
                if resp.data:
                    return resp.data[0]

                # Fallback: any Pending/Running
                resp = (self.client.table('tasks')
                        .select('*')
                        .eq('task_type', task_type)
                        .in_('status', task_status)
                        .order('created_at', desc=False)
                        .limit(1)
                        ).execute()
                return resp.data[0] if resp.data else None
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    self.logger.error(f"[SupabaseManager] Error finding task candidate after {max_retries} attempts: {e}")
                    return None
                else:
                    self.logger.warning(f"[SupabaseManager] Connection error on attempt {attempt + 1}/{max_retries}: {e}, retrying...")
                    import time
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
        
    
        
    #=============================
    # RPC function wrappers to handle entity updates with tasks transitions
    #=============================
    def pq_set_status_and_transition_task(
        self,
        questionnaire_id: str,
        new_status: str,
        failure_reason: Optional[str] = None,
        *,
        increment_generation: bool = False,
        task_status: str = "Running",   # 'Running' | 'Succeeded' | 'Failed'
        current_step: Optional[str] = None,
        release_worker: bool = True,
    ) -> bool:
        """
        Atomically:
        1) update post_questionnaire.status/failure_reason/generation_counter
        2) update matching tasks row (status/current_step/worker)
        """
        try:
            payload = {
                "p_questionnaire_id": questionnaire_id,
                "p_new_status": new_status,
                "p_failure_reason": failure_reason,
                "p_increment_generation": increment_generation,
                "p_task_status": task_status,
                "p_current_step": current_step,
                "p_release_worker": release_worker,
            }
            resp = self.client.rpc("pq_set_status_and_transition_task", payload).execute()
            err = getattr(resp, "error", None)
            if err:
                self.logger.error(f"RPC pq_set_status_and_transition_task({questionnaire_id}) error: {err}")
                return False
            data = getattr(resp, "data", None)
            if data is False:  # your RPC can return boolean false
                self.logger.info(f"❌ RPC pq_set_status_and_transition_task({questionnaire_id}) returned False")
                return False
            return True
        except Exception as e:
            self.logger.error(f"RPC pq_set_status_and_transition_task failed for {questionnaire_id}: {e}")
            return False

    def adspec_set_status_and_transition_task(
        self,
        ad_spec_id: str,
        new_status: str,
        failure_reason: Optional[str] = None,
        *,
        increment_generation: bool = False,
        task_status: str = "Running",   # 'Pending' | 'Running' | 'Succeeded' | 'Failed' | 'Cancelled'
        current_step: Optional[str] = None,
        release_worker: bool = True,
    ) -> bool:
        """
        Atomically:
        1) update ad_spec.status/failure_reason/generation_counter
        2) update matching tasks row (status/current_step/worker)
        """
        try:
            payload = {
                "p_adspec_id": ad_spec_id,
                "p_new_status": new_status,
                "p_failure_reason": failure_reason,
                "p_increment_generation": increment_generation,
                "p_task_status": task_status,
                "p_current_step": current_step,
                "p_release_worker": release_worker,
            }
            resp = self.client.rpc("as_set_status_and_transition_task", payload).execute()
            err = getattr(resp, "error", None)
            if err:
                self.logger.error(f"RPC as_set_status_and_transition_task({ad_spec_id}) error: {err}")
                return False
            data = getattr(resp, "data", None)
            if data is False:
                self.logger.info(f"❌ RPC as_set_status_and_transition_task({ad_spec_id}) returned False")
                return False
            return True
        except Exception as e:
            self.logger.error(f"RPC as_set_status_and_transition_task failed for {ad_spec_id}: {e}")
            return False
        
    @staticmethod
    def _prune_for_patch(d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Patch semantics:
          - Drop keys whose value is None.
          - Drop keys that are empty lists [] (avoid unintended clearing).
          - Keep empty strings if explicitly provided.
        """
        pruned = {}
        for k, v in d.items():
            if k == "id":  # always keep id
                pruned[k] = v
                continue
            if v is None:
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            pruned[k] = v
        return pruned

    def post_set_status_and_transition_task(
        self,
        post: PostModel, # PostModel
        *,
        task_status: str = "Running",           # 'Pending' | 'Running' | 'Succeeded' | 'Failed' | 'Cancelled'
        current_step: Optional[str] = None,
        release_worker: bool = True,
        increment_generation: bool = False,
        failure_reason: Optional[str] = None,   # overrides model.failure_reason if provided
        semantics: Literal["patch", "clear"] = "patch",
        only_keys: Optional[Iterable[str]] = None,  # limit which PostModel fields to send (besides id)
    ) -> bool:
        """
        Atomically update a Post row + its Task in one RPC.

        Args:
            post: PostModel instance. Must have a valid `id`.
            task_status: New Task status ('Pending'|'Running'|'Succeeded'|'Failed'|'Cancelled').
            current_step: Optional human-readable progress marker written to tasks.current_step.
            release_worker: If True, clears tasks.worker_id so other workers can pick up next stage.
            increment_generation: If True, atomically increments post.generation_counter in SQL.
            failure_reason: Optional reason string; when task_status='Failed', also stored in tasks.error_message.
            semantics:
                - "patch": Omit None/[] so untouched columns remain unchanged (default).
                - "clear": Keys you include will be applied literally:
                    * `None` → sets column to NULL
                    * `[]`   → sets array columns to an empty array
                Keys you do NOT include remain unchanged. Use `only_keys` to avoid touching unintended fields.
            only_keys: If provided, restricts which keys (from PostModel.to_dict()) are sent to the RPC
                       (the 'id' is always included). Useful with `semantics="clear"` to avoid clearing
                       default-empty arrays present on the model.

        Returns:
            bool: True if the RPC completed without error (DB and Task updated atomically), else False.
        """
        try:
            p_post: Dict[str, Any] = post.to_dict()
            if not p_post.get("id"):
                self.logger.info("❌ RPC post_set_status_and_transition_task: PostModel.id is required")
                return False

            # Optionally narrow the payload to a whitelist of keys (always include 'id').
            if only_keys is not None:
                allowed = set(only_keys) | {"id"}
                p_post = {k: v for k, v in p_post.items() if k in allowed}

            # Semantics handling:
            # - patch: prune None/[] so SQL coalesce keeps existing values
            # - clear: send as-is so SQL can set NULL/empty array where specified
            if semantics == "patch":
                p_post = self._prune_for_patch(p_post)
                p_clear_semantics = False
            else:
                p_clear_semantics = True  # respect explicit None/[] on provided keys

            payload = {
                "p_post": p_post,
                "p_task_status": task_status,
                "p_current_step": current_step,
                "p_release_worker": release_worker,
                "p_increment_generation": increment_generation,
                "p_failure_reason": failure_reason,
                "p_clear_semantics": p_clear_semantics,
            }

            resp = self.client.rpc("post_set_status_and_transition_task", payload).execute()
            err = getattr(resp, "error", None)
            if err:
                self.logger.error(f"RPC post_set_status_and_transition_task({p_post.get('id')}) error: {err}")
                return False
            data = getattr(resp, "data", None)
            if data is False:
                self.logger.info(f"❌ RPC post_set_status_and_transition_task({p_post.get('id')}) returned False")
                return False
            return True

        except Exception as e:
            self.logger.error(f"RPC post_set_status_and_transition_task failed for {getattr(post, 'id', 'unknown')}: {e}")
            return False

# Module-level instance initialization
supabase_manager = SupabaseManager(url=settings.supabase_url, key=settings.supabase_key)

def set_supabase_manager(instance):
    supabase_manager.set_instance(instance)

def get_supabase_manager():
    return supabase_manager.get_instance()
