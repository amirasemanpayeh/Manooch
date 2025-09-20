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
    

# Module-level instance initialization
supabase_manager = SupabaseManager(url=settings.supabase_url, key=settings.supabase_key)

def set_supabase_manager(instance):
    supabase_manager.set_instance(instance)

def get_supabase_manager():
    return supabase_manager.get_instance()
