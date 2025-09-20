import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

from dotenv import load_dotenv # type: ignore

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

class Settings(BaseSettings):
    # =============================================
    # CORE API SETTINGS
    # =============================================
    openai_api_key: str
    supabase_url: str
    supabase_key: str
    supabase_email: str
    supabase_password: str
    
    # =============================================
    # VIDEO GENERATION CONFIGURATION
    # =============================================
    
    # Image Editing Provider (openai or qwen) - applies to all I2I operations
    use_openai_for_image_editing: bool = Field(default=True, env="USE_OPENAI_FOR_IMAGE_EDITING")
    
    # Set Variant Generation Provider (openai or qwen) - DEPRECATED: use use_openai_for_image_editing instead
    use_openai_for_set_variants: bool = Field(default=True, env="USE_OPENAI_FOR_SET_VARIANTS")
    
    # =============================================
    # CREDIT MANAGER CONFIGURATION
    # =============================================
    
    # Worker Configuration
    credit_manager_num_workers: Optional[int] = None  # Auto-detect if None
    credit_manager_worker_timeout: float = 1.0  # Seconds to wait for queue items
    credit_manager_error_retry_delay: float = 1.0  # Seconds to wait after errors
    
    # Queue Size Configuration (Memory Management)
    credit_manager_transaction_queue_size: int = 500  # Max pending transactions
    credit_manager_profile_queue_size: int = 250      # Max pending profile events
    credit_manager_grant_queue_size: int = 250        # Max pending grant events
    
    # Performance Tuning for Different Dyno Types
    credit_manager_performance_dyno_workers: int = 6  # Workers for performance dynos
    credit_manager_standard_dyno_workers: int = 3     # Workers for standard dynos
    
    # Periodic Cleanup Configuration
    credit_manager_cleanup_interval: int = 60         # Seconds between cleanup runs
    credit_manager_stale_claim_timeout: int = 300     # Seconds before claims are stale
    
    # Signup Promotion Settings
    credit_manager_signup_bonus_amount: int = 500     # Credits granted on signup
    credit_manager_signup_bonus_note: str = "SIGNUP_PROMOTION_CREDIT_PERMANENT"
    
        # GLOBAL LOGGING CONFIGURATION
    # Controls application-wide log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_level: str = Field(default="WARNING", env="LOG_LEVEL")
    
    # CREDIT MANAGER CONFIGURATION
    # Monitoring and Logging
    credit_manager_enable_detailed_logging: bool = True    # Enable detailed operation logs
    credit_manager_log_queue_stats: bool = True           # Log queue statistics
    credit_manager_log_processing_stats: bool = True      # Log processing statistics

    # =============================================
    # BRAND MANAGER CONFIGURATION
    # =============================================
    
    # Worker Configuration
    brand_manager_num_workers: Optional[int] = None  # Auto-detect if None
    brand_manager_worker_timeout: float = 2.0  # Seconds to wait for queue items
    brand_manager_error_retry_delay: float = 2.0  # Seconds to wait after errors
    
    # Queue Size Configuration (Memory Management)
    brand_manager_brand_queue_size: int = 100         # Max pending brand events
    brand_manager_questionnaire_queue_size: int = 50  # Max pending questionnaire events
    brand_manager_waitlist_queue_size: int = 50       # Max pending waitlist events
    
    # Performance Tuning for Different Dyno Types
    brand_manager_performance_dyno_workers: int = 4   # Workers for performance dynos
    brand_manager_standard_dyno_workers: int = 2      # Workers for standard dynos
    
    # Periodic Cleanup Configuration
    brand_manager_cleanup_interval: int = 120         # Seconds between cleanup runs
    brand_manager_stale_claim_timeout: int = 600      # Seconds before claims are stale
    
    # Monitoring and Logging
    brand_manager_enable_detailed_logging: bool = True  # Enable detailed operation logs
    brand_manager_log_queue_stats: bool = True         # Log queue statistics
    brand_manager_log_processing_stats: bool = True    # Log processing statistics

    # =============================================
    # POST LOGIC EVENT-DRIVEN CONFIGURATION
    # =============================================
    
    # Worker Configuration (legacy compatibility - not used in new TaskEngine system)
    post_logic_worker_timeout: int = 5  # Seconds to wait for queue items
    post_logic_error_retry_delay: int = 2  # Seconds to wait after errors
    
    # NEW Stage Worker Concurrency Control (Production Critical)
    post_logic_max_concurrent_new_stage_workers: int = 10  # Max concurrent NEW stage workers across all types
    post_logic_max_concurrent_post_new_workers: int = 5     # Max concurrent post NEW stage workers
    post_logic_max_concurrent_adspec_new_workers: int = 3   # Max concurrent adspec NEW stage workers  
    post_logic_max_concurrent_questionnaire_new_workers: int = 2  # Max concurrent questionnaire NEW stage workers
    
    # AI Stage Workers - Performance Dyno Settings
    post_logic_post_ai_performance_workers: int = 3          # Post AI workers for performance dynos
    post_logic_questionnaire_ai_performance_workers: int = 2 # Questionnaire AI workers for performance dynos
    post_logic_adspec_ai_performance_workers: int = 2        # AdSpec AI workers for performance dynos
    
    # AI Stage Workers - Standard Dyno Settings  
    post_logic_post_ai_standard_workers: int = 2             # Post AI workers for standard dynos
    post_logic_questionnaire_ai_standard_workers: int = 1    # Questionnaire AI workers for standard dynos
    post_logic_adspec_ai_standard_workers: int = 1           # AdSpec AI workers for standard dynos
    
    # AI Stage Worker Intervals (seconds between task checks)
    post_logic_post_ai_scan_interval: int = 15               # Post AI stage scan interval
    post_logic_questionnaire_ai_scan_interval: int = 30      # Questionnaire AI stage scan interval
    post_logic_adspec_ai_scan_interval: int = 30             # AdSpec AI stage scan interval
    
    # Periodic Cleanup Configuration
    post_logic_cleanup_interval: int = 60         # Seconds between cleanup runs
    post_logic_stale_claim_timeout: int = 300     # Seconds before claims are stale
    post_logic_enable_periodic_sweep: bool = False  # Enable/disable periodic sweep for testing
    
    # Monitoring and Logging
    post_logic_enable_detailed_logging: bool = True  # Enable detailed operation logs
    post_logic_log_queue_stats: bool = True         # Log queue statistics
    post_logic_log_processing_stats: bool = True    # Log processing statistics

    # =============================================
    # REALTIME & SCHEDULER CONFIGURATION
    # =============================================
    
    # Realtime Connection Settings
    realtime_connection_retry_delay: float = 1.0      # Delay before reconnection attempts
    realtime_enable_structured_logging: bool = True   # Use structured logging vs print
    realtime_verbose_debug_prints: bool = False       # Gate verbose debug output
    
    # Scheduler Task Settings
    scheduler_enable_smart_backoff: bool = True       # Avoid double sleep on long delays
    scheduler_max_backoff_seconds: int = 30           # Cap exponential backoff
    
    # Scheduler Thread Pool Configuration (dyno-aware auto-detection)
    scheduler_performance_dyno_task_workers: int = 5  # Thread pool workers for performance dynos
    scheduler_standard_dyno_task_workers: int = 3     # Thread pool workers for standard dynos

    # =============================================
    # TASK ENGINE CONFIGURATION
    # =============================================
    
    # Task Engine Configuration
    task_engine_cleanup_max_age_hours: int = 24
    task_engine_enable_cleanup_logging: bool = True

    # Post Logic TaskEngine Settings
    post_logic_use_taskengine: bool = True
    post_logic_max_concurrent_posts: int = 5
    post_logic_task_timeout_seconds: int = 600  # 10 minutes
    post_logic_worker_claim_timeout: int = 30

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), '..', '.env')
        # Allow environment variables to override defaults
        env_prefix = ""  # No prefix, use exact variable names

settings = Settings()