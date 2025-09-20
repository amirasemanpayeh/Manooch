from pydantic import BaseModel
import uuid
from datetime import datetime, date, time, timezone

from typing import List, Optional, Dict, Any
from enum import Enum

class ChatGPTRequest(BaseModel):
    prompt: str


class TestAssetGeneratorRequest(BaseModel):
    rawImgPath: str


class GetBrandFromWebisteRequest(BaseModel):
    url: str

class CreateBrandInfoRequest(BaseModel):
    name: str
    industry: str
    short_description: str
    website_address: str

class CreateBrandInfoFromOnboarding(BaseModel):
    user_id: str


# ============================
#  FAILURE REASON ENUM
# ============================

class FailureReason(Enum):
    UNAUTHORIZED_USER = "UnauthorizedUser"
    INSUFFICIENT_FUNDS = "InsufficientFunds"
    INSUFFICIENT_FUNDS_FOR_CAMPAIGN = "InsufficientFundsForCampaign"
    AI_ENGINE_COMMS_FAILED = "AiEngineCommsFailed"
    AI_ENGINE_RETURNED_BAD_DATA = "AiEngineReturnedBadData"
    BRAND_DATA_NOT_VALID = "BrandDataNotValid"
    CAMPAIGN_DATA_NOT_VALID = "CampaignDataNotValid"
    AD_SPEC_DATA_NOT_VALID = "AdSpecDataNotValid"
    POST_DATA_NOT_VALID = "PostDataNotValid"
    ASSIGNED_CAMPAIGN_NOT_FOUND = "AssignedCampaignNotFound"
    ASSIGNED_AD_SPEC_NOT_FOUND = "AssignedAdSpecNotFound"
    ASSIGNED_POST_NOT_FOUND = "AssignedPostNotFound"
    INCORRECT_PUBLICATION_DATE = "IncorrectPublicationDate"
    INCORRECT_PUBLICATION_TIME = "IncorrectPublicationTime"
    USER_APPROVAL_EXPIRED = "UserApprovalExpired"
    SOCIAL_MEDIA_COMMS_FAILED = "SocialMediaCommsFailed"
    SOCIAL_MEDIA_CONNECTION_FAILED = "SocialMediaConnectionFailed"
    SOCIAL_MEDIA_CONNECTION_EXPIRED = "SocialMediaConnectionExpired"
    SOCIAL_MEDIA_CONNECTION_WILL_EXPIRE = "SocialMediaConnectionWillExpire"
    NO_FAILURE = "NoFailure"


# ============================
#  USER PROFILE MODEL
# ============================
class UserProfileModel:
    def __init__(self,
                 id: Optional[uuid.UUID] = None,
                 first_name: Optional[str] = '',
                 last_name: Optional[str] = '',
                 email: Optional[str] = None,
                 created_at: datetime = datetime.now(timezone.utc),
                 credits: int = 10):
        self.id = id or uuid.uuid4()
        self.first_name = first_name or ''
        self.last_name = last_name or ''
        self.email = email
        self.created_at = created_at
        self.credits = credits

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "credits": self.credits,
        }

    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'UserProfileModel':
        return UserProfileModel(
            id=uuid.UUID(data["id"]) if data.get("id") else None,
            first_name=data.get("first_name", ''),
            last_name=data.get("last_name", ''),
            email=data.get("email"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            credits=int(data.get("credits", 10)),
        )

# ============================
#  BRAND QUESTIONNAIRE MODEL
# ============================

class QuestionnaireInfo:
    def __init__(self, id: Optional[uuid.UUID] = None, created_at: Optional[uuid.UUID] = None, brand_name: Optional[str] = '', brand_industry: Optional[str] = '', brand_short_description: Optional[str] = '', brand_website_address: Optional[str] = '', user_id: Optional[uuid.UUID] = None, status: Optional[str] = ''):
        self.id = id
        self.created_at = created_at
        self.brand_name = brand_name
        self.brand_industry = brand_industry
        self.brand_short_description = brand_short_description
        self.brand_website_address = brand_website_address
        self.user_id = user_id
        self.status = status

    def __repr__(self):
        return f"BrandInfo(id={self.id}, created_at={self.created_at}, brand_name={self.brand_name}, brand_industry={self.brand_industry}, brand_short_description={self.brand_short_description}, brand_website_address={self.brand_website_address}, user_id={self.user_id}, status={self.status})"
    
# ============================
#  Waitlist Model
# ============================

class WaitListUserStatus(Enum):
    AWAITING_INVITATION = "AwaitingInvitation"
    INVITATION_REQUESTED = "InvitationRequested"
    INVITED = "Invited"
    SIGNED_UP = "SignedUp"

class WaitListUserModel:
    def __init__(self,
                 id: Optional[uuid.UUID] = None,
                 email: str = None,
                 created_at: datetime = datetime.now(timezone.utc),
                 invitation_code: Optional[uuid.UUID] = None,
                 status: WaitListUserStatus = WaitListUserStatus.AWAITING_INVITATION):
        self.id = id or uuid.uuid4()
        self.email = email
        self.created_at = created_at
        self.invitation_code = invitation_code
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id),
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'invitation_code': str(self.invitation_code) if self.invitation_code else None,
            'status': self.status.value if self.status else None,
        }

    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'WaitListUserModel':
        return WaitListUserModel(
            id=uuid.UUID(data['id']) if data.get('id') else None,
            email=data.get('email'),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(timezone.utc),
            invitation_code=uuid.UUID(data['invitation_code']) if data.get('invitation_code') else None,
            status=WaitListUserStatus(data['status']) if data.get('status') else WaitListUserStatus.AWAITING_INVITATION,
        )

# ============================
# INVITE TOEKN MODEL
# ============================
class InviteTokenModel:
    def __init__(self,
                 id: Optional[uuid.UUID] = None,
                 created_at: datetime = datetime.now(timezone.utc),
                 token: Optional[int] = None,
                 email: Optional[str] = None,
                 expires_at: Optional[datetime] = None,
                 used_at: Optional[datetime] = None):
        self.id = id or uuid.uuid4()
        self.created_at = created_at
        self.token = token
        self.email = email
        self.expires_at = expires_at
        self.used_at = used_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'token': self.token,
            'email': self.email,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'used_at': self.used_at.isoformat() if self.used_at else None,
        }

    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'InviteTokenModel':
        return InviteTokenModel(
            id=uuid.UUID(data['id']) if data.get('id') else None,
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(timezone.utc),
            token=int(data['token']) if data.get('token') is not None else None,
            email=data.get('email'),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            used_at=datetime.fromisoformat(data['used_at']) if data.get('used_at') else None,
        )

# ============================
#  ONBOARDING MODEL
# ============================
class OnboardingQuestionnaireModel:
    def __init__(self,
                 id: Optional[uuid.UUID] = None,
                 created_at: datetime = datetime.now(timezone.utc),
                 brand_name: Optional[str] = None,
                 brand_industry: Optional[str] = None,
                 brand_short_description: Optional[str] = None,
                 brand_website_address: Optional[str] = None,
                 user_id: Optional[uuid.UUID] = None,
                 status: Optional[str] = None):
        
        # Initialize with default values or given parameters
        self.id = id or uuid.uuid4()  # Generate UUID if not provided
        self.created_at = created_at
        self.brand_name = brand_name
        self.brand_industry = brand_industry
        self.brand_short_description = brand_short_description
        self.brand_website_address = brand_website_address
        self.user_id = user_id
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'brand_name': self.brand_name,
            'brand_industry': self.brand_industry,
            'brand_short_description': self.brand_short_description,
            'brand_website_address': self.brand_website_address,
            'user_id': str(self.user_id) if self.user_id else None,
            'status': self.status,
        }

    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'OnboardingQuestionnaireModel':
        return OnboardingQuestionnaireModel(
            id=uuid.UUID(data['id']) if data.get('id') else None,
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(timezone.utc),
            brand_name=data.get('brand_name'),
            brand_industry=data.get('brand_industry'),
            brand_short_description=data.get('brand_short_description'),
            brand_website_address=data.get('brand_website_address'),
            user_id=uuid.UUID(data['user_id']) if data.get('user_id') else None,
            status=data.get('status'),
        )

# ============================
#  BRAND MODEL
# ============================

class BrandMainStatus(Enum):
    NEW = "New"
    GENERATING = "Generating"
    READY = "Ready"
    FAILED = "Failed"

# Assuming there's a missing enum based on the usage in StateMachine
class BrandAssistantStatus(Enum):
    PRIMARY_DATA_LOADED_AWAITING_SCRAPING = "PrimaryDataLoadedAwaitingScraping"
    DATA_SCRAPED_AWAITING_GENERATION = "DataScrapedAwaitingGeneration"
    DATA_GENERATED_AWAITING_DB_UPDATE = "DataGeneratedAwaitingDBUpdate"
    DATA_UPDATED_AWAITING_USER_APPROVAL = "DataUpdatedAwaitingUserApproval"
    UPDATED_APPROVED = "UpdatedApproved"

class PersonalityTrait(Enum):
    SINCERE = "Sincere"
    EXCITING = "Exciting"
    COMPETENT = "Competent"
    SOPHISTICATED = "Sophisticated"
    RUGGED = "Rugged"
    HONEST = "Honest"
    WHOLESOME = "Wholesome"
    CHEERFUL = "Cheerful"
    DARING = "Daring"
    SPIRITED = "Spirited"
    IMAGINATIVE = "Imaginative"
    UP_TO_DATE = "UpToDate"
    RELIABLE = "Reliable"
    INTELLIGENT = "Intelligent"
    SUCCESSFUL = "Successful"
    UPPER_CLASS = "UpperClass"
    CHARMING = "Charming"
    OUTDOORSY = "Outdoorsy"
    TOUGH = "Tough"

class VoiceStyle(Enum):
    FORMAL = "Formal"
    CASUAL = "Casual"
    PROFESSIONAL = "Professional"
    FRIENDLY = "Friendly"
    INFORMATIVE = "Informative"
    ENTERTAINING = "Entertaining"
    AUTHORITATIVE = "Authoritative"
    ENTHUSIASTIC = "Enthusiastic"
    CONVERSATIONAL = "Conversational"
    INSTRUCTIONAL = "Instructional"
    INSPIRATIONAL = "Inspirational"
    EMPOWERING = "Empowering"
    HUMOROUS = "Humorous"
    SERIOUS = "Serious"
    RESPECTFUL = "Respectful"
    IRREVERENT = "Irreverent"
    BOLD = "Bold"
    PASSIONATE = "Passionate"
    SENSIBLE = "Sensible"
    MATTER_OF_FACT = "MatterOfFact"


class BrandModel:
    def __init__(self, id: Optional[uuid.UUID] = None, name: Optional[str] = '', industry: Optional[str] = None,
                 slogan: Optional[str] = '', description: Optional[str] = '',
                 unique_selling_proposition: Optional[str] = '', mission_vision: Optional[str] = '',
                 personality_trait: Optional[PersonalityTrait] = None, voice_style: Optional[VoiceStyle] = None,
                 key_messages_benefits_features: Optional[str] = '', logo_url: Optional[str] = None,
                 primary_font: Optional[str] = '', secondary_font: Optional[str] = '',
                 primary_color: Optional[str] = '', secondary_color: Optional[str] = '',
                 neutral_color: Optional[str] = '', accent_color: Optional[str] = '',
                 specific_ad_theme_guidelines: Optional[str] = '', address: Optional[str] = '',
                 telephone: Optional[str] = '', email: Optional[str] = '', website_url: Optional[str] = '', 
                 user_id: Optional[uuid.UUID] = None, 
                 status: Optional[BrandMainStatus] = BrandMainStatus.NEW, assistant_status: Optional[BrandAssistantStatus] = None, 
                 created_at: datetime = datetime.now(timezone.utc)):
        self.id = id
        self.name = name
        self.industry = industry # nullable
        self.slogan = slogan
        self.description = description
        self.unique_selling_proposition = unique_selling_proposition
        self.mission_vision = mission_vision
        self.personality_trait = personality_trait # nullable
        self.voice_style = voice_style # nullable
        self.key_messages_benefits_features = key_messages_benefits_features
        self.logo_url = logo_url  # nullable
        self.primary_font = primary_font
        self.secondary_font = secondary_font
        self.primary_color = primary_color
        self.secondary_color = secondary_color
        self.neutral_color = neutral_color
        self.accent_color = accent_color
        self.specific_ad_theme_guidelines = specific_ad_theme_guidelines
        self.address = address
        self.telephone = telephone
        self.email = email
        self.website_url = website_url
        self.user_id = user_id
        self.status = status  # nullable
        self.assistant_status = assistant_status  # nullable
        self.created_at = created_at

    def to_dict(self):
        return {
            'id': str(self.id) if self.id else None,
            'name': self.name,
            'industry': self.industry,
            'slogan': self.slogan,
            'description': self.description,
            'unique_selling_proposition': self.unique_selling_proposition,
            'mission_vision': self.mission_vision,
            'personality_trait': self.personality_trait.value if isinstance(self.personality_trait, Enum) else self.personality_trait,
            'voice_style': self.voice_style.value if isinstance(self.voice_style, Enum) else self.voice_style,
            'key_messages_benefits_features': self.key_messages_benefits_features,
            'logo_url': self.logo_url,
            'primary_font': self.primary_font,
            'secondary_font': self.secondary_font,
            'primary_color': self.primary_color,
            'secondary_color': self.secondary_color,
            'neutral_color': self.neutral_color,
            'accent_color': self.accent_color,
            'specific_ad_theme_guidelines': self.specific_ad_theme_guidelines,
            'address': self.address,
            'telephone': self.telephone,
            'email': self.email,
            'website_url': self.website_url,
            'user_id': str(self.user_id) if self.user_id else None,
            'status': self.status.value if isinstance(self.status, Enum) else self.status,
            'assistant_status': self.assistant_status.value if isinstance(self.assistant_status, Enum) else self.assistant_status,
            'created_at': self.created_at.isoformat()
        }
    
    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'BrandModel':
        return BrandModel(
            id=uuid.UUID(data['id']),
            name=data.get('name', ''),
            industry=data.get('industry', None),
            slogan=data.get('slogan', ''),
            description=data.get('description', ''),
            unique_selling_proposition=data.get('unique_selling_proposition', ''),
            mission_vision=data.get('mission_vision', ''),
            personality_trait=PersonalityTrait(data['personality_trait']) if data.get('personality_trait') else None,
            voice_style=VoiceStyle(data['voice_style']) if data.get('voice_style') else None,
            key_messages_benefits_features=data.get('key_messages_benefits_features', ''),
            logo_url=data.get('logo_url', None),
            primary_font=data.get('primary_font', ''),
            secondary_font=data.get('secondary_font', ''),
            primary_color=data.get('primary_color', ''),
            secondary_color=data.get('secondary_color', ''),
            neutral_color=data.get('neutral_color', ''),
            accent_color=data.get('accent_color', ''),
            specific_ad_theme_guidelines=data.get('specific_ad_theme_guidelines', ''),
            address=data.get('address', ''),
            telephone=data.get('telephone', ''),
            email=data.get('email', ''),
            website_url=data.get('website_url', None),
            user_id=uuid.UUID(data['user_id']) if data.get('user_id') else None,
            status=BrandMainStatus(data['status']) if data.get('status') else BrandMainStatus.NEW,
            assistant_status=BrandAssistantStatus(data['assistant_status']) if data.get('assistant_status') else None,
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now()
        )
    
    def is_valid(self) -> bool:
        return self.status == BrandMainStatus.READY
    
# ============================
#  CAMPAIGN MODEL
# ============================
class CampaignStatus(Enum):
    NEW = "New"
    ACTIVE = "Active"
    COMPLETED = "Completed"

class CampaignModel:
    def __init__(self,
                 id: Optional[uuid.UUID] = None,
                 name: Optional[str] = None,
                 colour: Optional[str] = None,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 brand_id: Optional[uuid.UUID] = None,
                 status: Optional[CampaignStatus] = CampaignStatus.NEW,
                 created_at: datetime = datetime.now(timezone.utc),
                 post_days: Optional[List[str]] = None,
                 post_time: Optional[time] = None,
                 credit_cost: Optional[int] = None,
                 credit_allowance: Optional[float] = None,
                 planned_budget: Optional[float] = None,
                 spent_budget: Optional[float] = None):
        
        # Initialize with default values or given parameters
        self.id = id or uuid.uuid4()  # Generate UUID if not provided
        self.name = name
        self.colour = colour
        self.start_date = start_date
        self.end_date = end_date
        self.brand_id = brand_id
        self.status = status
        self.created_at = created_at
        self.post_days = post_days or []
        self.post_time = post_time
        self.credit_cost = credit_cost
        self.credit_allowance = credit_allowance
        self.planned_budget = planned_budget
        self.spent_budget = spent_budget

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id),
            'name': self.name,
            'colour': self.colour,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'brand_id': str(self.brand_id) if self.brand_id else None,
            'status': self.status.value if self.status else None,
            'created_at': self.created_at.isoformat(),
            'post_days': self.post_days,
            'post_time': self.post_time.isoformat() if self.post_time else None,
            'credit_cost': self.credit_cost,
            'credit_allowance': self.credit_allowance,
            'planned_budget': self.planned_budget,
            'spent_budget': self.spent_budget,
        }

    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'CampaignModel':
        return CampaignModel(
            id=uuid.UUID(data['id']) if data.get('id') else None,
            name=data.get('name'),
            colour=data.get('colour'),
            start_date=datetime.fromisoformat(data['start_date']) if data.get('start_date') else None,
            end_date=datetime.fromisoformat(data['end_date']) if data.get('end_date') else None,
            brand_id=uuid.UUID(data['brand_id']) if data.get('brand_id') else None,
            status=CampaignStatus(data['status']) if data.get('status') else CampaignStatus.NEW,
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(timezone.utc),
            post_days=data.get('post_days', []),
            post_time=time.fromisoformat(data['post_time']) if data.get('post_time') else None,
            credit_cost=data.get('credit_cost'),
            credit_allowance=data.get('credit_allowance'),
            planned_budget=data.get('planned_budget'),
            spent_budget=data.get('spent_budget'),
        )

# ============================
# POST QUESTIONNAIRE MODEL
# ============================

class PostQuestionnaireMainStatus(Enum):
    NEW = "New"
    QUEUED_FOR_PROCESSING = "QueuedForProcessing"
    GENERATED_SUCCESSFULLY = "GeneratedSuccessfully"
    FAILED = "Failed"

class PostQuestionnaireModel:
    def __init__(self, 
                 created_at: datetime = datetime.now(),
                 post_title: Optional[str] = None,
                 objective: Optional[str] = None,
                 call_to_action: Optional[str] = None,
                 audience_emotion: Optional[str] = None,
                 key_message: Optional[str] = None,
                 ad_spec_id: Optional[str] = None,
                 user_id: Optional[uuid.UUID] = None,
                 status: PostQuestionnaireMainStatus = PostQuestionnaireMainStatus.NEW,
                 failure_reason: FailureReason = FailureReason.NO_FAILURE,
                 generation_counter: int = 0,
                 id: Optional[uuid.UUID] = None):
        
        self.created_at = created_at
        self.post_title = post_title
        self.objective = objective
        self.call_to_action = call_to_action
        self.audience_emotion = audience_emotion
        self.key_message = key_message
        self.ad_spec_id = ad_spec_id
        self.user_id = user_id
        self.status = status
        self.failure_reason = failure_reason
        self.generation_counter = generation_counter
        self.id = id

    def to_dict(self):
        return {
            'id': str(self.id) if self.id else None, 
            'created_at': self.created_at.isoformat(),
            'post_title': self.post_title,
            'objective': self.objective,
            'call_to_action': self.call_to_action,
            'audience_emotion': self.audience_emotion,
            'key_message': self.key_message,
            'ad_spec_id': str(self.ad_spec_id) if self.ad_spec_id else None,
            'user_id': str(self.user_id) if self.user_id else None,
            'status': self.status.value,
            'failure_reason': self.failure_reason.value,
            'generation_counter': self.generation_counter
        }
    
    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'PostQuestionnaireModel':
        return PostQuestionnaireModel(
            id=uuid.UUID(data['id']),
            created_at=datetime.fromisoformat(data['created_at']),
            post_title=data.get('post_title', None),
            objective=data.get('objective', None),
            call_to_action=data.get('call_to_action', None),
            audience_emotion=data.get('audience_emotion', None),
            key_message=data.get('key_message', None),
            ad_spec_id=uuid.UUID(data['ad_spec_id']) if data.get('ad_spec_id') else None,
            user_id=uuid.UUID(data['user_id']) if data.get('user_id') else None,
            status=PostQuestionnaireMainStatus(data['status']),
            failure_reason=FailureReason(data['failure_reason']) if data.get('failure_reason') else FailureReason.NO_FAILURE,
            generation_counter=int(data.get('generation_counter', 0))
        )

    def increment_generation_counter(self):
        """Increment the generation counter when generation is successful."""
        self.generation_counter += 1

# ============================
#  AD SPEC MODEL
# ============================

class AdSpecMainStatus(Enum):
    DRAFT = "Draft"
    NEW = "New"
    QUEUED_FOR_PROCESSING = "QueuedForProcessing"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    FAILED = "Failed"

class MarketingObjective(Enum):
    BRAND_AWARENESS = "BrandAwareness"
    PRODUCT_LAUNCH = "ProductLaunch"
    PRODUCT_REMINDER = "ProductReminder"
    EVENT_LAUNCH = "EventLaunch"
    EVENT_REMINDER = "EventReminder"
    LEAD_GENERATION = "LeadGeneration"
    SALES_AND_CONVERSIONS = "SalesAndConversions"
    WEBSITE_TRAFFIC = "WebsiteTraffic"
    APP_INSTALL = "AppInstall"

class AdSpecModel:
    def __init__(self, 
                 id: Optional[uuid.UUID] = None,
                 campaign_id: Optional[uuid.UUID] = None,
                 user_id: uuid.UUID = None,
                 created_at: datetime = datetime.now(),
                 # Objectives and Targets Tab
                 title: str = ' ',
                 objective: MarketingObjective = MarketingObjective.BRAND_AWARENESS,
                 target_audiences: Optional[List[str]] = None,
                 social_media_profiles: Optional[List[str]] = None,
                 # Content Specification Tab
                 content_specification: Optional[Dict[str, Any]] = None,
                 # Content Strategy Blueprint Tab
                 post_type: str = None,
                 organic_post_format: Optional[str] = None,
                 paid_ad_format: Optional[str] = None,
                 post_asset_template_id: Optional[uuid.UUID] = None,
                 hashtags: Optional[List[str]] = None,
                 # Auto-Generation and Publishing Tab
                    # always show
                    auto_generate_with_ai: bool = False,
                    # Only show if auto_generate_with_ai is true
                    auto_publish: bool = False,
                    ai_assistance_level_id: Optional[uuid.UUID] = None,
                    generation_style_id: Optional[uuid.UUID] = None,
                    hashtag_autogen_enabled: bool = False,
                 # Date and Time View
                 publication_target_date: Optional[date] = None,
                 publication_target_time: Optional[time] = None,
                 publication_date_time: Optional[datetime] = None,
                 assets: Optional[List[str]] = None,
                 status: AdSpecMainStatus = AdSpecMainStatus.DRAFT,
                 failure_reason: FailureReason = FailureReason.NO_FAILURE,
                 generation_counter: int = 0,
        ):
        # Initialize with default values based on SQL schema
        self.id = id or uuid.uuid4()  # If no ID is provided, generate a new UUID
        self.campaign_id = campaign_id  # Nullable, so no default needed
        self.user_id = user_id
        self.created_at = created_at if created_at else datetime.now()  # Use now() if not provided
        
        # Objectives and Targets
        self.title = title if title else ' '  # Default to single space if empty
        self.objective = objective  # Default is set to MarketingObjective.BRAND_AWARENESS
        self.target_audiences = target_audiences if target_audiences is not None else []  # Default to empty list
        self.social_media_profiles = social_media_profiles if social_media_profiles is not None else []  # Default to empty list
        
        # Content Specification
        self.content_specification = content_specification if content_specification is not None else {}  # Default to empty dict
        
        # Content Strategy Blueprint
        self.post_type = post_type  # No default in SQL, required field
        self.organic_post_format = organic_post_format  # Nullable, so no default needed
        self.paid_ad_format = paid_ad_format  # Nullable, so no default needed
        self.hashtags = hashtags if hashtags is not None else []  # Default to empty list
        
        # Auto-Generation and Publishing
        self.auto_generate_with_ai = auto_generate_with_ai
        self.auto_publish = auto_publish
        self.ai_assistance_level_id = ai_assistance_level_id  # Foreign key, nullable
        self.post_asset_template_id = post_asset_template_id  # Foreign key, required
        self.generation_style_id = generation_style_id  # Foreign key, required
        self.hashtag_autogen_enabled = hashtag_autogen_enabled
        
        # Date and Time View
        self.publication_target_date = publication_target_date if publication_target_date else date.today()  # Default to today's date
        self.publication_target_time = publication_target_time if publication_target_time else time(9, 0, 0)  # Default to 9:00 AM
        self.publication_date_time = publication_date_time  # Nullable, so no default needed
        
        self.assets = assets if assets is not None else []  # Default to empty list
        
        # Status
        self.status = status if status else AdSpecMainStatus.DRAFT  # Default to DRAFT status

        self.failure_reason = failure_reason
        self.generation_counter = generation_counter

    def to_dict(self):
        return {
            'id': str(self.id) if self.id else None,
            'campaign_id': str(self.campaign_id) if self.campaign_id else None,
            'user_id': str(self.user_id),
            'created_at': self.created_at.isoformat(),
            'title': self.title or "Untitled",
            'objective': self.objective.value,  # Convert enum to string value
            'target_audiences': self.target_audiences,
            'social_media_profiles': self.social_media_profiles,
            'content_specification': self.content_specification,
            'post_type': self.post_type or 'organicPost',
            'organic_post_format': self.organic_post_format,
            'paid_ad_format': self.paid_ad_format,
            'post_asset_template_id': str(self.post_asset_template_id) if self.post_asset_template_id else None,
            'hashtags': self.hashtags,
            'auto_generate_with_ai': self.auto_generate_with_ai,
            'auto_publish': self.auto_publish,
            'ai_assistance_level_id': str(self.ai_assistance_level_id) if self.ai_assistance_level_id else None,
            'generation_style_id': str(self.generation_style_id) if self.generation_style_id else None,
            'hashtag_autogen_enabled': self.hashtag_autogen_enabled,
            'publication_target_date': self.publication_target_date.isoformat() if self.publication_target_date else None,
            'publication_target_time': self.publication_target_time.isoformat() if self.publication_target_time else None,
            'publication_date_time': self.publication_date_time.astimezone(timezone.utc).isoformat() if self.publication_date_time else None,  # Include timezone info
            'assets': self.assets,
            'status': self.status.value,
            'failure_reason': self.failure_reason.value,
            'generation_counter': self.generation_counter
        }
    
    @staticmethod
    def create_default(user_id: uuid.UUID, title:str) -> 'AdSpecModel':
        """Creates a default AdSpecModel instance with minimal required values."""
        return AdSpecModel(
            user_id=user_id,
            title=title,
            objective=MarketingObjective.BRAND_AWARENESS,
            post_type= 'organicPost',
            organic_post_format='image',
            auto_generate_with_ai = True,
            auto_publish = False,
            #TODO: Add a settings file to store default values for the ai_assistance_level_id and generation_style_id
            post_asset_template_id = uuid.UUID('18a5a509-c17c-4aec-8586-24d16018b6f6'), # Default post asset template, single focus - minimalistic
            ai_assistance_level_id=uuid.UUID('c9441dde-77bb-41bc-b5a9-128ef6ec8b88'), # Default AI assistance level, Full creator
            generation_style_id=uuid.UUID('01bb5cf1-6621-400c-95df-0b7d5da29f15'), # Default generation style, Cinematic
            hashtag_autogen_enabled=False,
            status=AdSpecMainStatus.DRAFT
        )
    
    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'AdSpecModel':
        return AdSpecModel(
            id=uuid.UUID(data['id']) if data.get('id') else None,
            campaign_id=uuid.UUID(data['campaign_id']) if data.get('campaign_id') else None,
            user_id=uuid.UUID(data['user_id']),
            created_at=datetime.fromisoformat(data['created_at']),
            title=data.get('title', ' '),
            objective=MarketingObjective(data['objective']),
            target_audiences=data.get('target_audiences', []),
            social_media_profiles=data.get('social_media_profiles', []),
            content_specification=data.get('content_specification', {}),
            post_type=data.get('post_type'),
            organic_post_format=data.get('organic_post_format'),
            paid_ad_format=data.get('paid_ad_format'),
            post_asset_template_id=uuid.UUID(data['post_asset_template_id']) if data.get('post_asset_template_id') else None,
            hashtags=data.get('hashtags', []),
            auto_generate_with_ai=data.get('auto_generate_with_ai', False),
            auto_publish=data.get('auto_publish', False),
            ai_assistance_level_id=uuid.UUID(data['ai_assistance_level_id']) if data.get('ai_assistance_level_id') else None,
            generation_style_id=uuid.UUID(data['generation_style_id']) if data.get('generation_style_id') else None,
            hashtag_autogen_enabled=data.get('hashtag_autogen_enabled', False),
            publication_target_date=date.fromisoformat(data['publication_target_date']) if data.get('publication_target_date') else None,
            publication_target_time=time.fromisoformat(data['publication_target_time']) if data.get('publication_target_time') else None,
            publication_date_time=datetime.fromisoformat(data['publication_date_time']).astimezone(timezone.utc) if data.get('publication_date_time') else None,  # Ensure timezone-aware datetime
            assets=data.get('assets', []),
            status=AdSpecMainStatus(data['status']),
            failure_reason=FailureReason(data['failure_reason']) if data.get('failure_reason') else FailureReason.NO_FAILURE,
            generation_counter=int(data.get('generation_counter', 0))
        )
    
    @classmethod
    def to_obj_lite(cls, data):
        """Convert AI response into an AdSpec object using only provided keys."""
        allowed_keys = {"objective", "content_specification", "hashtags"}  # Keep only necessary fields
        filtered_data = {key: data[key] for key in allowed_keys if key in data}
        return cls(**filtered_data)  # Construct the model using only valid keys
    

# ============================
#  POST MODEL
# ============================
class PostMainStatus(Enum):
    DRAFT = "Draft"
    NEW = "New"
    QUEUED_FOR_PROCESSING = "QueuedForProcessing"
    GENERATING = "Generating"
    READY = "Ready"
    SCHEDULED = "Scheduled"
    PUBLISHED = "Published"
    FAILED = "Failed"

class PostAssistantStatus(Enum):
    PRIMARY_DATA_LOADED_AI_GENERATION_PENDING = "PrimaryDataLoadedAIGenerationPending"
    TEXT_ELEMENTS_GENERATED_ASSETS_GEN_PENDING = "TextElementsGeneratedAssetsGenPending"
    ASSETS_GENERATED_POST_FINALISATION_PENDING = "AssetsGeneratedPostFinalisationPending"
    POST_READY_USER_APPROVAL_PENDING = "PostReadyUserApprovalPending"
    USER_APPROVED_PUBLICATION_PENDING = "UserApprovedPublicationPending"
    APPROVED_PUBLISHED_SUCCESSFULLY = "ApprovedPublishedSuccessfully"

class PostProcessingStatus(Enum):
    OKAY = "Okay"
    FAILED_NO_BRAND_DATA = "NoBrandDataFound"
    FAILED_BRAND_DATA_NOT_READY = "BrandDataNotReady"
    FAILED_NO_AD_SPEC_DATA = "NoAdSpecDataFound"
    FAILED_AD_SPEC_NOT_READY = "AdSpecNotReady"

class PostGenerationRequestTypes(Enum):
    NONE = "None"
    ASSET_GENERATION = "AssetGeneration"
    CAPTION_GENERATION = "CaptionGeneration"
    HASHTAGS_GENERATION = "HashtagsGeneration"


class PostModel:
    def __init__(self,
                 id: Optional[uuid.UUID] = None,
                 created_at: datetime = datetime.now(timezone.utc),
                 title: Optional[str] = None,
                 campaign_id: Optional[uuid.UUID] = None,
                 ad_spec_id: uuid.UUID = None,  # Non-nullable
                 social_media_profiles: Optional[List[str]] = None,
                 target_publication_date_time: Optional[datetime] = None,  # timestamp with time zone
                 post_type: Optional[str] = None,
                 auto_publish_enabled: Optional[bool] = None,
                 template_html: Optional[str] = None,
                 template_css: Optional[str] = None,
                 base_prompt_for_text_generation: Optional[str] = None,
                 base_prompt_for_asset_generation: Optional[List[str]] = None,
                 headlines: Optional[List[str]] = None,
                 punchlines: Optional[List[str]] = None,
                 descriptions: Optional[List[str]] = None,
                 call2action_texts: Optional[List[str]] = None,
                 captions: Optional[List[str]] = None,
                 hashtags: Optional[List[str]] = None,
                 assets: Optional[List[str]] = None,
                 draft_publishable_assets: Optional[List[str]] = None,
                 draft_publishable_captions: Optional[List[str]] = None,
                 publishable_assets: Optional[List[str]] = None,
                 publishable_caption: Optional[str] = None,
                 user_prompt_for_text_generation: Optional[List[str]] = None,
                 user_prompt_for_asset_generation: Optional[List[str]] = None,
                 user_prompt_for_hashtag_generation: Optional[List[str]] = None,
                 main_status: PostMainStatus = PostMainStatus.DRAFT,  # Non-nullable
                 assistant_status: Optional[PostAssistantStatus] = None,
                 user_prompt_index: Optional[int] = 0,
                 user_no_assistance_generation_request_type: Optional[PostGenerationRequestTypes] = None,
                 failure_reason: FailureReason = FailureReason.NO_FAILURE,
                 generation_counter: int = 0,
                 user_id: uuid.UUID = None):  # Non-nullable
        
        # Initialize with default values or given parameters
        self.id = id or uuid.uuid4()  # Generate UUID if not provided
        self.created_at = created_at
        self.title = title
        self.campaign_id = campaign_id
        self.ad_spec_id = ad_spec_id  # Required field
        self.social_media_profiles = social_media_profiles if social_media_profiles is not None else []
        self.target_publication_date_time = target_publication_date_time  # timestamp with time zone
        self.post_type = post_type
        self.auto_publish_enabled = auto_publish_enabled
        self.template_html = template_html
        self.template_css = template_css
        self.base_prompt_for_text_generation = base_prompt_for_text_generation
        self.base_prompt_for_asset_generation = base_prompt_for_asset_generation if base_prompt_for_asset_generation is not None else []
        self.headlines = headlines if headlines is not None else []
        self.punchlines = punchlines if punchlines is not None else []
        self.descriptions = descriptions if descriptions is not None else []
        self.call2action_texts = call2action_texts if call2action_texts is not None else []
        self.captions = captions if captions is not None else []
        self.hashtags = hashtags if hashtags is not None else []
        self.assets = assets if assets is not None else []
        self.draft_publishable_assets = draft_publishable_assets if draft_publishable_assets is not None else []
        self.draft_publishable_captions = draft_publishable_captions if draft_publishable_captions is not None else []
        self.publishable_assets = publishable_assets if publishable_assets is not None else []
        self.publishable_caption = publishable_caption
        self.user_prompt_for_text_generation = user_prompt_for_text_generation if user_prompt_for_text_generation is not None else []
        self.user_prompt_for_asset_generation = user_prompt_for_asset_generation if user_prompt_for_asset_generation is not None else []
        self.user_prompt_for_hashtag_generation = user_prompt_for_hashtag_generation if user_prompt_for_hashtag_generation is not None else []
        self.main_status = main_status  # Required, defaults to DRAFT
        self.assistant_status = assistant_status
        self.user_prompt_index = user_prompt_index
        self.user_no_assistance_generation_request_type = user_no_assistance_generation_request_type
        self.user_id = user_id  # Required field
        self.failure_reason = failure_reason
        self.generation_counter = generation_counter

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'title': self.title,
            'campaign_id': str(self.campaign_id) if self.campaign_id else None,
            'ad_spec_id': str(self.ad_spec_id),  # Non-nullable UUID
            'social_media_profiles': self.social_media_profiles,
            'target_publication_date_time': self.target_publication_date_time.isoformat() if self.target_publication_date_time else None,  # Handle timestamp with timezone
            'post_type': self.post_type,
            'auto_publish_enabled': self.auto_publish_enabled,
            'template_html': self.template_html,
            'template_css': self.template_css,
            'base_prompt_for_text_generation': self.base_prompt_for_text_generation,
            'base_prompt_for_asset_generation': self.base_prompt_for_asset_generation,
            'headlines': self.headlines,
            'punchlines': self.punchlines,
            'descriptions': self.descriptions,
            'call2action_texts': self.call2action_texts,
            'captions': self.captions,
            'hashtags': self.hashtags,
            'assets': self.assets,
            'draft_publishable_assets': self.draft_publishable_assets,
            'draft_publishable_captions': self.draft_publishable_captions,
            'publishable_assets': self.publishable_assets,
            'publishable_caption': self.publishable_caption,
            'user_prompt_for_text_generation': self.user_prompt_for_text_generation,
            'user_prompt_for_asset_generation': self.user_prompt_for_asset_generation,
            'user_prompt_for_hashtag_generation': self.user_prompt_for_hashtag_generation,
            'main_status': self.main_status.value,  # Enum to string
            'assistant_status': self.assistant_status.value if self.assistant_status else None,  # Enum to string or None
            'user_no_assistance_generation_request_type': self.user_no_assistance_generation_request_type.value if self.user_no_assistance_generation_request_type else None,  # Enum to string or None
            'user_prompt_index': self.user_prompt_index,
            'failure_reason': self.failure_reason.value if self.failure_reason else None,  # Enum to string or None
            'generation_counter': self.generation_counter,
            'user_id': str(self.user_id)  # Non-nullable UUID
        }

    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'PostModel':
        return PostModel(
            id=uuid.UUID(data['id']) if data.get('id') else None,
            created_at=datetime.fromisoformat(data['created_at']),
            title=data.get('title'),
            campaign_id=uuid.UUID(data['campaign_id']) if data.get('campaign_id') else None,
            ad_spec_id=uuid.UUID(data['ad_spec_id']),  # Required UUID
            social_media_profiles=data.get('social_media_profiles', []),
            target_publication_date_time=datetime.fromisoformat(data['target_publication_date_time']) if data.get('target_publication_date_time') else None,
            post_type=data.get('post_type'),
            auto_publish_enabled=data.get('auto_publish_enabled'),
            template_html=data.get('template_html'),
            template_css=data.get('template_css'),
            base_prompt_for_text_generation=data.get('base_prompt_for_text_generation'),
            base_prompt_for_asset_generation=data.get('base_prompt_for_asset_generation', []),
            headlines=data.get('headlines', []),
            punchlines=data.get('punchlines', []),
            descriptions=data.get('descriptions', []),
            call2action_texts=data.get('call2action_texts', []),
            captions=data.get('captions', []),
            hashtags=data.get('hashtags', []),
            assets=data.get('assets', []),
            draft_publishable_assets=data.get('draft_publishable_assets', []),
            draft_publishable_captions=data.get('draft_publishable_captions', []),
            publishable_assets=data.get('publishable_assets', []),
            publishable_caption=data.get('publishable_caption'),
            user_prompt_for_text_generation=data.get('user_prompt_for_text_generation', []),
            user_prompt_for_asset_generation=data.get('user_prompt_for_asset_generation', []),
            user_prompt_for_hashtag_generation=data.get('user_prompt_for_hashtag_generation', []),
            main_status=PostMainStatus(data['main_status']),  # Enum from string
            assistant_status=PostAssistantStatus(data['assistant_status']) if data.get('assistant_status') else None,  # Enum from string
            user_prompt_index=data.get('user_prompt_index', 0),
            user_no_assistance_generation_request_type=PostGenerationRequestTypes(data['user_no_assistance_generation_request_type']) if data.get('user_no_assistance_generation_request_type') else None,  # Enum from string
            failure_reason=FailureReason(data['failure_reason']) if data.get('failure_reason') else FailureReason.NO_FAILURE,
            generation_counter=int(data.get('generation_counter', 0)),
            user_id=uuid.UUID(data['user_id'])  # Required UUID
        )


# ============================
#  POST AI ASSISTANCE LEVEL MODEL
# ============================
class PostAIAssistanceLevelModel:
    def __init__(self,
                 id: Optional[uuid.UUID] = None,
                 created_at: datetime = datetime.now(timezone.utc),
                 name: Optional[str] = None,
                 preview_url: Optional[str] = None,
                 description: Optional[str] = None,
                 assets_required: Optional[str] = None):
        
        # Initialize with default values or given parameters
        self.id = id or uuid.uuid4()  # Generate UUID if not provided
        self.created_at = created_at
        self.name = name
        self.preview_url = preview_url
        self.description = description
        self.assets_required = assets_required

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'name': self.name,
            'preview_url': self.preview_url,
            'description': self.description,
            'assets_required': self.assets_required,
        }

    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'PostAIAssistanceLevelModel':
        return PostAIAssistanceLevelModel(
            id=uuid.UUID(data['id']) if data.get('id') else None,
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(timezone.utc),
            name=data.get('name'),
            preview_url=data.get('preview_url'),
            description=data.get('description'),
            assets_required=data.get('assets_required'),
        )
    

# ============================
#  POST GENERATION STYLE MODEL
# ============================

class PostGenerationStyleModel:
    def __init__(self,
                 id: Optional[uuid.UUID] = None,
                 created_at: datetime = datetime.now(timezone.utc),
                 name: Optional[str] = None,
                 preview_url: Optional[str] = None,
                 description: Optional[str] = None):
        
        # Initialize with default values or given parameters
        self.id = id or uuid.uuid4()  # Generate UUID if not provided
        self.created_at = created_at
        self.name = name
        self.preview_url = preview_url
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'name': self.name,
            'preview_url': self.preview_url,
            'description': self.description,
        }

    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'PostGenerationStyleModel':
        return PostGenerationStyleModel(
            id=uuid.UUID(data['id']) if data.get('id') else None,
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(timezone.utc),
            name=data.get('name'),
            preview_url=data.get('preview_url'),
            description=data.get('description'),
        )
    

# ============================
#  POST TEMPLATE MODEL
# ============================
class PostTemplateModel:
    def __init__(self,
                 id: Optional[uuid.UUID] = None,
                 created_at: datetime = datetime.now(timezone.utc),
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 preview_url: Optional[str] = None,
                 html_templates: Optional[List[str]] = None,
                 owner_id: Optional[uuid.UUID] = None,
                 information_level: Optional[Dict[str, Any]] = None,
                 css_templates: Optional[List[str]] = None,
                 generation_guidance: Optional[str] = None,
                 svg_templates: Optional[List[str]] = None,
                 asset_type: Optional[str] = None):
        
        # Initialize with default values or given parameters
        self.id = id or uuid.uuid4()  # Generate UUID if not provided
        self.created_at = created_at
        self.name = name
        self.description = description
        self.preview_url = preview_url
        self.html_templates = html_templates or []
        self.owner_id = owner_id
        self.information_level = information_level or {}
        self.css_templates = css_templates or []
        self.generation_guidance = generation_guidance
        self.svg_templates = svg_templates or []
        self.asset_type = asset_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'name': self.name,
            'description': self.description,
            'preview_url': self.preview_url,
            'html_templates': self.html_templates,
            'owner_id': str(self.owner_id) if self.owner_id else None,
            'information_level': self.information_level,
            'css_templates': self.css_templates,
            'generation_guidance': self.generation_guidance,
            'svg_templates': self.svg_templates,
            'asset_type': self.asset_type,
        }

    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'PostTemplateModel':
        return PostTemplateModel(
            id=uuid.UUID(data['id']) if data.get('id') else None,
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(timezone.utc),
            name=data.get('name'),
            description=data.get('description'),
            preview_url=data.get('preview_url'),
            html_templates=data.get('html_templates', []),
            owner_id=uuid.UUID(data['owner_id']) if data.get('owner_id') else None,
            information_level=data.get('information_level', {}),
            css_templates=data.get('css_templates', []),
            generation_guidance=data.get('generation_guidance'),
            svg_templates=data.get('svg_templates', []),
            asset_type=data.get('asset_type'),
        )
    


# ============================
#  TARGET AUDIENCE MODEL
# ============================
class TargetAudienceModel:
    def __init__(self,
                 id: Optional[uuid.UUID] = None,
                 created_at: datetime = datetime.now(timezone.utc),
                 name: Optional[str] = None,
                 age_groups: Optional[List[str]] = None,
                 genders: Optional[List[str]] = None,
                 location: Optional[str] = None,
                 location_types: Optional[List[str]] = None,
                 interests: Optional[List[str]] = None,
                 income_levels: Optional[List[str]] = None,
                 education_levels: Optional[List[str]] = None,
                 relationship_statuses: Optional[List[str]] = None,
                 parental_statuses: Optional[List[str]] = None,
                 home_ownership_statuses: Optional[List[str]] = None,
                 occupation: Optional[str] = None,
                 brand_id: Optional[uuid.UUID] = None):
        
        # Initialize with default values or given parameters
        self.id = id or uuid.uuid4()  # Generate UUID if not provided
        self.created_at = created_at
        self.name = name
        self.age_groups = age_groups or []
        self.genders = genders or []
        self.location = location
        self.location_types = location_types or []
        self.interests = interests or []
        self.income_levels = income_levels or []
        self.education_levels = education_levels or []
        self.relationship_statuses = relationship_statuses or []
        self.parental_statuses = parental_statuses or []
        self.home_ownership_statuses = home_ownership_statuses or []
        self.occupation = occupation
        self.brand_id = brand_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'name': self.name,
            'age_groups': self.age_groups,
            'genders': self.genders,
            'location': self.location,
            'location_types': self.location_types,
            'interests': self.interests,
            'income_levels': self.income_levels,
            'education_levels': self.education_levels,
            'relationship_statuses': self.relationship_statuses,
            'parental_statuses': self.parental_statuses,
            'home_ownership_statuses': self.home_ownership_statuses,
            'occupation': self.occupation,
            'brand_id': str(self.brand_id) if self.brand_id else None,
        }

    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'TargetAudienceModel':
        return TargetAudienceModel(
            id=uuid.UUID(data['id']) if data.get('id') else None,
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(timezone.utc),
            name=data.get('name'),
            age_groups=data.get('age_groups', []),
            genders=data.get('genders', []),
            location=data.get('location'),
            location_types=data.get('location_types', []),
            interests=data.get('interests', []),
            income_levels=data.get('income_levels', []),
            education_levels=data.get('education_levels', []),
            relationship_statuses=data.get('relationship_statuses', []),
            parental_statuses=data.get('parental_statuses', []),
            home_ownership_statuses=data.get('home_ownership_statuses', []),
            occupation=data.get('occupation'),
            brand_id=uuid.UUID(data['brand_id']) if data.get('brand_id') else None,
        )
    
# ============================
#  Credit Transaction Model
# ============================

class CreditTransactionType(Enum):
    CREDIT = "Credit"
    DEBIT = "Debit"

class CreditTransactionReason(Enum):
    SUBSCRIPTION_GRANT = "SubscriptionGrant"
    IMAGE_POST_GENERATION = "ImagePostGeneration"
    VIDEO_POST_GENERATION = "VideoPostGeneration"
    TOP_UP = "TopUp"
    PROMOTION = "Promotion"
    EXPIRED = "Expired"
    REFUND = "Refund"
    ADMIN_ADJUSTMENT = "AdminAdjustment"

class CreditTransactionStatus(Enum):
    PENDING = "Pending"
    PROCESSED = "Processed"
    RESERVED = "Reserved"
    EXPIRED = "Expired"
    REFUNDED = "Refunded"

class CreditTransactionSource(Enum):
    STRIPE = "Stripe"
    USER = "User"
    ADMIN = "Admin"
    SYSTEM_CRON = "SystemCRON"

class CreditTransactionModel:
    def __init__(self,
                 id: Optional[uuid.UUID] = None,
                 created_at: datetime = datetime.now(timezone.utc),
                 user_id: uuid.UUID = None,
                 amount: int = 0,
                 type: CreditTransactionType = CreditTransactionType.CREDIT,
                 reason: CreditTransactionReason = CreditTransactionReason.SUBSCRIPTION_GRANT,
                 status: CreditTransactionStatus = CreditTransactionStatus.PENDING,
                 source: CreditTransactionSource = CreditTransactionSource.USER,
                 attachment_id: Optional[uuid.UUID] = None,
                 expires_at: Optional[datetime] = None,
                 notes: Optional[str] = None,
                 balance_after: int = 0):
        
        self.id = id or uuid.uuid4()
        self.created_at = created_at
        self.user_id = user_id
        self.amount = amount
        self.type = type
        self.reason = reason
        self.status = status
        self.source = source
        self.attachment_id = attachment_id
        self.expires_at = expires_at
        self.notes = notes
        self.balance_after = balance_after

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "user_id": str(self.user_id),
            "amount": self.amount,
            "type": self.type.value,
            "reason": self.reason.value,
            "status": self.status.value,
            "source": self.source.value,
            "attachment_id": str(self.attachment_id) if self.attachment_id else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "notes": self.notes,
            "balance_after": self.balance_after,
        }

    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'CreditTransactionModel':
        return CreditTransactionModel(
            id=uuid.UUID(data['id']) if data.get("id") else None,
            created_at=datetime.fromisoformat(data['created_at']) if data.get("created_at") else datetime.now(timezone.utc),
            user_id=uuid.UUID(data['user_id']),
            amount=int(data["amount"]),
            type=CreditTransactionType(data["type"]),
            reason=CreditTransactionReason(data["reason"]),
            status=CreditTransactionStatus(data["status"]),
            source=CreditTransactionSource(data["source"]),
            attachment_id=uuid.UUID(data["attachment_id"]) if data.get("attachment_id") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            notes=data.get("notes"),
            balance_after=int(data["balance_after"]),
        )
    
# ============================
#  Credit Grant Batch Model
# ============================
class CreditGrantBatchModel:
    def __init__(self,
                 credit_transaction_id: uuid.UUID = None,
                 id: Optional[uuid.UUID] = None,
                 created_at: datetime = datetime.now(timezone.utc),
                 user_id: uuid.UUID = None,
                 amount: int = 0,
                 consumed: int = 0,
                 granted_at: datetime = None,
                 expires_at: Optional[datetime] = None,):
        
        self.id = id or uuid.uuid4()
        self.created_at = created_at
        self.user_id = user_id
        self.amount = amount
        self.consumed = consumed
        self.granted_at = granted_at or datetime.now(timezone.utc)
        self.expires_at = expires_at
        self.credit_transaction_id = credit_transaction_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "user_id": str(self.user_id),
            "amount": self.amount,
            "consumed": self.consumed,
            "granted_at": self.granted_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "credit_transaction_id": str(self.credit_transaction_id)
        }

    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'CreditGrantBatchModel':
        return CreditGrantBatchModel(
            id=uuid.UUID(data['id']) if data.get('id') else None,
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(timezone.utc),
            user_id=uuid.UUID(data['user_id']),
            amount=int(data['amount']),
            consumed=int(data['consumed']),
            granted_at=datetime.fromisoformat(data['granted_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            credit_transaction_id=uuid.UUID(data['credit_transaction_id']),
        )
    
# ============================
#  Payment Segway Webhook Log Model
# ============================
    
class PaymentSegwayWebhookLogModel:
    def __init__(self,
                 id: Optional[uuid.UUID] = None,
                 created_at: datetime = datetime.now(timezone.utc),
                 user_id: Optional[uuid.UUID] = None,
                 event_type: Optional[str] = None,
                 payload: Optional[Dict[str, Any]] = None):
        
        self.id = id or uuid.uuid4()
        self.created_at = created_at
        self.user_id = user_id
        self.event_type = event_type
        self.payload = payload or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "user_id": str(self.user_id) if self.user_id else None,
            "event_type": self.event_type,
            "payload": self.payload,
        }

    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'PaymentSegwayWebhookLogModel':
        return PaymentSegwayWebhookLogModel(
            id=uuid.UUID(data["id"]) if data.get("id") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            user_id=uuid.UUID(data["user_id"]) if data.get("user_id") else None,
            event_type=data.get("event_type"),
            payload=data.get("payload") or {}
        )
    

# ============================
#  App Notification Model
# ============================
class NotificationType(Enum):
    """Enumeration for different types of notifications."""
    INFO = "Information"
    WARNING = "Warning"
    ERROR = "Error"

class NotificationDeliveryType(Enum):
    """Enumeration for notification delivery methods."""
    IN_APP = "InApp"
    EMAIL = "Email"
    SMS = "SMS"
    ALL = "All"

class NotificationsStatus(Enum):
    """Enumeration for notification status states."""
    UNREAD = "Unread"
    READ = "Read"
    ARCHIVED = "Archived"

class AppNotificationModel:
    """
    Model for application notifications sent to users.
    
    This model handles notifications created during various system events,
    such as when brand processing is complete and ready for user review.
    
    Attributes:
        id (UUID): Unique identifier for the notification
        user_id (UUID): ID of the user who will receive the notification
        title (str): Notification title/subject
        message (str): Notification body content
        type (NotificationType): Type/severity of notification (INFO, WARNING, ERROR)
        cta (str): Call-to-action text for notification buttons
        redirect_link (str): URL to redirect user when notification is clicked
        status (NotificationsStatus): Current status (UNREAD, READ, ARCHIVED)
        created_at (datetime): Timestamp when notification was created
        delivery_type (List[NotificationDeliveryType]): Methods for delivering notification
        priority (int): Priority level (1-10, higher = more important)
    """
    
    def __init__(self,
                 id: Optional[uuid.UUID] = None,
                 user_id: Optional[uuid.UUID] = None,
                 title: Optional[str] = None,
                 message: Optional[str] = None,
                 type: NotificationType = NotificationType.INFO,
                 cta: Optional[str] = None,
                 redirect_link: Optional[str] = None,
                 status: NotificationsStatus = NotificationsStatus.ARCHIVED,
                 created_at: datetime = datetime.now(timezone.utc),
                 delivery_type: Optional[List[NotificationDeliveryType]] = None,
                 priority: int = 1):
        """
        Initialize a new AppNotificationModel instance.
        
        Args:
            id: Unique identifier (auto-generated if not provided)
            user_id: Target user for the notification
            title: Notification title
            message: Notification content
            type: Notification type (defaults to INFO)
            cta: Call-to-action text
            redirect_link: URL for user redirection
            status: Initial status (defaults to ARCHIVED)
            created_at: Creation timestamp (defaults to current UTC time)
            delivery_type: List of delivery methods (defaults to empty list)
            priority: Priority level 1-10 (defaults to 1)
        """
        self.id = id or uuid.uuid4()
        self.user_id = user_id
        self.title = title
        self.message = message
        self.type = type
        self.cta = cta
        self.redirect_link = redirect_link
        self.status = status
        self.created_at = created_at
        self.delivery_type = delivery_type or []
        self.priority = priority

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the notification model to a dictionary representation.
        
        Returns:
            Dict containing all notification fields with proper type conversion
            for database storage and API responses.
        """
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "title": self.title,
            "message": self.message,
            "type": self.type.value if self.type else None,
            "cta": self.cta,
            "redirect_link": self.redirect_link,
            "status": self.status.value if self.status else None,
            "created_at": self.created_at.isoformat(),
            "delivery_type": [dt.value for dt in self.delivery_type] if self.delivery_type else None,
            "priority": self.priority,
        }

    @staticmethod
    def to_obj(data: Dict[str, Any]) -> 'AppNotificationModel':
        """
        Create an AppNotificationModel instance from dictionary data.
        
        This method safely handles data conversion from database responses
        or API requests, with proper null checking and type conversion.
        
        Args:
            data: Dictionary containing notification data from database or API
            
        Returns:
            AppNotificationModel instance with all fields properly converted
            
        Note:
            Uses empty list default for delivery_type to prevent iteration
            errors when the field is null in the database.
        """
        return AppNotificationModel(
            id=uuid.UUID(data["id"]) if data.get("id") else None,
            user_id=uuid.UUID(data["user_id"]) if data.get("user_id") else None,
            title=data.get("title"),
            message=data.get("message"),
            type=NotificationType(data["type"]) if data.get("type") else NotificationType.INFO,
            cta=data.get("cta"),
            redirect_link=data.get("redirect_link"),
            status=NotificationsStatus(data["status"]) if data.get("status") else NotificationsStatus.ARCHIVED,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            delivery_type=[NotificationDeliveryType(dt) for dt in data.get("delivery_type", [])] if data.get("delivery_type") else None,
            priority=int(data["priority"]) if data.get("priority") is not None else 1,
        )


