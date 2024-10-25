from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Literal, ClassVar
import datetime
import openai
from app import settings

openai.api_key = settings.OPEN_AI_KEY




class OpenAIModels():
    def __init__(self):
        self.response_models = openai.Engine.list(models = True)
        

    def check_model(self, model: str):
        open_ai_models = [data_object["id"] for data_object in self.response_models["data"]]
        if model not in open_ai_models:
            return False
        return True

    def list_models(self):
        open_ai_models = [data_object["id"] for data_object in self.response_models["data"]]
        return open_ai_models
    

class DemographicModel(BaseModel):
    sex_at_birth: Optional[str] = None
    age: Optional[str] = None
    marital_status: Optional[str] = None
    sexual_orientation: Optional[str] = None
    nationality: Optional[str] = None
    country_of_residence: Optional[str] = None
    state_province: Optional[str] = None
    city: Optional[str] = None
    rural_or_urban: Optional[str] = None
    type_of_residence: Optional[str] = None
    length_of_residence: Optional[str] = None
    level_of_education: Optional[str] = None
    student_status: Optional[str] = None
    field_of_study: Optional[str] = None
    occupational_area: Optional[str] = None
    annual_income_level: Optional[str] = None
    employment_status: Optional[str] = None
    home_ownership: Optional[str] = None
    ethnicity: Optional[str] = None
    languages_spoken: Optional[str] = None
    religion: Optional[str] = None
    cultural_practices: Optional[str] = None
    immigration_status: Optional[str] = None
    hobbies_and_interests: Optional[str] = None
    shopping_motivations: Optional[str] = None
    shopping_habits: Optional[str] = None
    shopping_channels: Optional[str] = None
    shopping_frequency: Optional[str] = None
    dietary_preferences: Optional[str] = None
    physical_activity_levels: Optional[str] = None
    social_media_usage: Optional[str] = None
    travel_habits: Optional[str] = None
    alcohol_use: Optional[str] = None
    tobacco_and_vape_use: Optional[str] = None
    technology_usage: Optional[str] = None
    family_structure: Optional[str] = None
    household_size: Optional[str] = None
    number_of_children: Optional[str] = None
    pet_ownership: Optional[str] = None
    number_of_pets: Optional[str] = None
    relationship_status: Optional[str] = None
    caregiving_responsibilities: Optional[str] = None
    general_health_status: Optional[str] = None
    disabilities_or_chronic_illnesses: Optional[str] = None
    mental_health_status: Optional[str] = None
    health_insurance_status: Optional[str] = None
    access_to_healthcare: Optional[str] = None
    political_affiliation: Optional[str] = None
    voting_behavior: Optional[str] = None
    political_engagement: Optional[str] = None

    class Config:
        extra = "forbid"  # Forbids any extra fields not defined in the model




class MultipleChoiceQuestion(BaseModel):
    type: str = Field("multiple choice", Literal=True)
    question: str
    choices: list[str]
    answer: None

class CheckboxesQuestion(BaseModel):
    type: str = Field("checkboxes", Literal=True)
    question: str
    choices: list[str]
    answer: None

class RankingQuestion(BaseModel):
    type: str = Field("ranking", Literal=True)
    answer_format: Optional[str] = "Any"
    question: str
    choices: list[str]
    answer: None

class LinearScaleQuestion(BaseModel):
    type: str = Field("linear scale", Literal=True)
    question: str
    min_value: int = Field(..., strict=True)
    max_value: int = Field(..., strict=True)
    answer: None
    @validator('min_value')
    def check_min_value(cls, v):
        if v < 0:
            raise ValueError('min_value cannot be less than 0')
        return v

    @validator('max_value')
    def check_max_value(cls, v):
        if v > 10:
            raise ValueError('max_value cannot be greater than 10')
        return v
    
class ShortAnswerQuestion(BaseModel):
    type: str = Field("short answer", Literal=True)
    answer_format: Optional[str] = "string"
    question: str
    answer: None
    
class LongAnswerQuestion(BaseModel):
    type: str = Field("long answer", Literal=True)
    answer_format: Optional[str] = "string"
    question: str
    answer: None

class SurveyModel(BaseModel):
    name: str
    context: Optional[list[str]] = None
    questions: List[Union[MultipleChoiceQuestion, CheckboxesQuestion, LinearScaleQuestion, RankingQuestion, ShortAnswerQuestion, LongAnswerQuestion]]

    @validator('questions', each_item=True)
    def check_question_type(cls, v):
        if v.type not in ["short answer", "long answer", "multiple choice", "checkboxes", "ranking", "linear scale"]:
            raise ValueError('Invalid question type')
        return v

    class Config:
        validate_assignment = True  # Enforces validation for nested assignments

class AgentProfile(BaseModel):
    persona: str
    demographic: dict

class AgentParameters(BaseModel):

    agent_model: Optional[str] = "gpt-3.5-turbo"
    embedding_model: Optional[str] = "text-embedding-3-small"
    llm_temperature: Optional[float] = 1.21  #for the llm temp
    agent_temperature: Optional[float] = 0.1   #between 0 and 1, if 1 means that the agent is erratic and nuts
    existance_date: Optional[str] = datetime.date.today().isoformat()
    json_mode: Optional[bool] = True

    #st_memory_params
    memory_context_length: Optional[int] = 4196 #in tokens
    max_output_length: Optional[int] = 512  #in tokens
    lt_memory_trigger_length: Optional[int] = 1024 #in tokens
    chunk_size: Optional[int] = 512   #in tokens

    embedding_dimension: Optional[int] = 512
    

    #lt_memory params , chunk size for lt_memory is calculated as chunk_size/(reconstruction_top_n+1)
    memory_limit: Optional[int] = 1000  #in chunks 
    memory_loss_factor: Optional[float] = 0.1
    sampling_top_n: Optional[int] = 3 #in chunks
    reconstruction_top_n: Optional[int] = 3 #in chunks
    reconstruction_factor: Optional[float] = 0.1  # percentage of memory limit
    class Config:
        extra="forbid"


class SimulationInstance(BaseModel):
    simulation_id: str
    agent_profile: AgentProfile
    agent_params: AgentParameters
    iterations: SurveyModel

    class Config:
        validate_assignment = True  # Enables recursive validation
        extra = "forbid"





    