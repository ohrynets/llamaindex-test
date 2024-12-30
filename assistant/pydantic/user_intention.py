from pydantic import BaseModel, Field

class UserIntent(BaseModel):
    """Data model for a user intent identification."""

    user_intent: str = Field(alias="user_intent", description="User intention from the list AUTODESK_SEARCH, JIRA_TICKET_CREATION, and RETRIEVAL")
    justification: str = Field(alias="justification", description="Justification why the model belives that the identified intent is correct")
    user_input: str = Field(alias="user_input", description="The specific user input that the model is trying to classify")