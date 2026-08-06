from typing import Any
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

class StartJob(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    jobId: str = Field(validation_alias=AliasChoices("jobId", "job_id"))
    attemptId: str = Field(validation_alias=AliasChoices("attemptId", "attempt_id"))
    request: dict[str, Any] = Field(default_factory=dict,
                                    validation_alias=AliasChoices("request", "payload", "input"))
    locale: str = "en"
    callbackBaseUrl: str = Field(validation_alias=AliasChoices(
        "callbackBaseUrl", "callback_base_url", "callbackUrl", "callback_url"))

class CandidateQuery(BaseModel):
    latitude: float
    longitude: float
    placeGroups: list[str] = Field(default_factory=list)
    limit: int = 250
