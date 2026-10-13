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

class StartTurn(BaseModel):
    """One question to Pengoo, sent by Java (AssistantContextBuilder)."""
    model_config = ConfigDict(extra="allow")
    turnId: str
    conversationId: str
    conversationType: str = "TRIP"
    locale: str = "vi"
    user: dict[str, Any] = Field(default_factory=dict)
    trip: dict[str, Any] | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    question: str = ""
    callbackBaseUrl: str
    limits: dict[str, Any] = Field(default_factory=dict)
    logPayloads: bool = True
