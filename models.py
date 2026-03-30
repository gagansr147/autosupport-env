from pydantic import BaseModel
from typing import List

class Observation(BaseModel):
    ticket_id: int
    customer_query: str
    sentiment: str
    category: str
    history: List[str]

class Action(BaseModel):
    action_type: str  # reply, escalate, request_info
    message: str

class Reward(BaseModel):
    score: float
    reason: str