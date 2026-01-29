# schema.py
from typing import Literal, Optional, List
from pydantic import BaseModel, Field

class QuerySpec(BaseModel):
    action: Literal["get_metric_timeseries"] = "get_metric_timeseries"
    patient_id: str = Field(..., description="e.g., '45_M'")
    metric: str = Field(..., description="one metric column name")
    date_start: str = Field(..., description="start date inclusive")
    date_end: str = Field(..., description="end date inclusive")
    game: Optional[str] = Field(None, description="optional e.g. 'game0'")
    session: Optional[str] = Field(None, description="optional e.g. 'session_2'")
    return_columns: List[str] = Field(default_factory=lambda: ["date", "patient_id", "metric_value"])