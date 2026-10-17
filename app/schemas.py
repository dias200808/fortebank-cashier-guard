from typing import Literal

from pydantic import BaseModel


class EvidenceFrame(BaseModel):
    timestamp_sec: float
    image_base64: str


class RuleRef(BaseModel):
    act: str
    point: str
    strength: str
    url: str | None = None


class Violation(BaseModel):
    type: str
    timecode_sec: float
    description: str
    evidence_frames: list[EvidenceFrame]
    rule_id: str | None = None
    rule: RuleRef | None = None
    confidence: float | None = None


class JobResult(BaseModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Literal["violation", "no_violation"] | None = None
    violations: list[Violation] = []
    error: str | None = None

    def to_api(self) -> dict:
        data = self.model_dump(exclude_none=True)
        if self.status != "done":
            data.pop("violations", None)
        return data
