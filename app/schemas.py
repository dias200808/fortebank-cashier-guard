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
    region: list[float] | None = None    # normalized [x1, y1, x2, y2] (0..1, top-left origin) of the area concerned


MAX_EVIDENCE = 3  # TZ section 10: 1-3 evidence frames per violation


class JobResult(BaseModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Literal["violation", "no_violation"] | None = None
    violations: list[Violation] = []
    hints: list[Violation] | None = None  # optional, not in the TZ: moments worth a look that do not affect the verdict
    error: str | None = None

    def to_api(self) -> dict:
        data = self.model_dump(exclude_none=True)
        if self.status != "done":
            data.pop("violations", None)
            data.pop("hints", None)
            return data
        if not data.get("hints"):  # an empty list adds nothing to the TZ response
            data.pop("hints", None)
        if self.verdict is None:  # a done result always carries the TZ verdict
            data["verdict"] = "violation" if self.violations else "no_violation"
        for v in data["violations"]:
            v["evidence_frames"] = v["evidence_frames"][:MAX_EVIDENCE]
        return data
