import os, json
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field, validator
import httpx

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

app = FastAPI(title="Learning Goal Pipeline API", version="1.0.0")

# ---- Types ----
Bloom = Literal["Remember","Understand","Apply","Analyze","Evaluate","Create"]
Knowledge = Literal["Factual","Conceptual","Procedural","Metacognitive"]

class ContextBlock(BaseModel):
    domain: str
    topic: str
    audience: str
    constraints: List[str]
    stakeholder_priorities: List[str]

class TaggedGoal(BaseModel):
    goal: str
    bloom_process: Bloom
    knowledge_type: Knowledge
    rationale: str = Field(max_length=120)

# ---- LLM helper ----
async def call_llm(messages, response_json: bool = True):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"model": MODEL, "messages": messages}
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, json=payload)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content) if response_json else content

def check_key(x_api_key: Optional[str]):
    expected = os.getenv("ACTION_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, "Invalid or missing API key.")

# ---- Step 1 endpoint: Ingest & Contextualize ----
class IngestReq(BaseModel):
    raw_text: str
class IngestResp(ContextBlock): pass

@app.post("/ingest-contextualize", response_model=IngestResp)
async def ingest(req: IngestReq, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    system = (
        "Extract domain/topic, audience, constraints, and stakeholder priorities from the text. "
        "Return ONLY JSON with keys domain, topic, audience, constraints[], stakeholder_priorities[]. "
        "If uncertain, infer conservatively and append '?' to the item. Do NOT list objectives."
    )
    user = f"Text:\n{req.raw_text}"
    data = await call_llm(
        [{"role":"system","content":system},{"role":"user","content":user}],
        response_json=True
    )
    required = ["domain","topic","audience","constraints","stakeholder_priorities"]
    if any(k not in data for k in required):
        raise HTTPException(500, "Model response missing required keys.")
    return data

# health check
@app.get("/healthz")
def health():
    return {"ok": True}


# ---- Step 2: Goal Harvest ----
class HarvestReq(BaseModel):
    source_text: str
    context: ContextBlock

class HarvestResp(BaseModel):
    broad_goals: List[str]
    @validator("broad_goals")
    def count_range(cls, v):
        if not (3 <= len(v) <= 7):
            raise ValueError("Must list 3–7 goals.")
        return v

@app.post("/harvest-goals", response_model=HarvestResp)
async def harvest(req: HarvestReq, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    prompt = (
        "List 3–7 broad, student-observable end-state abilities implied by the text. "
        "Avoid teaching activities. One sentence each, no metrics yet. "
        "Quality check: No bare verbs like 'learn/understand'—use observable verbs. "
        "Return JSON with key 'broad_goals'."
    )
    user = f"Context:\n{req.context.model_dump()}\n\nSource:\n{req.source_text}"
    data = await call_llm(
        [{"role":"system","content":prompt},{"role":"user","content":user}],
        response_json=True
    )
    if "broad_goals" not in data or not isinstance(data["broad_goals"], list):
        raise HTTPException(500, "Model did not return 'broad_goals' array.")
    return data

# ---- Step 3: Bloom & Knowledge Tagging ----
class TagReq(BaseModel):
    broad_goals: List[str]

class TaggedGoal(BaseModel):
    goal: str
    bloom_process: Literal["Remember","Understand","Apply","Analyze","Evaluate","Create"]
    knowledge_type: Literal["Factual","Conceptual","Procedural","Metacognitive"]
    rationale: str = Field(max_length=120)

class TagResp(BaseModel):
    tagged_goals: List[TaggedGoal]

@app.post("/tag-bloom-knowledge", response_model=TagResp)
async def tag(req: TagReq, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    prompt = (
        "For each broad goal, assign Bloom process {Remember, Understand, Apply, Analyze, Evaluate, Create} "
        "and knowledge {Factual, Conceptual, Procedural, Metacognitive}. "
        "Briefly justify tagging in ≤12 words. Prefer higher-order + conceptual/procedural at top layer. "
        "Return JSON: tagged_goals[]."
    )
    user = json.dumps({"broad_goals": req.broad_goals}, ensure_ascii=False)
    data = await call_llm(
        [{"role":"system","content":prompt},{"role":"user","content":user}],
        response_json=True
    )
    if "tagged_goals" not in data:
        raise HTTPException(500, "Missing 'tagged_goals'.")
    return data
