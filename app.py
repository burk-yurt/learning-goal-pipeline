import os, json
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field, field_validator

import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
app = FastAPI(title="Learning Goal Pipeline API", version="1.0.0")
def check_key(x_api_key: Optional[str]):
    expected = os.getenv("ACTION_API_KEY")
    # If you set ACTION_API_KEY in your environment, enforce it; otherwise skip
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ---- Types ----
Bloom = Literal["Remember","Understand","Apply","Analyze","Evaluate","Create"]
Knowledge = Literal["Factual","Conceptual","Procedural","Metacognitive"]

class ContextBlock(BaseModel):
    domain: str
    topic: str
    audience: str
    constraints: List[str]
    stakeholder_priorities: List[str]

# ---- LLM helper ----
async def call_llm(messages, response_json: bool = True):
    if not OPENAI_API_KEY:
        raise HTTPException(500, "Server misconfigured: OPENAI_API_KEY is not set.")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 8192}
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content) if response_json else content


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

    @field_validator("broad_goals")
    @classmethod
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
        "Return ONLY JSON with this exact shape:\n"
        "{ \"tagged_goals\": ["
        "  {\"goal\":\"...\",\"bloom_process\":\"...\",\"knowledge_type\":\"...\",\"rationale\":\"<=12 words\"}"
        "]}\n"
        "Keys must be exactly: goal, bloom_process, knowledge_type, rationale. "
        "Prefer higher-order + conceptual/procedural at top layer. Rationale ≤12 words."
    )
    user = json.dumps({"broad_goals": req.broad_goals}, ensure_ascii=False)
    data = await call_llm(
        [{"role":"system","content":prompt},{"role":"user","content":user}],
        response_json=True
    )

    # --- normalize common alias keys from the model ---
    def norm_case(s: str) -> str:
        return s.capitalize() if isinstance(s, str) else s

    raw = data.get("tagged_goals", [])
    fixed = []
    for g in raw:
        if not isinstance(g, dict): 
            continue
        fixed.append({
            "goal": g.get("goal") or g.get("Goal") or "",
            "bloom_process": g.get("bloom_process") or g.get("process") or g.get("Bloom") or "",
            "knowledge_type": g.get("knowledge_type") or g.get("knowledge") or g.get("Knowledge") or "",
            "rationale": g.get("rationale") or g.get("justification") or g.get("why") or ""
        })

    # sanitize enums & trim rationale
    allowed_bloom = {"Remember","Understand","Apply","Analyze","Evaluate","Create"}
    allowed_knowledge = {"Factual","Conceptual","Procedural","Metacognitive"}
    for g in fixed:
        # normalize case
        g["bloom_process"] = norm_case(g["bloom_process"])
        g["knowledge_type"] = norm_case(g["knowledge_type"])
        # clamp to allowed values if close matches
        if g["bloom_process"] not in allowed_bloom:
            # simple fallback: try title-case match or default to 'Apply'
            g["bloom_process"] = g["bloom_process"].title()
            if g["bloom_process"] not in allowed_bloom:
                g["bloom_process"] = "Apply"
        if g["knowledge_type"] not in allowed_knowledge:
            g["knowledge_type"] = g["knowledge_type"].title()
            if g["knowledge_type"] not in allowed_knowledge:
                g["knowledge_type"] = "Conceptual"
        # rationale max 120 chars
        if isinstance(g["rationale"], str) and len(g["rationale"]) > 120:
            g["rationale"] = g["rationale"][:117] + "..."
    return {"tagged_goals": fixed}

# ---- Step 4: Decomposition via Task Analysis ----
class Objective(BaseModel):
    id: str
    goal_id: Optional[str] = None     # which broad goal it came from
    behavior: str                     # measurable verb
    condition: str                    # tools/data/context
    degree: str                       # criteria, accuracy/speed/quality
    bloom: Literal["Remember","Understand","Apply","Analyze","Evaluate","Create"]
    knowledge: Literal["Factual","Conceptual","Procedural","Metacognitive"]

class DecomposeReq(BaseModel):
    tagged_goals: List[TaggedGoal]
    context: ContextBlock

class DecomposeResp(BaseModel):
    objectives: List[Objective]

@app.post("/decompose-objectives", response_model=DecomposeResp)
async def decompose(req: DecomposeReq, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    prompt = (
        "Decompose each broad goal into 3–8 measurable sub-objectives.\n"
        "Use ABCD structure: behavior (measurable verb), condition (tools/data), degree (criteria), "
        "and include Bloom + knowledge per objective.\n"
        "Return ONLY JSON {\"objectives\":[{"
        "\"id\":\"OBJ-1\",\"goal_id\":\"<index or text>\","
        "\"behavior\":\"...\",\"condition\":\"...\",\"degree\":\"...\","
        "\"bloom\":\"Apply\",\"knowledge\":\"Procedural\"}]} . "
        "Each statement must pass M.O.S.T. (Measurable verb, Observable product, Specific conditions, Threshold criteria)."
    )
    user = json.dumps({"tagged_goals":[tg.model_dump() for tg in req.tagged_goals],
                       "context": req.context.model_dump()}, ensure_ascii=False)
    data = await call_llm(
        [{"role":"system","content":prompt},{"role":"user","content":user}],
        response_json=True
    )
    # minimal sanity
    objs = data.get("objectives", [])
    if not isinstance(objs, list) or not objs:
        raise HTTPException(500, "Model did not return objectives.")
    # normalize enums
    ok_b = {"Remember","Understand","Apply","Analyze","Evaluate","Create"}
    ok_k = {"Factual","Conceptual","Procedural","Metacognitive"}
    fixed=[]
    for i,o in enumerate(objs,1):
        b=o.get("bloom","Apply").title()
        k=o.get("knowledge","Procedural").title()
        if b not in ok_b: b="Apply"
        if k not in ok_k: k="Procedural"
        fixed.append(Objective(
            id=o.get("id") or f"OBJ-{i}",
            goal_id=o.get("goal_id"),
            behavior=o.get("behavior",""),
            condition=o.get("condition",""),
            degree=o.get("degree",""),
            bloom=b, knowledge=k
        ))
    return {"objectives": fixed}

# ---- Step 5: Dependency Graphing ----
class Edge(BaseModel):
    src: str
    dst: str
    type: Literal["requires prior concept mastery","requires prior routine"]

class GraphReq(BaseModel):
    objectives: List[Objective]

class GraphResp(BaseModel):
    edges: List[Edge]
    from typing import Dict
# ...
prerequisites: Dict[str, List[str]]


@app.post("/graph-dependencies", response_model=GraphResp)
async def graph(req: GraphReq, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    prompt = (
        "Infer prerequisite edges among these objectives.\n"
        "Use types: 'requires prior concept mastery' (conceptual) or 'requires prior routine' (procedural).\n"
        "Return ONLY JSON {\"edges\":[{\"src\":\"OBJ-1\",\"dst\":\"OBJ-3\",\"type\":\"requires prior concept mastery\"}],"
        "\"prerequisites\":{\"OBJ-3\":[\"OBJ-1\"]}} . Acyclic graph; if you detect cycles, break them and prefer conceptual before procedural."
    )
    user = json.dumps({"objectives":[o.model_dump() for o in req.objectives]}, ensure_ascii=False)
    data = await call_llm(
        [{"role":"system","content":prompt},{"role":"user","content":user}],
        response_json=True
    )
    edges = data.get("edges", [])
    prereq = data.get("prerequisites", {})
    # make acyclic with simple cycle break if necessary
    G = {o.id: set() for o in req.objectives}
    for e in edges:
        s, d = e.get("src"), e.get("dst")
        if s in G and d in G and s != d:
            G[s].add(d)
    # detect cycles via DFS
    seen, stack = set(), set()
    acyclic = []
    def dfs(u):
        stack.add(u)
        for v in list(G[u]):
            if v in stack:
                G[u].remove(v)  # break cycle
            elif v not in seen:
                dfs(v)
        stack.remove(u); seen.add(u)
    for n in G: 
        if n not in seen: dfs(n)
    # rebuild edges list from G
    for s in G:
        for d in G[s]:
            # find matching type or default
            t = "requires prior concept mastery"
            for e in edges:
                if e.get("src")==s and e.get("dst")==d:
                    t = e.get("type", t)
                    break
            acyclic.append(Edge(src=s,dst=d,type=t))
    # recompute prerequisites
    prereq_fixed = {n: [] for n in G}
    for e in acyclic:
        prereq_fixed[e.dst].append(e.src)
    return {"edges": acyclic, "prerequisites": prereq_fixed}

# ---- Step 6: Difficulty & Priority Scoring ----
class ScoredObjective(Objective):
    difficulty: int
    priority: int
    rationale: str

class ScoreReq(BaseModel):
    objectives: List[Objective]
    edges: List[Edge]
    stakeholder_priorities: List[str]

class ScoreResp(BaseModel):
    scored: List[ScoredObjective]

@app.post("/score-difficulty-priority", response_model=ScoreResp)
async def score(req: ScoreReq, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    # compute simple centrality = in-degree + out-degree
    ids = {o.id: o for o in req.objectives}
    ind = {i:0 for i in ids}; outd = {i:0 for i in ids}
    for e in req.edges:
        if e.src in outd: outd[e.src]+=1
        if e.dst in ind: ind[e.dst]+=1
    # baseline difficulty by Bloom
    bloom_weight = {"Remember":1,"Understand":2,"Apply":3,"Analyze":4,"Evaluate":4,"Create":5}
    scored=[]
    for o in req.objectives:
        dep_load = ind[o.id]
        base = bloom_weight.get(o.bloom,3)
        diff = max(1, min(5, base + (1 if dep_load>=2 else 0)))
        central = ind[o.id] + outd[o.id]
        # priority heuristic: higher if central or mentioned by stakeholders
        pr = 3 + (1 if central>=2 else 0)
        sp = " ".join(req.stakeholder_priorities).lower()
        if any(k in sp for k in [o.behavior.lower(), o.knowledge.lower(), o.bloom.lower()]):
            pr = min(5, pr+1)
        rationale = f"Difficulty from Bloom={o.bloom} and deps={dep_load}; priority from centrality={central} and stakeholder cues."
        scored.append(ScoredObjective(**o.model_dump(), difficulty=diff, priority=pr, rationale=rationale))
    return {"scored": scored}

# ---- Step 7: SMART Refinement ----
class SmartReq(BaseModel):
    objectives: List[ScoredObjective]

class SmartResp(BaseModel):
    smart_objectives: List[ScoredObjective]

@app.post("/refine-smart", response_model=SmartResp)
async def smart(req: SmartReq, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    prompt = (
        "Rewrite each objective to be SMART. Keep the behavior verb, clarify conditions, and set numeric/qualitative criteria. "
        "No vague verbs (know/understand/learn). Prefer time/accuracy/quality metrics. "
        "Return ONLY JSON {\"smart_objectives\":[{"
        "\"id\":\"OBJ-1\",\"behavior\":\"...\",\"condition\":\"...\",\"degree\":\"...\","
        "\"bloom\":\"...\",\"knowledge\":\"...\",\"difficulty\":1,\"priority\":1}]} ."
    )
    user = json.dumps({"objectives":[o.model_dump() for o in req.objectives]}, ensure_ascii=False)
    data = await call_llm(
        [{"role":"system","content":prompt},{"role":"user","content":user}],
        response_json=True
    )
    arr = data.get("smart_objectives", [])
    if not arr: 
        # fall back: return as-is
        return {"smart_objectives": req.objectives}
    # cast back into ScoredObjective
    fixed=[]
    for o in arr:
        try:
            fixed.append(ScoredObjective(**{
                **{k:o.get(k) for k in ["id","goal_id","behavior","condition","degree","bloom","knowledge"]},
                "difficulty": int(o.get("difficulty",3)),
                "priority": int(o.get("priority",3)),
                "rationale": "Refined to SMART."
            }))
        except Exception:
            pass
    return {"smart_objectives": fixed or req.objectives}

# ---- Step 8: Coverage & Gap Analysis ----
class CoverageReq(BaseModel):
    objectives: List[ScoredObjective]

class CoverageResp(BaseModel):
    distribution: dict
    gaps: List[str]
    duplicates: List[List[str]]
    suggestions: List[str]

@app.post("/coverage-gap-analysis", response_model=CoverageResp)
async def coverage(req: CoverageReq, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    # distributions
    bloom_dist, know_dist = {}, {}
    for o in req.objectives:
        bloom_dist[o.bloom]=bloom_dist.get(o.bloom,0)+1
        know_dist[o.knowledge]=know_dist.get(o.knowledge,0)+1
    # naive near-duplicate by Jaccard on content words
    def tokens(s): 
        return {w.lower().strip(".,;:!?") for w in s.split() if len(w)>2}
    dup_pairs=[]
    objs = {o.id:o for o in req.objectives}
    ids = list(objs.keys())
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            a,b = objs[ids[i]], objs[ids[j]]
            ta, tb = tokens(a.behavior+" "+a.condition), tokens(b.behavior+" "+b.condition)
            jac = len(ta&tb)/max(1,len(ta|tb))
            if jac >= 0.7:
                dup_pairs.append([a.id,b.id])
    # quick LLM suggestions
    prompt = (
        "Given Bloom/Knowledge distributions and duplicate pairs, suggest merges/splits and identify taxonomy gaps. "
        "Return ONLY JSON {\"gaps\":[\"...\"],\"suggestions\":[\"...\"]} . "
        "Ensure at least one Analyze/Evaluate/Create objective unless scope forbids."
    )
    user = json.dumps({"bloom":bloom_dist,"knowledge":know_dist,"duplicates":dup_pairs}, ensure_ascii=False)
    data = await call_llm(
        [{"role":"system","content":prompt},{"role":"user","content":user}],
        response_json=True
    )
    return {
        "distribution":{"bloom":bloom_dist,"knowledge":know_dist},
        "gaps": data.get("gaps", []),
        "duplicates": dup_pairs,
        "suggestions": data.get("suggestions", [])
    }

# ---- Step 9: Sequencing Proposal ----
class SequenceReq(BaseModel):
    objectives: List[ScoredObjective]
    edges: List[Edge]
    from pydantic import Field
constraints: List[str] = Field(default_factory=list)

class Module(BaseModel):
    name: str
    objective_ids: List[str]
    checkpoint: Optional[str] = None
    rationale: Optional[str] = None

class SequenceResp(BaseModel):
    modules: List[Module]

@app.post("/sequence-modules", response_model=SequenceResp)
async def sequence(req: SequenceReq, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    # topological sort
    from collections import defaultdict, deque
    indeg=defaultdict(int); adj=defaultdict(list)
    ids={o.id for o in req.objectives}
    for e in req.edges:
        if e.src in ids and e.dst in ids:
            adj[e.src].append(e.dst); indeg[e.dst]+=1
    q=deque([i for i in ids if indeg[i]==0])
    order=[]
    while q:
        u=q.popleft(); order.append(u)
        for v in adj[u]:
            indeg[v]-=1
            if indeg[v]==0: q.append(v)
    if len(order)!=len(ids):
        # fallback: append remaining
        order += [i for i in ids if i not in order]
    # simple grouping into 4–8 modules by contiguous chunks
    mcount = min(8, max(4, (len(order)+4)//5))  # ~5 per module
    size = max(1, len(order)//mcount)
    groups=[order[i:i+size] for i in range(0,len(order),size)]
    groups=groups[:mcount] + ([] if len(groups)<=mcount else [sum(groups[mcount:],[])])
    # brief LLM rationale
    prompt = (
        "Provide brief rationales and checkpoints for each module grouping. "
        "Return ONLY JSON {\"modules\":[{\"name\":\"Module 1\",\"checkpoint\":\"...\",\"rationale\":\"...\"}]} ."
    )
    user = json.dumps({"groups":groups,"constraints":req.constraints}, ensure_ascii=False)
    data = await call_llm(
        [{"role":"system","content":prompt},{"role":"user","content":user}],
        response_json=True
    )
    metas = data.get("modules", [])
    modules=[]
    for i,g in enumerate(groups,1):
        meta = metas[i-1] if i-1 < len(metas) else {}
        modules.append(Module(
            name=meta.get("name", f"Module {i}"),
            objective_ids=g,
            checkpoint=meta.get("checkpoint"),
            rationale=meta.get("rationale")
        ))
    return {"modules": modules}


# ---- Step 10: Final Sanity & Risk Review ----
class ReviewReq(BaseModel):
    context: ContextBlock
    objectives: List[ScoredObjective]
    edges: List[Edge]
    modules: List[Module]

class ReviewResp(BaseModel):
    assumptions: List[str]
    low_confidence: List[str]
    questions: List[str]

@app.post("/sanity-risk-review", response_model=ReviewResp)
async def review(req: ReviewReq, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)
    prompt = (
        "List assumptions, low-confidence items (≤0.6), and 3–6 clarifying questions to improve validity. "
        "Must include at least one scope/constraint question. "
        "Return ONLY JSON {\"assumptions\":[\"...\"],\"low_confidence\":[\"...\"],\"questions\":[\"...\"]} ."
    )
    user = json.dumps({"context":req.context.model_dump(),
                       "objectives":[o.model_dump() for o in req.objectives],
                       "edges":[e.model_dump() for e in req.edges],
                       "modules":[m.model_dump() for m in req.modules]}, ensure_ascii=False)
    data = await call_llm(
        [{"role":"system","content":prompt},{"role":"user","content":user}],
        response_json=True
    )
    return {
        "assumptions": data.get("assumptions", []),
        "low_confidence": data.get("low_confidence", []),
        "questions": data.get("questions", [])
    }

# ---- Export Relational Types ----
class Table(BaseModel):
    name: str
    columns: List[str]
    rows: List[List[str]]

class ExportReq(BaseModel):
    context: ContextBlock
    tagged_goals: List[TaggedGoal]
    objectives: List[Objective]  # or ScoredObjective if you have that class; otherwise Objective
    edges: List[Edge]
    modules: List[Module]

class ExportResp(BaseModel):
    tables: List[Table]
    sqlite_ddl: str

@app.post("/export-relational", response_model=ExportResp)
async def export_relational(req: ExportReq, x_api_key: Optional[str] = Header(None)):
    check_key(x_api_key)

    # 1) Mint stable GOAL ids (GOAL-1..N) and a resolver to map any incoming goal_id field
    goal_ids = {i: f"GOAL-{i}" for i in range(1, len(req.tagged_goals) + 1)}

    def resolve_goal_id(gid):
        if gid is None:
            return None
        s = str(gid).strip()
        if s.isdigit():
            i = int(s)
            return goal_ids.get(i)
        if s.upper().startswith("GOAL-"):
            return s.upper()
        return None  # unknown mapping

    # 2) Build tables
    goals_tbl = Table(
        name="goals",
        columns=["id", "goal_text", "bloom_process", "knowledge_type"],
        rows=[
            [f"GOAL-{i}", tg.goal, tg.bloom_process, tg.knowledge_type]
            for i, tg in enumerate(req.tagged_goals, start=1)
        ],
    )

    # Ensure each objective has an id; if your pipeline already assigns ids, use those
    def obj_id(o, idx):
        oid = getattr(o, "id", None)
        return oid if oid else f"OBJ-{idx}"

    objectives_rows = []
    for idx, o in enumerate(req.objectives, start=1):
        # Support SMART fields if present; fall back to blanks
        behavior = getattr(o, "behavior", "")
        condition = getattr(o, "condition", "")
        degree = getattr(o, "degree", "")
        bloom = getattr(o, "bloom", getattr(o, "bloom_process", ""))
        knowledge = getattr(o, "knowledge", getattr(o, "knowledge_type", ""))
        difficulty = getattr(o, "difficulty", None)
        priority = getattr(o, "priority", None)
        goal_id = resolve_goal_id(getattr(o, "goal_id", None))
        objectives_rows.append([
            obj_id(o, idx), goal_id, behavior, condition, degree,
            bloom, knowledge, difficulty, priority
        ])

    objectives_tbl = Table(
        name="objectives",
        columns=["id","goal_id","behavior","condition","degree","bloom","knowledge","difficulty","priority"],
        rows=objectives_rows,
    )

    edges_tbl = Table(
        name="edges",
        columns=["src_objective_id","dst_objective_id","type"],
        rows=[[e.src, e.dst, e.type] for e in req.edges],
    )

    # Module ids M-1..N
    mod_ids = {i: f"M-{i}" for i in range(1, len(req.modules) + 1)}
    modules_tbl = Table(
        name="modules",
        columns=["id","name","checkpoint","rationale"],
        rows=[
            [mod_ids[i], getattr(m, "name", f"Module {i}"),
             getattr(m, "checkpoint", "") or "",
             getattr(m, "rationale", "") or ""]
            for i, m in enumerate(req.modules, start=1)
        ],
    )

    modobj_tbl = Table(
        name="module_objectives",
        columns=["module_id","objective_id"],
        rows=[
            [mod_ids[i], oid]
            for i, m in enumerate(req.modules, start=1)
            for oid in getattr(m, "objective_ids", [])
        ],
    )

    context_tbl = Table(
        name="context",
        columns=["key","value"],
        rows=[
            ["domain", req.context.domain],
            ["topic", req.context.topic],
            ["audience", req.context.audience],
            *[["constraint", c] for c in req.context.constraints],
            *[["stakeholder_priority", s] for s in req.context.stakeholder_priorities],
        ],
    )

    # 3) DDL for a clean, minimal relational schema
    ddl = """
CREATE TABLE goals(
  id TEXT PRIMARY KEY,
  goal_text TEXT,
  bloom_process TEXT,
  knowledge_type TEXT
);
CREATE TABLE objectives(
  id TEXT PRIMARY KEY,
  goal_id TEXT REFERENCES goals(id),
  behavior TEXT,
  condition TEXT,
  degree TEXT,
  bloom TEXT,
  knowledge TEXT,
  difficulty INTEGER,
  priority INTEGER
);
CREATE TABLE edges(
  src_objective_id TEXT REFERENCES objectives(id),
  dst_objective_id TEXT REFERENCES objectives(id),
  type TEXT,
  PRIMARY KEY (src_objective_id, dst_objective_id)
);
CREATE TABLE modules(
  id TEXT PRIMARY KEY,
  name TEXT,
  checkpoint TEXT,
  rationale TEXT
);
CREATE TABLE module_objectives(
  module_id TEXT REFERENCES modules(id),
  objective_id TEXT REFERENCES objectives(id),
  PRIMARY KEY (module_id, objective_id)
);
CREATE TABLE context(
  key TEXT,
  value TEXT
);
""".strip()

    return {"tables": [goals_tbl, objectives_tbl, edges_tbl, modules_tbl, modobj_tbl, context_tbl],
            "sqlite_ddl": ddl}
