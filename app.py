import os
import json
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from loguru import logger
from router_logic import decide_route

MODEL_DIR = os.getenv("MODEL_DIR","saved_model")
confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

class InferenceRequest(BaseModel):
    text: str
    metadata: Optional[dict] = {}   # e.g., {"is_vip": True}
    session_id: Optional[str] = None

class InferenceResponse(BaseModel):
    predicted_intent: str
    confidence: float
    department: Optional[str]
    decision_reason: str
    clarification_needed: bool
    clarification_prompt: Optional[str] = None
    
app = FastAPI(title="Chat Routing ML Service")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
with open(os.path.join(MODEL_DIR, "id2intent.json"), "r") as f:
    id2intent = json.load(f)

def predict_intent(text: str):
    inputs = tokenizer(text,truncation=True,padding=True,max_length = 128,return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        top_idx = int(probs.argmax())
        confidence = float(probs[top_idx])
        intent_label = id2intent.get(str(top_idx), id2intent.get(top_idx,"unknown"))
        return intent_label, confidence, probs

def create_clarification_prompt(text, top_intents):
    if len(top_intents) == 1:
        return f"Do you want help with: {top_intents[0].replace('_',' ')}?"
    else:
        options = ", or ".join([t.replace("_"," ") for t in top_intents])
        return f"I'm not sure — did you mean {options}? Please choose one."
    
@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    intent, confidence, probs = predict_intent(req.text)
    import numpy as np
    top_indices = np.argsort(probs)[-3:][::-1]
    top_intents = [id2intent.get(str(int(idx)), id2intent.get(int(idx), "unknown")) for idx in top_indices]
    clarification_needed = confidence < confidence_threshold
    clarification_prompt = None
    department = None
    decision_reason = ""
    if clarification_needed:
        clarification_prompt = create_clarification_prompt(req.text, top_intents[:2])
        decision_reason = "low_confidence_clarification"
    else:
        department, decision_reason = decide_route(intent, confidence, req.metadata)

    logger.info({
        "text": req.text,
        "predicted_intent": intent,
        "confidence": confidence,
        "top_intents": top_intents,
        "department": department,
        "decision_reason": decision_reason,
        "session_id": req.session_id
    })

    return InferenceResponse(
        predicted_intent=intent,
        confidence=confidence,
        department=department,
        decision_reason=decision_reason,
        clarification_needed=clarification_needed,
        clarification_prompt=clarification_prompt
    )

@app.post("/clarify/{chosen_intent}", response_model=InferenceResponse)
def clarify(chosen_intent: str, req: InferenceRequest):
    department, decision_reason = decide_route(chosen_intent, 1.0, req.metadata)
    logger.info({
        "text": req.text,
        "clarified_intent": chosen_intent,
        "department": department,
        "session_id": req.session_id
    })
    return InferenceResponse(
        predicted_intent=chosen_intent,
        confidence=1.0,
        department=department,
        decision_reason="clarified_by_user",
        clarification_needed=False,
        clarification_prompt=None
    )

@app.post("/feedback")
def feedback(session_id: str, correct_intent: str, correct_department: Optional[str] = None):
    import json, time
    entry = {
        "session_id": session_id,
        "correct_intent": correct_intent,
        "correct_department": correct_department,
        "timestamp": int(time.time())
    }
    os.makedirs("feedback", exist_ok=True)
    with open("feedback/feedback_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(f"Feedback recorded: {entry}")
    return {"status": "ok"}
        