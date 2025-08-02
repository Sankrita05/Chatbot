from fastapi import FastAPI, APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict
from uuid import uuid4
import os
import requests
from motor.motor_asyncio import AsyncIOMotorClient

from app.auth import get_current_user, User  # adjust if auth is elsewhere
from app.upload import collection
from app.config.settings import settings

app = FastAPI()
router = APIRouter()

# === MongoDB Setup ===
client = AsyncIOMotorClient(settings.MONGODB_URI)
db = client.chatbot_db
session_collection = db.chat_sessions

# === Prompt Paths ===
PROMPT_PATHS = {
    "role1": "app/prompts/hr.txt",
    "role2": "app/prompts/employee.txt",
    "role3": "app/prompts/employer.txt"
}

# === Models ===
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    role: str
    language: Optional[str] = "English"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[str] = []

# === Utilities ===
def search_similar_chunks(query: str, top_k: int = 5) -> List[str]:
    results = collection.query(query_texts=[query], n_results=top_k)
    return results.get("documents", [[]])[0]

async def get_session_messages(session_id: str) -> List[Dict[str, str]]:
    session = await session_collection.find_one({"session_id": session_id})
    return session.get("messages", []) if session else []

async def save_message(session_id: str, role: str, content: str):
    await session_collection.update_one(
        {"session_id": session_id},
        {"$push": {"messages": {"role": role, "content": content}}},
        upsert=True
    )

def load_prompt(role: str, context: str, query: str, language: str) -> str:
    prompt_file = PROMPT_PATHS.get(role)
    if not prompt_file or not os.path.exists(prompt_file):
        raise HTTPException(status_code=400, detail=f"Prompt not found for role '{role}'")
    with open(prompt_file, "r", encoding="utf-8") as f:
        template = f.read()
    return template.replace("{context}", context.strip()).replace("{query}", query.strip()).replace("{language}", language.strip())

def is_relevant(chunk: str, answer: str, threshold: int = 5) -> bool:
    chunk_words = set(chunk.lower().split())
    answer_words = set(answer.lower().split())
    return len(chunk_words & answer_words) >= threshold

async def generate_response(session_id: str, query: str, role: str, language: str, context_chunks: List[str]) -> (str, List[str]):
    await save_message(session_id, "user", query)
    context = "\n".join(context_chunks)
    prompt = load_prompt(role, context, query, language)
    history = [{"role": "system", "content": prompt}] + await get_session_messages(session_id)

    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": history,
        "temperature": 0.7
    }

    response = requests.post(settings.DEEPSEEK_API_URL, json=payload, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    assistant_reply = response.json()["choices"][0]["message"]["content"]
    await save_message(session_id, "assistant", assistant_reply)

    used_sources = [chunk for chunk in context_chunks if is_relevant(chunk, assistant_reply)]
    return assistant_reply, used_sources

# === Main Chat Endpoint ===
@router.post("/chat/ask", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, current_user: User = Depends(get_current_user)):
    try:
        if payload.role not in PROMPT_PATHS:
            raise HTTPException(status_code=400, detail="Invalid role. Use 'role1', 'role2', or 'role3'.")

        session_id = payload.session_id or str(uuid4())
        context_chunks = search_similar_chunks(payload.query)

        if not context_chunks:
            fallback_msg = "Sorry, we do not have a response available in our sources. Please try different terms."
            await save_message(session_id, "user", payload.query)
            await save_message(session_id, "system", fallback_msg)
            return ChatResponse(response=fallback_msg, session_id=session_id, sources=[])

        answer, sources_used = await generate_response(
            session_id=session_id,
            query=payload.query,
            role=payload.role,
            language=payload.language,
            context_chunks=context_chunks
        )

        fail_keywords = [
            "i cannot answer that", "i couldn't find", "not related to the provided context",
            "no information", "not covered", "unable to locate", "i'm sorry, but"
        ]

        if any(phrase in answer.lower() for phrase in fail_keywords):
            fallback_msg = "Sorry, we do not have a response available in our sources. Please try different terms."
            await save_message(session_id, "system", fallback_msg)
            return ChatResponse(response=fallback_msg, session_id=session_id, sources=[])

        return ChatResponse(response=answer, session_id=session_id, sources=sources_used)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Mount router ===
app.include_router(router)
