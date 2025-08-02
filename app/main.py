from fastapi import FastAPI
from app.auth import router as auth_router
from app.upload import router as upload_router
from app.chatbot import router as chat_router

app = FastAPI(title="APIs", version="1.0.0")

# Register all endpoints
app.include_router(auth_router, prefix="/api", tags=["Authentication"])
app.include_router(upload_router, prefix="/api", tags=["Upload"])
app.include_router(chat_router, prefix="/api", tags=["Chatbot"]) 
