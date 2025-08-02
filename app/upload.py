# ========= Imports ========= #
import os
import io
from pprint import pprint
import uuid
import tempfile
import pdfplumber
import easyocr
import cv2
import numpy as np
import chromadb
from PIL import Image


from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from app.config.settings import settings


from app.auth import get_current_user
from app.auth import User

# ========= Pydantic Schema ========= #
class UploadResponse(BaseModel):
    id: str
    filename: str
    status: str
    user: str
    text: str

# ========= Utils ========= #
def save_upload_file(file: UploadFile, upload_dir: str) -> str:
    os.makedirs(upload_dir, exist_ok=True)
    filename = file.filename.replace(" ", "_")
    file_path = os.path.join(upload_dir, filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path

# ========= OCR ========= #
def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(file_path)
    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        return extract_text_from_video(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    
def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Failed to read .txt file: {e}")


    
def extract_text_from_pdf(file_path: str) -> str:
    reader = easyocr.Reader(['en'], gpu=False)
    text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # 1. Extract selectable text
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    # 2. If no text, try OCR on image
                    image = page.to_image(resolution=300).original
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()

                    ocr_result = reader.readtext(img_byte_arr, detail=0)
                    if ocr_result:
                        text += " ".join(ocr_result) + "\n"
    except Exception as e:
        # Fallback to OCR if PDF is invalid
        print(f"[WARN] pdfplumber failed: {e}. Falling back to image-based OCR.")
        try:
            # Try to convert to image and OCR the whole thing
            from pdf2image import convert_from_path
            pages = convert_from_path(file_path)
            for page in pages:
                temp_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
                page.save(temp_path, "JPEG")
                ocr_result = reader.readtext(temp_path, detail=0)
                if ocr_result:
                    text += " ".join(ocr_result) + "\n"
                os.remove(temp_path)
        except Exception as ocr_error:
            raise ValueError(f"PDF processing failed: {ocr_error}")

    return text.strip()


def extract_text_from_image(file_path: str) -> str:
    reader = easyocr.Reader(['en'])
    results = reader.readtext(file_path, detail=0)
    return " ".join(results)

def extract_text_from_video(file_path: str, sample_rate_sec: int = 2) -> str:
    reader = easyocr.Reader(['en'], gpu=False)
    text = []
    
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {file_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("Failed to get FPS from video.")

    frame_interval = int(fps * sample_rate_sec)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_num in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        if not success:
            continue

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
            temp_path = temp.name
            cv2.imwrite(temp_path, frame)

        try:
            result = reader.readtext(temp_path, detail=0)
            if result:
                text.append(" ".join(result))
        finally:
            os.remove(temp_path)

    cap.release()
    return "\n".join(text)

# ========= Vectorizer ========= #
def load_model():
    model_path = os.path.join(settings.MODEL_DIR, settings.MODEL_NAME)
    if not os.path.exists(model_path):
        model = SentenceTransformer(settings.MODEL_NAME)
        model.save(model_path)
    return SentenceTransformer(model_path)

model = load_model()

def get_text_embeddings(text: str):
    return model.encode([text])[0]


# ========= ChromaDB Handler ========= #
client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
collection = client.get_or_create_collection(name=settings.COLLECTION_NAME)

def store_vector(doc_id, text, embedding):
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    collection.add(
        documents=[text],
        embeddings=[embedding.tolist()],
        ids=[doc_id]
    )

def search_vectors(query_vector, k=3):
    results = collection.query(query_embeddings=[query_vector.tolist()], n_results=k)
    return results['documents'][0]

def view_all_vectors():
    return collection.get(include=["documents", "embeddings", "metadatas"])



# ========= FastAPI Route ========= #
router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    try:
        file_path = save_upload_file(file, settings.UPLOAD_DIR)

        text = extract_text(file_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No extractable text found.")

        embedding = get_text_embeddings(text)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Embedding failed.")

        doc_id = str(uuid.uuid4())
        store_vector(doc_id, text, embedding)
        return UploadResponse(
            id=doc_id,
            filename=file.filename,
            status="success",
            user=current_user.username,
            text=text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
