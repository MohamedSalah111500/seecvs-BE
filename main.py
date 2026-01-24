from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import List
import os, json, uuid, re
import PyPDF2
import docx2txt
from openai import OpenAI

# ================= CONFIG & SECURITY =================
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="SeeCVs PRO")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = ["https://seecvs.com", "https://www.seecvs.com", "http://localhost:4200"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_FILE_SIZE = 5 * 1024 * 1024 

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'), 
    base_url="https://api.deepseek.com"
)

# ================= HELPERS =================
def read_pdf(path: str) -> str:
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception: return ""
    return text

def read_docx(path: str) -> str:
    try: return docx2txt.process(path)
    except Exception: return ""

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"score": 0, "comment": ["AI response format error."]}

# ================= AI LOGIC =================
def analyze_cv_with_ai(cv_text: str, job_desc: str, mode: str = "rank", lang: str = "en") -> dict:
    target_lang = "Arabic" if lang == "ar" else "English"
    
    system_content = f"You are an expert HR. Respond in JSON. Language: {target_lang}."
    
    if mode == "ats":
        user_prompt = f"""Score CV (0-100) vs JD. 
        'comment' MUST be a JSON array of 3 strings in {target_lang}. No bullets.
        JD: {job_desc} | CV: {cv_text[:6000]}"""
    else:
        user_prompt = f"""Score CV (0-100) vs JD. 
        'comment' MUST be a JSON array of 3-4 strings in {target_lang}. No bullets.
        JD: {job_desc} | CV: {cv_text[:7000]}"""
        
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        response_format={ 'type': 'json_object' }
    )
    return extract_json(response.choices[0].message.content)

# ================= API ENDPOINTS =================
@app.post("/analyze-cvs") # Matches your Angular Call
@limiter.limit("20/minute")
async def analyze_cvs(
    request: Request,
    files: List[UploadFile] = File(...),
    job_description: str = Form(...),
    notes: str = Form(""),
    lang: str = Form("en") # Captures the 'ar' or 'en'
):
    mode = "ats" if notes == "ats-mode" else "rank"
    results = []

    for file in files:
        # 1. Validation
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            results.append({"filename": file.filename, "score": 0, "comment": ["File too large"]})
            continue

        ext = file.filename.split(".")[-1].lower()
        file_id = str(uuid.uuid4())
        path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")

        try:
            with open(path, "wb") as f:
                f.write(file_content)

            # 2. Text Extraction (Fixed NameError)
            raw_text = read_pdf(path) if ext == "pdf" else read_docx(path)
            text = clean_text(raw_text)

            if len(text) < 50:
                results.append({"filename": file.filename, "score": 0, "comment": ["Insufficient text found"]})
            else:
                # 3. AI Analysis
                ai_result = analyze_cv_with_ai(text, job_description, mode, lang)
                results.append({
                    "filename": file.filename,
                    "score": ai_result.get("score", 0),
                    "comment": ai_result.get("comment", [])
                })
        finally:
            if os.path.exists(path): os.remove(path)

    return {"results": results}

@app.get("/health")
def health(): return {"status": "online"}