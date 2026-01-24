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
# Initialize Rate Limiter (e.g., 5 requests per minute per IP for analysis)
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="SeeCVs PRO - Production Backend")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Security: Restrict to your production domain
origins = [
    "https://seecvs.com",
    "https://www.seecvs.com",
    "http://localhost:4200" # Keep for local testing if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"], # Restrict methods
    allow_headers=["Content-Type", "Authorization"],
)

# Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response

app.add_middleware(SecurityHeadersMiddleware)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB limit

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
            if len(reader.pages) > 15: # Prevent extremely long PDFs
                return "Error: PDF too long."
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception:
        return ""
    return text

def read_docx(path: str) -> str:
    try:
        return docx2txt.process(path)
    except Exception:
        return ""

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"score": 0, "comment": "AI response format error."}

# ================= AI LOGIC =================
# ================= AI LOGIC UPDATED =================

def analyze_cv_with_ai(cv_text: str, job_desc: str, mode: str = "rank", lang: str = "en") -> dict:
    # Set the target language instruction
    target_lang = "Arabic" if lang == "ar" else "English"
    
    if mode == "ats":
        system_content = f"You are an expert Resume Strategist. You always respond in JSON format. All text in 'comment' MUST be in {target_lang}."
        user_prompt = f"""
        Compare the CV with the Job Description. Provide a compatibility score from 0-100.
        
        CRITICAL: The "comment" field MUST be a JSON array containing exactly 3 separate strings.
        Do NOT include bullet points (like • or -) inside the strings.
        The analysis MUST be written in {target_lang}.
        
        Example JSON:
        {{
            "score": 85,
            "comment": ["Point 1 in {target_lang}", "Point 2 in {target_lang}", "Point 3 in {target_lang}"]
        }}

        JD: {job_desc}
        CV: {cv_text[:6000]}
        """
    else:
        system_content = f"You are an expert HR Recruiter. You always respond in JSON format. All text in 'comment' MUST be in {target_lang}."
        user_prompt = f"""
        Compare the CV with the Job Description. Provide a match score from 0-100.
        "comment" MUST be a JSON array of 3-4 strings summarizing the match.
        Do NOT use bullet point symbols inside the strings.
        The entire analysis MUST be in {target_lang}.
        
        JD: {job_desc}
        CV CONTENT: {cv_text[:7000]}
        """
        
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1, # Lower temperature for stricter formatting
        response_format={ 'type': 'json_object' }
    )
    
    return extract_json(response.choices[0].message.content)
# ================= API ENDPOINTS =================
@app.post("/analyze-cvs")
@limiter.limit("10/minute") # Rate limit: 10 requests per minute per IP
async def analyze_cvs(
    request: Request,
    files: List[UploadFile] = File(...),
    job_description: str = Form(...),
    notes: str = Form(""),
    lang: str = Form("en")  # <--- CRITICAL: Capture the language here
):
    if not files:
        raise HTTPException(400, "No files uploaded")
    
    mode = "ats" if notes == "ats-mode" else "rank"
    results = []

    for file in files:
        # ... (Your existing file validation and reading logic) ...
        file_content = await file.read()
        # ... (Assuming 'text' is extracted here) ...

        if len(text) < 100:
            results.append({"filename": file.filename, "score": 0, "comment": ["Insufficient text."]})
        else:
            # CRITICAL: Pass 'lang' to the AI function
            ai_result = analyze_cv_with_ai(text, job_description, mode=mode, lang=lang)
            results.append({
                "filename": file.filename,
                "score": ai_result.get("score", 0),
                "comment": ai_result.get("comment", [])
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"mode": mode, "results": results}
@app.get("/health")
def health():
    return {"status": "online"}