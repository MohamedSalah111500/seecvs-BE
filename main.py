from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import List
import os, json, uuid, re, io
import PyPDF2
import docx2txt
from openai import OpenAI

# ================= CONFIG =================
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="SeeCVs PRO")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = [
    "https://seecvs.com",
    "https://www.seecvs.com",
    "http://localhost:4200"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 5 * 1024 * 1024
UPLOAD_BASE_DIR = "uploads"
os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")

# ================= FILE STORAGE =================
def safe_folder_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s_-]", "", name)
    return re.sub(r"\s+", "_", name)

def save_cv_locally(file_bytes: bytes, filename: str, job_title: str) -> str:
    job_folder = safe_folder_name(job_title)
    job_path = os.path.join(UPLOAD_BASE_DIR, job_folder)
    os.makedirs(job_path, exist_ok=True)

    unique_name = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(job_path, unique_name)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    return file_path

# ================= HELPERS =================
def read_pdf(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""

def read_docx(file_bytes: bytes) -> str:
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        text = docx2txt.process(tmp_path)
        os.remove(tmp_path)
        return text
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
        return {"score": 0, "comment": ["AI response parsing failed"]}

def normalize_score(value) -> int:
    try:
        score = int(float(str(value).replace("%", "")))
        return max(0, min(score, 100))
    except Exception:
        return 0

# ================= AI ANALYSIS =================
def analyze_cv_with_ai(cv_text: str, job_desc: str, mode: str, lang: str) -> dict:
    target_lang = "Arabic" if lang == "ar" else "English"

    system_prompt = (
        f"You are a Senior Recruiter and ATS Expert. "
        f"Respond in JSON only. Language: {target_lang}."
    )

    user_prompt = f"""
Analyze the CV against the Job Description.
Mode: {mode.upper()}

RULES:
- Score: 0-100
- Comment: List of strings

Job Description:
{job_desc[:4000]}

CV Content:
{cv_text[:4000]}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    parsed = extract_json(response.choices[0].message.content)
    return {
        "score": normalize_score(parsed.get("score")),
        "comment": parsed.get("comment", [])
    }

# ================= API =================
@app.post("/analyze-cvs")
@limiter.limit("20/minute")
async def analyze_cvs(
    request: Request,
    files: List[UploadFile] = File(...),
    job_title: str = Form(...),
    job_description: str = Form(...),
    notes: str = Form(""),
    lang: str = Form("en")
):
    mode = "ats" if "ats" in notes.lower() else "rank"
    results = []

    for file in files:
        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            results.append({
                "filename": file.filename,
                "score": 0,
                "comment": ["File too large (Max 5MB)"]
            })
            continue

        ext = file.filename.split(".")[-1].lower()
        if ext not in ["pdf", "docx"]:
            results.append({
                "filename": file.filename,
                "score": 0,
                "comment": ["Unsupported format"]
            })
            continue

        # 1️⃣ Save file locally under job folder
        stored_path = save_cv_locally(content, file.filename, job_title)

        # 2️⃣ Extract text
        raw_text = read_pdf(content) if ext == "pdf" else read_docx(content)
        text = clean_text(raw_text)

        if len(text) < 100:
            results.append({
                "filename": file.filename,
                "score": 0,
                "comment": ["Could not read CV content"]
            })
            continue

        # 3️⃣ AI analysis
        ai_result = analyze_cv_with_ai(text, job_description, mode, lang)

        results.append({
            "filename": file.filename,
            "score": ai_result["score"],
            "comment": ai_result["comment"],
            "stored_at": stored_path
        })

    return {
        "job": job_title,
        "mode": mode,
        "results": results
    }

@app.get("/health")
def health():
    return {
        "status": "online",
        "storage": "local",
        "uploads_dir": UPLOAD_BASE_DIR
    }
