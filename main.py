from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import List
import os, json, uuid, re, io
import PyPDF2
import docx2txt
from openai import OpenAI

# Google Drive Imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# ================= CONFIG =================
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

MAX_FILE_SIZE = 5 * 1024 * 1024
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")

client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")

# ================= LOCAL SAVE =================
def save_file_locally(file_bytes: bytes, filename: str) -> str:
    unique_name = f"{uuid.uuid4()}_{filename}"
    path = os.path.join(UPLOAD_DIR, unique_name)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path

# ================= GOOGLE DRIVE LOGIC =================
def get_drive_service():
    if not GOOGLE_CREDS_JSON:
        return None
    try:
        creds_dict = json.loads(GOOGLE_CREDS_JSON)
        scopes = ['https://www.googleapis.com/auth/drive.file']
        creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return build('drive', 'v3', credentials=creds)
    except Exception:
        return None

def upload_to_drive(file_content: bytes, filename: str):
    service = get_drive_service()
    if not service or not DRIVE_FOLDER_ID:
        return None
    try:
        file_metadata = {
            'name': f"{uuid.uuid4()}_{filename}",
            'parents': [DRIVE_FOLDER_ID]
        }
        media = MediaIoBaseUpload(
            io.BytesIO(file_content),
            mimetype='application/pdf',
            resumable=True
        )
        uploaded_file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id',
            supportsAllDrives=True
        ).execute()
        return uploaded_file.get('id')
    except Exception:
        return None

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
def analyze_cv_with_ai(cv_text: str, job_desc: str, mode: str = "rank", lang: str = "en") -> dict:
    target_lang = "Arabic" if lang == "ar" else "English"
    system_prompt = f"You are a Senior Recruiter and ATS Expert. Respond in JSON only. Language: {target_lang}."
    user_prompt = f"""
Analyze the CV against the Job Description.
Mode: {mode.upper()}

RULES:
- Score: 0-100
- Comment: List of strings

Job Description: {job_desc[:4000]}
CV Content: {cv_text[:4000]}
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
    job_description: str = Form(...),
    notes: str = Form(""),
    lang: str = Form("en")
):
    mode = "ats" if "ats" in notes.lower() else "rank"
    results = []

    for file in files:
        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            results.append({"filename": file.filename, "score": 0, "comment": ["File too large"]})
            continue

        ext = file.filename.split(".")[-1].lower()
        if ext not in ["pdf", "docx"]:
            results.append({"filename": file.filename, "score": 0, "comment": ["Unsupported format"]})
            continue

        # ✅ SAVE LOCALLY (NEW)
        local_path = save_file_locally(content, file.filename)

        # OPTIONAL: Google Drive (can be removed later)
        drive_id = upload_to_drive(content, file.filename)

        raw_text = read_pdf(content) if ext == "pdf" else read_docx(content)
        text = clean_text(raw_text)

        if len(text) < 100:
            results.append({"filename": file.filename, "score": 0, "comment": ["Unreadable CV"]})
            continue

        ai_result = analyze_cv_with_ai(text, job_description, mode, lang)

        results.append({
            "filename": file.filename,
            "score": ai_result["score"],
            "comment": ai_result["comment"],
            "local_path": local_path,
            "drive_id": drive_id
        })

    return {"mode": mode, "results": results}

@app.get("/health")
def health():
    return {"status": "online"}
