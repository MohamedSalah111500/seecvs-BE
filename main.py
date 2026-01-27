from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import List
import os, json, uuid, re
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

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_FILE_SIZE = 5 * 1024 * 1024

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ================= HELPERS =================
def read_pdf(path: str) -> str:
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""

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
        return {"score": 0, "comment": ["AI response parsing failed"]}

def normalize_score(value) -> int:
    try:
        score = int(float(str(value).replace("%", "")))
        return max(0, min(score, 100))
    except Exception:
        return 0

# ================= AI ANALYSIS =================
def analyze_cv_with_ai(
    cv_text: str,
    job_desc: str,
    mode: str = "rank",
    lang: str = "en"
) -> dict:

    target_lang = "Arabic" if lang == "ar" else "English"

    if mode == "ats":
        system_prompt = (
            "You are an ATS scoring engine.\n"
            "Score CV strictly using:\n"
            "- Keyword match\n"
            "- Role relevance\n"
            "- ATS formatting\n"
            "Return ONLY valid JSON."
        )

        user_prompt = f"""
        Analyze CV against Job Description for ATS compatibility.

        RULES:
        - Score MUST be a number between 0 and 100.
        - Comments can be ANY number of strings.
        - No bullets or numbering.

        JSON FORMAT:
        {{
          "score": number,
          "comment": ["string", "string", "..."]
        }}

        Job Description:
        {job_desc[:4000]}

        CV:
        {cv_text[:4000]}
        """

    else:
        system_prompt = (
            f"You are a senior HR recruiter. Respond in JSON only. Language: {target_lang}."
        )

        user_prompt = f"""
        Evaluate candidate suitability for the role.

        RULES:
        - Score between 0 and 100.
        - Comments can be ANY number of strings.

        JSON FORMAT:
        {{
          "score": number,
          "comment": ["string", "string", "..."]
        }}

        Job Description:
        {job_desc[:5000]}

        CV:
        {cv_text[:5000]}
        """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
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
            results.append({
                "filename": file.filename,
                "score": 0,
                "comment": ["File size exceeds 5MB limit"]
            })
            continue

        ext = file.filename.split(".")[-1].lower()
        if ext not in ["pdf", "docx"]:
            results.append({
                "filename": file.filename,
                "score": 0,
                "comment": ["Unsupported file format"]
            })
            continue

        file_id = str(uuid.uuid4())
        path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")

        try:
            with open(path, "wb") as f:
                f.write(content)

            raw_text = read_pdf(path) if ext == "pdf" else read_docx(path)
            text = clean_text(raw_text)

            if len(text) < 100:
                results.append({
                    "filename": file.filename,
                    "score": 0,
                    "comment": ["Insufficient readable content"]
                })
            else:
                ai_result = analyze_cv_with_ai(
                    cv_text=text,
                    job_desc=job_description,
                    mode=mode,
                    lang=lang
                )

                results.append({
                    "filename": file.filename,
                    "score": ai_result["score"],
                    "comment": ai_result["comment"]
                })

        finally:
            if os.path.exists(path):
                os.remove(path)

    return {"mode": mode, "results": results}

@app.get("/health")
def health():
    return {"status": "online"}
