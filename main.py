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
            f"You are a strict ATS (Applicant Tracking System) scoring engine. "
            f"You must respond ONLY in {target_lang}. Return ONLY valid JSON.\n"
            "You evaluate CVs harshly and realistically. Most CVs have significant issues.\n"
            "A score of 90+ means near-perfect ATS optimization — this is extremely rare.\n"
            "A typical CV scores between 40-65. Only truly outstanding CVs score above 80."
        )

        user_prompt = f"""
        Analyze this CV against the Job Description for ATS compatibility.
        Be STRICT and REALISTIC. Do NOT inflate the score.

        SCORING RUBRIC (deduct points for each issue):
        - Keyword match with job description (0-25 pts): Exact skill/tool matches, industry terms, certifications mentioned in JD
        - Role relevance & experience alignment (0-25 pts): Years of experience match, seniority level match, industry match
        - ATS formatting compliance (0-20 pts): No tables/columns/graphics, standard section headings, clean text parsing, no headers/footers with critical info
        - Quantified achievements (0-15 pts): Metrics, numbers, percentages showing impact. Vague statements like "responsible for" score 0 here
        - Structure & completeness (0-15 pts): Contact info, summary, experience with dates, education, skills section

        DEDUCTIONS:
        - Missing critical keywords from JD: -10 to -20
        - No quantified achievements at all: -15
        - Generic objective/summary not tailored to role: -10
        - Gaps or unclear employment dates: -5
        - Missing skills section: -10
        - Too short (under 300 words) or too long (over 1000 words for 1 page): -5
        - Irrelevant experience listed: -5

        You MUST return two lists:
        - "improvements": actionable things the user SHOULD DO to improve their CV (add, fix, enhance)
        - "warnings": things the user should REMOVE or STOP doing (bad practices, irrelevant content, formatting issues)

        ALL text in improvements and warnings MUST be in {target_lang}.

        JSON FORMAT:
        {{
          "score": number,
          "improvements": ["actionable suggestion 1", "actionable suggestion 2", "..."],
          "warnings": ["thing to remove or avoid 1", "thing to remove or avoid 2", "..."]
        }}

        Job Description:
        {job_desc[:4000]}

        CV:
        {cv_text[:4000]}
        """

    else:
        system_prompt = (
            f"You are a strict senior HR recruiter. You must respond ONLY in {target_lang}. "
            f"Return ONLY valid JSON.\n"
            "You evaluate candidates harshly and realistically. Most candidates are average.\n"
            "A score of 90+ means an exceptional candidate — this is extremely rare.\n"
            "A typical candidate scores between 40-65. Only truly outstanding ones score above 80."
        )

        user_prompt = f"""
        Evaluate this candidate's suitability for the role.
        Be STRICT and REALISTIC. Do NOT inflate the score.

        SCORING RUBRIC:
        - Direct skill match with requirements (0-30 pts): Hard skills, tools, technologies explicitly required
        - Experience relevance & depth (0-25 pts): Years, seniority, industry alignment
        - Achievements & impact (0-20 pts): Quantified results, promotions, notable projects
        - Education & certifications (0-15 pts): Relevant degrees, professional certifications
        - Overall presentation & clarity (0-10 pts): Well-structured, concise, professional

        DEDUCTIONS:
        - Missing critical required skills: -10 to -20
        - Experience in unrelated field: -15
        - No measurable achievements: -10
        - Over-qualified or under-qualified: -10
        - Poorly structured or hard to read: -5

        You MUST return:
        - "comment": a brief 2-3 sentence summary of the candidate's profile (who they are, key skills, experience level)
        - "improvements": actionable things the user SHOULD DO to improve their CV (add, fix, enhance)
        - "warnings": things the user should REMOVE or STOP doing (bad practices, irrelevant content, formatting issues)

        ALL text in comment, improvements and warnings MUST be in {target_lang}.

        JSON FORMAT:
        {{
          "score": number,
          "comment": ["short summary sentence 1", "short summary sentence 2"],
          "improvements": ["actionable suggestion 1", "actionable suggestion 2", "..."],
          "warnings": ["thing to remove or avoid 1", "thing to remove or avoid 2", "..."]
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

    # Build backward-compatible comment from improvements + warnings
    improvements = parsed.get("improvements", [])
    warnings = parsed.get("warnings", [])
    comment = parsed.get("comment", [])

    return {
        "score": normalize_score(parsed.get("score")),
        "comment": comment,
        "improvements": improvements,
        "warnings": warnings,
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
                    "comment": ai_result["comment"],
                    "improvements": ai_result.get("improvements", []),
                    "warnings": ai_result.get("warnings", []),
                })

        finally:
            if os.path.exists(path):
                os.remove(path)

    return {"mode": mode, "results": results}

@app.get("/health")
def health():
    return {"status": "online"}
