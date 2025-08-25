# Streamlit Library Manager — Students registry with Pinecone semantic search
# Extended: adds mobile number, ID proof, host authentication, and optional SMS reminders on registration.
# Save as: library_manager_with_mobile_and_auth.py

import os
import sys
import uuid
import time
import hashlib
import sqlite3
import tempfile
import csv
import calendar
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Tuple

# Optional libraries — guarded imports
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None
    STREAMLIT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    ST_AVAILABLE = False

# Pinecone SDK (optional)
try:
    # keep import generic — user may have different pinecone package versions
    import pinecone
    PINECONE_SDK = True
except Exception:
    pinecone = None
    PINECONE_SDK = False

# Twilio optional for SMS
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TwilioClient = None
    TWILIO_AVAILABLE = False

# Pandas is optional — used only for nicer displays and CSV convenience
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    pd = None
    PANDAS_AVAILABLE = False

# -----------------------------
# Config & constants
# -----------------------------
INDEX_NAME = os.environ.get("PINECONE_INDEX", "library-students-idx")
INDEX_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
INDEX_REGION = os.environ.get("PINECONE_REGION", "us-east-1")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384 if ST_AVAILABLE else 128  # fallback dim if no ST model
DB_PATH_DEFAULT = os.environ.get("STUDENTS_DB", "students.db")

# SMS / Twilio env vars
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER")
# Use DISABLE_SMS=1 to prevent sending SMS even if Twilio is configured
DISABLE_SMS = os.environ.get("DISABLE_SMS", "0") in ("1", "true", "True")

# Admin credentials (put secure values into environment in production)
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "admin123")

# -----------------------------
# Utilities
# -----------------------------

def mask_aadhaar(aadhaar: str) -> str:
    a = ''.join(ch for ch in aadhaar if ch.isdigit())
    if len(a) < 4:
        return "****"
    return f"XXXX-XXXX-{a[-4:]}"


def mask_mobile(mobile: str) -> str:
    d = ''.join(ch for ch in mobile if ch.isdigit())
    if len(d) < 4:
        return "****"
    return f"XXXX-XXX-{d[-4:]}"


def hash_aadhaar(aadhaar: str) -> str:
    a = ''.join(ch for ch in aadhaar if ch.isdigit())
    return hashlib.sha256(a.encode("utf-8")).hexdigest()


def validate_mobile(mobile: str) -> bool:
    d = ''.join(ch for ch in mobile if ch.isdigit())
    return len(d) == 10

# -----------------------------
# Embedding (sentence-transformers if available, else a small hashing embedder)
# -----------------------------
class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL_NAME, dim: int = EMBED_DIM):
        self.dim = dim
        if ST_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.encode = lambda texts: self.model.encode(texts, show_progress_bar=False)
                self.using = "sentence-transformers"
            except Exception:
                # fallback
                self.model = None
                self.encode = self._hash_encode
                self.using = "fallback-hash"
        else:
            self.model = None
            self.encode = self._hash_encode
            self.using = "fallback-hash"

    def _hash_encode(self, texts):
        # Accepts single string or list
        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            # simple token counts hashed into dim buckets
            for w in str(t).lower().split():
                idx = abs(hash(w)) % self.dim
                vec[idx] += 1.0
            # L2 normalize
            norm = sum(x * x for x in vec) ** 0.5
            if norm == 0.0:
                norm = 1.0
            vec = [x / norm for x in vec]
            out.append(vec)
        return out[0] if single else out

# -----------------------------
# Local in-memory index (fallback for Pinecone)
# -----------------------------
class LocalIndex:
    def __init__(self, dim: int):
        self.dim = dim
        # storage: id -> {values, metadata}
        self._store: Dict[str, Dict[str, Any]] = {}

    def upsert(self, vectors: List[Dict[str, Any]]):
        for v in vectors:
            _id = v["id"]
            vals = v.get("values")
            meta = v.get("metadata", {})
            self._store[_id] = {"values": vals, "metadata": meta}

    def query(self, vector: List[float], top_k: int = 10, include_values=False, include_metadata=True):
        # cosine similarity
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        results: List[Tuple[float, str, Dict[str, Any]]] = []
        for _id, it in self._store.items():
            vals = it.get("values")
            if vals is None or len(vals) != len(vector):
                continue
            score = dot(vector, vals)
            results.append((score, _id, it.get("metadata", {})))
        results.sort(key=lambda x: x[0], reverse=True)
        matches = []
        for score, _id, meta in results[:top_k]:
            matches.append({"id": _id, "score": float(score), "metadata": meta})
        return {"matches": matches}

    def update(self, id: str, set_metadata: Dict[str, Any]):
        if id in self._store:
            self._store[id]["metadata"].update(set_metadata)

# -----------------------------
# Pinecone wrapper (uses real Pinecone if available & configured, else LocalIndex)
# -----------------------------
class PineconeWrapper:
    def __init__(self, dim: int, index_name: str = "library"):
        self.dim = dim
        self.index_name = index_name
        self.index = None
        self.using = None
        api_key = os.environ.get("pcsk_3S8Nh_3DwAJWeZoTzrRgFpwrcjuQM5rNqw2tJRoFsP6QH558xKhsBoDDR2BW4vBs8P8ry")
        if PINECONE_SDK and api_key:
            try:
                pinecone.init(api_key=api_key)
                existing = pinecone.list_indexes()
                if self.index_name not in (existing or []):
                    pinecone.create_index(name=self.index_name, dimension=self.dim, metric="cosine")
                    time.sleep(1)
                self.index = pinecone.Index(self.index_name)
                self.using = "pinecone"
            except Exception:
                # fallback
                self.index = LocalIndex(dim)
                self.using = "local-fallback"
        else:
            self.index = LocalIndex(dim)
            self.using = "local-fallback"

    def upsert(self, vectors: List[Dict[str, Any]]):
        if self.using == "pinecone":
            # real pinecone Index.upsert expects certain signature
            self.index.upsert(vectors=vectors)
        else:
            self.index.upsert(vectors)

    def query(self, vector: List[float], top_k: int = 10, include_values=False, include_metadata=True):
        return self.index.query(vector=vector, top_k=top_k, include_values=include_values, include_metadata=include_metadata)

    def update(self, id: str, set_metadata: Dict[str, Any]):
        try:
            if self.using == "pinecone":
                # pinecone update signature may differ across versions
                self.index.update(id=id, set_metadata=set_metadata)
            else:
                self.index.update(id, set_metadata)
        except Exception:
            # best-effort fallback
            try:
                self.index.update(id, set_metadata)
            except Exception:
                pass

# -----------------------------
# SQLite operations (canonical store) — now includes mobile_number and id_proof
# -----------------------------

def init_sqlite(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            aadhaar TEXT NOT NULL UNIQUE,
            mobile_number TEXT,
            id_proof TEXT,
            joining_date TEXT NOT NULL,
            address TEXT NOT NULL,
            fee_paid REAL NOT NULL,
            fee_due_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def insert_student(db_path: str, row: Dict[str, Any]) -> Optional[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO students (id, name, aadhaar, mobile_number, id_proof, joining_date, address, fee_paid, fee_due_date, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"], row["name"], row["aadhaar"], row.get("mobile_number"), row.get("id_proof"), row["joining_date"], row["address"],
                float(row["fee_paid"]), row["fee_due_date"], row["created_at"], row["updated_at"]
            ),
        )
        conn.commit()
        return row["id"]
    except sqlite3.IntegrityError as e:
        print(f"ERROR: Aadhaar already exists or invalid: {e}")
        return None
    finally:
        conn.close()


def update_student_fee(db_path: str, aadhaar: str, fee_paid: float, fee_due_date: str) -> Optional[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id FROM students WHERE aadhaar = ?", (aadhaar,))
    r = cur.fetchone()
    if not r:
        conn.close()
        return None
    sid = r[0]
    cur.execute(
        "UPDATE students SET fee_paid = ?, fee_due_date = ?, updated_at = ? WHERE id = ?",
        (float(fee_paid), fee_due_date, datetime.utcnow().isoformat(), sid),
    )
    conn.commit()
    conn.close()
    return sid


def get_student_by_id(db_path: str, student_id: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name, aadhaar, mobile_number, id_proof, joining_date, address, fee_paid, fee_due_date, created_at, updated_at FROM students WHERE id = ?", (student_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    cols = ["id","name","aadhaar","mobile_number","id_proof","joining_date","address","fee_paid","fee_due_date","created_at","updated_at"]
    return dict(zip(cols, row))


def fetch_all_students(db_path: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name, aadhaar, mobile_number, id_proof, joining_date, address, fee_paid, fee_due_date, created_at, updated_at FROM students ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "name": r[1],
            "aadhaar": r[2],
            "mobile_number": r[3],
            "id_proof": r[4],
            "joining_date": r[5],
            "address": r[6],
            "fee_paid": r[7],
            "fee_due_date": r[8],
            "created_at": r[9],
            "updated_at": r[10],
        })
    return out

# -----------------------------
# Search helpers
# -----------------------------

def build_search_text(name: str, address: str, joining_date: str, fee_due_date: str, aadhaar_last4: str, mobile_last4: str = "") -> str:
    return f"Name: {name}\nAddress: {address}\nJoining: {joining_date}\nFeeDue: {fee_due_date}\nAadhaarLast4: {aadhaar_last4}\nMobileLast4: {mobile_last4}"


def upsert_to_index(index_wrapper: PineconeWrapper, embedder: Embedder, student_id: str, name: str, address: str, joining_date: str, fee_paid: float, fee_due_date: str, aadhaar: str, mobile_number: Optional[str] = None):
    aadhaar_last4 = ''.join(ch for ch in aadhaar if ch.isdigit())[-4:]
    mobile_last4 = ''.join(ch for ch in (mobile_number or '') if ch.isdigit())[-4:] if mobile_number else ''
    text = build_search_text(name, address, joining_date, fee_due_date, aadhaar_last4, mobile_last4)
    vec = embedder.encode(text)
    meta = {
        "name": name,
        "address": address,
        "joining_date": joining_date,
        "fee_paid": float(fee_paid),
        "fee_due_date": fee_due_date,
        "aadhaar_last4": aadhaar_last4,
        "mobile_last4": mobile_last4,
        "sqlite_id": student_id,
        "aadhaar_hash": hash_aadhaar(aadhaar),
    }
    index_wrapper.upsert(vectors=[{"id": student_id, "values": vec, "metadata": meta}])


def query_index(index_wrapper: PineconeWrapper, embedder: Embedder, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    qvec = embedder.encode(query)
    res = index_wrapper.query(vector=qvec, top_k=top_k, include_values=False, include_metadata=True)
    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    out = []
    for m in matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
        sid = meta.get("sqlite_id")
        if not sid:
            continue
        out.append({"sqlite_id": sid, "score": float(m.get("score", 0.0)), "metadata": meta})
    return out

# -----------------------------
# SMS helpers (Twilio optional) - safe fallback to simulation
# -----------------------------

def send_fee_due_sms(student: Dict[str, Any]) -> Tuple[bool, str]:
    """Attempt to send an SMS. Returns (sent, info).
    If Twilio is not configured it will simulate and return (False, message_text).
    """
    mobile = student.get("mobile_number")
    if not mobile:
        return False, "no mobile"
    # normalize mobile to E.164 if user provided 10-digit Indian number; naive approach
    digits = ''.join(ch for ch in mobile if ch.isdigit())
    if len(digits) == 10:
        to_number = "+91" + digits
    elif digits.startswith("+") or len(digits) > 10:
        to_number = mobile
    else:
        to_number = mobile

    message_text = f"Dear {student.get('name')}, your library fee of Rs {student.get('fee_paid')} is due on {student.get('fee_due_date')}. Please pay by the due date. - Library"

    if DISABLE_SMS:
        # explicit disable
        print("SMS disabled by DISABLE_SMS environment variable. Simulating SMS:", message_text)
        return False, message_text

    if TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER:
        try:
            client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            msg = client.messages.create(body=message_text, from_=TWILIO_FROM_NUMBER, to=to_number)
            return True, getattr(msg, 'sid', 'sent')
        except Exception as e:
            print("Twilio send error:", e)
            return False, str(e)
    else:
        # simulation / fallback
        print("Twilio not configured or library missing. Simulating SMS:", message_text)
        return False, message_text


def should_send_due_sms_on_registration(created_at_iso: str, fee_due_iso: str) -> bool:
    """Decide whether to send an immediate SMS reminder when registering.
    Policy (best-effort interpretation of user request):
    - Only for students registered in the current month (created_at)
    - If fee_due_date is on or before the last day of the created_at month (i.e., due before month end)
    """
    try:
        created = datetime.fromisoformat(created_at_iso)
        fee_due = datetime.fromisoformat(fee_due_iso)
    except Exception:
        return False
    last_day = calendar.monthrange(created.year, created.month)[1]
    last_of_month = datetime(created.year, created.month, last_day)
    return fee_due <= last_of_month and fee_due >= created.replace(hour=0, minute=0, second=0, microsecond=0)

# -----------------------------
# CLI fallback UI
# -----------------------------

def cli_login() -> bool:
    print("Host login required.")
    tries = 0
    while tries < 3:
        user = input("Username: ").strip()
        pwd = input("Password: ").strip()
        if user == ADMIN_USER and pwd == ADMIN_PASS:
            print("Login successful.")
            return True
        else:
            print("Invalid credentials, try again.")
            tries += 1
    print("Failed login attempts. Exiting.")
    return False


def cli_main(db_path: str, index_wrapper: PineconeWrapper, embedder: Embedder):
    if not cli_login():
        return
    print("Library Manager — CLI mode")
    print("Type the number for the action and press Enter.")
    while True:
        print("\n1) Register student\n2) Search\n3) List all\n4) Update fee\n5) Export CSV\n6) Run self-tests\n0) Exit")
        cmd = input("Choice: ").strip()
        if cmd == "0":
            break
        elif cmd == "1":
            name = input("Full name: ").strip()
            aadhaar = input("Aadhaar (12 digits): ").strip()
            mobile = input("Mobile (10 digits): ").strip()
            id_proof = input("ID proof (type/number): ").strip()
            joining = input("Joining date (YYYY-MM-DD) [today]: ").strip() or date.today().isoformat()
            address = input("Address: ").strip()
            fee_paid = float(input("Fee paid (number): ").strip() or 0)
            fee_due = input("Fee due date (YYYY-MM-DD) [today]: ").strip() or date.today().isoformat()
            digits = ''.join(ch for ch in aadhaar if ch.isdigit())
            if len(digits) != 12:
                print("ERROR: Aadhaar must be 12 digits.")
                continue
            if not validate_mobile(mobile):
                print("ERROR: Mobile must be 10 digits.")
                continue
            now = datetime.utcnow().isoformat()
            sid = str(uuid.uuid4())
            row = {"id": sid, "name": name, "aadhaar": digits, "mobile_number": mobile, "id_proof": id_proof, "joining_date": joining, "address": address, "fee_paid": float(fee_paid), "fee_due_date": fee_due, "created_at": now, "updated_at": now}
            new_id = insert_student(db_path, row)
            if new_id:
                upsert_to_index(index_wrapper, embedder, new_id, name, address, joining, fee_paid, fee_due, digits, mobile)
                print(f"Saved student {name} (Aadhaar {mask_aadhaar(digits)})")
                # send SMS if policy applies
                if should_send_due_sms_on_registration(now, fee_due):
                    sent, info = send_fee_due_sms(row)
                    if sent:
                        print("Fee reminder SMS sent (sid:", info, ")")
                    else:
                        print("Fee reminder SMS not sent (info):", info)
        elif cmd == "2":
            q = input("Search text: ").strip()
            if not q:
                print("Empty query")
                continue
            results = query_index(index_wrapper, embedder, q, top_k=10)
            if not results:
                print("No matches")
            else:
                for r in results:
                    sid = r["sqlite_id"]
                    row = get_student_by_id(db_path, sid)
                    print(f"- {row['name']} | Aadhaar: {mask_aadhaar(row['aadhaar'])} | Mobile: {mask_mobile(row.get('mobile_number') or '')} | Fee: {row['fee_paid']} | Due: {row['fee_due_date']} | score: {r['score']}")
        elif cmd == "3":
            rows = fetch_all_students(db_path)
            if not rows:
                print("No students yet.")
            else:
                for r in rows:
                    print(f"- {r['name']} | Aadhaar: {mask_aadhaar(r['aadhaar'])} | Mobile: {mask_mobile(r.get('mobile_number') or '')} | Fee: {r['fee_paid']} | Due: {r['fee_due_date']}")
        elif cmd == "4":
            aadhaar_u = input("Aadhaar (12): ").strip()
            digits = ''.join(ch for ch in aadhaar_u if ch.isdigit())
            if len(digits) != 12:
                print("ERROR: Aadhaar must be 12 digits.")
                continue
            fee_paid_u = float(input("New fee paid: ").strip() or 0)
            fee_due_u = input("New fee due date (YYYY-MM-DD): ").strip() or date.today().isoformat()
            sid = update_student_fee(db_path, digits, fee_paid_u, fee_due_u)
            if not sid:
                print("No student found for that Aadhaar.")
            else:
                index_wrapper.update(sid, {"fee_paid": float(fee_paid_u), "fee_due_date": fee_due_u, "updated_at": datetime.utcnow().isoformat()})
                print("Updated.")
        elif cmd == "5":
            out_file = input("CSV filename [students_export.csv]: ").strip() or "students_export.csv"
            rows = fetch_all_students(db_path)
            if not rows:
                print("No students to export.")
            else:
                with open(out_file, "w", newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=["id","name","aadhaar","mobile_number","id_proof","joining_date","address","fee_paid","fee_due_date","created_at","updated_at"])
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(r)
                print(f"Wrote {len(rows)} rows to {out_file}")
        elif cmd == "6":
            run_self_tests()
        else:
            print("Unknown command")

# -----------------------------
# Self-tests
# -----------------------------

def run_self_tests():
    print("Running lightweight self-tests...")
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    db_path = tmp.name
    try:
        init_sqlite(db_path)
        now = datetime.utcnow().isoformat()
        # test insert
        sid1 = str(uuid.uuid4())
        r1 = {"id": sid1, "name": "Test User 1", "aadhaar": "111111111111", "mobile_number": "9999999999", "id_proof": "Voter:1234", "joining_date": now, "address": "Addr 1", "fee_paid": 100.0, "fee_due_date": now, "created_at": now, "updated_at": now}
        assert insert_student(db_path, r1) == sid1
        # duplicate aadhaar should fail
        sid2 = str(uuid.uuid4())
        r2 = {"id": sid2, "name": "Test User 2", "aadhaar": "111111111111", "mobile_number": "8888888888", "id_proof": "Aadhar:4321", "joining_date": now, "address": "Addr 2", "fee_paid": 200.0, "fee_due_date": now, "created_at": now, "updated_at": now}
        assert insert_student(db_path, r2) is None
        # fetch
        rows = fetch_all_students(db_path)
        assert len(rows) == 1
        # update fee
        updated = update_student_fee(db_path, "111111111111", 150.0, now)
        assert updated == sid1
        row = get_student_by_id(db_path, sid1)
        assert float(row["fee_paid"]) == 150.0
        print("SELF-TESTS: PASS")
    except AssertionError as e:
        print("SELF-TESTS: FAIL", e)
    except Exception as e:
        print("SELF-TESTS: ERROR", e)
    finally:
        try:
            os.unlink(db_path)
        except Exception:
            pass

# -----------------------------
# Streamlit UI (if available) with authentication
# -----------------------------

def run_streamlit_app(db_path: str, index_wrapper: PineconeWrapper, embedder: Embedder):
    # NOTE: This function is run inside Streamlit. Keep state in st.session_state.
    st.set_page_config(page_title="Library Manager — Students", page_icon="📚", layout="wide")

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    st.sidebar.header("Host Login")
    if not st.session_state["authenticated"]:
        user = st.sidebar.text_input("Username")
        pwd = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if user == ADMIN_USER and pwd == ADMIN_PASS:
                st.session_state["authenticated"] = True
                st.experimental_rerun()
            else:
                st.sidebar.error("Invalid credentials")
                return
    else:
        st.sidebar.success("Logged in as host")

    st.sidebar.markdown("Set your Pinecone API key in the environment: `PINECONE_API_KEY`.\nSet TWILIO vars to enable SMS reminders.")
    st.title("📚 Library Manager — Students (host)")
    st.success(f"Using embedder: {embedder.using} — Using index: {index_wrapper.using}")

    TAB_REGISTER, TAB_SEARCH, TAB_ALL, TAB_UPDATE = st.tabs(["➕ Register", "🔎 Search", "📋 All students", "💳 Update fee"])

    with TAB_REGISTER:
        st.subheader("Register a new student")
        with st.form("register_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Full name *")
                aadhaar = st.text_input("Aadhaar number (12 digits) *")
                mobile = st.text_input("Mobile number (10 digits)")
                joining = st.date_input("Date of joining *", value=date.today())
            with col2:
                fee_paid = st.number_input("Amount of fee paid *", min_value=0.0, step=100.0, value=0.0, format="%.2f")
                fee_due = st.date_input("Last day of fee due *", value=date.today())
                id_proof = st.text_input("ID proof (type/number)")
            address = st.text_area("Address *", height=80)
            submitted = st.form_submit_button("Save student")

        if submitted:
            digits = ''.join(ch for ch in aadhaar if ch.isdigit())
            if len(name.strip()) == 0:
                st.warning("Name is required.")
            elif len(digits) != 12:
                st.warning("Aadhaar must be 12 digits.")
            elif len(address.strip()) == 0:
                st.warning("Address is required.")
            elif mobile and not validate_mobile(mobile):
                st.warning("Mobile must be 10 digits if provided.")
            else:
                now = datetime.utcnow().isoformat()
                sid = str(uuid.uuid4())
                row = {"id": sid, "name": name.strip(), "aadhaar": digits, "mobile_number": mobile, "id_proof": id_proof, "joining_date": joining.isoformat(), "address": address.strip(), "fee_paid": float(fee_paid), "fee_due_date": fee_due.isoformat(), "created_at": now, "updated_at": now}
                new_id = insert_student(db_path, row)
                if new_id:
                    try:
                        upsert_to_index(index_wrapper, embedder, new_id, row["name"], row["address"], row["joining_date"], row["fee_paid"], row["fee_due_date"], row["aadhaar"], row.get("mobile_number"))
                    except Exception as e:
                        st.warning(f"Saved to SQLite but failed to index: {e}")

                    st.success(f"Saved student {row['name']} (Aadhaar {mask_aadhaar(row['aadhaar'])}).")

                    # Send SMS if condition met
                    if should_send_due_sms_on_registration(now, row["fee_due_date"]):
                        sent, info = send_fee_due_sms(row)
                        if sent:
                            st.info("Fee reminder SMS sent.")
                        else:
                            st.warning(f"Fee reminder SMS not sent: {info}")

    with TAB_SEARCH:
        st.subheader("Semantic search")
        q = st.text_input("Search by name, address, Aadhaar last 4, mobile last 4, or anything relevant")
        k = st.slider("Results", min_value=1, max_value=20, value=10)
        if q:
            results = query_index(index_wrapper, embedder, q, top_k=k)
            if not results:
                st.info("No matches.")
            else:
                rows = []
                for r in results:
                    row = get_student_by_id(db_path, r["sqlite_id"]) if isinstance(r, dict) else get_student_by_id(db_path, r)
                    if row:
                        row["score"] = r.get("score", 0.0)
                        rows.append(row)
                if PANDAS_AVAILABLE:
                    df = pd.DataFrame(rows)
                    df["aadhaar_masked"] = df["aadhaar"].apply(mask_aadhaar)
                    df["mobile_masked"] = df["mobile_number"].apply(lambda x: mask_mobile(x) if x else "")
                    st.dataframe(df[["name","aadhaar_masked","mobile_masked","joining_date","address","fee_paid","fee_due_date","score"]].sort_values("score", ascending=False), use_container_width=True)
                else:
                    for row in rows:
                        st.write(f"- {row['name']} | Aadhaar: {mask_aadhaar(row['aadhaar'])} | Mobile: {mask_mobile(row.get('mobile_number') or '')} | Fee: {row['fee_paid']} | Due: {row['fee_due_date']} | score: {row.get('score')}")

    with TAB_ALL:
        st.subheader("All students")
        rows = fetch_all_students(db_path)
        if not rows:
            st.info("No students yet.")
        else:
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(rows)
                df["aadhaar"] = df["aadhaar"].apply(mask_aadhaar)
                df["mobile_number"] = df["mobile_number"].apply(lambda x: mask_mobile(x) if x else "")
                st.dataframe(df.drop(columns=["created_at","updated_at"]).sort_values("joining_date", ascending=False), use_container_width=True)
                csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV (full data, unmasked Aadhaar & mobile)", csv, file_name="students_export.csv", mime="text/csv")
            else:
                for r in rows:
                    st.write(f"- {r['name']} | Aadhaar: {mask_aadhaar(r['aadhaar'])} | Mobile: {mask_mobile(r.get('mobile_number') or '')} | Fee: {r['fee_paid']} | Due: {r['fee_due_date']}")

    with TAB_UPDATE:
        st.subheader("Update fee & due date")
        with st.form("fee_form"):
            aadhaar_u = st.text_input("Aadhaar number (exact 12 digits) *")
            fee_paid_u = st.number_input("New fee paid *", min_value=0.0, step=100.0, value=0.0, format="%.2f")
            fee_due_u = st.date_input("New last day of fee due *", value=date.today())
            ok = st.form_submit_button("Update")
        if ok:
            digits = ''.join(ch for ch in aadhaar_u if ch.isdigit())
            if len(digits) != 12:
                st.warning("Aadhaar must be 12 digits.")
            else:
                sid = update_student_fee(db_path, digits, fee_paid_u, fee_due_u.isoformat())
                if not sid:
                    st.error("No student found for that Aadhaar.")
                else:
                    try:
                        index_wrapper.update(sid, {"fee_paid": fee_paid_u, "fee_due_date": fee_due_u.isoformat(), "updated_at": datetime.utcnow().isoformat()})
                    except Exception as e:
                        st.warning(f"Updated SQLite but failed to update index metadata: {e}")
                    st.success(f"Updated fee for Aadhaar {mask_aadhaar(digits)}.")

    st.caption("Security notes: Your Pinecone index stores only name, address, dates, fee, Aadhaar last 4, mobile last 4, and a hash. Full Aadhaar and mobile stay in SQLite. Consider stronger protection for production.")

# -----------------------------
# Entrypoint
# -----------------------------

def main():
    db_path = os.environ.get("STUDENTS_DB", DB_PATH_DEFAULT)
    init_sqlite(db_path)
    embedder = Embedder()
    index_wrapper = PineconeWrapper(dim=embedder.dim)

    if len(sys.argv) > 1 and sys.argv[1] in ("--test", "-t"):
        run_self_tests()
        return

    if STREAMLIT_AVAILABLE:
        try:
            run_streamlit_app(db_path, index_wrapper, embedder)
        except Exception as e:
            print("Error while running Streamlit UI:", e)
            print("Falling back to CLI mode.")
            cli_main(db_path, index_wrapper, embedder)
    else:
        print("Streamlit not available in this environment. Running CLI fallback.")
        print("If you prefer Streamlit UI, install streamlit (pip install streamlit) and rerun with `streamlit run <this_file>.py`.")
        cli_main(db_path, index_wrapper, embedder)


if __name__ == '__main__':
    main()
