"""
IPC Navigator — AI Legal Assistant
==================================
A Streamlit "real app" over the Indian Penal Code (IPC, 1860).

Stack
-----
* NLP   : regex parsing, sentence segmentation, script detection
* ML    : TF-IDF, BM25 (rank_bm25), cosine similarity, RRF hybrid fusion
* DL    : Sentence-BERT (all-MiniLM-L6-v2, offline) dense embeddings
* LLM   : Groq API (Llama-3 / Mistral) for case analysis, chain-of-thought,
          chat, translation, NER, contradiction checking, transcription
* Store : SQLite (sections, session logs, feedback, A/B) + on-disk vector index

Everything degrades gracefully: without a Groq key the LLM panels explain how
to enable them, and without a cached embedding model the search falls back to
BM25 + TF-IDF.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import zipfile
import hashlib
import datetime as _dt
import difflib
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── optional dependencies (graceful fallback) ────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
    BM25_OK = True
except Exception:                                    # pragma: no cover
    BM25_OK = False

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except Exception:                                    # pragma: no cover
    ST_OK = False

try:
    from fpdf import FPDF
    FPDF_OK = True
except Exception:                                    # pragma: no cover
    FPDF_OK = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PX_OK = True
except Exception:                                    # pragma: no cover
    PX_OK = False


# ═════════════════════════════════════════════════════════════════════════════
# 0.  CONFIG
# ═════════════════════════════════════════════════════════════════════════════
APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "navigator.db"
INDEX_DIR = APP_DIR / "vector_index"

EMBED_MODEL = "all-MiniLM-L6-v2"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

FOOTNOTE_PREFIXES = [
    "The Indian", "Ins.", "Subs.", "Added", "Rep.", "The words",
    "Cl.", "Omitted", "Certain words", "The original", "Cf.",
    "Vide", "See", "Now see",
]

LEGAL_DISCLAIMER = (
    "⚠️ **Reference aid only.** Bail / cognizable / compoundability tags and the "
    "IPC→BNS mapping are curated for common sections and are *indicative*, not "
    "authoritative. Always verify against the bare Act, Schedule I of the CrPC "
    "(now BNSS, 2023) and the Bharatiya Nyaya Sanhita, 2023."
)

st.set_page_config(
    page_title="IPC Navigator — AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  CURATED LEGAL REFERENCE TABLES
#     (indicative only — see LEGAL_DISCLAIMER)
# ═════════════════════════════════════════════════════════════════════════════
# IPC → BNS (Bharatiya Nyaya Sanhita, 2023) — section-level correspondence
IPC_BNS_MAP: dict[str, str] = {
    "34": "3(5)", "109": "48", "120A": "61", "120B": "61(2)", "124A": "152",
    "141": "189", "143": "190", "147": "191", "148": "191", "149": "190",
    "153A": "196", "201": "238", "212": "251", "279": "281", "294": "296",
    "299": "100", "300": "101", "301": "102", "302": "103", "303": "104",
    "304": "105", "304A": "106", "304B": "80", "305": "107", "306": "108",
    "307": "109", "308": "110", "309": "226", "323": "115", "324": "117",
    "325": "117", "326": "117", "332": "121", "341": "126", "342": "127",
    "354": "74", "354A": "75", "354B": "76", "354C": "77", "354D": "78",
    "363": "137", "364": "138", "365": "139", "366": "137", "366A": "140",
    "366B": "141", "367": "142", "368": "143", "370": "143", "375": "63",
    "376": "64", "376A": "66", "376B": "67", "376C": "68", "376D": "70",
    "377": "— (omitted)", "379": "303", "380": "305", "381": "306",
    "382": "304", "392": "309", "393": "310", "394": "311", "395": "310",
    "396": "310", "399": "310", "400": "310", "402": "310", "403": "314",
    "406": "316", "407": "317", "408": "316", "409": "316", "415": "318",
    "417": "318", "419": "319", "420": "318", "425": "324", "441": "329",
    "447": "329", "448": "330", "463": "336", "465": "336", "468": "336",
    "471": "340", "497": "— (omitted)", "498A": "85", "499": "356",
    "500": "356", "503": "351", "504": "352", "506": "351", "509": "79",
}

# Section-level Schedule I style tags (indicative)
# keys: bail (Bailable/Non-bailable), cog (Cognizable/Non-cognizable), comp (Yes/No), court
LEGAL_TAGS: dict[str, dict[str, str]] = {
    "120B": {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "124A": {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "153A": {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Magistrate First Class"},
    "201":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "279":  {"bail": "Bailable", "cog": "Cognizable", "comp": "Yes", "court": "Magistrate First Class"},
    "302":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "303":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "304":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "304A": {"bail": "Bailable", "cog": "Cognizable", "comp": "No", "court": "Magistrate First Class"},
    "304B": {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "305":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "306":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "307":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "308":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "323":  {"bail": "Bailable", "cog": "Non-cognizable", "comp": "Yes", "court": "Any Magistrate"},
    "324":  {"bail": "Bailable", "cog": "Cognizable", "comp": "Yes", "court": "Any Magistrate"},
    "325":  {"bail": "Bailable", "cog": "Cognizable", "comp": "Yes", "court": "Any Magistrate"},
    "326":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Magistrate First Class"},
    "332":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Magistrate First Class"},
    "354":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Any Magistrate"},
    "363":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Magistrate First Class"},
    "366":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "366A": {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "375":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "376":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "379":  {"bail": "Bailable", "cog": "Cognizable", "comp": "Yes", "court": "Any Magistrate"},
    "380":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Any Magistrate"},
    "392":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Magistrate First Class"},
    "395":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Court of Session"},
    "406":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "Yes", "court": "Any Magistrate"},
    "409":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Magistrate First Class"},
    "415":  {"bail": "Bailable", "cog": "Non-cognizable", "comp": "Yes", "court": "Any Magistrate"},
    "417":  {"bail": "Bailable", "cog": "Cognizable", "comp": "Yes", "court": "Any Magistrate"},
    "420":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "Yes", "court": "Any Magistrate"},
    "447":  {"bail": "Bailable", "cog": "Cognizable", "comp": "Yes", "court": "Any Magistrate"},
    "498A": {"bail": "Non-bailable", "cog": "Cognizable", "comp": "No", "court": "Magistrate First Class"},
    "499":  {"bail": "Bailable", "cog": "Non-cognizable", "comp": "Yes", "court": "Magistrate First Class"},
    "500":  {"bail": "Bailable", "cog": "Non-cognizable", "comp": "Yes", "court": "Magistrate First Class"},
    "506":  {"bail": "Non-bailable", "cog": "Cognizable", "comp": "Yes", "court": "Any Magistrate"},
}

# Curated "related sections" clusters (thematic / cross-referenced)
RELATED_CLUSTERS: list[list[str]] = [
    ["299", "300", "301", "302", "303", "304", "304A", "304B", "305", "306", "307", "308", "309"],
    ["375", "376", "376A", "376B", "376C", "376D", "376E", "377", "354", "354A", "354B", "354C", "354D", "509"],
    ["378", "379", "380", "381", "382", "411", "412", "413", "414"],
    ["383", "384", "385", "386", "387", "388", "389"],
    ["390", "391", "392", "393", "394", "395", "396", "397", "398", "399", "400", "401", "402"],
    ["403", "404", "405", "406", "407", "408", "409", "410", "411", "412", "413", "414"],
    ["415", "416", "417", "418", "419", "420", "421", "422", "423", "424"],
    ["141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "157", "158", "159", "160"],
    ["363", "364", "365", "366", "366A", "366B", "367", "368", "369", "370", "370A", "371", "372", "373", "374"],
    ["497", "498", "498A", "304B", "494", "495", "496"],
    ["499", "500", "501", "502"],
    ["503", "504", "505", "506", "507", "508"],
    ["425", "426", "427", "428", "429", "430", "431", "432", "433", "434", "435", "436", "437", "438", "439", "440"],
    ["34", "35", "36", "37", "38", "109", "114", "120A", "120B", "149"],
]

RELATED_GRAPH: dict[str, set[str]] = {}
for _cluster in RELATED_CLUSTERS:
    for _s in _cluster:
        RELATED_GRAPH.setdefault(_s, set()).update(x for x in _cluster if x != _s)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  DATA LOADING + PARSING  (NLP layer)
# ═════════════════════════════════════════════════════════════════════════════
def _find_ipc_file() -> Path | None:
    for name in ("ipc_act.txt", "ipc.txt", "IPC.txt", "ipc.pdf.txt"):
        p = APP_DIR / name
        if p.exists():
            return p
    return None


def read_ipc_text(path: Path) -> str:
    """Read the IPC source, transparently unzipping archives and decoding bytes."""
    data = path.read_bytes()

    # zip archive (e.g. a .txt that is really a zipped ipc_act.txt)
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            txt = next((n for n in names if n.lower().endswith(".txt")), names[0])
            raw = z.read(txt)
    else:
        raw = data

    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def _clean_num(num: str) -> str:
    return re.sub(r"\s+", "", num).upper()


def _valid_num(num: str) -> bool:
    m = re.match(r"(\d+)", num)
    return bool(m) and 1 <= int(m.group(1)) <= 511


def _is_footnote(title: str) -> bool:
    return any(title.strip().startswith(p) for p in FOOTNOTE_PREFIXES)


def _parse_arrangement_chapters(raw: str, body_start: int) -> dict[str, str]:
    """Chapter code → readable title, taken from the arrangement of sections."""
    head = raw[:body_start]
    out: dict[str, str] = {}
    matches = list(re.finditer(r"(?m)^\s*CHAPTER\s+([IVXL]+[A-Z]?)\s*$", head))
    for i, m in enumerate(matches):
        code = m.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(head)
        block = head[m.end():end]
        for line in block.splitlines():
            line = line.strip()
            if line and not re.match(r"^\d+[A-Z]*\.?$", line) and line.upper() == line:
                out[code] = re.sub(r"\s+", " ", line).strip()
                break
    return out


@st.cache_data(show_spinner=False)
def parse_ipc(path_str: str, _mtime: float) -> dict:
    """Parse raw IPC text → {section_num: {num,title,content,full,chapter_*,amendments}}."""
    raw = read_ipc_text(Path(path_str)).replace("\f", "").replace("\r", "")

    # Body starts at the *second* "CHAPTER I" (first is the arrangement of sections).
    ch1 = list(re.finditer(r"(?m)^\s*CHAPTER\s+I\s*$", raw))
    body_start = ch1[1].start() if len(ch1) >= 2 else (ch1[0].start() if ch1 else 0)
    arr_titles = _parse_arrangement_chapters(raw, body_start)
    body = raw[body_start:]

    # strip page-number-only lines
    body = re.sub(r"\n\s*\d{1,3}\s*\n", "\n\n", body)

    # ---- chapter banner positions (a banner precedes the first section of its chapter)
    chap_positions: list[tuple[int, str, str]] = []
    for m in re.finditer(r"(?m)^\s*\[?CHAPTER\s+([IVXL]+[A-Z]*)\]?\s*$", body):
        code = m.group(1)
        title = ""
        for line in body[m.end():].splitlines():
            s = line.strip()
            if s:
                title = re.sub(r"\s+", " ", s).strip()
                break
        chap_positions.append((m.start(), code, arr_titles.get(code, title) or title))

    def chapter_for(pos: int) -> tuple[str, str]:
        code, title = "—", "Unclassified"
        for p, c, t in chap_positions:
            if p <= pos:
                code, title = c, t
            else:
                break
        return code, title

    # ---- section boundaries (allow an optional leading "[" e.g. "[124A.")
    bounds = [m.start() for m in re.finditer(r"(?m)^(?=\[?\d+[A-Z]*\.\s)", body)]
    bounds.append(len(body))

    sections: dict[str, dict] = {}

    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        part = body[start:end]

        secm = re.match(r"(?s)^\[?(\d+[A-Z]*)\.\s+(.+?)\.\s*—\s*(.*)$", part)
        if not secm:
            continue

        chapter_code, chapter_title = chapter_for(start)

        num, title, content = secm.groups()
        num = _clean_num(num)
        title = re.sub(r"\s+", " ", title).strip()

        # content must stop before the next chapter banner (which lands in this part's tail)
        content = re.split(r"(?m)^\s*\[?CHAPTER\s+[IVXL]", content)[0]
        content = re.sub(r"\s+", " ", content).strip()

        # extract amendment / footnote notes (for the amendments timeline)
        amendments = []
        for line in part.splitlines():
            s = line.strip()
            if _is_footnote(s) and len(s) > 12:
                if s not in amendments:
                    amendments.append(s)

        # clean footnote brackets
        content = re.sub(r"\d+\[", "[", content)
        content = re.sub(r"\]\d+", "]", content)
        content = re.sub(r"\[\d+\]", "", content)
        content = re.sub(r"\s+", " ", content).strip()

        # guards
        if not _valid_num(num):       continue
        if not (3 <= len(title) <= 250): continue
        if _is_footnote(title):       continue
        if len(content) < 20:         continue
        if re.match(r"^\[?(Repealed|Omitted)", title, re.I): continue
        if num in sections:           continue

        sections[num] = {
            "num": num,
            "title": title,
            "content": content,
            "full": f"Section {num}: {title}. {content}",
            "chapter_code": chapter_code,
            "chapter_title": chapter_title,
            "amendments": amendments,
        }

    # second pass: fill any missing chapter titles by proximity
    for num, sec in sections.items():
        if not sec["chapter_code"]:
            sec["chapter_code"] = "—"
            sec["chapter_title"] = "Unclassified"
    return sections


def load_sections() -> dict:
    p = _find_ipc_file()
    if p is None:
        st.error("⚠️ Could not find `ipc_act.txt` / `ipc.txt` next to `app.py`.")
        st.stop()
    secs = parse_ipc(str(p), p.stat().st_mtime)
    if not secs:
        st.error("No sections were parsed — please check the source file format.")
        st.stop()
    return secs


# ═════════════════════════════════════════════════════════════════════════════
# 3.  STORAGE  (SQLite  +  on-disk vector index)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sections(
            num TEXT PRIMARY KEY, title TEXT, content TEXT, full TEXT,
            chapter_code TEXT, chapter_title TEXT);
        CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT,
            query TEXT, section TEXT, extra TEXT);
        CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, query TEXT,
            section TEXT, vote INTEGER, model TEXT);
        CREATE TABLE IF NOT EXISTS abtest(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, query TEXT,
            engine TEXT, top1 TEXT, score REAL);
        """
    )
    conn.commit()
    return conn


def db_save_sections(sections: dict) -> None:
    try:
        conn = db_conn()
        conn.executemany(
            "INSERT OR REPLACE INTO sections(num,title,content,full,chapter_code,chapter_title)"
            " VALUES(?,?,?,?,?,?)",
            [(s["num"], s["title"], s["content"], s["full"],
              s["chapter_code"], s["chapter_title"]) for s in sections.values()],
        )
        conn.commit()
    except Exception:
        pass


def log_event(kind: str, query: str = "", section: str = "", extra: str = "") -> None:
    try:
        c = db_conn()
        c.execute("INSERT INTO events(ts,kind,query,section,extra) VALUES(?,?,?,?,?)",
                  (_dt.datetime.now().isoformat(timespec="seconds"), kind, query, section, extra))
        c.commit()
    except Exception:
        pass


def log_feedback(query: str, section: str, vote: int, model: str) -> None:
    try:
        c = db_conn()
        c.execute("INSERT INTO feedback(ts,query,section,vote,model) VALUES(?,?,?,?,?)",
                  (_dt.datetime.now().isoformat(timespec="seconds"), query, section, vote, model))
        c.commit()
    except Exception:
        pass


def feedback_boost() -> dict[str, float]:
    """section -> net 👍 minus 👎 (used as a light re-ranking signal)."""
    try:
        rows = db_conn().execute(
            "SELECT section, SUM(vote) s FROM feedback GROUP BY section").fetchall()
        return {r["section"]: float(r["s"] or 0) for r in rows}
    except Exception:
        return {}


def log_abtest(query: str, engine: str, top1: str, score: float) -> None:
    try:
        c = db_conn()
        c.execute("INSERT INTO abtest(ts,query,engine,top1,score) VALUES(?,?,?,?,?)",
                  (_dt.datetime.now().isoformat(timespec="seconds"), query, engine, top1, score))
        c.commit()
    except Exception:
        pass


def analytics() -> dict:
    out: dict = {}
    try:
        c = db_conn()
        out["top_sections"] = c.execute(
            "SELECT section, COUNT(*) n FROM events WHERE kind='click' AND section<>''"
            " GROUP BY section ORDER BY n DESC LIMIT 15").fetchall()
        out["top_queries"] = c.execute(
            "SELECT query, COUNT(*) n FROM events WHERE kind='search' AND query<>''"
            " GROUP BY query ORDER BY n DESC LIMIT 15").fetchall()
        out["heat"] = c.execute(
            "SELECT substr(ts,1,10) d, substr(ts,12,2) h, COUNT(*) n FROM events"
            " GROUP BY d,h").fetchall()
        out["counts"] = c.execute(
            "SELECT kind, COUNT(*) n FROM events GROUP BY kind").fetchall()
        out["abtest"] = c.execute(
            "SELECT engine, COUNT(*) n, AVG(score) s FROM abtest GROUP BY engine").fetchall()
        out["feedback"] = c.execute(
            "SELECT section, SUM(vote) s, COUNT(*) n FROM feedback GROUP BY section"
            " ORDER BY n DESC LIMIT 15").fetchall()
    except Exception:
        pass
    return out


# ── vector index persistence ────────────────────────────────────────────────
def _index_paths(sig: str) -> tuple[Path, Path]:
    INDEX_DIR.mkdir(exist_ok=True)
    return INDEX_DIR / f"{sig}.npy", INDEX_DIR / f"{sig}.keys.json"


# ═════════════════════════════════════════════════════════════════════════════
# 4.  MODELS  (DL embeddings)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_embedder(offline: bool = True):
    """Sentence-BERT embedder, loaded from cache only (no downloads when offline)."""
    if not ST_OK:
        return None
    try:
        return SentenceTransformer(EMBED_MODEL, local_files_only=offline)
    except Exception:
        return None


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ═════════════════════════════════════════════════════════════════════════════
# 5.  SEARCH  (BM25 + dense + RRF hybrid + optional LLM rerank)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔎 Building hybrid search index…")
def build_index(sig: str, keys: tuple, texts: tuple, use_dense: bool):
    """Returns a dict describing the searchable index (persisted to disk)."""
    emb_path, keys_path = _index_paths(sig)

    # ---- dense embeddings (persist / reload) ----
    embs = None
    model = load_embedder() if use_dense else None
    if model is not None:
        if emb_path.exists() and keys_path.exists():
            try:
                if json.loads(keys_path.read_text()) == list(keys):
                    embs = np.load(emb_path)
            except Exception:
                embs = None
        if embs is None:
            embs = model.encode(list(texts), normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
            try:
                np.save(emb_path, embs)
                keys_path.write_text(json.dumps(list(keys)))
            except Exception:
                pass

    # ---- BM25 ----
    bm25 = None
    if BM25_OK:
        try:
            bm25 = BM25Okapi([tokenize(t) for t in texts])
        except Exception:
            bm25 = None

    # ---- TF-IDF fallback ----
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30000)
    tfidf = vec.fit_transform(list(texts))

    engines = [e for e, ok in (("BM25", bm25 is not None),
                               ("Dense", embs is not None), ("TF-IDF", True)) if ok]
    return {"kind": "hybrid", "keys": list(keys), "texts": list(texts),
            "model": model, "embs": embs, "bm25": bm25, "tfidf": (vec, tfidf),
            "engines": engines}


def _dense_scores(index: dict, query: str) -> np.ndarray | None:
    if index["embs"] is None or index["model"] is None:
        return None
    q = index["model"].encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return (q @ index["embs"].T).flatten()


def _bm25_scores(index: dict, query: str) -> np.ndarray | None:
    if index["bm25"] is None:
        return None
    return np.asarray(index["bm25"].get_scores(tokenize(query)))


def _tfidf_scores(index: dict, query: str) -> np.ndarray:
    vec, mat = index["tfidf"]
    return cosine_similarity(vec.transform([query]), mat).flatten()


def _rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion over several ranked index lists."""
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return fused


def hybrid_search(index: dict, query: str, top_k: int = 5,
                  allowed: set[str] | None = None,
                  rerank: bool = False) -> list[tuple[str, float]]:
    """Hybrid retrieval (BM25 + dense + TF-IDF via RRF), then optional LLM rerank."""
    keys = index["keys"]
    n = len(keys)

    rank_lists: list[list[int]] = []
    score_map: dict[str, float] = {}

    bm = _bm25_scores(index, query)
    if bm is not None:
        rank_lists.append(list(np.argsort(-bm)[: max(top_k * 8, 50)]))
        denom = bm.max() or 1.0
        for i, s in enumerate(bm):
            score_map[keys[i]] = max(score_map.get(keys[i], 0.0), float(s) / denom)

    dn = _dense_scores(index, query)
    if dn is not None:
        rank_lists.append(list(np.argsort(-dn)[: max(top_k * 8, 50)]))
        for i, s in enumerate(dn):
            score_map[keys[i]] = max(score_map.get(keys[i], 0.0), float(s))

    tf = _tfidf_scores(index, query)
    rank_lists.append(list(np.argsort(-tf)[: max(top_k * 8, 50)]))
    for i, s in enumerate(tf):
        score_map[keys[i]] = max(score_map.get(keys[i], 0.0), float(s))

    fused = _rrf(rank_lists)

    # blend RRF with normalised similarity + feedback boost
    boost = feedback_boost()
    cand = []
    for idx, rrf in fused.items():
        k = keys[idx]
        if allowed is not None and k not in allowed:
            continue
        sim = score_map.get(k, 0.0)
        cand.append((k, 0.6 * rrf / (1.0) + 0.4 * sim + 0.02 * boost.get(k, 0.0), sim))

    cand.sort(key=lambda x: -x[1])
    cand = cand[: max(top_k * 3, top_k)]

    if rerank and cand:
        cand = llm_rerank(query, cand, top_k)

    return [(k, float(sim)) for k, _blend, sim in cand[:top_k]]


def llm_rerank(query: str, cand: list[tuple], top_k: int) -> list[tuple]:
    """Optional cross-encoder substitute: ask the LLM to order the candidates."""
    if not groq_available() or len(cand) <= 1:
        return cand
    listing = "\n".join(f"{i}. Section {k}: {t[:200]}" for i, (k, _s, _sim) in enumerate(cand))
    prompt = (
        "You are a legal retrieval re-ranker. Given a case description and candidate IPC "
        "sections, reply ONLY with a JSON array of candidate indices ordered from most to "
        "least relevant, e.g. [2,0,1].\n\n"
        f"CASE:\n{query[:800]}\n\nCANDIDATES:\n{listing}"
    )
    try:
        out = groq_complete([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=120)
        order = json.loads(re.search(r"\[[\d,\s]*\]", out).group(0))
        ordered = [cand[i] for i in order if 0 <= i < len(cand)]
        ordered += [c for c in cand if c not in ordered]
        return ordered
    except Exception:
        return cand


def filter_sections(sections: dict, chapter: str | None = None,
                    ptype: str | None = None, bail: str | None = None) -> set[str]:
    """Keyword + facet filters → set of allowed section numbers."""
    allowed: set[str] = set()
    for num, sec in sections.items():
        if chapter and chapter != "All" and sec["chapter_code"] != chapter:
            continue
        if ptype and ptype != "Any":
            calc = punishment_calc(sec)
            if ptype not in calc["types"]:
                continue
        if bail and bail != "Any":
            tag = LEGAL_TAGS.get(num, {}).get("bail", "Unknown")
            if tag != bail:
                continue
        allowed.add(num)
    return allowed


# ── fuzzy code matching ─────────────────────────────────────────────────────
_OCR_MAP = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5",
                          "Z": "2", "G": "6", "Q": "0"})


def normalize_code(s: str) -> str:
    s = s.strip().upper()
    s = re.sub(r"SECTION|SEC\.|SEC|#", "", s).strip()
    s = re.sub(r"\s+", "", s)
    return s.translate(_OCR_MAP)


def fuzzy_lookup(raw: str, keys) -> tuple[str | None, str]:
    """Resolve a (possibly mangled) section query to a real key. Returns (key, how)."""
    q = normalize_code(raw)
    if not q:
        return None, "empty"
    if q in keys:
        return q, "exact"

    # letter→digit confusion already normalised; try difflib closeness
    close = difflib.get_close_matches(q, list(keys), n=3, cutoff=0.6)
    if close:
        return close[0], "fuzzy"

    # numeric prefix suggestion (e.g. 42O -> 420)
    digits = re.sub(r"\D", "", q)
    if digits:
        pref = [k for k in keys if k.startswith(digits)]
        if pref:
            return pref[0], "prefix"
    return None, "none"


def parse_range(raw: str) -> tuple[int, int] | None:
    m = re.search(r"(\d{1,3})\s*(?:-|–|—|to)\s*(\d{1,3})", raw)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    if a > b:
        a, b = b, a
    return a, b


# ═════════════════════════════════════════════════════════════════════════════
# 6.  LLM LAYER  (Groq)
# ═════════════════════════════════════════════════════════════════════════════
class GroqUnavailable(Exception):
    pass


def get_groq_key() -> str:
    key = ""
    try:
        key = st.session_state.get("groq_key", "") or ""
    except Exception:
        key = ""
    if not key:
        key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            key = ""
    return key.strip()


def groq_available() -> bool:
    return bool(get_groq_key())


def get_model() -> str:
    try:
        return st.session_state.get("groq_model", GROQ_MODELS[0])
    except Exception:
        return GROQ_MODELS[0]


def groq_complete(messages: list[dict], temperature: float = 0.2,
                  max_tokens: int = 900) -> str:
    key = get_groq_key()
    if not key:
        raise GroqUnavailable("No Groq API key configured.")
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": get_model(), "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens, "stream": False},
        timeout=90,
    )
    if r.status_code != 200:
        raise GroqUnavailable(f"Groq {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"]


def groq_stream(messages: list[dict], temperature: float = 0.2,
                max_tokens: int = 900):
    """Yield tokens as they arrive (for token-by-token streaming)."""
    key = get_groq_key()
    if not key:
        raise GroqUnavailable("No Groq API key configured.")
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": get_model(), "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens, "stream": True},
        stream=True, timeout=120,
    )
    if r.status_code != 200:
        raise GroqUnavailable(f"Groq {r.status_code}: {r.text[:200]}")
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            break
        try:
            delta = json.loads(payload)["choices"][0]["delta"].get("content")
        except Exception:
            continue
        if delta:
            yield delta


def _extract_json(text: str):
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        raise ValueError("no json")
    return json.loads(m.group(1))


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]


def get_gemini_key() -> str:
    key = ""
    try:
        key = st.session_state.get("gemini_key", "") or ""
    except Exception:
        key = ""
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            key = ""
    return key.strip()


def gemini_available() -> bool:
    return bool(get_gemini_key())


def gemini_complete(prompt: str, system: str = "", temperature: float = 0.3,
                     max_tokens: int = 700) -> str:
    """Call Google's Gemini API (Generative Language API) for the law chatbot."""
    key = get_gemini_key()
    if not key:
        raise GroqUnavailable("No Google (Gemini) API key configured.")
    model = st.session_state.get("gemini_model", GEMINI_MODELS[0])
    url = GEMINI_URL.format(model=model) + f"?key={key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    r = requests.post(url, json=payload, timeout=60)
    if r.status_code != 200:
        raise GroqUnavailable(f"Gemini {r.status_code}: {r.text[:200]}")
    data = r.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except Exception:
        raise GroqUnavailable("Gemini returned an unexpected response.")


def law_chatbot_reply(question: str, sections: dict, index: dict) -> str:
    """Grounded answer: pulls the most relevant IPC sections, then asks Gemini."""
    try:
        hits = hybrid_search(index, question, top_k=4, allowed=None)
    except Exception:
        hits = []
    ctx_lines = []
    for k, _score in hits:
        s = sections.get(k)
        if s:
            ctx_lines.append(f"Section {k} ({s['title']}): {s['content'][:350]}")
    context = "\n".join(ctx_lines)
    system = (
        "You are a concise Indian law chatbot grounded in the Indian Penal Code (IPC, 1860). "
        "Use the provided candidate sections when relevant, cite section numbers, keep answers "
        "short and in plain language, and always end with: 'Not legal advice — verify with a "
        "lawyer / the bare Act.'"
    )
    prompt = f"Relevant IPC sections:\n{context}\n\nQuestion: {question}"
    return gemini_complete(prompt, system=system, max_tokens=600)


def groq_transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    key = get_groq_key()
    if not key:
        raise GroqUnavailable("No Groq API key configured.")
    r = requests.post(
        GROQ_AUDIO_URL,
        headers={"Authorization": f"Bearer {key}"},
        files={"file": (filename, audio_bytes)},
        data={"model": "whisper-large-v3"},
        timeout=180,
    )
    if r.status_code != 200:
        raise GroqUnavailable(f"Groq {r.status_code}: {r.text[:200]}")
    return r.json().get("text", "")


# ── multilingual ────────────────────────────────────────────────────────────
_SCRIPT_RANGES = {
    "Devanagari (Hindi)": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Gurmukhi (Punjabi)": (0x0A00, 0x0A7F),
    "Arabic/Urdu": (0x0600, 0x06FF),
}


def detect_script(text: str) -> str | None:
    for ch in text:
        cp = ord(ch)
        for name, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                return name
    return None


def translate_to_english(text: str) -> str:
    prompt = (
        "Translate the following legal case description into clear English. "
        "Return ONLY the translation, no notes.\n\n" + text
    )
    return groq_complete([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=500)


# ── NER / reasoning / contradiction ─────────────────────────────────────────
def extract_entities(text: str) -> dict:
    prompt = (
        "Extract entities from this legal case description. Reply ONLY with JSON:\n"
        '{"victim":[], "accused":[], "weapon":[], "location":[], "offence":[]}\n\n'
        f"TEXT:\n{text[:1200]}"
    )
    out = groq_complete([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=400)
    data = _extract_json(out)
    return {k: (v if isinstance(v, list) else [v]) for k, v in data.items()}


def case_analysis_stream(case: str, results: list[tuple], sections: dict):
    blocks = []
    for k, score in results:
        s = sections.get(k)
        if s:
            blocks.append(f"- Section {k} ({s['title']}) [similarity {score:.0%}]: {s['content'][:400]}")
    sys_msg = (
        "You are an Indian criminal-law research assistant. Be precise, cite section "
        "numbers, and add a short disclaimer that this is not legal advice."
    )
    user = (
        f"CASE FACTS:\n{case}\n\nCANDIDATE IPC SECTIONS:\n" +
        "\n".join(blocks) +
        "\n\nFor each candidate section, explain in 1-2 sentences WHY it may apply, and "
        "rate applicability (Strong/Moderate/Weak)."
    )
    yield from groq_stream([{"role": "system", "content": sys_msg},
                            {"role": "user", "content": user}], max_tokens=900)


def cot_reasoning_stream(case: str, section: dict):
    sys_msg = ("You reason step by step like a criminal-law analyst. Show numbered "
               "reasoning steps, then a conclusion.")
    user = (
        f"Section {section['num']}: {section['title']}\nText: {section['content'][:900]}\n\n"
        f"Case: {case}\n\nShow step-by-step whether this section applies."
    )
    yield from groq_stream([{"role": "system", "content": sys_msg},
                            {"role": "user", "content": user}], max_tokens=800)


def contradiction_stream(case: str, results: list[tuple], sections: dict):
    names = ", ".join(k for k, _ in results)
    sys_msg = "You are a meticulous legal reviewer. Flag inconsistencies explicitly."
    user = (
        f"Case: {case}\n\nCandidate sections: {names}\n\n"
        "Do any of these facts point to mutually inconsistent offences (e.g. conflicting "
        "intent/negligence, or sections that cannot co-exist)? List concerns, or say "
        "'No material contradiction detected.'"
    )
    yield from groq_stream([{"role": "system", "content": sys_msg},
                            {"role": "user", "content": user}], max_tokens=700)


def chat_stream(history: list[dict], context: str):
    sys_msg = (
        "You are the IPC Navigator assistant. Answer questions about the Indian Penal "
        "Code. Use plain language, give examples on request, and cite sections. "
        "Add a brief 'not legal advice' caveat when giving guidance.\n\n"
        + (f"Context:\n{context}" if context else "")
    )
    msgs = [{"role": "system", "content": sys_msg}] + history
    yield from groq_stream(msgs, max_tokens=800)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  NLP / LEGAL HELPERS
# ═════════════════════════════════════════════════════════════════════════════
_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "hundred": 100, "thousand": 1000,
}
_UNIT_MONTHS = {"year": 12, "years": 12, "month": 1, "months": 1, "day": 0}


def _to_num(word: str):
    word = word.strip().lower()
    if word.isdigit():
        return int(word)
    parts = re.split(r"[\s\-]+", word)
    total, found = 0, False
    for p in parts:
        if p in _NUM_WORDS:
            total += _NUM_WORDS[p]
            found = True
    return total if found else None


def _parse_duration(phrase: str):
    """'ten years' -> 120 (months)."""
    m = re.search(r"([a-z\- ]+?)\s+(year|years|month|months|day|days)", phrase.lower())
    if not m:
        return None
    val = _to_num(m.group(1))
    if val is None:
        return None
    return val * _UNIT_MONTHS.get(m.group(2), 12)


def punishment_calc(section: dict) -> dict:
    """Auto-compute punishment bounds and types from the section text."""
    content = section["content"]
    types, detail = set(), {}

    if re.search(r"punished with death|shall be punished with death|with death or", content, re.I):
        types.add("Death")
        detail["death"] = "Death penalty provided."
    if re.search(r"imprisonment for life|transportation for life", content, re.I):
        types.add("Life imprisonment")
        detail["life"] = "Imprisonment for life."

    m = re.search(r"may extend to ([a-z\- ]+? (?:year|years|month|months|day|days))", content, re.I)
    if m:
        months = _parse_duration(m.group(1))
        if months is not None:
            types.add("Imprisonment")
            detail["max"] = m.group(1).strip()
            detail["max_months"] = months

    m = re.search(r"not less than ([a-z\- ]+? (?:year|years|month|months|day|days))", content, re.I)
    if m:
        months = _parse_duration(m.group(1))
        if months is not None:
            types.add("Imprisonment")
            detail["min"] = m.group(1).strip()
            detail["min_months"] = months

    if re.search(r"\bfine\b", content, re.I):
        types.add("Fine")
    if not types:
        types.add("Not explicit")

    return {"types": types, "detail": detail}


def analyze_section(section: dict) -> dict:
    content = section["content"]
    m = re.search(r"shall be punished with ([^.]+?)(?:\.|$)", content, re.I)
    punishment = m.group(1).strip() if m else "Not explicitly stated in this section."
    m = re.search(r"(?:Whoever|If any person|Any person who)\s+(.+?)"
                  r"(?:(?:,\s*shall)|(?:\.\s*shall)|(?:commits))", content)
    trigger = m.group(1).strip() if m else content[:300]
    return {"punishment": punishment, "trigger": trigger}


def _extractive_summary(text: str, n: int = 2) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]
    if not sentences:
        return [text[:300]]
    if len(sentences) <= n:
        return sentences
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        X = vec.fit_transform(sentences)
        scores = np.asarray(X.sum(axis=1)).flatten()
        top = scores.argsort()[-n:][::-1]
        return [sentences[i] for i in sorted(top)]
    except Exception:
        return sentences[:n]


def summarize(text: str, n: int = 2) -> list[str]:
    return _extractive_summary(text, n)


def legal_tag_badge_html(num: str) -> str:
    t = LEGAL_TAGS.get(num)
    if not t:
        return '<span class="pill pill-grey">tags: refer to statute</span>'
    bail_cls = "pill-red" if t["bail"] == "Non-bailable" else "pill-green"
    cog_cls = "pill-amber" if t["cog"] == "Cognizable" else "pill-grey"
    comp = t["comp"]
    comp_cls = "pill-green" if comp == "Yes" else "pill-red"
    return (
        f'<span class="pill {bail_cls}">{t["bail"]}</span>'
        f'<span class="pill {cog_cls}">{t["cog"]}</span>'
        f'<span class="pill {comp_cls}">Compoundable: {comp}</span>'
        f'<span class="pill pill-grey">{t["court"]}</span>'
    )


def confidence_badge(score: float) -> str:
    s = max(0.0, min(1.0, float(score)))
    if s >= 0.55:
        cls, label = "pill-green", "High"
    elif s >= 0.30:
        cls, label = "pill-amber", "Medium"
    else:
        cls, label = "pill-red", "Low"
    return f'<span class="pill {cls}">{label} · {s:.0%}</span>'


def related_sections(num: str) -> list[str]:
    return sorted(RELATED_GRAPH.get(num, set()), key=lambda x: (len(x), x))


def compare_sections(a: dict, b: dict) -> list[tuple[str, str, str]]:
    """Line-level diff of two section texts."""
    ta = re.split(r"(?<=[.;])\s+", a["content"])
    tb = re.split(r"(?<=[.;])\s+", b["content"])
    sm = difflib.SequenceMatcher(None, ta, tb)
    rows = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                rows.append(("same", ta[k], tb[j1 + (k - i1)]))
        elif tag in ("replace", "delete"):
            left = " ".join(ta[i1:i2]) or "—"
            right = " ".join(tb[j1:j2]) or "—"
            rows.append(("diff", left, right))
        elif tag == "insert":
            rows.append(("diff", "—", " ".join(tb[j1:j2])))
    return rows


# ── external legal data (optional) ──────────────────────────────────────────
def get_kanoon_token() -> str:
    try:
        tok = st.session_state.get("kanoon_token", "") or ""
    except Exception:
        tok = ""
    if not tok:
        tok = os.environ.get("INDIAN_KANOON_TOKEN", "")
    return tok.strip()


def kanoon_search(query: str, limit: int = 5):
    """Best-effort landmark-judgment lookup via the Indian Kanoon API."""
    tok = get_kanoon_token()
    if not tok:
        return None
    try:
        r = requests.post(
            f"https://api.indiankanoon.org/search/?formInput={quote(query)}",
            headers={"Authorization": f"Token {tok}"}, timeout=30)
        if r.status_code != 200:
            return None
        docs = r.json().get("docs", [])[:limit]
        return [{"title": d.get("title", "").strip(), "court": d.get("docsource", ""),
                 "date": d.get("publishdate", "")} for d in docs]
    except Exception:
        return None


def check_source_update() -> str:
    """Best-effort reachability check of the official IPC source."""
    try:
        r = requests.get("https://www.indiacode.nic.in/handle/123456789/2263",
                         timeout=15)
        return f"Source reachable (HTTP {r.status_code}). Verify amendments manually."
    except Exception as e:
        return f"Could not reach source: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# 7B.  LEGAL NEWS / STORIES  (RSS, no API key required)
# ═════════════════════════════════════════════════════════════════════════════
NEWS_FEEDS = [
    ("LiveLaw", "https://www.livelaw.in/rss.xml"),
    ("Bar & Bench", "https://www.barandbench.com/feed"),
]
STORIES_FEEDS = [
    ("SCC Online Blog", "https://www.scconline.com/blog/feed/"),
    ("The Leaflet", "https://theleaflet.in/feed/"),
]

_FALLBACK_NEWS = [
    {"title": "Supreme Court updates on criminal law — check LiveLaw / Bar & Bench",
     "link": "https://www.livelaw.in", "source": "LiveLaw"},
]


def _parse_rss(url: str, source: str, limit: int = 5) -> list[dict]:
    import xml.etree.ElementTree as ET
    out: list[dict] = []
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return out
        root = ET.fromstring(r.content)
        for item in root.iter():
            if item.tag.split("}")[-1] != "item":
                continue
            title, link = "", ""
            for child in item:
                tag = child.tag.split("}")[-1]
                if tag == "title" and not title:
                    title = (child.text or "").strip()
                elif tag == "link" and not link:
                    link = (child.text or child.get("href") or "").strip()
            if title:
                out.append({"title": title, "link": link or url, "source": source})
            if len(out) >= limit:
                break
    except Exception:
        return out
    return out


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_feed_items(feeds: tuple, limit_each: int = 3, total: int = 5) -> list[dict]:
    items: list[dict] = []
    for source, url in feeds:
        items.extend(_parse_rss(url, source, limit=limit_each))
        if len(items) >= total:
            break
    return items[:total] if items else []


def render_news_strip() -> None:
    st.markdown("<div class='sec-h'>📰 Legal News & Stories</div>", unsafe_allow_html=True)
    st.caption("Latest from Indian legal-news outlets — refreshed every 30 min.")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='col-h'>📜 Top stories</div>", unsafe_allow_html=True)
        stories = fetch_feed_items(tuple(STORIES_FEEDS), total=5)
        if not stories:
            stories = _FALLBACK_NEWS
        for s in stories:
            st.markdown(
                f"<div class='newsitem'>📘 <a href='{s['link']}' target='_blank'>{s['title']}</a>"
                f"<div class='muted'>{s['source']}</div></div>",
                unsafe_allow_html=True,
            )

    with c2:
        st.markdown("<div class='col-h'>🗞️ Top news</div>", unsafe_allow_html=True)
        news = fetch_feed_items(tuple(NEWS_FEEDS), total=5)
        if not news:
            news = _FALLBACK_NEWS
        for n in news:
            st.markdown(
                f"<div class='newsitem'>🔸 <a href='{n['link']}' target='_blank'>{n['title']}</a>"
                f"<div class='muted'>{n['source']}</div></div>",
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# 8.  PDF REPORT
# ═════════════════════════════════════════════════════════════════════════════
def _lat(text: str) -> str:
    repl = {"—": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"',
            "•": "-", "₹": "Rs.", "\u00a0": " "}
    for k, v in repl.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(title: str, blocks: list[tuple[str, str]]) -> bytes:
    if not FPDF_OK:
        raise RuntimeError("fpdf2 not installed")
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9, _lat(title))
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _lat(LEGAL_DISCLAIMER.replace("**", "").replace("⚠️", "")))
    pdf.ln(3)
    for heading, body in blocks:
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 7, _lat(heading))
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, _lat(body))
        pdf.ln(2)
    out = pdf.output(dest="S")
    return bytes(out)


# ═════════════════════════════════════════════════════════════════════════════
# 9.  UI  —  theme, layout, pages
# ═════════════════════════════════════════════════════════════════════════════
import base64
from streamlit.components.v1 import html as st_html


import html as _html

THEMES = {
    "dark": dict(bg="#090d1b", bg2="#0e1426", card="#131b30", card2="#1a2340", text="#e9edf8",
                 muted="#9aa5c0", line="#252f50", accent="#7c8cff", accent2="#a06bff",
                 gold="#e4b95c", shadow="0 10px 30px rgba(0,0,0,.35)",
                 glow1="rgba(124,140,255,.16)", glow2="rgba(160,107,255,.12)", scheme="dark",
                 ok="#4ade80", warn="#fbbf24", bad="#f87171", info="#8da2ff", vio="#b794ff"),
    "light": dict(bg="#f4f6fc", bg2="#ffffff", card="#ffffff", card2="#eef1fa", text="#0f1830",
                  muted="#59637d", line="#dfe4f1", accent="#4f46e5", accent2="#7c3aed",
                  gold="#c9962f", shadow="0 8px 24px rgba(30,41,90,.08)",
                  glow1="rgba(79,70,229,.10)", glow2="rgba(124,58,237,.08)", scheme="light",
                  ok="#15803d", warn="#b45309", bad="#b91c1c", info="#3730a3", vio="#6d28d9"),
}

FONT_IMPORT = ("@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800"
               "&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');")

CSS_BODY = r"""
html, body, .stApp, [class*="css"] { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
.stApp {
    background:
      radial-gradient(900px 520px at 6% -8%, var(--glow1), transparent 62%),
      radial-gradient(760px 480px at 100% 0%, var(--glow2), transparent 58%),
      var(--bg) !important;
    background-attachment: fixed !important; color: var(--text);
}
[data-testid="stAppViewContainer"], [data-testid="stMain"], .main { background: transparent !important; color: var(--text); }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stStatusWidget"] { visibility: hidden; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
h1, h2, h3, h4, h5 { color: var(--text) !important; letter-spacing: -.015em; }
h3 { font-family: 'Fraunces', Georgia, serif; font-weight: 650; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p, label { color: var(--text); }
button [data-testid="stMarkdownContainer"] p { color: inherit !important; }
[data-testid="stCaptionContainer"] { color: var(--muted) !important; }
a { color: var(--accent); }
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] > div, div[data-testid="stSidebarContent"] { background: transparent !important; }
.side-brand { display: flex; align-items: center; gap: 12px; padding: 4px 2px 8px; }
.logo {
    width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
    font-size: 21px; color: #1b1405; flex-shrink: 0;
    background: linear-gradient(135deg, var(--gold), color-mix(in srgb, var(--gold) 55%, #7c4a00));
    box-shadow: 0 6px 18px color-mix(in srgb, var(--gold) 35%, transparent);
}
.side-brand .nm { font-family: 'Fraunces', Georgia, serif; font-weight: 700; font-size: 18px; line-height: 1.1; color: var(--text); }
.side-brand .sb { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
.side-h { font-size: 10.5px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin: 6px 0 8px; }

div[role="radiogroup"] { flex-direction: column; align-items: stretch; gap: 4px; }
div[role="radiogroup"] > label {
    background: transparent; border: 1px solid transparent; border-radius: 12px;
    padding: 9px 12px; margin: 0 !important; width: 100%; cursor: pointer; transition: all .18s ease;
}
div[role="radiogroup"] > label:hover { background: var(--card2); }
div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 22%, transparent), color-mix(in srgb, var(--accent2) 16%, transparent));
    border-color: color-mix(in srgb, var(--accent) 45%, transparent); box-shadow: inset 3px 0 0 var(--accent);
}
div[role="radiogroup"] > label:has(input:checked) p { color: var(--text) !important; font-weight: 700; }
div[role="radiogroup"] > label > div:first-child { display: none; }

/* ---------- top bar + hero ---------- */
.topbar {
    display: flex; align-items: center; gap: 14px; padding: 12px 18px; margin-bottom: 20px;
    border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow);
    background: color-mix(in srgb, var(--card) 82%, transparent); backdrop-filter: blur(10px);
}
.topbar .brand { font-family: 'Fraunces', Georgia, serif; font-size: 20px; font-weight: 700; color: var(--text); line-height: 1.1; }
.topbar .brand span { color: var(--gold); }
.topbar .brandsub { font-size: 12px; color: var(--muted); margin-top: 2px; }
.topbar .tb-right { margin-left: auto; display: flex; gap: 6px; flex-wrap: wrap; }

.hero {
    position: relative; overflow: hidden; border-radius: 20px; padding: 26px 30px; margin-bottom: 22px;
    border: 1px solid var(--line); box-shadow: var(--shadow); animation: fadeUp .5s ease both;
    background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 20%, var(--card)), color-mix(in srgb, var(--accent2) 14%, var(--card)));
}
.hero::before {
    content: ""; position: absolute; inset: 0; opacity: .35; pointer-events: none;
    background-image: linear-gradient(var(--line) 1px, transparent 1px), linear-gradient(90deg, var(--line) 1px, transparent 1px);
    background-size: 36px 36px;
    -webkit-mask-image: linear-gradient(120deg, transparent 30%, #000 100%); mask-image: linear-gradient(120deg, transparent 30%, #000 100%);
}
.hero::after {
    content: ""; position: absolute; right: -70px; top: -90px; width: 280px; height: 280px; border-radius: 50%;
    background: radial-gradient(circle, var(--gold) 0%, transparent 70%); opacity: .20; filter: blur(12px); pointer-events: none;
}
.hero > * { position: relative; z-index: 1; }
.hero .t { font-family: 'Fraunces', Georgia, serif; font-size: 32px; font-weight: 700; color: var(--text); margin: 0; letter-spacing: -.02em; }
.hero .s { color: var(--muted); font-size: 14.5px; margin: 6px 0 0; max-width: 680px; }
.hero .chips { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 6px; }

/* ---------- pills ---------- */
.pills { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 10px; }
.pill { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 11.5px; font-weight: 600; border: 1px solid transparent; white-space: nowrap; }
.pill-green  { background: color-mix(in srgb, var(--ok) 14%, transparent);   color: var(--ok);   border-color: color-mix(in srgb, var(--ok) 35%, transparent); }
.pill-amber  { background: color-mix(in srgb, var(--warn) 14%, transparent); color: var(--warn); border-color: color-mix(in srgb, var(--warn) 35%, transparent); }
.pill-red    { background: color-mix(in srgb, var(--bad) 14%, transparent);  color: var(--bad);  border-color: color-mix(in srgb, var(--bad) 35%, transparent); }
.pill-blue   { background: color-mix(in srgb, var(--info) 14%, transparent); color: var(--info); border-color: color-mix(in srgb, var(--info) 35%, transparent); }
.pill-purple { background: color-mix(in srgb, var(--vio) 14%, transparent);  color: var(--vio);  border-color: color-mix(in srgb, var(--vio) 35%, transparent); }
.pill-grey   { background: var(--card2); color: var(--muted); border-color: var(--line); }

/* ---------- cards (a container holding .card-marker becomes a card) ---------- */
.element-container:has(.card-marker), [data-testid="stElementContainer"]:has(.card-marker) { display: none; }
div[data-testid="stVerticalBlock"]:has(> .element-container .card-marker, > [data-testid="stElementContainer"] .card-marker) {
    background: var(--card); border: 1px solid var(--line); border-radius: 18px;
    padding: 22px 24px; margin-bottom: 14px; box-shadow: var(--shadow);
    animation: fadeUp .45s ease both; transition: border-color .2s ease, box-shadow .2s ease;
}
div[data-testid="stVerticalBlock"]:has(> .element-container .card-marker, > [data-testid="stElementContainer"] .card-marker):hover {
    border-color: color-mix(in srgb, var(--accent) 55%, var(--line));
}
.sec-head { display: flex; gap: 16px; align-items: center; margin-bottom: 4px; }
.sec-num {
    min-width: 70px; padding: 9px 10px; border-radius: 14px; text-align: center; color: #1b1405;
    font-family: 'Fraunces', Georgia, serif; font-size: 24px; font-weight: 700; line-height: 1;
    background: linear-gradient(135deg, var(--gold), color-mix(in srgb, var(--gold) 60%, #fff));
    box-shadow: 0 6px 16px color-mix(in srgb, var(--gold) 30%, transparent);
}
.sec-num small { display: block; font-family: 'Inter', sans-serif; font-size: 9px; letter-spacing: .2em; font-weight: 700; opacity: .7; margin-bottom: 3px; }
.sectitle { font-family: 'Fraunces', Georgia, serif; font-size: 24px; font-weight: 700; line-height: 1.2; color: var(--text); }
.muted { color: var(--muted); font-size: 12.5px; }
.lawtext {
    font-family: 'Fraunces', Georgia, serif; font-size: 16px; line-height: 1.75; color: var(--text);
    padding: 16px 20px; border-radius: 14px; background: var(--card2);
    border: 1px solid var(--line); border-left: 3px solid var(--gold);
}
.kv { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
.kv .box { flex: 1 1 160px; background: var(--card2); border: 1px solid var(--line); border-radius: 12px; padding: 11px 14px; }
.kv .box .k { font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; font-weight: 600; }
.kv .box .v { font-size: 14.5px; font-weight: 650; margin-top: 4px; color: var(--text); }
.diffrow { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 8px; }
.diffcell { background: var(--card2); border: 1px solid var(--line); border-radius: 12px; padding: 11px 13px; font-size: 13px; color: var(--text); }
.diffcell.chg { border-color: color-mix(in srgb, var(--warn) 55%, transparent); background: color-mix(in srgb, var(--warn) 8%, var(--card2)); }

.sec-step { display: flex; align-items: center; gap: 10px; margin: 20px 0 8px; }
.sec-step .step-no {
    width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 800; color: #fff; flex-shrink: 0;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
}
.sec-step .step-label { font-weight: 700; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.rank { font-size: 11px; letter-spacing: .16em; text-transform: uppercase; color: var(--gold); font-weight: 700; margin: 18px 0 8px; }

/* ---------- home / stats / features ---------- */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin: 4px 0 8px; }
.stat {
    position: relative; overflow: hidden; background: var(--card); border: 1px solid var(--line);
    border-radius: 16px; padding: 16px 18px 16px 20px; box-shadow: var(--shadow); animation: fadeUp .5s ease both;
}
.stat::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: linear-gradient(var(--accent), var(--accent2)); }
.stat .n { font-family: 'Fraunces', Georgia, serif; font-size: 30px; font-weight: 700; line-height: 1; color: var(--text); }
.stat .l { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; margin-top: 7px; font-weight: 600; }
.sec-h { font-family: 'Fraunces', Georgia, serif; font-size: 21px; font-weight: 650; margin: 28px 0 14px; display: flex; align-items: center; gap: 12px; color: var(--text); }
.sec-h::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, var(--line), transparent); }
.col-h { font-weight: 700; font-size: 14px; margin: 4px 0 10px; color: var(--text); }
.features { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
.feature {
    background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 18px;
    transition: transform .2s ease, border-color .2s ease; animation: fadeUp .5s ease both;
}
.feature:hover { transform: translateY(-3px); border-color: var(--accent); }
.feature:nth-child(2) { animation-delay: .06s; } .feature:nth-child(3) { animation-delay: .12s; } .feature:nth-child(4) { animation-delay: .18s; }
.feature .ico {
    width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
    font-size: 19px; margin-bottom: 12px; background: color-mix(in srgb, var(--accent) 16%, transparent);
}
.feature .ft { font-weight: 700; font-size: 15px; color: var(--text); }
.feature .fd { font-size: 13px; color: var(--muted); margin-top: 4px; line-height: 1.5; }

.newsitem {
    background: var(--card); border: 1px solid var(--line); border-left: 3px solid var(--accent);
    border-radius: 12px; padding: 10px 14px; margin-bottom: 9px; font-size: 13.5px; transition: transform .15s ease;
}
.newsitem:hover { transform: translateX(3px); }
.newsitem a { text-decoration: none; font-weight: 600; }
.newsitem a:hover { text-decoration: underline; }
.chat-bubble { border-radius: 14px; padding: 9px 12px; margin-bottom: 8px; font-size: 13px; line-height: 1.45; border: 1px solid var(--line); color: var(--text); }
.chat-bubble.user { background: var(--card2); }
.chat-bubble.bot  { background: color-mix(in srgb, var(--accent) 10%, transparent); border-color: color-mix(in srgb, var(--accent) 30%, transparent); }
.footer { margin-top: 38px; padding: 20px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; text-align: center; line-height: 1.7; }
.footer b { color: var(--text); }

/* ---------- streamlit widgets ---------- */
.stButton > button, .stDownloadButton > button, button[data-testid^="stBaseButton"] {
    border-radius: 12px; border: 1px solid var(--line); background: var(--card); color: var(--text);
    font-weight: 600; transition: all .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--accent); color: var(--accent); transform: translateY(-1px);
    box-shadow: 0 6px 16px color-mix(in srgb, var(--accent) 20%, transparent);
}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"], .stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important; border: none !important; color: #fff !important; font-weight: 700;
}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.1); color: #fff !important; }
.stTextInput input, .stTextArea textarea, div[data-baseweb="select"] > div, div[data-baseweb="input"] {
    background: var(--card) !important; color: var(--text) !important; border-radius: 12px !important; border-color: var(--line) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--muted) !important; opacity: .8; }
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important; box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 22%, transparent) !important;
}
div[data-baseweb="popover"] div, div[data-baseweb="popover"] ul { background: var(--card) !important; color: var(--text) !important; }
div[data-testid="stExpander"] { border: 1px solid var(--line) !important; border-radius: 14px !important; background: var(--card) !important; }
div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary p { color: var(--text) !important; font-weight: 600; }
button[data-baseweb="tab"] { color: var(--muted); font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent); }
div[data-baseweb="tab-highlight"] { background: var(--accent) !important; }
div[data-baseweb="tab-border"] { background: var(--line) !important; }
div[data-testid="stAlert"] {
    background: var(--card2) !important; border: 1px solid var(--line) !important;
    border-left: 3px solid var(--accent) !important; border-radius: 12px !important;
}
div[data-testid="stAlert"] * { color: var(--text) !important; }
div[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; box-shadow: var(--shadow); }
div[data-testid="stMetricValue"] { font-family: 'Fraunces', Georgia, serif; font-size: 26px; color: var(--text); }
div[data-testid="stMetricLabel"] p { color: var(--muted) !important; }
[data-testid="stTable"] table { color: var(--text); border-collapse: collapse; }
[data-testid="stTable"] th, [data-testid="stTable"] td { border-color: var(--line) !important; background: transparent !important; }
hr, div[data-testid="stDivider"] { border-color: var(--line) !important; opacity: .7; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
::-webkit-scrollbar-track { background: transparent; }

@media (max-width: 780px) {
    .hero { padding: 20px; } .hero .t { font-size: 24px; }
    .topbar .tb-right, .topbar .brandsub { display: none; }
    .block-container { padding-left: .8rem; padding-right: .8rem; padding-top: 1rem; }
    .diffrow { grid-template-columns: 1fr; } .kv .box { flex: 1 1 100%; }
    .sec-head { flex-direction: column; align-items: flex-start; } .sectitle { font-size: 20px; }
}
"""


def _palette() -> dict:
    return THEMES["dark" if st.session_state.get("dark_mode", True) else "light"]


def inject_css() -> None:
    p = _palette()
    keys = ("bg", "bg2", "card", "card2", "text", "muted", "line", "accent", "accent2",
            "gold", "shadow", "glow1", "glow2", "ok", "warn", "bad", "info", "vio")
    vars_ = "".join(f"--{k}:{p[k]};" for k in keys) + f"color-scheme:{p['scheme']};"
    st.markdown("<style>" + FONT_IMPORT + ":root{" + vars_ + "}" + CSS_BODY + "</style>",
                unsafe_allow_html=True)


def _goto(code: str) -> None:
    """Button callback: jump to Code Lookup for a section (safe before widgets re-instantiate)."""
    st.session_state["code_query"] = code
    st.session_state["nav"] = "🔎 Code Lookup"


def side_h(text: str) -> None:
    st.markdown(f"<div class='side-h'>{text}</div>", unsafe_allow_html=True)


def style_fig(fig, height: int = 300):
    p = _palette()
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=p["text"], family="Inter, sans-serif"))
    fig.update_xaxes(gridcolor=p["line"], zeroline=False)
    fig.update_yaxes(gridcolor=p["line"], zeroline=False)
    fig.update_traces(marker_color=p["accent"], selector=dict(type="bar"))
    return fig


def copy_widget(text: str, label: str = "Copy section text") -> None:
    p = _palette()
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    js = ("navigator.clipboard.writeText("
          "new TextDecoder().decode(Uint8Array.from(atob('" + b64 + "'),"
          "c=>c.charCodeAt(0))));"
          "this.innerText='✓ Copied';"
          "setTimeout(()=>{this.innerText='" + label + "'},1300);")
    css = (".cpy{font-family:Inter,sans-serif;font-size:13px;font-weight:600;padding:7px 14px;"
           "border-radius:12px;border:1px solid %LINE%;background:%CARD%;color:%TEXT%;"
           "cursor:pointer;transition:all .18s ease;}"
           ".cpy:hover{border-color:%ACC%;color:%ACC%;}"
           "body{margin:0;background:transparent;}")
    css = (css.replace("%LINE%", p["line"]).replace("%CARD%", p["card"])
              .replace("%TEXT%", p["text"]).replace("%ACC%", p["accent"]))
    st_html(f"<style>{css}</style><button class='cpy' onclick=\"{js}\">{label}</button>",
            height=46, scrolling=False)


def hero(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    ch = ""
    if chips:
        ch = "<div class='chips'>" + "".join(
            f"<span class='pill pill-blue'>{c}</span>" for c in chips) + "</div>"
    st.markdown(
        f"<div class='hero'><p class='t'>{title}</p>"
        f"<p class='s'>{subtitle}</p>{ch}</div>",
        unsafe_allow_html=True,
    )


def pills(html_bits: str) -> None:
    st.markdown(f"<div class='pills'>{html_bits}</div>", unsafe_allow_html=True)


def render_topbar() -> None:
    st.markdown(
        "<div class='topbar'><div class='logo'>⚖️</div>"
        "<div><div class='brand'>IPC <span>Navigator</span></div>"
        "<div class='brandsub'>AI legal assistant for the Indian Penal Code</div></div>"
        "<div class='tb-right'><span class='pill pill-blue'>IPC · 1860</span>"
        "<span class='pill pill-purple'>BNS · 2023 mapped</span></div></div>",
        unsafe_allow_html=True)


def render_home(sections: dict) -> None:
    """Landing view for Code Lookup: stats, popular sections, feature tiles."""
    chapters = {s["chapter_code"] for s in sections.values()}
    tiles = [(len(sections), "Sections indexed"), (len(chapters), "Chapters"),
             (len(IPC_BNS_MAP), "IPC → BNS mappings"), (len(LEGAL_TAGS), "Bail / cognizable tags")]
    st.markdown("<div class='stats'>" + "".join(
        f"<div class='stat'><div class='n'>{n}</div><div class='l'>{l}</div></div>"
        for n, l in tiles) + "</div>", unsafe_allow_html=True)

    st.markdown("<div class='sec-h'>Popular sections</div>", unsafe_allow_html=True)
    popular = [c for c in ("302", "420", "376", "379", "498A", "304B") if c in sections]
    for row in range(0, len(popular), 3):
        cols = st.columns(3)
        for col, code in zip(cols, popular[row:row + 3]):
            title = sections[code]["title"]
            title = title if len(title) <= 34 else title[:32] + "…"
            col.button(f"§{code} · {title}", key=f"pop_{code}", on_click=_goto,
                       args=(code,), use_container_width=True)

    st.markdown("<div class='sec-h'>What you can do</div>", unsafe_allow_html=True)
    feats = [("🧩", "Case study → code", "Describe a scenario; hybrid BM25 + dense search finds the sections."),
             ("⚖️", "Compare sections", "Side-by-side diff of two sections, with punishment and BNS mapping."),
             ("🌳", "Browse by chapter", "Walk the Code like a tree: Chapter → Section."),
             ("💬", "Ask the assistant", "Grounded chat in plain language, with section citations.")]
    st.markdown("<div class='features'>" + "".join(
        f"<div class='feature'><div class='ico'>{i}</div><div class='ft'>{t}</div><div class='fd'>{d}</div></div>"
        for i, t, d in feats) + "</div>", unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        "<div class='footer'><b>IPC Navigator</b> · Reference aid only — bail / cognizable / "
        "compoundability tags and the IPC→BNS mapping are indicative. Verify against the bare Act, "
        "BNSS 2023 and the Bharatiya Nyaya Sanhita, 2023. Not legal advice.</div>",
        unsafe_allow_html=True)


def remember_lookup(code: str) -> None:
    hist = st.session_state.setdefault("history", [])
    hist = [h for h in hist if h != code]
    hist.insert(0, code)
    st.session_state["history"] = hist[:25]


def toggle_bookmark(code: str) -> None:
    bm = st.session_state.setdefault("bookmarks", [])
    if code in bm:
        bm.remove(code)
    else:
        bm.append(code)
    st.session_state["bookmarks"] = bm


def run_search(index, query, engine, top_k, allowed):
    """Engine-selectable retrieval (used for A/B testing)."""
    if engine == "BM25 only":
        sc = _bm25_scores(index, query)
        if sc is None:
            return hybrid_search(index, query, top_k, allowed)
        order = np.argsort(-sc)[: max(top_k * 6, 30)]
        return [(index["keys"][i], float(sc[i] / (sc.max() or 1.0))) for i in order
                if allowed is None or index["keys"][i] in allowed][:top_k]
    if engine == "Dense only" and index["embs"] is not None:
        sc = _dense_scores(index, query)
        order = np.argsort(-sc)[: max(top_k * 6, 30)]
        return [(index["keys"][i], float(sc[i])) for i in order
                if allowed is None or index["keys"][i] in allowed][:top_k]
    if engine == "TF-IDF only":
        sc = _tfidf_scores(index, query)
        order = np.argsort(-sc)[: max(top_k * 6, 30)]
        return [(index["keys"][i], float(sc[i])) for i in order
                if allowed is None or index["keys"][i] in allowed][:top_k]
    return hybrid_search(index, query, top_k, allowed)


# ── section card ────────────────────────────────────────────────────────────
def render_section_card(num: str, sec: dict, sections: dict, *,
                        score: float | None = None, prefix: str = "",
                        query: str = "", allow_actions: bool = True) -> None:
    with st.container():
        st.markdown("<span class='card-marker'></span>", unsafe_allow_html=True)
        _section_card_body(num, sec, sections, score=score, prefix=prefix,
                           query=query, allow_actions=allow_actions)


def _section_card_body(num: str, sec: dict, sections: dict, *,
                        score: float | None = None, prefix: str = "",
                        query: str = "", allow_actions: bool = True) -> None:
    st.markdown(
        "<div class='sec-head'>"
        f"<div class='sec-num'><small>SEC</small>{num}</div>"
        "<div>"
        f"<div class='sectitle'>{_html.escape(sec['title'])}</div>"
        f"<div class='muted'>Indian Penal Code, 1860 · Chapter {sec['chapter_code']} · "
        f"{_html.escape(sec['chapter_title'])}</div>"
        "</div></div>", unsafe_allow_html=True)

    bits = ""
    if score is not None:
        bits += confidence_badge(score)
    bits += legal_tag_badge_html(num)
    bns = IPC_BNS_MAP.get(num)
    if bns:
        bits += f"<span class='pill pill-purple'>BNS § {bns}</span>"
    pills(bits)

    # 2) WHAT THIS SECTION IS ------------------------------------------------
    st.markdown(
        f"<div class='sec-step'><span class='step-no'>1</span>"
        f"<span class='step-label'>What this section is</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='lawtext'>{_html.escape(sec['content'])}</div>", unsafe_allow_html=True)

    calc = punishment_calc(sec)
    detail = calc["detail"]
    lo = detail.get("min")
    hi = detail.get("max")
    relation = "—"
    if lo and hi:
        relation = f"{lo}  →  {hi}"
    elif hi:
        relation = f"up to {hi}"
    elif lo:
        relation = f"not less than {lo}"
    elif "Life imprisonment" in calc["types"]:
        relation = "imprisonment for life"
    elif "Death" in calc["types"]:
        relation = "death"
    st.markdown(
        "<div class='kv'>"
        f"<div class='box'><div class='k'>Punishment types</div><div class='v'>{', '.join(sorted(calc['types']))}</div></div>"
        f"<div class='box'><div class='k'>Term range (auto)</div><div class='v'>{relation}</div></div>"
        f"<div class='box'><div class='k'>Bailable</div><div class='v'>{LEGAL_TAGS.get(num, {}).get('bail', 'Refer to statute')}</div></div>"
        "</div>", unsafe_allow_html=True)

    # 3) WHY IT IS USED -------------------------------------------------------
    why = analyze_section(sec)
    st.markdown(
        f"<div class='sec-step'><span class='step-no'>2</span>"
        f"<span class='step-label'>Why it is used</span></div>",
        unsafe_allow_html=True,
    )
    st.write(why["trigger"])
    st.caption(f"Punishment clause: {why['punishment']}")

    # 4) SUMMARY, POINT-WISE ---------------------------------------------------
    st.markdown(
        f"<div class='sec-step'><span class='step-no'>3</span>"
        f"<span class='step-label'>Summary — point-wise</span></div>",
        unsafe_allow_html=True,
    )
    for i, pt in enumerate(summarize(sec["content"], 3), 1):
        st.markdown(f"- {pt}")

    rel = [r for r in related_sections(num) if r in sections]
    if rel:
        st.markdown("**🕸️ Related sections**")
        cols = st.columns(min(len(rel), 8))
        for c, r in zip(cols, rel[:8]):
            c.button(r, key=f"{prefix}rel_{num}_{r}", on_click=_goto, args=(r,))

    if sec.get("amendments"):
        with st.expander("🕓 Amendment notes"):
            for a in sec["amendments"][:6]:
                st.markdown(f"- {a}")

    if allow_actions:
        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.4, 3])
        with c1:
            star = "⭐" if num in st.session_state.get("bookmarks", []) else "☆"
            if st.button(f"{star} Save", key=f"{prefix}bm_{num}"):
                toggle_bookmark(num)
                st.rerun()
        with c2:
            if st.button("👍 Helpful", key=f"{prefix}up_{num}"):
                log_feedback(query, num, 1, "local")
                st.toast(f"Thanks — noted 👍 for §{num}")
        with c3:
            if st.button("👎 Not relevant", key=f"{prefix}dn_{num}"):
                log_feedback(query, num, -1, "local")
                st.toast(f"Thanks — noted 👎 for §{num}")
        with c4:
            copy_widget(sec["full"], "📋 Copy section text")



# ── pages ───────────────────────────────────────────────────────────────────
def page_code(sections: dict) -> None:
    raw = st.text_input(
        "Section number — try 302, 42O (letter O), 124A, or a range like 299-304",
        key="code_query", placeholder="e.g. 302")

    if not raw:
        render_home(sections)
        return
    log_event("search", query=raw)

    rng = parse_range(raw)
    if rng:
        a, b = rng
        st.markdown(f"### Sections {a}–{b}")
        for n in range(a, b + 1):
            k = str(n)
            if k in sections:
                render_section_card(k, sections[k], sections, prefix="rng_")
        return

    code, how = fuzzy_lookup(raw, sections.keys())
    if not code:
        st.error(f"❌ Nothing close to `{raw}` was found in the IPC database.")
        return
    if how == "fuzzy":
        st.warning(f"🔤 Interpreted `{raw}` as **Section {code}** (fuzzy match).")
    elif how == "prefix":
        st.info(f"ℹ️ Showing the closest match: **Section {code}**.")

    remember_lookup(code)
    log_event("click", query=raw, section=code)
    render_section_card(code, sections[code], sections, prefix="lookup_", query=raw)


def page_case(sections: dict, index: dict) -> None:
    hero("🧩 Case Study → Code", "Describe a scenario and find the applicable IPC sections.",
         ["hybrid BM25+dense", "LLM rerank", "NER", "multi-lingual", "voice"])

    # ---- controls ----
    with st.expander("⚙️ Search options", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            top_k = st.slider("Top-k matches", 1, 10, 5)
        with c2:
            engine = st.selectbox("Engine (for A/B)", ["Hybrid", "BM25 only", "Dense only", "TF-IDF only"])
        with c3:
            rerank = st.toggle("LLM re-rank", value=False,
                               help="Needs a Groq key. Re-orders results with the LLM.")

        st.markdown("**Keyword + facet filters**")
        f1, f2, f3 = st.columns(3)
        chapters = ["All"] + sorted({s["chapter_code"] for s in sections.values()})
        with f1:
            chapter = st.selectbox("Chapter", chapters)
        with f2:
            ptype = st.selectbox("Punishment type",
                                 ["Any", "Death", "Life imprisonment", "Imprisonment", "Fine"])
        with f3:
            bail = st.selectbox("Bail", ["Any", "Bailable", "Non-bailable"])
        kq = st.text_input("Must contain keyword(s)", placeholder="e.g. consent, dwelling")

    # ---- voice input ----
    with st.expander("🎙️ Voice input (Whisper via Groq)", expanded=False):
        if not groq_available():
            st.caption("Add a Groq API key in the sidebar to enable dictation.")
        elif hasattr(st, "audio_input"):
            audio = st.audio_input("Record your case description")
            if audio is not None and st.button("Transcribe"):
                try:
                    with st.spinner("Transcribing…"):
                        st.session_state["case_text"] = groq_transcribe(audio.read(), "case.wav")
                    st.success("Transcribed — it's now in the case box below.")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
        else:
            st.caption("st.audio_input unavailable in this Streamlit build.")

    # ---- case input ----
    st.session_state.setdefault("case_text", "")
    case = st.text_area("Case study", key="case_text", height=170,
                        placeholder="Example: A man takes a minor girl away from her parents "
                                    "without their consent with the intention of marrying her.")

    detected = detect_script(case) if case else None
    if detected:
        st.info(f"🌐 Detected **{detected}** script — it will be translated to English before search.")

    go = st.button("🔎 Find applicable IPC sections", type="primary")

    if go and case.strip():
        query = case
        if detected and groq_available():
            try:
                with st.spinner("Translating to English…"):
                    query = translate_to_english(case)
                st.success("**Translated query:** " + query)
            except Exception as e:
                st.warning(f"Translation unavailable ({e}); searching the original text.")

        allowed = filter_sections(sections, chapter, ptype, bail)
        if kq.strip():
            kws = [k.strip().lower() for k in kq.split(",") if k.strip()]
            allowed = {n for n in allowed
                       if all(w in sections[n]["content"].lower() for w in kws)}

        log_event("search", query=query)
        with st.spinner("Running retrieval…"):
            results = run_search(index, query, engine, top_k, allowed if allowed else None)
            if rerank and results:
                results = [(k, s) for k, _b, s in llm_rerank(
                    query, [(k, 0.0, s) for k, s in results], top_k)]
        if results:
            log_abtest(query, engine, results[0][0], results[0][1])
        st.session_state["last_results"] = results
        st.session_state["last_query"] = query

    results = st.session_state.get("last_results") or []
    query = st.session_state.get("last_query", "")

    if results:
        # ---- NER + LLM analysis ----
        if groq_available():
            t1, t2, t3 = st.tabs(["🧠 LLM case analysis", "🏷️ Entities (NER)", "⚠️ Contradiction check"])
            with t1:
                try:
                    st.write_stream(case_analysis_stream(query, results, sections))
                except Exception as e:
                    st.error(f"LLM error: {e}")
            with t2:
                if st.button("Extract entities"):
                    try:
                        with st.spinner("Extracting…"):
                            st.session_state["ner"] = extract_entities(query)
                    except Exception as e:
                        st.error(f"NER failed: {e}")
                ner = st.session_state.get("ner")
                if ner:
                    cc = st.columns(len(ner) or 1)
                    for c, (k, v) in zip(cc, ner.items()):
                        c.markdown(f"**{k.title()}**")
                        c.write(", ".join(v) if v else "—")
            with t3:
                if st.button("Check for contradictions"):
                    try:
                        st.write_stream(contradiction_stream(query, results, sections))
                    except Exception as e:
                        st.error(f"LLM error: {e}")
        else:
            st.caption("🔑 Add a Groq API key in the sidebar to unlock LLM analysis, NER and "
                       "contradiction checking.")

        st.markdown("### 🎯 Most relevant IPC sections")
        for rank, (code, score) in enumerate(results, 1):
            st.markdown(f"<div class='rank'>Match {rank}</div>", unsafe_allow_html=True)
            render_section_card(code, sections[code], sections,
                                score=score, prefix=f"case{rank}_", query=query)

        # ---- chain-of-thought ----
        if groq_available() and results:
            with st.expander("🪜 Chain-of-thought reasoning (top match)"):
                if st.button("Generate step-by-step reasoning"):
                    try:
                        st.write_stream(cot_reasoning_stream(query, sections[results[0][0]]))
                    except Exception as e:
                        st.error(f"LLM error: {e}")

        # ---- landmark judgments ----
        with st.expander("⚖️ Landmark judgments (Indian Kanoon)"):
            if not get_kanoon_token():
                st.caption("Provide an Indian Kanoon API token in the sidebar to fetch citations.")
            else:
                if st.button("Fetch judgments for top match"):
                    sec = sections[results[0][0]]
                    hits = kanoon_search(f"{sec['title']} section {sec['num']} IPC")
                    if not hits:
                        st.info("No results (or the API rejected the request).")
                    for h in hits or []:
                        st.markdown(f"- **{h['title']}** · {h['court']} · {h['date']}")

        # ---- report ----
        st.markdown("### 📄 Export")
        if st.button("Build PDF report"):
            blocks = []
            for code, score in results:
                s = sections[code]
                blocks.append((f"Section {code} — {s['title']}",
                               f"Similarity: {score:.0%}\n{s['content'][:1500]}"))
            try:
                st.session_state["pdf"] = build_pdf("IPC Navigator — Case Analysis", blocks)
            except Exception as e:
                st.error(f"PDF build failed: {e}")
        if st.session_state.get("pdf"):
            st.download_button("⬇️ Download report (PDF)", st.session_state["pdf"],
                               file_name="ipc_analysis.pdf", mime="application/pdf")


def page_compare(sections: dict) -> None:
    hero("⚖️ Compare Sections", "Side-by-side diff of two sections — e.g. 299 vs 300, 375 vs 376.",
         ["diff view", "punishment diff", "BNS mapping"])
    keys = list(sections.keys())
    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("Section A", keys, index=keys.index("299") if "299" in keys else 0)
    with c2:
        b = st.selectbox("Section B", keys, index=keys.index("300") if "300" in keys else 1)
    if not a or not b:
        return
    sa, sb = sections[a], sections[b]
    ca, cb = punishment_calc(sa), punishment_calc(sb)

    st.markdown("### Head-to-head")
    st.markdown(
        "<div class='kv'>"
        f"<div class='box'><div class='k'>A · §{a}</div><div class='v'>{sa['title']}</div></div>"
        f"<div class='box'><div class='k'>B · §{b}</div><div class='v'>{sb['title']}</div></div>"
        f"<div class='box'><div class='k'>A punishment</div><div class='v'>{', '.join(sorted(ca['types']))}</div></div>"
        f"<div class='box'><div class='k'>B punishment</div><div class='v'>{', '.join(sorted(cb['types']))}</div></div>"
        f"<div class='box'><div class='k'>A → BNS</div><div class='v'>{IPC_BNS_MAP.get(a, '—')}</div></div>"
        f"<div class='box'><div class='k'>B → BNS</div><div class='v'>{IPC_BNS_MAP.get(b, '—')}</div></div>"
        "</div>", unsafe_allow_html=True)

    st.markdown("### Textual diff (changed rows highlighted)")
    for kind, left, right in compare_sections(sa, sb)[:120]:
        cls = "diffcell chg" if kind == "diff" else "diffcell"
        st.markdown(
            f"<div class='diffrow'><div class='{cls}'>{left}</div>"
            f"<div class='{cls}'>{right}</div></div>",
            unsafe_allow_html=True)


def page_browse(sections: dict) -> None:
    hero("🌳 Chapter Taxonomy", "Browse the IPC like a tree: Chapter → Section.",
         [f"{len(sections)} sections"])
    by_ch: dict[str, list[str]] = {}
    for num, sec in sections.items():
        by_ch.setdefault(sec["chapter_code"], []).append(num)
    for code in sorted(by_ch, key=lambda x: (len(x), x)):
        nums = sorted(by_ch[code], key=lambda n: (len(n), n))
        title = sections[nums[0]]["chapter_title"]
        with st.expander(f"Chapter {code} — {title}  ({len(nums)} sections)"):
            for n in nums:
                c1, c2 = st.columns([1, 5])
                c1.button(f"§{n}", key=f"br_{n}", on_click=_goto, args=(n,))
                c2.markdown(f"{sections[n]['title']}")


def page_chat(sections: dict, index: dict) -> None:
    hero("💬 Chat", "Multi-turn conversation — ask 'explain in simpler words' or 'give an example'.",
         ["streaming", "context-aware"])
    if not groq_available():
        st.warning("🔑 Add a Groq API key in the sidebar to use chat mode.")
        return

    st.session_state.setdefault("chat", [])
    context_sec = st.selectbox("Optional context section",
                               ["— none —"] + list(sections.keys()))
    if context_sec != "— none —":
        context = f"Section {context_sec}: {sections[context_sec]['title']}\n{sections[context_sec]['content'][:1200]}"
    else:
        context = ""

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    with st.form("chat_form", clear_on_submit=True):
        q = st.text_input("Your question")
        submitted = st.form_submit_button("Send")

    if submitted and q.strip():
        st.session_state["chat"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(chat_stream(st.session_state["chat"], context))
            except Exception as e:
                answer = f"⚠️ LLM error: {e}"
                st.error(answer)
        st.session_state["chat"].append({"role": "assistant", "content": answer})

    if st.button("🧹 Clear conversation"):
        st.session_state["chat"] = []
        st.rerun()


def page_analytics(sections: dict) -> None:
    hero("📊 Analytics & Feedback", "Usage insights, feedback loop and A/B results.",
         ["SQLite-backed", "RLHF-lite"])
    data = analytics()
    counts = {r["kind"]: r["n"] for r in data.get("counts", [])}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lookups", counts.get("click", 0))
    m2.metric("Searches", counts.get("search", 0))
    m3.metric("Sections indexed", len(sections))
    m4.metric("Feedback votes", sum(r["n"] for r in data.get("feedback", [])))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Most-requested sections**")
        rows = data.get("top_sections", [])
        if rows and PX_OK:
            fig = px.bar(x=[r["section"] for r in rows], y=[r["n"] for r in rows],
                         labels={"x": "Section", "y": "Lookups"})
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        elif rows:
            st.table([{"Section": r["section"], "Lookups": r["n"]} for r in rows])
        else:
            st.caption("No lookups recorded yet.")
    with c2:
        st.markdown("**Query heatmap (day × hour)**")
        heat = data.get("heat", [])
        if heat and PX_OK:
            days = sorted({r["d"] for r in heat})
            hours = [f"{h:02d}" for h in range(24)]
            z = [[0] * 24 for _ in days]
            for r in heat:
                try:
                    z[days.index(r["d"])][int(r["h"])] = r["n"]
                except Exception:
                    pass
            fig = px.imshow(z, x=hours, y=days, aspect="auto", color_continuous_scale=["#1a2340", "#7c8cff", "#e4b95c"])
            style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Not enough activity for a heatmap yet.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Top queries**")
        tq = data.get("top_queries", [])
        if tq:
            st.table([{"Query": r["query"][:60], "n": r["n"]} for r in tq])
        else:
            st.caption("—")
    with c4:
        st.markdown("**A/B test — engine vs avg top score**")
        ab = data.get("abtest", [])
        if ab:
            rows_ab = [{"Engine": r["engine"], "Runs": r["n"], "Avg top score": f"{r['s']:.3f}"}
                       for r in ab]
            st.table(rows_ab)
        else:
            st.caption("No A/B runs yet.")

    st.markdown("**Feedback (👍 minus 👎)**")
    fb = data.get("feedback", [])
    if fb:
        st.table([{"Section": r["section"], "Net score": r["s"], "Votes": r["n"]} for r in fb])
    else:
        st.caption("No feedback yet — vote on results to train the re-ranker.")

    st.markdown("---")
    st.caption(LEGAL_DISCLAIMER.replace("**", ""))


# ═════════════════════════════════════════════════════════════════════════════
# 10.  SIDEBAR + MAIN
# ═════════════════════════════════════════════════════════════════════════════
def render_law_chatbot_sidebar(sections: dict, index: dict) -> None:
    """Compact Law Chatbot pinned in the left sidebar, powered by Google's Gemini API."""
    side_h("⚖️ Law chatbot · Google API")
    st.text_input("Google (Gemini) API key", key="gemini_key", type="password",
                  help="Get a free key at aistudio.google.com/app/apikey. "
                       "Stored only in this session, or set GEMINI_API_KEY env var.")
    st.selectbox("Gemini model", GEMINI_MODELS, key="gemini_model")

    if gemini_available():
        st.success("Gemini connected")
    else:
        st.info("Add a key above to chat.")

    msgs = st.session_state.setdefault("sb_chat", [])
    with st.container(height=260):
        if not msgs:
            st.caption("Ask me e.g. *“What is Section 302?”* or *“Punishment for theft?”*")
        for m in msgs[-12:]:
            css = "user" if m["role"] == "user" else "bot"
            tag = "🧑 You" if m["role"] == "user" else "⚖️ Bot"
            st.markdown(
                f"<div class='chat-bubble {css}'><b>{tag}:</b> {m['content']}</div>",
                unsafe_allow_html=True,
            )

    with st.form("sb_chat_form", clear_on_submit=True):
        q = st.text_input("Ask about IPC…", key="sb_chat_input", label_visibility="collapsed",
                          placeholder="Ask a law question…")
        sent = st.form_submit_button("Send ➤", use_container_width=True)

    if sent and q.strip():
        msgs.append({"role": "user", "content": q.strip()})
        if gemini_available():
            try:
                ans = law_chatbot_reply(q.strip(), sections, index)
            except GroqUnavailable as e:
                ans = f"⚠️ {e}"
            except Exception as e:
                ans = f"⚠️ Chatbot error: {e}"
        else:
            ans = "⚠️ Add a Google (Gemini) API key above to enable the chatbot."
        msgs.append({"role": "assistant", "content": ans})
        st.session_state["sb_chat"] = msgs[-40:]
        st.rerun()

    if msgs and st.button("🧹 Clear chat", key="sb_chat_clear"):
        st.session_state["sb_chat"] = []
        st.rerun()


def sidebar(sections: dict, index: dict) -> str:
    with st.sidebar:
        st.markdown(
            "<div class='side-brand'><div class='logo'>⚖️</div><div>"
            "<div class='nm'>IPC Navigator</div><div class='sb'>AI Legal Assistant</div>"
            "</div></div>", unsafe_allow_html=True)

        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["🔎 Code Lookup", "🧩 Case Study", "⚖️ Compare", "🌳 Browse", "💬 Chat", "📊 Analytics"],
            key="nav", label_visibility="collapsed",
        )

        st.markdown("---")
        st.toggle("🌙 Dark mode", value=True, key="dark_mode")

        st.markdown("---")
        render_law_chatbot_sidebar(sections, index)

        st.markdown("---")
        side_h("System status")
        st.markdown(f"**Sections parsed:** {len(sections)}")
        st.markdown(f"**Engines:** {' + '.join(index['engines'])}")
        if index["embs"] is not None:
            st.success("Dense: Sentence-BERT (MiniLM, offline)")
        else:
            st.info("Dense: off → BM25 + TF-IDF")

        st.markdown("---")
        side_h("Recent")
        hist = st.session_state.get("history", [])
        if hist:
            for h in hist[:8]:
                st.button(f"§{h}", key=f"hist_{h}", on_click=_goto, args=(h,))
        else:
            st.caption("No lookups yet.")

        bms = st.session_state.get("bookmarks", [])
        side_h("Bookmarks")
        if bms:
            for h in bms:
                st.button(f"§{h}", key=f"bmk_{h}", on_click=_goto, args=(h,))
        else:
            st.caption("Nothing saved yet.")

        st.markdown("---")
        if st.button("🔄 Check source updates"):
            st.info(check_source_update())

    return page


def main() -> None:
    st.session_state.setdefault("use_dense", True)
    inject_css()

    sections = load_sections()
    db_save_sections(sections)

    texts = tuple(v["full"] for v in sections.values())
    keys = tuple(sections.keys())
    sig = hashlib.sha1(("|".join(keys) + EMBED_MODEL).encode()).hexdigest()[:16]
    index = build_index(sig, keys, texts, st.session_state.get("use_dense", True))

    render_topbar()
    page = sidebar(sections, index)

    if page == "🔎 Code Lookup":
        page_code(sections)
    elif page == "🧩 Case Study":
        page_case(sections, index)
    elif page == "⚖️ Compare":
        page_compare(sections)
    elif page == "🌳 Browse":
        page_browse(sections)
    elif page == "💬 Chat":
        page_chat(sections, index)
    elif page == "📊 Analytics":
        page_analytics(sections)

    render_news_strip()
    render_footer()


if __name__ == "__main__":
    main()
