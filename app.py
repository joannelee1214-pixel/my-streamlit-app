import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import streamlit as st
import plotly.graph_objects as go
import requests

# ======================================================
# System Prompt (UI 미노출)
# ======================================================
SYSTEM_PROMPT = """
당신은 음악, 도서, 영화, 미술 전반에 깊은 식견을 가진 큐레이터이자 평론가입니다.
사용자의 취향 또는 기준 작품을 바탕으로 실제 존재하는 작품을 추천해야 합니다.

각 작품에 대해 반드시 다음 두 가지를 모두 포함해 설명하세요.
1. 추천 이유
2. 감상 포인트 (어떤 관점으로 보면 좋은지, 무엇에 주목하면 좋은지)

중요 제약:
- 음악은 반드시 '앨범(정규/EP/컴필레이션 포함)' 단위로만 추천하세요. (곡/트랙 금지)
- 미술은 가능한 한 '작품의 영문 제목(English title)'도 함께 떠올려 추천하고,
  작품 제목이 현지어/번역명인 경우 괄호로 영문 제목을 덧붙이세요. 예: 절규 (The Scream)
- 존재하지 않는 작품을 만들어내면 안 됩니다.
- JSON 형식만 출력하세요. 추가 텍스트 금지.
"""

# ======================================================
# Constants
# ======================================================
DIMENSIONS = ["복잡성", "직관성", "대중성", "감정 톤", "개방성", "각성도"]

DIM_LABELS = {
    "복잡성": ("simple", "complex"),
    "직관성": ("analytical", "intuitive"),
    "대중성": ("niche", "mainstream"),
    "감정 톤": ("dark", "bright"),
    "개방성": ("conventional", "exploratory"),
    "각성도": ("calm", "intense"),
}

CATEGORIES = ["도서", "음악", "영화", "미술"]

CATEGORY_EMOJI = {
    "도서": "📚",
    "음악": "🎵",
    "영화": "🎬",
    "미술": "🖼️",
}

# ======================================================
# Data Model
# ======================================================
@dataclass
class Item:
    category: str
    title: str
    creator: str
    reason: str
    image: Optional[str] = None

# ======================================================
# Utils
# ======================================================
def radar_chart(values: List[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=DIMENSIONS + [DIMENSIONS[0]],
        fill="toself",
        line=dict(width=4),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 10])),
        showlegend=False,
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def placeholder_image(text: str) -> str:
    safe = "".join(c for c in text if c.isalnum() or c in " _-")[:20]
    return f"https://placehold.co/600x800?text={safe.replace(' ', '+')}"


def clean_title_for_search(title: str) -> str:
    """
    API 검색 성공률을 높이기 위해 괄호 안 보조정보를 제거하거나 줄이는 정도의 정리만 수행.
    (기존 기능 변경 없이 '검색만' 개선)
    """
    t = title.strip()
    # "절규 (The Scream)" -> "절규"와 "The Scream" 둘 다 시도할 거라 원문은 유지하고,
    # 괄호 내용만 따로 뽑을 수 있게 반환은 원문 그대로 사용.
    return t


def extract_parenthetical_english(title: str) -> Optional[str]:
    # "절규 (The Scream)" -> "The Scream"
    t = title.strip()
    if "(" in t and ")" in t:
        inside = t.split("(", 1)[1].split(")", 1)[0].strip()
        if inside:
            return inside
    return None

# ======================================================
# External APIs (이미지용)
# ======================================================
def fetch_tmdb(title: str, key: str) -> Optional[str]:
    if not key:
        return None
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": key, "query": title, "language": "ko-KR"},
            timeout=8,
        ).json()
        if r.get("results"):
            p = r["results"][0].get("poster_path")
            if p:
                return f"https://image.tmdb.org/t/p/w500{p}"
    except Exception:
        return None
    return None


def fetch_kakao_book(title: str, key: str) -> Optional[str]:
    if not key:
        return None
    try:
        r = requests.get(
            "https://dapi.kakao.com/v3/search/book",
            headers={"Authorization": f"KakaoAK {key}"},
            params={"query": title},
            timeout=8,
        ).json()
        if r.get("documents"):
            return r["documents"][0].get("thumbnail")
    except Exception:
        return None
    return None


# ---------- Last.fm 개선: album.getinfo 실패 시 album.search로 폴백 ----------
def _lastfm_album_getinfo(artist: str, album: str, key: str) -> Optional[str]:
    r = requests.get(
        "http://ws.audioscrobbler.com/2.0/",
        params={
            "method": "album.getinfo",
            "api_key": key,
            "artist": artist,
            "album": album,
            "format": "json",
        },
        timeout=8,
    ).json()
    try:
        url = r["album"]["image"][-1]["#text"]
        return url or None
    except Exception:
        return None


def _lastfm_album_search(album: str, key: str, limit: int = 5) -> List[Tuple[str, str]]:
    """
    album.search로 후보(artist, album)를 몇 개 가져옴.
    """
    r = requests.get(
        "http://ws.audioscrobbler.com/2.0/",
        params={
            "method": "album.search",
            "api_key": key,
            "album": album,
            "limit": limit,
            "format": "json",
        },
        timeout=8,
    ).json()
    out: List[Tuple[str, str]] = []
    try:
        matches = r["results"]["albummatches"]["album"]
        if isinstance(matches, dict):
            matches = [matches]
        for m in matches:
            a = (m.get("artist") or "").strip()
            t = (m.get("name") or "").strip()
            if a and t:
                out.append((a, t))
    except Exception:
        pass
    return out


def fetch_lastfm(artist: str, album: str, key: str) -> Optional[str]:
    """
    기존 기능 유지 + 성공률만 올림:
    1) album.getinfo(원래 방식)
    2) 실패하면 album.search로 가장 그럴듯한 후보를 찾고 getinfo 재시도
    """
    if not key:
        return None

    a = (artist or "").strip()
    t = (album or "").strip()
    if not t:
        return None

    try:
        # 1) 원래 방식
        img = _lastfm_album_getinfo(a, t, key)
        if img:
            return img

        # 2) 폴백: album.search로 후보를 찾아 getinfo
        candidates = _lastfm_album_search(t, key, limit=6)
        for cand_artist, cand_album in candidates:
            img2 = _lastfm_album_getinfo(cand_artist, cand_album, key)
            if img2:
                return img2

        return None
    except Exception:
        return None


# ---------- The Met 개선: 결과 여러 개 순회 + (영문 괄호/작가명) 보조 검색 ----------
def _met_search_object_ids(query: str, limit: int = 25) -> List[int]:
    search = requests.get(
        "https://collectionapi.metmuseum.org/public/collection/v1/search",
        params={"q": query, "hasImages": "true"},
        timeout=10,
    ).json()
    ids = search.get("objectIDs") or []
    # 너무 많으면 앞쪽만
    return ids[:limit]


def _met_get_image_for_object(obj_id: int) -> Optional[str]:
    obj = requests.get(
        f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}",
        timeout=10,
    ).json()
    # primaryImageSmall 우선, 없으면 primaryImage
    return obj.get("primaryImageSmall") or obj.get("primaryImage") or None


def fetch_met_artwork(title: str, artist: Optional[str] = None) -> Optional[str]:
    """
    기존: 첫 번째 결과만 사용 → 실패 잦음
    개선: 여러 ID를 순회하면서 이미지 있는 걸 찾음.
    또한, 제목에 (English) 괄호가 있으면 그 영문으로도 검색.
    가능하면 'artist + title' 결합 검색도 시도.
    """
    try:
        raw_title = clean_title_for_search(title)
        english_in_paren = extract_parenthetical_english(raw_title)

        queries = []
        if raw_title:
            queries.append(raw_title)
        if english_in_paren and english_in_paren != raw_title:
            queries.append(english_in_paren)

        if artist:
            a = artist.strip()
            if a and raw_title:
                queries.insert(0, f"{a} {raw_title}")
            if a and english_in_paren:
                queries.insert(0, f"{a} {english_in_paren}")

        # 중복 제거
        seen = set()
        queries = [q for q in queries if q and not (q in seen or seen.add(q))]

        for q in queries:
            ids = _met_search_object_ids(q, limit=30)
            for obj_id in ids:
                img = _met_get_image_for_object(obj_id)
                if img:
                    return img

        return None
    except Exception:
        return None

# ======================================================
# OpenAI Recommendation (핵심)
# ======================================================
def recommend_with_llm(prompt: str, openai_key: str) -> Dict[str, Dict]:
    if not openai_key:
        raise RuntimeError("OpenAI API Key가 필요합니다.")

    from openai import OpenAI
    client = OpenAI(api_key=openai_key)

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return json.loads(res.choices[0].message.content)

# ======================================================
# Streamlit App
# ======================================================
st.set_page_config(page_title="My Curator", page_icon="✨", layout="wide")

# --- UI: mode 선택을 더 눈에 띄게(기능은 동일) ---
st.markdown(
    """
    <style>
    /* 라디오를 버튼처럼 보이게 */
    div[data-testid="stRadio"] > div {
        background: rgba(127,127,127,0.08);
        padding: 0.6rem 0.8rem;
        border-radius: 16px;
        border: 1px solid rgba(127,127,127,0.18);
    }
    div[data-testid="stRadio"] label {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("✨ My Curator")

# ---------------- Sidebar ----------------
st.sidebar.header("🔑 API Keys")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
tmdb_key = st.sidebar.text_input("TMDb API Key", type="password")
lastfm_key = st.sidebar.text_input("Last.fm API Key", type="password")
kakao_key = st.sidebar.text_input("Kakao Book API Key", type="password")

mode_choice = st.radio(
    "검색 방식 선택",
    ["🎛️ 취향 검색", "🔗 연관 검색"],
    horizontal=True
)
mode = "취향 검색" if "취향" in mode_choice else "연관 검색"

# ======================================================
# 취향 검색
# ======================================================
if mode == "취향 검색":
    values: List[float] = []

    for dim in DIMENSIONS:
        left, right = DIM_LABELS[dim]
        st.markdown(f"**{dim}**")

        cols = st.columns([1, 6, 1])
        with cols[0]:
            st.markdown(
                f"<div style='font-size:0.85em; opacity:0.8'>{left}</div>",
                unsafe_allow_html=True
            )
        with cols[1]:
            v = st.slider(dim, 0.0, 10.0, 6.0, 0.5, label_visibility="collapsed")
            values.append(v)
        with cols[2]:
            st.markdown(
                f"<div style='font-size:0.85em; opacity:0.8; text-align:right'>{right}</div>",
                unsafe_allow_html=True
            )

    st.plotly_chart(radar_chart(values), use_container_width=True)

    if st.button("✨ curate", type="primary"):
        taste_desc = "\n".join(f"- {DIMENSIONS[i]}: {values[i]}" for i in range(6))

        # ✅ 음악은 '앨범'만 추천하도록 명시 강화(기능 추가/삭제 없이 프롬프트만 강화)
        prompt = f"""
다음은 사용자의 취향입니다:
{taste_desc}

이 취향에 가장 잘 맞는 작품을 아래 형식의 JSON으로 추천하세요.

엄격 규칙:
- 음악은 반드시 앨범 단위로만 추천 (곡/트랙 금지). title에는 '앨범명'만.
- 미술 title에는 가능하면 영문 제목을 괄호로 병기. 예: 절규 (The Scream)
- reason에는 '추천 이유'와 '감상 포인트'를 모두 포함.

형식(키 이름/구조 그대로):
{{
  "도서": {{"title": "", "creator": "", "reason": ""}},
  "음악": {{"title": "", "creator": "", "reason": ""}},
  "영화": {{"title": "", "creator": "", "reason": ""}},
  "미술": {{"title": "", "creator": "", "reason": ""}}
}}
"""

        recs = recommend_with_llm(prompt, openai_key)

        items: List[Item] = []
        for cat in CATEGORIES:
            r = recs[cat]
            item = Item(cat, r["title"], r["creator"], r["reason"])

            if cat == "도서":
                item.image = fetch_kakao_book(item.title, kakao_key)
            elif cat == "음악":
                # ✅ Last.fm 폴백 강화된 fetch_lastfm 사용
                item.image = fetch_lastfm(item.creator, item.title, lastfm_key)
            elif cat == "영화":
                item.image = fetch_tmdb(item.title, tmdb_key)
            else:
                # ✅ Met 검색 성공률 강화(작가명도 함께 전달)
                item.image = fetch_met_artwork(item.title, artist=item.creator)

            item.image = item.image or placeholder_image(item.title)
            items.append(item)

        st.divider()
        cols = st.columns(4)
        for i, item in enumerate(items):
            with cols[i]:
                st.markdown(f"### {CATEGORY_EMOJI[item.category]} {item.category}")
                st.image(item.image, use_container_width=True)
                st.markdown(f"**{item.title}**")
                st.caption(item.creator)
                st.markdown(item.reason)

# ======================================================
# 연관 검색
# ======================================================
if mode == "연관 검색":
    base_cat = st.selectbox("기준 카테고리", CATEGORIES)
    base_creator = st.text_input("창작자")
    base_title = st.text_input("작품 제목")

    if st.button("✨ curate", type="primary"):
        prompt = f"""
다음 작품과 함께 감상하면 좋은 작품을 추천하세요.

기준 작품:
- 카테고리: {base_cat}
- 제목: {base_title}
- 창작자: {base_creator}

아래 형식의 JSON으로 추천하세요.
(기준 작품과 같은 카테고리는 제외)

엄격 규칙:
- 음악은 반드시 앨범 단위로만 추천 (곡/트랙 금지). title에는 '앨범명'만.
- 미술 title에는 가능하면 영문 제목을 괄호로 병기. 예: 절규 (The Scream)
- reason에는 '추천 이유'와 '감상 포인트'를 모두 포함.

형식(키 이름/구조 그대로):
{{
  "도서": {{"title": "", "creator": "", "reason": ""}},
  "음악": {{"title": "", "creator": "", "reason": ""}},
  "영화": {{"title": "", "creator": "", "reason": ""}},
  "미술": {{"title": "", "creator": "", "reason": ""}}
}}
"""

        recs = recommend_with_llm(prompt, openai_key)

        items: List[Item] = []
        for cat in CATEGORIES:
            if cat == base_cat:
                continue

            r = recs[cat]
            item = Item(cat, r["title"], r["creator"], r["reason"])

            if cat == "도서":
                item.image = fetch_kakao_book(item.title, kakao_key)
            elif cat == "음악":
                item.image = fetch_lastfm(item.creator, item.title, lastfm_key)
            elif cat == "영화":
                item.image = fetch_tmdb(item.title, tmdb_key)
            else:
                item.image = fetch_met_artwork(item.title, artist=item.creator)

            item.image = item.image or placeholder_image(item.title)
            items.append(item)

        st.divider()
        cols = st.columns(3)
        for i, item in enumerate(items):
            with cols[i]:
                st.markdown(f"### {CATEGORY_EMOJI[item.category]} {item.category}")
                st.image(item.image, use_container_width=True)
                st.markdown(f"**{item.title}**")
                st.caption(item.creator)
                st.markdown(item.reason)
