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


def safe_text(s: str) -> str:
    return (s or "").strip()


# ======================================================
# Naver Search API (도서/영화/미술 이미지)
# ======================================================
def naver_headers(client_id: str, client_secret: str) -> Dict[str, str]:
    return {
        "X-Naver-Client-Id": client_id.strip(),
        "X-Naver-Client-Secret": client_secret.strip(),
    }


def fetch_naver_image(
    query: str,
    client_id: str,
    client_secret: str,
    display: int = 5,
) -> Optional[str]:
    """
    네이버 이미지 검색 API로 첫 번째 결과 이미지 링크를 가져옴.
    """
    if not client_id or not client_secret:
        return None

    try:
        r = requests.get(
            "https://openapi.naver.com/v1/search/image",
            headers=naver_headers(client_id, client_secret),
            params={
                "query": query,
                "display": display,
                "sort": "sim",  # 유사도순
                "filter": "all",
            },
            timeout=10,
        ).json()

        items = r.get("items") or []
        if not items:
            return None

        # 첫 번째 링크 우선
        link = items[0].get("link")
        return link or None

    except Exception:
        return None


def fetch_naver_book_image(
    title: str,
    author: str,
    client_id: str,
    client_secret: str
) -> Optional[str]:
    """
    책은 이미지 검색으로도 되지만,
    책 검색 API가 더 정확하긴 해서 책 API 먼저 시도 후 없으면 이미지 검색.
    """
    if not client_id or not client_secret:
        return None

    q = f"{title} {author}".strip()

    # 1) book search
    try:
        r = requests.get(
            "https://openapi.naver.com/v1/search/book.json",
            headers=naver_headers(client_id, client_secret),
            params={"query": q, "display": 5, "sort": "sim"},
            timeout=10,
        ).json()

        items = r.get("items") or []
        if items:
            img = items[0].get("image")
            if img:
                return img
    except Exception:
        pass

    # 2) fallback: image search
    return fetch_naver_image(q, client_id, client_secret)


def fetch_naver_movie_image(
    title: str,
    director: str,
    client_id: str,
    client_secret: str
) -> Optional[str]:
    """
    영화는 movie 검색 API를 먼저 시도하고,
    실패하면 이미지 검색으로 폴백.
    """
    if not client_id or not client_secret:
        return None

    q = f"{title} {director}".strip()

    # 1) movie search
    try:
        r = requests.get(
            "https://openapi.naver.com/v1/search/movie.json",
            headers=naver_headers(client_id, client_secret),
            params={"query": q, "display": 5},
            timeout=10,
        ).json()

        items = r.get("items") or []
        if items:
            img = items[0].get("image")
            if img:
                return img
    except Exception:
        pass

    # 2) fallback: image search
    return fetch_naver_image(q, client_id, client_secret)


def fetch_naver_art_image(
    title: str,
    artist: str,
    client_id: str,
    client_secret: str
) -> Optional[str]:
    """
    미술은 전용 API가 없으니 이미지 검색을 씀.
    작품명+작가명으로 검색하면 성공률이 훨씬 올라감.
    """
    q = f"{title} {artist} artwork".strip()
    return fetch_naver_image(q, client_id, client_secret)


# ======================================================
# Last.fm (음악 앨범 커버) - 폴백 강화 유지
# ======================================================
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
    if not key:
        return None

    a = safe_text(artist)
    t = safe_text(album)
    if not t:
        return None

    try:
        img = _lastfm_album_getinfo(a, t, key)
        if img:
            return img

        candidates = _lastfm_album_search(t, key, limit=6)
        for cand_artist, cand_album in candidates:
            img2 = _lastfm_album_getinfo(cand_artist, cand_album, key)
            if img2:
                return img2

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

# --- UI: mode 선택을 더 눈에 띄게(기능 동일) ---
st.markdown(
    """
    <style>
    div[data-testid="stRadio"] > div {
        background: rgba(127,127,127,0.08);
        padding: 0.7rem 0.9rem;
        border-radius: 16px;
        border: 1px solid rgba(127,127,127,0.18);
    }
    div[data-testid="stRadio"] label {
        font-size: 1.05rem !important;
        font-weight: 800 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("✨ My Curator")

# ---------------- Sidebar ----------------
st.sidebar.header("🔑 API Keys")

openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
lastfm_key = st.sidebar.text_input("Last.fm API Key", type="password")

st.sidebar.divider()
st.sidebar.subheader("네이버 검색 API")
naver_client_id = st.sidebar.text_input("Naver Client ID", type="password")
naver_client_secret = st.sidebar.text_input("Naver Client Secret", type="password")

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

        prompt = f"""
다음은 사용자의 취향입니다:
{taste_desc}

이 취향에 가장 잘 맞는 작품을 아래 형식의 JSON으로 추천하세요.

엄격 규칙:
- 음악은 반드시 앨범 단위로만 추천 (곡/트랙 금지). title에는 '앨범명'만.
- reason에는 '추천 이유'와 '감상 포인트'를 모두 포함.
- 너무 길지 않게: 5~8줄 정도로 간결하게.

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
                item.image = fetch_naver_book_image(
                    item.title, item.creator, naver_client_id, naver_client_secret
                )
            elif cat == "음악":
                item.image = fetch_lastfm(item.creator, item.title, lastfm_key)
            elif cat == "영화":
                item.image = fetch_naver_movie_image(
                    item.title, item.creator, naver_client_id, naver_client_secret
                )
            else:
                item.image = fetch_naver_art_image(
                    item.title, item.creator, naver_client_id, naver_client_secret
                )

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
- reason에는 '추천 이유'와 '감상 포인트'를 모두 포함.
- 너무 길지 않게: 5~8줄 정도로 간결하게.

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
                item.image = fetch_naver_book_image(
                    item.title, item.creator, naver_client_id, naver_client_secret
                )
            elif cat == "음악":
                item.image = fetch_lastfm(item.creator, item.title, lastfm_key)
            elif cat == "영화":
                item.image = fetch_naver_movie_image(
                    item.title, item.creator, naver_client_id, naver_client_secret
                )
            else:
                item.image = fetch_naver_art_image(
                    item.title, item.creator, naver_client_id, naver_client_secret
                )

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
