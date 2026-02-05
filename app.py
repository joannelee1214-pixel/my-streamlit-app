import json
from dataclasses import dataclass
from typing import List, Dict, Optional

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

존재하지 않는 작품을 만들어내면 안 됩니다.
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
    )
    return fig


def placeholder_image(text: str) -> str:
    safe = "".join(c for c in text if c.isalnum() or c in " _-")[:20]
    return f"https://placehold.co/600x800?text={safe.replace(' ', '+')}"

# ======================================================
# External APIs (이미지용)
# ======================================================
def fetch_tmdb(title: str, key: str) -> Optional[str]:
    if not key:
        return None
    r = requests.get(
        "https://api.themoviedb.org/3/search/movie",
        params={"api_key": key, "query": title, "language": "ko-KR"},
        timeout=5,
    ).json()
    if r.get("results"):
        p = r["results"][0].get("poster_path")
        if p:
            return f"https://image.tmdb.org/t/p/w500{p}"
    return None


def fetch_lastfm(artist: str, album: str, key: str) -> Optional[str]:
    if not key:
        return None
    r = requests.get(
        "http://ws.audioscrobbler.com/2.0/",
        params={
            "method": "album.getinfo",
            "api_key": key,
            "artist": artist,
            "album": album,
            "format": "json",
        },
        timeout=5,
    ).json()
    try:
        return r["album"]["image"][-1]["#text"]
    except Exception:
        return None


def fetch_kakao_book(title: str, key: str) -> Optional[str]:
    if not key:
        return None
    r = requests.get(
        "https://dapi.kakao.com/v3/search/book",
        headers={"Authorization": f"KakaoAK {key}"},
        params={"query": title},
        timeout=5,
    ).json()
    if r.get("documents"):
        return r["documents"][0].get("thumbnail")
    return None


def fetch_met_artwork(title: str) -> Optional[str]:
    search = requests.get(
        "https://collectionapi.metmuseum.org/public/collection/v1/search",
        params={"q": title},
        timeout=5,
    ).json()
    if not search.get("objectIDs"):
        return None
    obj_id = search["objectIDs"][0]
    obj = requests.get(
        f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}",
        timeout=5,
    ).json()
    return obj.get("primaryImageSmall")

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
st.title("✨ My Curator")

# ---------------- Sidebar ----------------
st.sidebar.header("🔑 API Keys")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
tmdb_key = st.sidebar.text_input("TMDb API Key", type="password")
lastfm_key = st.sidebar.text_input("Last.fm API Key", type="password")
kakao_key = st.sidebar.text_input("Kakao Book API Key", type="password")

mode = st.radio("검색 방식 선택", ["취향 검색", "연관 검색"], horizontal=True)

# ======================================================
# 취향 검색
# ======================================================
if mode == "취향 검색":
    values = []

    for dim in DIMENSIONS:
        left, right = DIM_LABELS[dim]
        st.markdown(f"**{dim}**")

        cols = st.columns([1, 6, 1])
        with cols[0]:
            st.markdown(f"<div style='font-size:0.85em; opacity:0.8'>{left}</div>", unsafe_allow_html=True)
        with cols[1]:
            v = st.slider(dim, 0.0, 10.0, 6.0, 0.5, label_visibility="collapsed")
            values.append(v)
        with cols[2]:
            st.markdown(f"<div style='font-size:0.85em; opacity:0.8; text-align:right'>{right}</div>", unsafe_allow_html=True)

    st.plotly_chart(radar_chart(values), use_container_width=True)

    if st.button("✨ curate", type="primary"):
        taste_desc = "\n".join(f"- {DIMENSIONS[i]}: {values[i]}" for i in range(6))

        prompt = f"""
다음은 사용자의 취향입니다:
{taste_desc}

이 취향에 가장 잘 맞는 작품을 아래 형식의 JSON으로 추천하세요.

형식:
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
                item.image = fetch_lastfm(item.creator, item.title, lastfm_key)
            elif cat == "영화":
                item.image = fetch_tmdb(item.title, tmdb_key)
            else:
                item.image = fetch_met_artwork(item.title)

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

형식:
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
                item.image = fetch_met_artwork(item.title)

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
