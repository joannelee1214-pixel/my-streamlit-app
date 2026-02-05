# app.py
# My Curator – Full Version (Python 3.13 compatible)

import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from difflib import get_close_matches

# ======================================================
# System Prompt
# ======================================================
SYSTEM_PROMPT = (
    "당신은 음악, 도서, 미술, 영화를 포함해 문화 전반에 식견이 넓고 깊은 "
    "큐레이터이자 평론가입니다. 문화적인 식견을 바탕으로 사용자가 만족할만한 "
    "작품을 추천하고 어떤 관점으로 감상하면 좋을지 자세히 설명해주세요."
)

# ======================================================
# Taste Dimensions
# ======================================================
DIMENSIONS = ["복잡성", "직관성", "대중성", "감정 톤", "개방성", "각성도"]
CATEGORIES = ["도서", "음악", "영화", "미술"]

# ======================================================
# Data Model
# ======================================================
@dataclass
class Item:
    category: str
    title: str
    creator: str
    year: str
    vector: np.ndarray
    tagline: str


# ======================================================
# Utility Functions
# ======================================================
def clamp(value: float) -> float:
    return max(0.0, min(10.0, float(value)))


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


def placeholder_image(text: str) -> str:
    safe = "".join(c for c in text if c.isalnum() or c in " _-")[:22]
    return f"https://placehold.co/600x800?text={safe.replace(' ', '+')}"


def stable_vector(seed: str) -> np.ndarray:
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    values = []
    for i in range(6):
        chunk = h[i * 8:(i + 1) * 8]
        values.append((int(chunk, 16) % 1000) / 100)
    return np.array(values, dtype=float)


# ======================================================
# Catalog (Example Data)
# ======================================================
CATALOG: List[Item] = [
    # Books
    Item("도서", "데미안", "헤르만 헤세", "1919",
         np.array([6, 8, 8, 7, 7, 4]), "자기 탐색의 서사"),
    Item("도서", "백년 동안의 고독", "가브리엘 가르시아 마르케스", "1967",
         np.array([9, 4, 6, 8, 8, 5]), "시간과 신화의 소용돌이"),

    # Music
    Item("음악", "OK Computer", "Radiohead", "1997",
         np.array([8, 6, 7, 7, 8, 7]), "기술 시대의 불안"),
    Item("음악", "Kind of Blue", "Miles Davis", "1959",
         np.array([6, 9, 9, 7, 7, 4]), "여백과 즉흥"),

    # Movies
    Item("영화", "이터널 선샤인", "미셸 공드리", "2004",
         np.array([6, 8, 8, 9, 7, 6]), "기억과 사랑"),
    Item("영화", "기생충", "봉준호", "2019",
         np.array([7, 9, 9, 7, 7, 8]), "장르의 전복"),

    # Art
    Item("미술", "별이 빛나는 밤", "Vincent van Gogh", "1889",
         np.array([6, 9, 9, 9, 7, 7]), "감정의 소용돌이"),
    Item("미술", "게르니카", "Pablo Picasso", "1937",
         np.array([8, 6, 8, 8, 8, 8]), "폭력의 파편"),
]

# ======================================================
# Recommendation Logic
# ======================================================
def recommend_by_vector(target: np.ndarray, exclude: Optional[Item] = None) -> Dict[str, Item]:
    results: Dict[str, Item] = {}
    for category in CATEGORIES:
        items = [i for i in CATALOG if i.category == category]
        if exclude:
            items = [
                i for i in items
                if not (i.category == exclude.category and i.title == exclude.title)
            ]
        best = max(items, key=lambda i: cosine_similarity(target, i.vector))
        results[category] = best
    return results


def find_anchor(category: str, creator: str, title: str) -> Optional[Item]:
    candidates = [i for i in CATALOG if i.category == category]
    query = f"{creator} {title}".strip().lower()

    for item in candidates:
        if query and query in f"{item.creator} {item.title}".lower():
            return item

    matches = get_close_matches(
        title, [i.title for i in candidates], n=1, cutoff=0.6
    )
    if matches:
        for item in candidates:
            if item.title == matches[0]:
                return item
    return None


def curator_reason(item: Item, user_vec: np.ndarray, anchor: Optional[Item] = None) -> str:
    diffs = np.abs(user_vec - item.vector)
    best_axis = DIMENSIONS[int(np.argmin(diffs))]

    text = [
        f"**{item.tagline}**",
        f"이 작품은 특히 **{best_axis}** 축에서 당신의 성향과 잘 맞습니다."
    ]

    if anchor:
        sim = cosine_similarity(anchor.vector, item.vector)
        text.append(
            f"입력한 작품 **{anchor.title}**와도 정서적 결이 이어지며 "
            f"(유사도 {sim:.2f}), 함께 감상하면 맥락이 확장됩니다."
        )

    text.append(
        "감상 시에는 작품의 분위기뿐 아니라 구조와 리듬이 "
        "어떤 감정을 유도하는지에 주목해 보세요."
    )

    return "\n\n".join(text)


# ======================================================
# Radar Chart
# ======================================================
def radar_chart(values: List[float], scale: float = 1.0) -> go.Figure:
    values = [clamp(v * scale) for v in values]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=DIMENSIONS + [DIMENSIONS[0]],
            fill="toself",
            line=dict(width=4),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 10])),
        showlegend=False,
        height=420,
    )
    return fig


def animate_radar(values: List[float]):
    slot = st.empty()
    for scale in [1.0, 1.05, 1.1, 1.15]:
        slot.plotly_chart(radar_chart(values, scale), use_container_width=True)
        time.sleep(0.08)


# ======================================================
# Streamlit App
# ======================================================
st.set_page_config(page_title="My Curator", page_icon="✨", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "main"
if "mode" not in st.session_state:
    st.session_state.mode = None
if "taste" not in st.session_state:
    st.session_state.taste = [6.0] * 6

st.title("✨ My Curator")
st.caption("취향의 별을 조율하거나, 한 작품에서 다른 세계로 확장하세요.")

# ---------------- MAIN ----------------
if st.session_state.page == "main":
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("검색 방식 선택")

        if st.button("🎛️ 취향 검색", use_container_width=True):
            st.session_state.mode = "taste"

        if st.button("🔗 연관 검색", use_container_width=True):
            st.session_state.mode = "related"

        st.divider()

        # Taste Search
        if st.session_state.mode == "taste":
            st.subheader("취향 설정")

            values: List[float] = []
            for i, dim in enumerate(DIMENSIONS):
                values.append(
                    st.slider(dim, 0.0, 10.0, st.session_state.taste[i], 0.5)
                )

            st.session_state.taste = values
            st.plotly_chart(radar_chart(values), use_container_width=True)

            if st.button("✨ curate", type="primary"):
                animate_radar(values)
                user_vec = np.array(values)
                st.session_state.results = recommend_by_vector(user_vec)
                st.session_state.reasons = {
                    k: curator_reason(v, user_vec) for k, v in st.session_state.results.items()
                }
                st.session_state.page = "results"
                st.rerun()

        # Related Search
        if st.session_state.mode == "related":
            category = st.selectbox("카테고리", CATEGORIES)
            creator = st.text_input("창작자")
            title = st.text_input("작품 제목")

            if st.button("✨ curate", type="primary"):
                anchor = find_anchor(category, creator, title)
                if anchor:
                    vec = anchor.vector
                else:
                    vec = stable_vector(f"{category}-{creator}-{title}")
                    anchor = Item(category, title, creator, "—", vec, "입력 기반 연관점")

                st.session_state.results = recommend_by_vector(vec, exclude=anchor)
                st.session_state.reasons = {
                    k: curator_reason(v, vec, anchor) for k, v in st.session_state.results.items()
                }
                st.session_state.anchor = anchor
                st.session_state.page = "results"
                st.rerun()

    with col2:
        st.subheader("System Prompt")
        st.text_area("큐레이터 성격", SYSTEM_PROMPT, height=200)

# ---------------- RESULTS ----------------
if st.session_state.page == "results":
    st.subheader("추천 결과")

    cols = st.columns(4)
    for i, cat in enumerate(CATEGORIES):
        item = st.session_state.results[cat]
        with cols[i]:
            st.image(placeholder_image(item.title), use_container_width=True)
            st.markdown(f"**{item.title}**")
            st.caption(f"{item.creator} · {item.year}")

    st.divider()
    st.subheader("큐레이터의 설명")

    for cat in CATEGORIES:
        item = st.session_state.results[cat]
        with st.expander(f"[{cat}] {item.title}"):
            st.markdown(st.session_state.reasons[cat])

    if st.button("🔄 초기화"):
        st.session_state.clear()
        st.rerun()
