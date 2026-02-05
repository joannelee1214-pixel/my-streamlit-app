```python
# app.py
# Streamlit 추천 앱: My Curator
# 실행: streamlit run app.py

import time
import math
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from difflib import get_close_matches

# -----------------------------
# 설정 & 시스템 프롬프트(앱 내 사용)
# -----------------------------
SYSTEM_PROMPT = """당신은 음악, 도서, 미술, 영화를 포함해 문화 전반에 식견이 넓고 깊은 큐레이터이자 평론가입니다.
문화적인 식견을 바탕으로 사용자가 만족할만한 작품을 추천하고 어떤 관점으로 감상하면 좋을지 자세히 설명해주세요."""

DIMENSIONS = ["복잡성", "직관성", "대중성", "감정 톤", "개방성", "각성도"]  # 6축
DIM_HELP = {
    "복잡성": "구조/서사/형식이 촘촘하고 다층적인 정도",
    "직관성": "처음 접했을 때 바로 감이 오는 정도",
    "대중성": "많은 사람들이 접근하기 쉬운 정도",
    "감정 톤": "정서의 농도(서정/우울/따뜻함/격정 등)의 강도",
    "개방성": "새로움/실험성/낯섦을 기꺼이 받아들이는 정도",
    "각성도": "긴장/에너지/몰입을 끌어올리는 정도",
}

# -----------------------------
# 데이터(예시 카탈로그)
# - 실제 서비스라면 API/DB로 대체하면 좋습니다.
# - 이미지 URL은 안정적인 placeholder 사용
# -----------------------------
@dataclass
class Item:
    category: str  # "도서" | "음악" | "영화" | "미술"
    title: str
    creator: str
    year: str
    vector: np.ndarray  # shape (6,)
    tagline: str  # 한 줄 느낌

def ph_image(text: str) -> str:
    # placehold.co는 간단히 텍스트 이미지를 만들어줍니다.
    # (한글이 깨질 수 있어, 영어/숫자 위주로 표기)
    safe = "".join([c for c in text if c.isalnum() or c in " -_"])[:24]
    if not safe:
        safe = "Artwork"
    return f"https://placehold.co/600x800?text={safe.replace(' ', '+')}"

CATALOG: List[Item] = [
    # 도서
    Item("도서", "백년 동안의 고독", "가브리엘 가르시아 마르케스", "1967",
         np.array([9, 4, 6, 8, 8, 5], dtype=float), "신화처럼 번지는 가족과 시간"),
    Item("도서", "데미안", "헤르만 헤세", "1919",
         np.array([6, 8, 8, 7, 7, 4], dtype=float), "성장의 상처를 통과하는 자아"),
    Item("도서", "노르웨이의 숲", "무라카미 하루키", "1987",
         np.array([5, 8, 8, 8, 6, 4], dtype=float), "상실과 기억의 습기"),
    Item("도서", "1984", "조지 오웰", "1949",
         np.array([7, 7, 7, 6, 7, 7], dtype=float), "감시와 언어의 미래"),
    Item("도서", "연금술사", "파울로 코엘료", "1988",
         np.array([4, 9, 9, 7, 6, 5], dtype=float), "징후를 따라가는 우화"),

    # 음악(앨범)
    Item("음악", "Kind of Blue", "Miles Davis", "1959",
         np.array([6, 9, 9, 7, 7, 4], dtype=float), "여백이 숨 쉬는 쿨 재즈"),
    Item("음악", "OK Computer", "Radiohead", "1997",
         np.array([8, 6, 7, 7, 8, 7], dtype=float), "기계화된 불안의 서정"),
    Item("음악", "In Rainbows", "Radiohead", "2007",
         np.array([7, 7, 8, 8, 7, 6], dtype=float), "부드러운 긴장과 관능"),
    Item("음악", "Discovery", "Daft Punk", "2001",
         np.array([5, 9, 9, 7, 6, 9], dtype=float), "빛나는 멜로디의 추진력"),
    Item("음악", "Blue", "Joni Mitchell", "1971",
         np.array([6, 9, 8, 9, 7, 4], dtype=float), "가까이 들려오는 고백"),

    # 영화
    Item("영화", "기생충", "봉준호", "2019",
         np.array([7, 9, 9, 7, 7, 8], dtype=float), "장르를 접어 만든 사회의 단면"),
    Item("영화", "인셉션", "크리스토퍼 놀란", "2010",
         np.array([8, 7, 9, 6, 7, 9], dtype=float), "꿈의 구조물을 설계하다"),
    Item("영화", "이터널 선샤인", "미셸 공드리", "2004",
         np.array([6, 8, 8, 9, 7, 6], dtype=float), "기억을 지우는 사랑의 역설"),
    Item("영화", "2001: 스페이스 오디세이", "스탠리 큐브릭", "1968",
         np.array([9, 5, 7, 6, 9, 6], dtype=float), "인류와 미지의 침묵"),
    Item("영화", "라라랜드", "데이미언 셔젤", "2016",
         np.array([5, 9, 9, 8, 6, 8], dtype=float), "꿈과 현실의 스텝"),

    # 미술(작품)
    Item("미술", "별이 빛나는 밤", "Vincent van Gogh", "1889",
         np.array([6, 9, 9, 9, 7, 7], dtype=float), "소용돌이치는 밤의 신경"),
    Item("미술", "게르니카", "Pablo Picasso", "1937",
         np.array([8, 6, 8, 8, 8, 8], dtype=float), "폭력의 파편을 한 화면에"),
    Item("미술", "인상, 해돋이", "Claude Monet", "1872",
         np.array([5, 9, 9, 7, 7, 5], dtype=float), "빛이 주인공이 되는 순간"),
    Item("미술", "키스", "Gustav Klimt", "1907",
         np.array([6, 9, 9, 9, 6, 6], dtype=float), "황금빛에 감싸인 밀도"),
    Item("미술", "기억의 지속", "Salvador Dalí", "1931",
         np.array([7, 7, 8, 7, 9, 6], dtype=float), "시간이 녹아내리는 초현실"),
]

CATEGORIES = ["도서", "음악", "영화", "미술"]


# -----------------------------
# 유틸
# -----------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(10.0, float(x)))

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a2 = normalize(a)
    b2 = normalize(b)
    return float(np.dot(a2, b2))

def stable_random_vector(seed_text: str) -> np.ndarray:
    h = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    # 6개 값 생성: 0~10 범위
    vals = []
    for i in range(6):
        chunk = h[i*8:(i+1)*8]
        vals.append((int(chunk, 16) % 1000) / 1000 * 10.0)
    return np.array(vals, dtype=float)

def find_item_in_category(category: str, creator: str, title: str) -> Optional[Item]:
    candidates = [it for it in CATALOG if it.category == category]
    if not candidates:
        return None

    # 1) 정확 포함 매칭 우선
    q = f"{creator} {title}".strip().lower()
    for it in candidates:
        t = f"{it.creator} {it.title}".lower()
        if q and q in t:
            return it

    # 2) 유사 문자열(creator+title)을 대상으로 근접 매칭
    hay = [f"{it.creator} {it.title}" for it in candidates]
    matches = get_close_matches(f"{creator} {title}".strip(), hay, n=1, cutoff=0.6)
    if matches:
        m = matches[0]
        for it in candidates:
            if f"{it.creator} {it.title}" == m:
                return it

    # 3) title만으로 근접 매칭
    titles = [it.title for it in candidates]
    matches = get_close_matches(title.strip(), titles, n=1, cutoff=0.6)
    if matches:
        m = matches[0]
        for it in candidates:
            if it.title == m:
                return it

    return None

def pick_best_per_category(target_vec: np.ndarray, exclude: Optional[Item] = None) -> Dict[str, Item]:
    recs: Dict[str, Item] = {}
    for cat in CATEGORIES:
        items = [it for it in CATALOG if it.category == cat]
        if exclude is not None:
            items = [it for it in items if not (it.category == exclude.category and it.title == exclude.title and it.creator == exclude.creator)]
        best = max(items, key=lambda it: cosine_similarity(target_vec, it.vector))
        recs[cat] = best
    return recs

def describe_taste(vec: np.ndarray) -> List[str]:
    # 상위 2개 축, 하위 1개 축을 뽑아 간단한 성향 요약
    v = np.array([clamp01(x) for x in vec], dtype=float)
    idx_sorted = np.argsort(v)
    low = int(idx_sorted[0])
    high1 = int(idx_sorted[-1])
    high2 = int(idx_sorted[-2])

    def level(x: float) -> str:
        if x >= 8: return "매우 높고"
        if x >= 6: return "높은 편이고"
        if x >= 4: return "중간 정도이며"
        if x >= 2: return "낮은 편이고"
        return "매우 낮고"

    lines = [
        f"당신의 취향은 **{DIMENSIONS[high1]}**이 {level(v[high1])}, **{DIMENSIONS[high2]}**도 {level(v[high2])} 보여요.",
        f"반면 **{DIMENSIONS[low]}**은(는) {level(v[low])} 그 축을 과도하게 요구하는 작품은 피하는 편이 편안할 수 있어요.",
    ]
    return lines

def curator_reason(item: Item, user_vec: np.ndarray, mode: str, anchor: Optional[Item] = None) -> str:
    """
    mode: "취향" | "연관"
    anchor: 연관검색에서 사용자가 입력한 원작
    """
    # 축별 차이를 기반으로 설명(가까운 축, 강하게 맞물리는 축)
    diffs = np.abs(user_vec - item.vector)
    closest = int(np.argmin(diffs))
    farthest = int(np.argmax(diffs))

    # 작품 고유 태그라인 + 감상 포인트(축 기반)
    base = []
    base.append(f"**{item.tagline}**")
    base.append(f"특히 당신의 성향과 **{DIMENSIONS[closest]}** 축에서 결이 잘 맞아요 "
                f"(당신 {user_vec[closest]:.1f} ↔ 작품 {item.vector[closest]:.1f}).")

    if mode == "연관" and anchor is not None:
        # 앵커와의 관계를 한 문장 더
        a_sim = cosine_similarity(anchor.vector, item.vector)
        base.append(f"또한 입력한 작품(**{anchor.title}**)과도 정서/리듬의 접점이 있어 "
                    f"함께 감상하면 연결 고리가 선명해질 가능성이 큽니다 (유사도 {a_sim:.2f}).")
    else:
        base.append(f"한편 **{DIMENSIONS[farthest]}** 축에서는 대비가 조금 나는데 "
                    f"그 차이가 오히려 ‘새로운 즐거움’으로 작동할 수도 있어요 "
                    f"(당신 {user_vec[farthest]:.1f} ↔ 작품 {item.vector[farthest]:.1f}).")

    # 감상 관점 제안(축 조합)
    # 감정 톤/복잡성/각성도 위주로 코멘트
    emo_i = DIMENSIONS.index("감정 톤")
    comp_i = DIMENSIONS.index("복잡성")
    arou_i = DIMENSIONS.index("각성도")
    emo = item.vector[emo_i]
    comp = item.vector[comp_i]
    arou = item.vector[arou_i]

    pov = []
    if emo >= 7:
        pov.append("감정의 결을 ‘이야기(혹은 장면) 바깥의 공기’처럼 따라가 보세요. 여운이 핵심입니다.")
    else:
        pov.append("감정보다 구조/아이디어가 앞서는 편이에요. ‘무엇을 말하려는가’보다 ‘어떻게 만들었는가’를 관찰해보세요.")
    if comp >= 7:
        pov.append("구조가 촘촘해 재감상 가치가 큽니다. 두 번째에는 디테일(반복/대칭/모티프)에 집중해보면 좋아요.")
    else:
        pov.append("직관적 흐름이 강점이라 ‘속도’와 ‘톤’을 편하게 타면 만족도가 올라갑니다.")
    if arou >= 7:
        pov.append("에너지 레벨이 높아 몰입을 끌어올립니다. 방해 요소를 줄이고 한 번에 끝까지 가보세요.")
    else:
        pov.append("잔잔한 집중을 요구합니다. 밤이나 조용한 시간대에 감상하면 장점이 더 잘 드러납니다.")

    return "\n\n".join(base + ["**감상 포인트**", "- " + "\n- ".join(pov)])

def make_radar(values: List[float], glow: bool = False, scale: float = 1.0) -> go.Figure:
    # values: length 6, 0~10
    vals = [clamp01(v) * scale for v in values]
    vals = [min(10.0, v) for v in vals]
    theta = DIMENSIONS + [DIMENSIONS[0]]
    r = vals + [vals[0]]

    line_width = 3 if not glow else 8
    fill_alpha = 0.20 if not glow else 0.35
    marker_size = 6 if not glow else 12

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        mode="lines+markers",
        line=dict(width=line_width),
        marker=dict(size=marker_size),
        fill="toself",
        opacity=1.0,
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10]),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=420,
    )

    if glow:
        # 간단한 "빛남" 느낌: 배경/타이틀 강조(색 지정 없이도 두꺼운 선+면적으로 효과)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
        )
    return fig

def animate_glow_radar(values: List[float]):
    slot = st.empty()
    # 6프레임: 점점 커지고 두꺼워지는 레이더
    for s in [1.00, 1.03, 1.06, 1.09, 1.12, 1.15]:
        fig = make_radar(values, glow=True, scale=s)
        slot.plotly_chart(fig, use_container_width=True)
        time.sleep(0.08)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="My Curator",
    page_icon="✨",
    layout="wide",
)

# 기본 상태
if "page" not in st.session_state:
    st.session_state.page = "main"  # "main" | "results"
if "mode" not in st.session_state:
    st.session_state.mode = None  # "취향" | "연관"
if "taste_values" not in st.session_state:
    st.session_state.taste_values = [6, 6, 6, 6, 6, 6]
if "anchor_category" not in st.session_state:
    st.session_state.anchor_category = "도서"
if "anchor_creator" not in st.session_state:
    st.session_state.anchor_creator = ""
if "anchor_title" not in st.session_state:
    st.session_state.anchor_title = ""
if "results" not in st.session_state:
    st.session_state.results = None  # dict
if "results_reason" not in st.session_state:
    st.session_state.results_reason = None  # dict reasons
if "anchor_item" not in st.session_state:
    st.session_state.anchor_item = None  # Item or None


# 상단 헤더
st.markdown(
    """
    <style>
    .title-wrap {
        padding: 0.2rem 0 0.8rem 0;
    }
    .subtitle {
        opacity: 0.8;
        font-size: 0.95rem;
        margin-top: -0.4rem;
    }
    .pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(127,127,127,0.35);
        font-size: 0.85rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    </style>
    <div class="title-wrap">
        <h1>✨ My Curator</h1>
        <div class="subtitle">취향의 별을 조율하거나, 한 작품을 고리로 다른 세계를 연결해 추천해드립니다.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 사이드바: 시스템 프롬프트 표시(참고용)
with st.sidebar:
    st.header("큐레이터 설정")
    st.caption("아래 문장은 앱의 ‘큐레이터 톤’의 기준입니다.")
    st.text_area("System Prompt", SYSTEM_PROMPT, height=140)
    st.markdown("---")
    st.caption("축 설명")
    for d in DIMENSIONS:
        st.markdown(f"- **{d}**: {DIM_HELP[d]}")


def go_main():
    st.session_state.page = "main"
    st.session_state.results = None
    st.session_state.results_reason = None
    st.session_state.anchor_item = None


def reset_all():
    st.session_state.page = "main"
    st.session_state.mode = None
    st.session_state.taste_values = [6, 6, 6, 6, 6, 6]
    st.session_state.anchor_category = "도서"
    st.session_state.anchor_creator = ""
    st.session_state.anchor_title = ""
    st.session_state.results = None
    st.session_state.results_reason = None
    st.session_state.anchor_item = None


def run_taste_curate(values: List[float]):
    user_vec = np.array([clamp01(v) for v in values], dtype=float)
    recs = pick_best_per_category(user_vec)

    reasons = {}
    for cat, it in recs.items():
        reasons[cat] = curator_reason(it, user_vec, mode="취향", anchor=None)

    st.session_state.results = recs
    st.session_state.results_reason = reasons
    st.session_state.anchor_item = None
    st.session_state.page = "results"


def run_related_curate(category: str, creator: str, title: str):
    anchor = find_item_in_category(category, creator, title)
    if anchor is None:
        # 카탈로그에 없으면 입력 텍스트 기반 임의 벡터(안정적)
        seed = f"{category}|{creator}|{title}"
        user_vec = stable_random_vector(seed)
        # 앵커가 없으면 "입력작품"을 가상의 앵커로 취급(설명은 조금 덜 정확)
        anchor_vec = user_vec
        anchor_item = Item(category, title or "(제목 미입력)", creator or "(창작자 미입력)", "—", anchor_vec, "입력 기반 연관 고리")
    else:
        anchor_item = anchor
        anchor_vec = anchor.vector
        user_vec = anchor_vec

    recs = pick_best_per_category(anchor_vec, exclude=anchor_item)

    reasons = {}
    for cat, it in recs.items():
        reasons[cat] = curator_reason(it, user_vec, mode="연관", anchor=anchor_item)

    st.session_state.results = recs
    st.session_state.results_reason = reasons
    st.session_state.anchor_item = anchor_item
    st.session_state.page = "results"


# -----------------------------
# MAIN PAGE
# -----------------------------
if st.session_state.page == "main":
    colA, colB = st.columns([1.2, 1.0], vertical_alignment="top")

    with colA:
        st.subheader("메인 화면")
        st.write("원하는 검색 방식을 선택하세요.")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🎛️ 취향 검색", use_container_width=True):
                st.session_state.mode = "취향"
        with b2:
            if st.button("🔗 연관 검색", use_container_width=True):
                st.session_state.mode = "연관"

        st.markdown("---")

        if st.session_state.mode is None:
            st.info("위 버튼에서 **취향 검색** 또는 **연관 검색**을 선택해 주세요.")
        elif st.session_state.mode == "취향":
            st.markdown("#### 1) 취향 검색: 별의 꼭짓점을 조절해 주세요")
            st.caption("각 축을 안쪽/바깥쪽으로 움직이듯 슬라이더로 조절합니다. (0~10)")

            # 슬라이더 6개
            vals = []
            for i, d in enumerate(DIMENSIONS):
                v = st.slider(d, 0.0, 10.0, float(st.session_state.taste_values[i]), 0.5, key=f"taste_{i}")
                vals.append(v)

            st.session_state.taste_values = vals

            st.markdown("#### 나의 취향 별")
            st.plotly_chart(make_radar(vals, glow=False, scale=1.0), use_container_width=True)

            st.markdown("#### Curate")
            if st.button("✨ curate", type="primary", use_container_width=True):
                # 버튼 누르면 동일 별이 빛나며 커지는 효과
                animate_glow_radar(vals)
                run_taste_curate(vals)

        elif st.session_state.mode == "연관":
            st.markdown("#### 2) 연관 검색: 한 작품을 입력하면 나머지 카테고리를 추천")
            st.caption("도서/음악/영화/미술 중 하나를 고르고, 창작자와 제목을 입력해 주세요.")

            st.session_state.anchor_category = st.selectbox(
                "카테고리", CATEGORIES, index=CATEGORIES.index(st.session_state.anchor_category)
            )
            st.session_state.anchor_creator = st.text_input("창작자(저자/아티스트/감독/작가)", value=st.session_state.anchor_creator)
            st.session_state.anchor_title = st.text_input("작품 제목", value=st.session_state.anchor_title)

            if st.button("✨ curate", type="primary", use_container_width=True):
                run_related_curate(
                    st.session_state.anchor_category,
                    st.session_state.anchor_creator,
                    st.session_state.anchor_title,
                )

    with colB:
        st.subheader("사용 팁")
        st.markdown(
            """
            <span class="pill">취향 검색</span>
            <span class="pill">연관 검색</span>
            <span class="pill">도서·음악·영화·미술</span>
            """,
            unsafe_allow_html=True,
        )
        st.write("**취향 검색**은 6개 축으로 당신의 ‘감상 성향’을 별 모양으로 만들고, 그 결에 맞는 작품을 각 카테고리에서 1개씩 추천합니다.")
        st.write("**연관 검색**은 한 작품을 ‘고리’로 삼아, 다른 카테고리에서 같이 감상하면 좋은 작품을 1개씩 골라줍니다.")
        st.markdown("---")
        st.write("※ 현재는 예시 카탈로그로 동작합니다. 실제 커버/포스터/작품 이미지를 붙이려면 각 도메인 API(예: 도서/영화/음악/미술 데이터베이스)를 연결하면 좋습니다.")


# -----------------------------
# RESULTS PAGE
# -----------------------------
if st.session_state.page == "results":
    recs: Dict[str, Item] = st.session_state.results or {}
    reasons: Dict[str, str] = st.session_state.results_reason or {}

    st.subheader("결과 화면")

    # 모드에 따라 상단 설명
    if st.session_state.mode == "취향":
        user_vec = np.array([clamp01(v) for v in st.session_state.taste_values], dtype=float)
        st.markdown("#### 당신의 취향 요약")
        for line in describe_taste(user_vec):
            st.write("• " + line)

        st.markdown("#### 취향 별")
        st.plotly_chart(make_radar(st.session_state.taste_values, glow=True, scale=1.0), use_container_width=True)

    elif st.session_state.mode == "연관":
        anchor = st.session_state.anchor_item
        st.markdown("#### 입력한 작품")
        if anchor is not None:
            st.write(f"**[{anchor.category}] {anchor.title}** — {anchor.creator} ({anchor.year})")
        else:
            st.write("입력한 작품 정보가 충분하지 않아, 입력 텍스트 기반으로 연관 추천을 구성했어요.")

    st.markdown("---")
    st.markdown("### 추천 작품")

    # 4개 카테고리 카드형 표시
    cols = st.columns(4, vertical_alignment="top")
    cat_to_col = {"도서": 0, "음악": 1, "영화": 2, "미술": 3}

    for cat in CATEGORIES:
        it = recs.get(cat)
        if it is None:
            continue
        c = cols[cat_to_col[cat]]
        with c:
            st.markdown(f"#### {cat}")
            # 이미지 (placeholder)
            img_text = f"{cat} {it.title}"
            st.image(ph_image(img_text), use_container_width=True)
            st.markdown(f"**{it.title}**")
            st.write(f"{it.creator} · {it.year}")
            st.caption(it.tagline)

    st.markdown("---")
    st.markdown("### 추천 이유 & 감상 관점(큐레이터 코멘트)")

    # 이유는 카테고리별로 접기
    for cat in CATEGORIES:
        it = recs.get(cat)
        if it is None:
            continue
        with st.expander(f"[{cat}] {it.title} — {it.creator}", expanded=(cat == "도서")):
            st.markdown(reasons.get(cat, ""))

    st.markdown("---")
    bL, bR = st.columns([1, 1])
    with bL:
        if st.button("🔄 초기화", use_container_width=True):
            reset_all()
            st.rerun()
    with bR:
        if st.button("⬅️ 메인으로", use_container_width=True):
            go_main()
            st.rerun()
```
