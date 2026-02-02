import streamlit as st
import requests

# ---------------------------
# 페이지 설정
# ---------------------------
st.set_page_config(page_title="🎬 나와 어울리는 영화는?", page_icon="🎬")

# ---------------------------
# 사이드바: TMDB API Key
# ---------------------------
st.sidebar.title("🔑 TMDB 설정")
api_key = st.sidebar.text_input("TMDB API Key", type="password")

# ---------------------------
# 세션 상태 초기화
# ---------------------------
if "answers" not in st.session_state:
    st.session_state.answers = {}

if "show_result" not in st.session_state:
    st.session_state.show_result = False

# ---------------------------
# 제목
# ---------------------------
st.title("🎬 나와 어울리는 영화는?")
st.write("당신의 선택을 바탕으로 지금 딱 맞는 영화를 추천해드려요 😊")
st.divider()

# ---------------------------
# 질문 & 장르
# ---------------------------
genres = ["로맨스/드라마", "액션/어드벤처", "SF/판타지", "코미디"]

questions = {
    "Q1": ("주말에 갑자기 하루가 비었다! 너의 선택은?", [
        "카페에 앉아 음악 들으면서 일기 쓰거나 영화 한 편 몰아보기 ☕",
        "즉흥으로 여행 가거나 새로운 액티비티 도전 🚗",
        "집에서 세계관 탄탄한 영화 정주행, 상상력 풀가동 ✨",
        "친구들이랑 만나서 웃다가 하루 순삭 🤣"
    ]),
    "Q2": ("영화 볼 때 가장 중요한 건 뭐야?", [
        "감정선과 여운, 보고 나서 한참 생각나면 최고 💭",
        "속도감 있는 전개와 손에 땀 나는 장면 🔥",
        "“이런 설정을 생각했다고?” 싶은 신선함 🪐",
        "아무 생각 없이 웃을 수 있는 포인트 😂"
    ]),
    "Q3": ("과제 폭탄 맞은 시험 기간 밤, 너의 기분은?", [
        "괜히 센치해져서 플레이리스트부터 튼다 🎧",
        "끝까지 버티겠다는 의지로 에너지 충전 💪",
        "현실 도피하고 싶어서 다른 세계를 상상한다 🌌",
        "“아 망했다” 하면서도 밈 찾아본다 🤪"
    ]),
    "Q4": ("네가 영화 속 주인공이라면?", [
        "관계와 감정 속에서 성장하는 인물",
        "위기의 순간마다 몸부터 움직이는 히어로",
        "특별한 능력이나 운명을 가진 존재",
        "사건을 더 꼬이게 만드는 분위기 메이커"
    ]),
    "Q5": ("영화 엔딩은 이랬으면 좋겠어", [
        "조용하지만 마음에 오래 남는 결말 🌙",
        "모든 갈등이 해결되고 짜릿한 마무리 💥",
        "“그래서 그 세계는 계속될까?” 여운 남김 🧩",
        "크레딧 올라가도 웃음이 멈추지 않음 😆"
    ])
}

# ---------------------------
# 질문 표시
# ---------------------------
for q, (text, opts) in questions.items():
    st.session_state.answers[q] = st.radio(f"{q}. {text}", opts, key=q)

st.divider()

# ---------------------------
# 버튼
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("🎥 결과 보기"):
        st.session_state.show_result = True
with col2:
    if st.button("🔄 다시 테스트하기"):
        st.session_state.answers = {}
        st.session_state.show_result = False
        st.experimental_rerun()

# ---------------------------
# 결과 처리
# ---------------------------
if st.session_state.show_result:

    if not api_key:
        st.warning("TMDB API Key를 입력해주세요.")
        st.stop()

    scores = {g: 0 for g in genres}
    for q, ans in st.session_state.answers.items():
        idx = questions[q][1].index(ans)
        scores[genres[idx]] += 1

    result_genre = max(scores, key=scores.get)

    st.subheader("🎯 당신의 영화 취향")
    st.markdown(f"## **{result_genre}**")

    tmdb_genre_map = {
        "로맨스/드라마": "18,10749",
        "액션/어드벤처": "28",
        "SF/판타지": "878,14",
        "코미디": "35"
    }

    # ---------------------------
    # 영화 검색
    # ---------------------------
    discover_url = (
        f"https://api.themoviedb.org/3/discover/movie"
        f"?api_key={api_key}"
        f"&with_genres={tmdb_genre_map[result_genre]}"
        "&language=ko-KR"
        "&sort_by=popularity.desc"
    )

    movies = requests.get(discover_url).json().get("results", [])[:5]

    st.divider()
    st.subheader("🍿 추천 영화 TOP 5")

    for movie in movies:
        movie_id = movie["id"]

        # 상세 정보
        detail = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=ko-KR"
        ).json()

        credits = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={api_key}&language=ko-KR"
        ).json()

        providers = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={api_key}"
        ).json()

        # 데이터 정리
        countries = ", ".join([c["name"] for c in detail.get("production_countries", [])])
        cast = ", ".join([c["name"] for c in credits.get("cast", [])[:3]])
        ott = ", ".join([p["provider_name"] for p in providers.get("results", {}).get("KR", {}).get("flatrate", [])])

        col1, col2 = st.columns([1, 3])

        with col1:
            if movie.get("poster_path"):
                st.image("https://image.tmdb.org/t/p/w500" + movie["poster_path"], use_container_width=True)

        with col2:
            st.markdown(f"### 🎬 {movie['title']}")
            st.write(f"⭐ 평점: {movie['vote_average']}")
            st.write(f"🌍 국가: {countries or '정보 없음'}")
            st.write(f"🎭 주연: {cast or '정보 없음'}")
            st.write(f"📺 OTT: {ott or '국내 제공 OTT 없음'}")
            st.write(movie.get("overview", "줄거리 정보가 없습니다."))
            st.caption("👉 당신의 취향과 가장 잘 맞는 장르의 인기 작품이에요.")

        st.divider()
