import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. 데이터 로드 함수 (캐싱 적용으로 속도 향상)
@st.cache_data
def load_data():
    # 현재 폴더 안의 data/cars.csv 파일을 읽어옵니다.
    df = pd.read_csv("./data/cars.csv")
    return df

# 2. 홈 화면 기능
def cars_home():
    st.title("🚗 자동차 연비 분석 대시보드")
    st.markdown("### 실시간 데이터 분석 및 머신러닝 예측 서비스")
    st.write("이 대시보드는 자동차 성능 데이터를 기반으로 연비와 다양한 변수 간의 관계를 분석합니다.")
    st.info("왼쪽 사이드바에서 메뉴를 선택하여 분석을 시작하세요!")

# 3. 탐색적 자료분석(EDA) 기능
def cars_EDA(df):
    st.title("🔍 탐색적 자료 분석 (EDA)")
    
    # [나만의 기능 추가]: 데이터의 기초 통계량을 보여주는 확장 레이아웃
    with st.expander("📊 나만의 기능: 데이터 기초 통계량 확인하기"):
        st.write("전체 데이터의 평균, 표준편차, 최솟값, 최댓값 등을 확인합니다.")
        st.write(df.describe())

    st.subheader("📄 데이터 미리보기")
    st.dataframe(df.head())
    st.markdown("---")

    # 시각화 1: 대륙별 평균 연비 (막대 그래프)
    st.subheader("🌏 대륙별 평균 연비")
    with st.spinner("그래프를 그리는 중입니다..."):
        continent_mpg = df.groupby("continent")["mpg"].mean().reset_index()
        fig1 = px.bar(
            continent_mpg, 
            x="continent", 
            y="mpg", 
            color="mpg", 
            title="대륙별 평균 연비 비교",
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig1, use_container_width=True)
    st.info("💡 **분석 포인트**: 미국 차량은 비교적 연비가 낮고 일본 차량은 연비가 높은 편입니다.")
    st.markdown("---")

    # 시각화 2: 마력 대비 연비 (산점도 그래프)
    st.subheader("⚡ 마력(hp) 대비 연비 관계")
    fig2 = px.scatter(
        df, 
        x="hp", 
        y="mpg", 
        color="continent", 
        size="weightlbs", 
        hover_name="continent", 
        title="마력과 연비의 상관관계"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.info("💡 **분석 포인트**: 마력이 높을수록 연비가 낮아지는 경향이 있습니다.")

# 4. 머신러닝 연비 예측 기능
def cars_predict(df):
    st.title("🤖 머신러닝 연비 예측")
    st.write("선형회귀(Linear Regression) 모델을 활용하여 자동차의 연비(mpg)를 예측합니다.")
    
    # 입력 변수(X)와 목표 변수(y) 설정
    X = df[["cylinders", "cubicinches", "hp", "weightlbs", "time-to-60"]]
    y = df["mpg"]
    
    # 학습 데이터와 테스트 데이터 분리 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 예측 성능 표시
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    st.write(f"📊 모델 성능 (R²): **{score:.3f}**")

    # 사용자 제원 입력 공간 (슬라이더)
    st.subheader("🚗 자동차 제원 입력")
    col1, col2 = st.columns(2)
    with col1:
        cylinders = st.slider("실린더 수 (cylinders)", 3, 12, 6)
        cubicinches = st.slider("배기량 (cubicinches)", 60, 500, 200)
        hp = st.slider("마력 (horsepower)", 50, 400, 150)
    with col2:
        weightlbs = st.slider("무게 (weightlbs)", 1500, 6000, 3000)
        time_to_60 = st.slider("시속 60마일 도달 시간 (초)", 4.0, 20.0, 10.0)

    # 사용자가 입력한 값으로 데이터프레임 생성
    input_data = pd.DataFrame({
        "cylinders": [cylinders],
        "cubicinches": [cubicinches],
        "hp": [hp],
        "weightlbs": [weightlbs],
        "time-to-60": [time_to_60]
    })

    # 예측 수행 및 결과 출력
    mpg_pred = model.predict(input_data)[0]
    st.markdown("---")
    st.success(f"### 예상 연비 결과: **{mpg_pred:.2f} mpg** 🚘")

# 5. 메인 함수 (사이드바 메뉴 제어)
def main():
    st.set_page_config(page_title="자동차 연비 대시보드", layout="wide")
    
    # 데이터 불러오기
    try:
        df = load_data()
    except Exception as e:
        st.error(f"데이터 파일을 읽어오지 못했습니다. data 폴더 안에 cars.csv 파일이 있는지 확인하세요. 에러 내용: {e}")
        return

    # 사이드바 메뉴 구성
    menu = st.sidebar.radio(
        "대시보드 메뉴",
        ["홈", "탐색적 자료분석(EDA)", "연비 예측"]
    )

    # 메뉴 선택에 따른 함수 실행
    if menu == "홈":
        cars_home()
    elif menu == "탐색적 자료분석(EDA)":
        cars_EDA(df)
    elif menu == "연비 예측":
        cars_predict(df)

if __name__ == "__main__":
    main()