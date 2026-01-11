import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import sys

# 모듈 임포트
from modules.data_quality import DataQualityAnalyzer
from modules.visualization import DataVisualizer
from modules.ml_analysis import MLAnalyzer

# 페이지 설정
st.set_page_config(
    page_title="데이터 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# 제목
st.title("📊 데이터 분석 및 시각화 대시보드")
st.markdown("---")

# 사이드바
with st.sidebar:
    st.header("📁 데이터 업로드")
    
    # 샘플 다운로드
    with open('sample_data.csv', 'r', encoding='utf-8') as f:
        sample_data = f.read()
    st.download_button(
        label="📥 샘플 CSV 다운로드",
        data=sample_data,
        file_name="sample_data.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # 파일 업로드
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=['csv'])
    
    if uploaded_file is not None:
        st.success("✅ 파일이 업로드되었습니다!")
    
    st.markdown("---")
    st.markdown("### 📋 템플릿 정보")
    st.info("""
    **필수 컬럼:**
    - customer_id: 고객 ID
    - name: 이름
    - age: 나이
    - gender: 성별
    - region: 지역
    - product_category: 제품 카테고리
    - purchase_amount: 구매 금액
    - purchase_date: 구매 날짜
    - satisfaction_score: 만족도 점수
    - loyalty_member: 충성도 멤버 여부
    """)

# 데이터 로드 및 처리
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # 탭 생성
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 데이터 개요",
            "🔍 데이터 품질",
            "📈 시각화",
            "🤖 머신러닝",
            "📉 고급 분석"
        ])
        
        # 탭 1: 데이터 개요
        with tab1:
            st.header("데이터 개요")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 전체 행 수", len(df))
            with col2:
                st.metric("📋 컬럼 수", len(df.columns))
            with col3:
                st.metric("❌ 결측치 수", df.isnull().sum().sum())
            with col4:
                st.metric("🔄 중복 행 수", df.duplicated().sum())
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📄 데이터 샘플")
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                st.subheader("📊 컬럼 정보")
                col_info = pd.DataFrame({
                    '컬럼명': df.columns,
                    '데이터타입': df.dtypes.values,
                    '결측치': df.isnull().sum().values,
                    '비어있음 %': (df.isnull().sum() / len(df) * 100).round(2).values
                })
                st.dataframe(col_info, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📈 기본 통계")
            st.dataframe(df.describe().T, use_container_width=True)
        
        # 탭 2: 데이터 품질
        with tab2:
            st.header("🔍 데이터 품질 분석")
            
            analyzer = DataQualityAnalyzer(df)
            
            # 요약 리포트
            summary = analyzer.get_summary_report()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 전체 레코드", summary['Total Records'])
            with col2:
                st.metric("📋 전체 컬럼", summary['Total Columns'])
            with col3:
                st.metric("❌ 결측치 비율", summary['Overall Missing Percentage'])
            with col4:
                st.metric("🔄 중복 레코드", summary['Duplicate Records'])
            
            st.markdown("---")
            
            # 분석 수행
            results = analyzer.analyze()
            
            # 결측치 분석
            st.subheader("1️⃣ 결측치 분석")
            missing_df = results['missing_values']
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
                
                fig = px.bar(missing_df, x='Column', y='Missing Percentage',
                           title='컬럼별 결측치 비율',
                           labels={'Missing Percentage': '결측치 비율 (%)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ 결측치가 없습니다!")
            
            # 중복 분석
            st.subheader("2️⃣ 중복 분석")
            duplicate_info = results['duplicate_rows']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("중복 행 수", duplicate_info['Duplicate Rows'])
            with col2:
                st.metric("중복 비율", duplicate_info['Duplicate Percentage'])
            
            if duplicate_info['Duplicate Rows'] > 0:
                st.warning("⚠️ 중복 행이 발견되었습니다.")
                duplicate_rows = df[df.duplicated(keep=False)].sort_values(by=list(df.columns))
                st.dataframe(duplicate_rows, use_container_width=True)
            
            # 데이터 타입 분석
            st.subheader("3️⃣ 데이터 타입 분석")
            st.dataframe(results['data_types'], use_container_width=True)
            
            # 이상치 분석
            st.subheader("4️⃣ 이상치 분석")
            outliers = results['outliers']
            if outliers:
                outlier_df = pd.DataFrame(outliers).T
                st.dataframe(outlier_df, use_container_width=True)
            else:
                st.info("수치형 컬럼이 없습니다.")
            
            # 범위 분석
            st.subheader("5️⃣ 수치형 컬럼 범위 분석")
            range_df = pd.DataFrame(results['value_ranges']).T
            st.dataframe(range_df, use_container_width=True)
        
        # 탭 3: 시각화
        with tab3:
            st.header("📈 데이터 시각화")
            
            visualizer = DataVisualizer(df)
            
            # 상관계수 히트맵
            st.subheader("상관계수 히트맵")
            corr_fig = visualizer.create_correlation_heatmap()
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
            else:
                st.info("상관계수 분석을 위한 수치형 변수가 2개 이상 필요합니다.")
            
            st.markdown("---")
            
            # 개별 변수 분석
            st.subheader("개별 변수 분석")
            
            col1, col2 = st.columns(2)
            with col1:
                selected_col = st.selectbox("분석할 컬럼 선택", df.columns)
            with col2:
                chart_type = st.radio("차트 유형", ["분포도", "상자 그림", "원형 그래프"])
            
            if selected_col:
                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    if chart_type == "분포도":
                        fig = visualizer.create_numerical_distribution(selected_col)
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "상자 그림":
                        fig = visualizer.create_box_plot(selected_col)
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "원형 그래프":
                        fig = visualizer.create_pie_chart(selected_col)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    if chart_type == "분포도":
                        fig = visualizer.create_categorical_distribution(selected_col)
                        st.plotly_chart(fig, use_container_width=True)
                    elif chart_type == "원형 그래프":
                        fig = visualizer.create_pie_chart(selected_col)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("범주형 변수는 상자 그림을 지원하지 않습니다.")
            
            st.markdown("---")
            
            # 다중 변수 분석
            st.subheader("다중 변수 산점도")
            multi_fig = visualizer.create_multi_dimensional_analysis()
            if multi_fig:
                st.plotly_chart(multi_fig, use_container_width=True)
            else:
                st.info("다중 변수 분석을 위한 수치형 변수가 2개 이상 필요합니다.")
            
            st.markdown("---")
            
            # 두 변수 비교
            st.subheader("두 변수 비교")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X축 선택", numeric_cols)
                with col2:
                    y_col = st.selectbox("Y축 선택", numeric_cols, 
                                        index=min(1, len(numeric_cols)-1))
                
                if x_col != y_col:
                    scatter_fig = visualizer.create_scatter_plot(x_col, y_col)
                    st.plotly_chart(scatter_fig, use_container_width=True)
        
        # 탭 4: 머신러닝
        with tab4:
            st.header("🤖 머신러닝 분석")
            
            ml_analyzer = MLAnalyzer(df)
            
            # 탭 분할
            ml_tab1, ml_tab2, ml_tab3 = st.tabs(["클러스터링", "특성 중요도", "엘보우 곡선"])
            
            with ml_tab1:
                st.subheader("K-Means 클러스터링")
                
                col1, col2 = st.columns(2)
                with col1:
                    n_clusters = st.slider("클러스터 수 선택", 2, 10, 3)
                with col2:
                    exclude_cols = st.multiselect(
                        "제외할 컬럼 (ID, 이름 등)",
                        df.columns,
                        default=['customer_id', 'name'] if 'customer_id' in df.columns else []
                    )
                
                if st.button("클러스터링 실행"):
                    results, message = ml_analyzer.clustering_analysis(
                        n_clusters=n_clusters,
                        exclude_cols=exclude_cols
                    )
                    
                    if results:
                        st.success(f"✅ {message}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("실루엣 점수", results['silhouette_score'])
                        with col2:
                            st.metric("관성값", results['inertia'])
                        
                        # 클러스터 할당
                        result_df = df.copy()
                        result_df['Cluster'] = results['clusters']
                        
                        st.subheader("클러스터 할당 결과")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # 클러스터 분포
                        cluster_counts = pd.Series(results['clusters']).value_counts().sort_index()
                        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                                   title='클러스터별 데이터 분포',
                                   labels={'x': 'Cluster', 'y': 'Count'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"❌ {message}")
            
            with ml_tab2:
                st.subheader("특성 중요도 분석")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                all_target_cols = numeric_cols + categorical_cols
                
                col1, col2 = st.columns(2)
                with col1:
                    target_col = st.selectbox(
                        "타겟 변수 선택",
                        all_target_cols
                    )
                with col2:
                    task_type = st.radio(
                        "작업 유형",
                        ["classification", "regression"],
                        format_func=lambda x: "분류" if x == "classification" else "회귀"
                    )
                
                exclude_cols = st.multiselect(
                    "제외할 컬럼",
                    df.columns,
                    default=['customer_id', 'name'] if 'customer_id' in df.columns else [],
                    key="feature_importance_exclude"
                )
                
                if st.button("특성 중요도 분석 실행"):
                    results, message = ml_analyzer.feature_importance_analysis(
                        target_col=target_col,
                        task_type=task_type,
                        exclude_cols=exclude_cols
                    )
                    
                    if results:
                        st.success(f"✅ {message}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(results['score_type'], results['model_score'])
                        with col2:
                            st.metric("분석 샘플 수", results['n_samples'])
                        
                        # 특성 중요도 테이블
                        st.subheader("특성 중요도")
                        st.dataframe(results['feature_importance'], use_container_width=True)
                        
                        # 특성 중요도 차트
                        fig = px.bar(
                            results['feature_importance'],
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='상위 특성 중요도',
                            labels={'Importance': '중요도', 'Feature': '특성'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"❌ {message}")
            
            with ml_tab3:
                st.subheader("최적 클러스터 수 결정 (엘보우 곡선)")
                
                exclude_cols = st.multiselect(
                    "제외할 컬럼",
                    df.columns,
                    default=['customer_id', 'name'] if 'customer_id' in df.columns else [],
                    key="elbow_exclude"
                )
                
                if st.button("엘보우 곡선 생성"):
                    k_range, inertias, silhouettes = ml_analyzer.get_elbow_curve_data(
                        max_k=10,
                        exclude_cols=exclude_cols
                    )
                    
                    # 엘보우 곡선
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(
                        x=k_range, y=inertias,
                        mode='lines+markers',
                        name='관성값',
                        line=dict(color='blue', width=2),
                        marker=dict(size=8)
                    ))
                    fig1.update_layout(
                        title='엘보우 곡선 - 관성값',
                        xaxis_title='클러스터 수 (K)',
                        yaxis_title='관성값',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # 실루엣 스코어 곡선
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=k_range, y=silhouettes,
                        mode='lines+markers',
                        name='실루엣 스코어',
                        line=dict(color='green', width=2),
                        marker=dict(size=8)
                    ))
                    fig2.update_layout(
                        title='실루엣 스코어 곡선',
                        xaxis_title='클러스터 수 (K)',
                        yaxis_title='실루엣 스코어',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        
        # 탭 5: 고급 분석
        with tab5:
            st.header("📉 고급 분석")
            
            st.subheader("요약 통계")
            visualizer = DataVisualizer(df)
            summary_stats = visualizer.create_summary_statistics_table()
            st.dataframe(summary_stats, use_container_width=True)
            
            st.markdown("---")
            
            st.subheader("데이터 다운로드")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 필터링된 데이터 다운로드
                st.write("필터링된 데이터 다운로드")
                
                filter_col = st.selectbox("필터링할 컬럼", df.columns, key="filter_download")
                
                if pd.api.types.is_numeric_dtype(df[filter_col]):
                    min_val = float(df[filter_col].min())
                    max_val = float(df[filter_col].max())
                    selected_range = st.slider(
                        f"{filter_col} 범위",
                        min_val, max_val, (min_val, max_val),
                        key="filter_range"
                    )
                    filtered_df = df[
                        (df[filter_col] >= selected_range[0]) &
                        (df[filter_col] <= selected_range[1])
                    ]
                else:
                    unique_values = df[filter_col].unique()
                    selected_values = st.multiselect(
                        f"{filter_col} 선택",
                        unique_values,
                        default=list(unique_values)[:5],
                        key="filter_values"
                    )
                    filtered_df = df[df[filter_col].isin(selected_values)]
                
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 필터링 데이터 다운로드 (CSV)",
                    data=csv_data,
                    file_name="filtered_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                # 전체 분석 리포트 다운로드
                st.write("전체 분석 리포트")
                
                if st.button("📊 분석 리포트 생성"):
                    report = "=== 데이터 분석 리포트 ===\n\n"
                    report += f"생성일시: {pd.Timestamp.now()}\n\n"
                    report += f"1. 데이터 개요\n"
                    report += f"   - 전체 행: {len(df)}\n"
                    report += f"   - 컬럼: {len(df.columns)}\n"
                    report += f"   - 결측치: {df.isnull().sum().sum()}\n\n"
                    report += f"2. 컬럼 정보\n{df.dtypes}\n\n"
                    report += f"3. 통계\n{df.describe()}\n"
                    
                    st.download_button(
                        label="📥 텍스트 리포트 다운로드",
                        data=report,
                        file_name="analysis_report.txt",
                        mime="text/plain"
                    )

    except Exception as e:
        st.error(f"❌ 파일을 읽는 중 오류가 발생했습니다: {str(e)}")
        st.info("CSV 파일 형식을 확인하고 다시 시도해주세요.")

else:
    st.info("👈 왼쪽 사이드바에서 CSV 파일을 업로드하세요. 또는 먼저 샘플 파일을 다운로드해서 템플릿을 확인할 수 있습니다.")
    
    # 시작 가이드
    st.markdown("""
    ## 🚀 시작 가이드
    
    ### 1단계: 샘플 파일 다운로드
    사이드바의 **"📥 샘플 CSV 다운로드"** 버튼을 클릭하여 샘플 파일을 다운로드합니다.
    
    ### 2단계: 데이터 준비
    샘플 파일과 같은 형식으로 CSV 파일을 준비합니다.
    
    ### 3단계: 파일 업로드
    준비한 CSV 파일을 사이드바의 **"CSV 파일을 업로드하세요"** 영역에 업로드합니다.
    
    ### 4단계: 분석 시작
    다음 기능을 사용할 수 있습니다:
    - 📊 **데이터 개요**: 기본 통계 및 데이터 정보
    - 🔍 **데이터 품질**: 결측치, 중복, 이상치 분석
    - 📈 **시각화**: 다양한 차트 및 그래프
    - 🤖 **머신러닝**: 클러스터링 및 특성 중요도 분석
    - 📉 **고급 분석**: 통계 요약 및 데이터 다운로드
    
    ---
    
    ### 📋 데이터 템플릿 구조
    - **customer_id**: 고객 고유 ID
    - **name**: 고객 이름
    - **age**: 연령대
    - **gender**: 성별 (M/F)
    - **region**: 지역 (North/South/East/West)
    - **product_category**: 제품 분류
    - **purchase_amount**: 구매 금액
    - **purchase_date**: 구매 날짜 (YYYY-MM-DD)
    - **satisfaction_score**: 만족도 (1-5)
    - **loyalty_member**: 충성도 멤버 여부 (Yes/No)
    """)
