#!/bin/bash

# 데이터 분석 대시보드 실행 스크립트

echo "========================================="
echo "📊 데이터 분석 및 시각화 대시보드"
echo "========================================="
echo ""

# 현재 디렉토리 확인
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt를 찾을 수 없습니다."
    echo "프로젝트 루트 디렉토리에서 실행하세요."
    exit 1
fi

# 라이브러리 설치 확인
echo "📦 라이브러리 설치 확인 중..."
pip install -r requirements.txt --quiet 2>/dev/null && echo "✅ 라이브러리 준비 완료" || {
    echo "❌ 라이브러리 설치 실패"
    exit 1
}

echo ""
echo "🚀 Streamlit 애플리케이션 시작..."
echo ""
echo "브라우저가 자동으로 열립니다."
echo "http://localhost:8501"
echo ""
echo "종료하려면 Ctrl+C를 누르세요."
echo ""

# Streamlit 앱 실행
streamlit run app.py --client.toolbarMode=viewer
