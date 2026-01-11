import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report, r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

class MLAnalyzer:
    """머신러닝 분석 클래스"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def prepare_data(self, exclude_cols=None):
        """데이터 전처리"""
        if exclude_cols is None:
            exclude_cols = []
        
        df_prepared = self.df.drop(columns=exclude_cols, errors='ignore')
        
        # 범주형 변수 인코딩
        categorical_cols = df_prepared.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df_prepared[col] = le.fit_transform(df_prepared[col].astype(str))
            self.label_encoders[col] = le
        
        # 날짜형 처리
        for col in df_prepared.columns:
            if pd.api.types.is_datetime64_any_dtype(df_prepared[col]):
                df_prepared[col] = (df_prepared[col] - df_prepared[col].min()).dt.days
        
        return df_prepared
    
    def clustering_analysis(self, n_clusters=3, exclude_cols=None):
        """K-Means 클러스터링"""
        df_prepared = self.prepare_data(exclude_cols)
        
        # 결측치 제거
        df_prepared = df_prepared.dropna()
        
        if len(df_prepared) < n_clusters:
            return None, "데이터 행 수가 클러스터 수보다 적습니다."
        
        # 스케일링
        X_scaled = self.scaler.fit_transform(df_prepared)
        
        # 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # 실루엣 점수
        silhouette = silhouette_score(X_scaled, clusters)
        
        results = {
            'clusters': clusters,
            'n_clusters': n_clusters,
            'silhouette_score': f"{silhouette:.4f}",
            'inertia': f"{kmeans.inertia_:.2f}",
            'centers': kmeans.cluster_centers_
        }
        
        return results, "클러스터링 완료"
    
    def feature_importance_analysis(self, target_col, task_type='classification', exclude_cols=None):
        """특성 중요도 분석"""
        if exclude_cols is None:
            exclude_cols = []
        
        exclude_cols.append(target_col)
        df_prepared = self.prepare_data(exclude_cols)
        
        # 결측치 제거
        df_prepared = df_prepared.dropna()
        
        if target_col not in self.df.columns:
            return None, f"{target_col} 컬럼을 찾을 수 없습니다."
        
        y = self.df[target_col].copy()
        
        # 타겟 인코딩
        if task_type == 'classification':
            if pd.api.types.is_object_dtype(y):
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
        
        # 결측치 일치
        valid_idx = df_prepared.index & y.index
        X = df_prepared.loc[valid_idx]
        y = y.loc[valid_idx]
        
        if len(X) < 2:
            return None, "유효한 데이터가 부족합니다."
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        if task_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            score_type = "정확도"
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            score = r2_score(y_test, model.predict(X_test))
            score_type = "R² 점수"
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        results = {
            'feature_importance': feature_importance,
            'model_score': f"{score:.4f}",
            'score_type': score_type,
            'n_samples': len(X)
        }
        
        return results, "특성 중요도 분석 완료"
    
    def get_elbow_curve_data(self, max_k=10, exclude_cols=None):
        """엘보우 곡선 데이터"""
        df_prepared = self.prepare_data(exclude_cols)
        df_prepared = df_prepared.dropna()
        
        X_scaled = self.scaler.fit_transform(df_prepared)
        
        inertias = []
        silhouettes = []
        k_range = range(2, min(max_k + 1, len(df_prepared)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
        
        return list(k_range), inertias, silhouettes
