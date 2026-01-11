import pandas as pd
import numpy as np
from datetime import datetime

class DataQualityAnalyzer:
    """데이터 품질 분석 클래스"""
    
    def __init__(self, df):
        self.df = df
        self.analysis_results = {}
    
    def analyze(self):
        """종합 품질 분석 수행"""
        self.analysis_results = {
            'missing_values': self.check_missing_values(),
            'duplicate_rows': self.check_duplicates(),
            'data_types': self.check_data_types(),
            'outliers': self.check_outliers(),
            'value_ranges': self.check_value_ranges()
        }
        return self.analysis_results
    
    def check_missing_values(self):
        """결측치 검사"""
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        return pd.DataFrame({
            'Column': self.df.columns,
            'Missing Count': missing.values,
            'Missing Percentage': missing_percent.values
        })
    
    def check_duplicates(self):
        """중복 행 검사"""
        total_rows = len(self.df)
        duplicate_count = self.df.duplicated().sum()
        duplicate_percent = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0
        
        return {
            'Total Rows': total_rows,
            'Duplicate Rows': duplicate_count,
            'Duplicate Percentage': f"{duplicate_percent:.2f}%"
        }
    
    def check_data_types(self):
        """데이터 타입 검사"""
        return pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': self.df.dtypes.values
        })
    
    def check_outliers(self):
        """이상치 검사 (수치형 컬럼만)"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = len(self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)])
            outliers_info[col] = {
                'Outlier Count': outlier_count,
                'Lower Bound': f"{lower_bound:.2f}",
                'Upper Bound': f"{upper_bound:.2f}"
            }
        
        return outliers_info
    
    def check_value_ranges(self):
        """수치형 컬럼의 범위 검사"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        ranges = {}
        
        for col in numeric_cols:
            ranges[col] = {
                'Min': f"{self.df[col].min():.2f}",
                'Max': f"{self.df[col].max():.2f}",
                'Mean': f"{self.df[col].mean():.2f}",
                'Std Dev': f"{self.df[col].std():.2f}"
            }
        
        return ranges
    
    def get_summary_report(self):
        """요약 리포트 반환"""
        total_records = len(self.df)
        total_columns = len(self.df.columns)
        missing_percent = (self.df.isnull().sum().sum() / (total_records * total_columns)) * 100
        
        return {
            'Total Records': total_records,
            'Total Columns': total_columns,
            'Overall Missing Percentage': f"{missing_percent:.2f}%",
            'Duplicate Records': self.df.duplicated().sum()
        }
