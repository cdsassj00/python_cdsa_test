import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

class DataVisualizer:
    """데이터 시각화 클래스"""
    
    def __init__(self, df):
        self.df = df
    
    def create_numerical_distribution(self, column):
        """수치형 데이터 분포도"""
        fig = px.histogram(self.df, x=column, nbins=30, 
                          title=f"{column} 분포도",
                          labels={column: column})
        return fig
    
    def create_categorical_distribution(self, column):
        """카테고리형 데이터 분포도"""
        value_counts = self.df[column].value_counts()
        fig = px.bar(x=value_counts.index, y=value_counts.values,
                    title=f"{column} 분포도",
                    labels={'x': column, 'y': 'Count'})
        return fig
    
    def create_correlation_heatmap(self):
        """상관계수 히트맵"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return None
        
        corr_matrix = numeric_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(title='수치형 변수 상관계수 히트맵',
                         width=600, height=600)
        return fig
    
    def create_scatter_plot(self, x_col, y_col, color_col=None):
        """산점도"""
        fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col,
                        title=f"{x_col} vs {y_col}",
                        hover_data=self.df.columns)
        return fig
    
    def create_box_plot(self, column):
        """상자 그림"""
        fig = px.box(self.df, y=column, title=f"{column} 상자 그림")
        return fig
    
    def create_multi_dimensional_analysis(self):
        """다중 변수 분석"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            fig = px.scatter_matrix(
                self.df[numeric_cols],
                title='수치형 변수 다중 산점도',
                height=800
            )
            return fig
        return None
    
    def create_summary_statistics_table(self):
        """요약 통계 테이블"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        summary = numeric_df.describe().T
        return summary
    
    def create_pie_chart(self, column):
        """원형 그래프"""
        value_counts = self.df[column].value_counts()
        fig = px.pie(values=value_counts.values, names=value_counts.index,
                    title=f"{column} 비율")
        return fig
