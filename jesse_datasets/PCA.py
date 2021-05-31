from IPython.display import JSON
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px


def PCA_graph(df,PCA_components,y):
    '''
    This is a function that plots the PCA of the dataset you pass in. 
    Parameters:
    df = the dataset you are passing in
    PCA_components = the number of PCA components you would like PCA to plot in matrix
    y = The label for prediction in a supervised learning problem. If your dataset doesn't have a label feature 
    then find a column that you care the least about :)
    '''
    df1 = df.drop([y,'longitude','latitude'], axis = 1)
    categorical_cols = [cname for cname in df1.columns if df1[cname].nunique() < 10 and 
                        df1[cname].dtype == "object"]
    numerical_cols = [cname for cname in df1.columns if df1[cname].dtype in ['int64', 'float64']]
    my_cols = categorical_cols + numerical_cols
    df1=df1[my_cols].copy()
    numerical_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='constant')),
                                       ('scaler', MinMaxScaler())
                                      ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    df1=preprocessor.fit_transform(df1)
    pca = PCA(n_components=PCA_components)
    
    components = pca.fit_transform(df1)

    labels ={
        str(i): f'PC {i+1} ( {var:.1f}%)'
        for i , var in enumerate(pca.explained_variance_ratio_* 100)
    }

    fig = px.scatter_matrix(

    components, labels = labels, 
    
    dimensions= range(PCA_components))
    fig.update_traces(diagonal_visible=False)
    
    
    return fig.show()