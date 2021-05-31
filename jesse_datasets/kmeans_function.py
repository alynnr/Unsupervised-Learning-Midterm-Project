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


def kmeans_tower(df,kmeans_cluster,y):
    from sklearn.pipeline import Pipeline
    '''
    this is a function that clusters based on kmeans and append the clustering result as a result to your    DataFrame.
    parameter:
    df = the DataFrame that you are passing in
    kmeans_cluster = number of kmeans clusters you would like the function to predict
    y = the label feature in the data for a supervised learning problem. If your data doesn't have a prediction label then just drop the column that you care the least about :)
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
    kmeans = KMeans(n_clusters=kmeans_cluster, n_init=10)
    cluster = kmeans.fit_predict(df1)
    df['Cluster'] = cluster
    
    return df