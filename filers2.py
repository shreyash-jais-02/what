import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import pickle

def conv_to_mnt(value):
    try:
        parts = value.split('yrs')
        years = int(parts[0])
        months = int(parts[1].split('mon')[0])
        total_history_months = years * 12 + months
        return total_history_months
    except:
        return 0

def Slope(row):
    row = row.fillna(0)
    y = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    xx = row.tolist()
    train_x = [[el] for el in xx]
    scaler = StandardScaler()
    model = scaler.fit(train_x)
    scaled_data = model.transform(train_x)
    regressor = LinearRegression()
    regressor.fit(scaled_data, y)
    return regressor.coef_[0]

def Rename(df):
    df.rename(columns={'LIMIT': 'Education_loan_Limit'}, inplace=True)
    df.rename(columns={'OUTS': 'Education_loan_outstanding_amount'}, inplace=True)
    df['SI_flag'] = df['SI_flag'].replace({'Y': 1, 'N': 0})
    df['EM_FLG'] = df['EM_FLG'].astype(int)
    df['REPDATE'] = pd.to_datetime(df['REPDATE'])
    return df

def FeatureTransform(df):
    df['Education_loan_tenure_completed'] = ((df['REPDATE'] - df['SANCDATE']).dt.days // 365) / ((df['LIMIT_EXP_DATE'] - df['SANCDATE']).dt.days // 365)
    return df

def PCA2(df, columns, model_name, Flag):
    df["RG_3M_SUM"] = df[columns].sum(axis=1)
    for col in columns:
        df[f'{col}_RATIO'] = df[col] / df["RG_3M_SUM"]

    for col in columns:
        ratio_col = f'{col}_RATIO'
        df[ratio_col].replace([np.nan, -np.inf], df[ratio_col].mean(), inplace=True)
    
    ratio_columns = [f'{col}_RATIO' for col in columns]
    pca_cr = PCA(n_components=1)
    
    if Flag == 1:
        days_3m = pca_cr.fit(df[ratio_columns])
        with open(f'{model_name}.pkl', 'wb') as f:
            pickle.dump(days_3m, f)
    else:
        with open(f'{model_name}.pkl', 'rb') as f:
            days_3m = pickle.load(f)
    
    days_rg_3m_pca = days_3m.transform(df[ratio_columns])
    df.drop(columns=ratio_columns, inplace=True)
    return days_rg_3m_pca

def Preprocess(df, Flag):
    print("Dropping cols")
    cols_to_drop = ['col1', 'col2', 'col3', ..., 'col50']
    df['CREDIT_HISTORY_LENGTH'] = df['CREDIT_HISTORY_LENGTH'].fillna(0)
    df['CREDIT_HISTORY_LENGTH_MNTH'] = df['CREDIT_HISTORY_LENGTH'].apply(conv_to_mnt)
    df.drop('CREDIT_HISTORY_LENGTH', axis=1, inplace=True)
    df.drop(cols_to_drop, axis=1, inplace=True)
    
    print("Renaming")
    df = Rename(df)
    
    print("Transforming")
    df = FeatureTransform(df)
    
    cols_to_drop2 = ['col51', 'col52', 'col53', ..., 'col60']
    df.drop(cols_to_drop2, axis=1, inplace=True)
    
    print("CREDS")
    X = ['ONEMNTHSCR', 'TWOMNTHSCR', 'THREEMNTHSCR', 'FOURMNTHSCR', 'FIVEMNTHSCR', 'SIXMNTHSCR', 'SEVENMNTHSCR', 'EIGHTMNTHSCR', 'NINEMNTHSCR', 'TENMNTHSCR', 'ELEVENMNTHSCR', 'TWELVEMNTHSCR']
    df["one_year_CR_LR"] = df[X].apply(Slope, axis=1)
    df.drop(columns=X, inplace=True)
    
    print("Finding slope Of Debits")
    X = ['ONEMNTHSDR', 'TWOMNTHSDR', 'THREEMNTHSDR', 'FOURMNTHSDR', 'FIVEMNTHSDR', 'SIXMNTHSDR', 'SEVENMNTHSDR', 'EIGHTMNTHSDR', 'NINEMNTHSDR', 'TENMNTHSDR', 'ELEVENMNTHSDR', 'TWELVEMNTHSDR']
    df["one_year_DR_LR"] = df[X].apply(Slope, axis=1)
    df.drop(columns=X, inplace=True)
    
    print("PCA_for_DAYS_model_3M")
    model_name = "DAYS_RG_3M_PCA"
    df[model_name] = PCA2(df, ['DAYS_RG0_3M', 'DAYS_RG1_3M', 'DAYS_RG2_3M', 'DAYS_RG3_3M', 'DAYS_RG4_3M'], model_name, Flag)
    
    print("PCA For DAYS 6M")
    model_name = "DAYS_RG_6M_PCA"
    df[model_name] = PCA2(df, ['DAYS_RG0_6M', 'DAYS_RG1_6M', 'DAYS_RG2_6M', 'DAYS_RG3_6M', 'DAYS_RG4_6M'], model_name, Flag)
    
    print("PCA_for_TIMES_UPG_3M")
    model_name = "TIMES_UPG_RG_3M_PCA"
    df[model_name] = PCA2(df, ['TIMES_UPG_RG0_3M', 'TIMES_UPG_RG1_3M', 'TIMES_UPG_RG2_3M', 'TIMES_UPG_RG3_3M'], model_name, Flag)
    
    return df
