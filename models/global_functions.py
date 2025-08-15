# global function for models

#Import modules
import numpy as np
import itertools
from scipy import stats
from pmdarima import auto_arima
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox



def test_stationarity(series):
    """Test stationarity of a time series"""
    result = adfuller(series)
    print(f"p-value: {result[1]}")

    if result[1] < 0.05:
        print("La série est stationnaire")
        return True
    
    else:
        print("La série n'est pas stationnaire, essayez une différenciation")
        return False



def calcul_nbr_lags(df):
    """Calculate and return the number of lags (min and max)"""
    #min_lags = max(2, int(len(df) * 0.02))
    min_lags = 0
    max_lags = max(40, int(len(df) * 0.20))
    return min_lags, max_lags




def split_df(df):
    """Split 80% of train data and 20 of test data"""
    train_size = int(len(df) * 0.8)
    train, test = df['value'].iloc[:train_size], df['value'].iloc[train_size:]
    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")
    return train, test




def calculate_optimal_lags(df, max_lags=None):
    """Calculate number of lags with AIC."""
    if max_lags is None:
        max_lags = int(np.sqrt(len(df)))
    
    best_aic = float('inf')
    best_p = 0
    
    for lag in range(1, max_lags + 1):
        try:
            model = AutoReg(df, lags=lag)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_p = lag
        except:
            continue
    
    return best_p




def calculate_q(df, colonne, max_q=100, max_p=100):
    "Calculate q"
    if colonne not in df.columns:
        raise ValueError(f"La colonne '{colonne}' n'existe pas dans le DataFrame.")

    series = df[colonne]

    best_aic = float('inf')
    q_optimal = 0

    for p, q in itertools.product(range(max_q), range(max_p)):
        try:
            model = ARIMA(series, order=(p, 0, q)).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                q_optimal = q

        except:
            continue

    return q_optimal


def calculate_optimal_q(df, colonne, max_q=100):
    if colonne not in df.columns:
        raise ValueError(f"La colonne '{colonne}' n'existe pas dans le DataFrame.")
    
    serie = df[colonne]
    modele = auto_arima(serie, start_p=0, start_q=0,
                        max_p=0, max_q=max_q,
                        seasonal=False, trace=True,
                        error_action='ignore', suppress_warnings=True)
    q_optimal = modele.order[2]
    
    return q_optimal




def calculate_optimal_p_and_q(df, colonne, max_q=100, max_p=100):
    "Calculate p and q"
    if colonne not in df.columns:
        raise ValueError(f"La colonne '{colonne}' n'existe pas dans le DataFrame.")

    series = df[colonne]

    best_aic = float('inf')
    q_optimal = 0

    for p, q in itertools.product(range(max_q), range(max_p)):
        try:
            model = ARIMA(series, order=(p, 0, q)).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                p_optimal = p
                q_optimal = q

        except:
            continue

    return p_optimal, q_optimal



def calculate_optimal_qp_arma(df, colonne, max_p=100, max_q=100):
    "Calculate p and q"
    if colonne not in df.columns:
        raise ValueError(f"La colonne '{colonne}' n'existe pas dans le DataFrame.")
    
    serie = df[colonne]
    modele = auto_arima(serie, start_p=0, start_q=0,
                        max_p=max_p, max_q=max_q,
                        seasonal=False, trace=True,
                        error_action='ignore', suppress_warnings=True)
    
    p_optimal = modele.order[0]
    q_optimal = modele.order[2]
    
    return p_optimal, q_optimal



def calculate_optimal_p_and_d_and_q(df, colonne, max_p=100, max_d=100, max_q=100):
    "Calculate p and q and d"
    if colonne not in df.columns:
        raise ValueError(f"La colonne '{colonne}' n'existe pas dans le DataFrame.")

    series = df[colonne]

    best_aic = float('inf')
    p_optimal = 0
    d_optimal = 0
    q_optimal = 0

    for p, d, q in itertools.product(range(max_p), range(max_d), range(max_q)):
        try:
            model = ARIMA(series, order=(p, d, q)).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                p_optimal = 0
                d_optimal = 0
                q_optimal = 0

        except:
            continue

    return p_optimal, d_optimal, q_optimal



def calculate_optimal_pdq_arima(df, colonne, max_p=100, max_d=100, max_q=100):
    "Calculate p, d and q"
    if colonne not in df.columns:
        raise ValueError(f"La colonne '{colonne}' n'existe pas dans le DataFrame.")
    
    serie = df[colonne]
    modele = auto_arima(serie, start_p=0, start_d=0, start_q=0,
                        max_p=max_p, max_d=max_d, max_q=max_q,
                        seasonal=False, trace=True,
                        error_action='ignore', suppress_warnings=True)
    
    p_optimal = modele.order[0]
    d_optimal = modele.order[1]
    q_optimal = modele.order[2]
    
    return p_optimal, d_optimal, q_optimal





def test_jb(model, h):
    """Test : ljun_box"""
    h = 10
    residual = model.resid
    test_result = acorr_ljungbox(residual, lags=[h], return_df=True)
    q_stat = test_result.lb_stat.iloc[0]
    p_value = test_result.lb_pvalue.iloc[0]
    return q_stat, p_value



def test_shapiro(model):
    """Test : shapiro"""
    residual = model.resid
    shapiro_test = stats.shapiro(residual)
    s_q_stat = shapiro_test[0]
    s_p_value = shapiro_test[1]
    return s_q_stat, s_p_value

