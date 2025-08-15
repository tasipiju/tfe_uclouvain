# MA model


# Import librairies
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.simplefilter('ignore', ValueWarning)


# Import global functions
from global_functions import test_stationarity, calculate_optimal_qp_arma, calcul_nbr_lags



def display_one_series(df, tittle):
    "Displaytime series"
    col1, col2 = st.columns([1, 3])
    col1.subheader("Données")
    col1.write(df)
    col2.subheader("Graphique")
    fig = px.line(df, x=df.index, y="value", title=tittle)
    col2.plotly_chart(fig)




def prediction_auto(df):
    "Prévision automatique"
    if 'compute_auto' not in st.session_state:
        st.session_state.compute_auto = False

    if st.sidebar.button("Computer"):
        st.session_state.compute_auto = True

    if st.session_state.compute_auto:
        #on vérifie que le DataFrame est chargé et contient la colonne 'value'
        if df is None:
            st.write("Erreur : le DataFrame n'a pas été chargé.")
            return
        
        if 'value' not in df.columns:
            st.write("Erreur : la colonne 'value' est absente du DataFrame.")
            return
        
        #Test de stationatité
        if not test_stationarity(df['value']):
            # La série n'est pas stationnaire, on effectue n différenciation(s)
            n = 0
            df_diff = df.copy()
            while not test_stationarity(df_diff['value']):
                df_diff = df_diff.diff().dropna()
                n += 1

            st.write(f"Série non stationaire => {n} différenciation(s) effectuée(s)")
            auto_p, auto_q = calculate_optimal_qp_arma(df_diff, "value")
            st.write(f"Les valeurs de p et q sont respectivement: {auto_p} et {auto_q}")

            with st.spinner("Entraînement <<auto>> du modèle (différentié) en cours..."):
                #PARTIE 1 : VALIDATION CLASSIQUE (80% train / 20% test)
                #division des données
                df_diff_size = len(df_diff)
                train_size = int(df_diff_size * 0.8)
                train = df_diff.iloc[:train_size]
                test = df_diff.iloc[train_size:]
                #modelisation
                model = ARIMA(train, order=(auto_p,0,auto_q))
                model_fit = model.fit()
                #predictions différentiées
                pred_train = model_fit.predict(start=train.index[0], end=train.index[-1], dynamic=False)
                pred_test = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
                #inverse de la différenciation sur le test et le train
                nn = n
                while nn > 0:
                    start_pos_train = df.index.get_loc(train.index[0])
                    last_train_value = df['value'].iloc[start_pos_train - 1]
                    start_pos_test = df.index.get_loc(test.index[0])
                    last_test_value = df['value'].iloc[start_pos_test - 1]
                    forecast_train = pred_train.cumsum() + last_train_value
                    forecast_test = pred_test.cumsum() + last_test_value
                    nn -= 1
                
                #ATTENTION les 2 séries (diff et pas) doivent avoir la meme taille => meme index
                forecast_train.index = train.index
                forecast_test.index = test.index

                #PARTIE 2 : PRÉDICTIONS (20% avec 100% des données)
                #ré-entraînement sur TOUTE la série différenciée
                full_model = ARIMA(df_diff, order=(auto_p,0,auto_q))
                full_model_fitted = full_model.fit()
                n_future = int(len(df) * 0.2)  # 20% de la série originale
                pred_future_diff = full_model_fitted.predict(start=len(df_diff), end=len(df_diff) + n_future - 1)
                #inverse de la différenciation sur les valeurs futures
                nnn = n
                while nnn > 0:
                    last_future_value = df['value'].iloc[-1]
                    forecast_future = pred_future_diff.cumsum() + last_future_value
                    nnn -= 1
                #Génération des valeurs futures (généerer les dates duture et conserver la fréquence des donées)
                future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=n_future,freq=df.index.inferred_freq)
                forecast_future.index = future_dates

                #AFFICHAGE DE TOUS LES RESULTATS
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                ax.plot(forecast_train, label='Prédictions entrainements (80% Données réelles)', linestyle="--", color='orange')
                ax.plot(forecast_test, label='Prédictions tests (20% Données réelles)', linestyle="--", color='red')
                ax.plot(forecast_future, label='Prédictions futures (20% Valeurs futures)', linestyle="--", color='green')
                ax.axvline(df.index[train_size], alpha=0.6, linestyle=':', color="gray", label='Train/Test split')
                ax.axvline(df.index[-1], alpha=0.6, linestyle=':', color="black", label='Next data')
                ax.set_title(f"Prédictions automatiques <<différentiées>> avec le modèle ARMA(q={auto_p}, p={auto_q})")
                ax.legend()
                st.pyplot(fig)

                #st.write("Critères de performance (automatique)")
                mse_auto = mean_squared_error(test, pred_test)
                rmse_auto = np.sqrt(mse_auto)
                mae_auto = mean_absolute_error(test, pred_test)
                aic_auto = model_fit.aic
                bic_auto = model_fit.bic
                st.write(f"Erreur Quadratique Moyenne (MSE) : {mse_auto:.3f}")
                st.write(f"Racine de l'Erreur Quadratique Moyenne (RMSE) : {rmse_auto:.3f}")
                st.write(f"Erreur Absolue Moyenne (MAE) : {mae_auto:.3f}")
                st.write(f"Critère d'Information d'Akaike (AIC) : {aic_auto:.3f}")
                st.write(f"Critère d'Information Bayésien (BIC) : {bic_auto:.3f}")
                st.dataframe(forecast_future)
        
        else:
            st.write("Série stationaire => PAS besoin de différenciation")
            auto_p, auto_q = calculate_optimal_qp_arma(df, "value")
            st.write(f"Les valeurs de p et q sont respectivement: {auto_p} et {auto_q}")

            with st.spinner("Entraînement <<auto>> du modèle (NON différentié) en cours..."):
                #PARTIE 1 : VALIDATION CLASSIQUE (80% train / 20% test)
                #division des données
                df_size = len(df)
                train_size = int(df_size * 0.8)
                train = df.iloc[:train_size]
                test = df.iloc[train_size:]
                #modelisation
                model = ARIMA(train, order=(auto_p,0,auto_q))
                model_fit = model.fit()
                #predictions différentiées
                pred_train = model_fit.predict(start=train.index[0], end=train.index[-1], dynamic=False)
                pred_test = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
                #Pas d'inversion (car pas de différentiation), On utilise directement les prédictions
                #train 
                forecast_train = pred_train
                #test
                forecast_test = pred_test

                #PARTIE 2 : PRÉDICTIONS (20% avec 100% des données)
                #ré-entraînement sur TOUTE la série différenciée
                full_model = ARIMA(df, order=(auto_p,0,auto_q))
                full_model_fitted = full_model.fit()
                #calcul des 20% FUTUR (basé sur la taille totale des données)
                n_future = int(len(df) * 0.2)  # 20% de la série future
                pred_future = full_model_fitted.predict(start=len(df), end=len(df) + n_future - 1)
                forecast_future = pred_future

                #AFFICHAGE DE TOUS LES RESULTATS
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                ax.plot(forecast_train, label='Prédictions entrainement (80% Données réelles)', linestyle="--", color='orange')
                ax.plot(forecast_test, label='Prédictions tests (20% Données réelles)', linestyle="--", color='red')
                ax.plot(forecast_future, label='Prédictions futures (20% Valeurs futures)', linestyle="--", color='green')
                ax.axvline(df.index[train_size], alpha=0.6, linestyle=':', color="gray", label='Train/Test split')
                ax.axvline(df.index[-1], alpha=0.6, linestyle=':', color="black", label='Next data')
                ax.set_title(f"Prédictions automatiques <<non différentiées>> avec le modèle ARMA(p={auto_p}, q={auto_q})")
                ax.legend()
                st.pyplot(fig)

                #st.subheader("Critères de performance (automatique)")
                mse_auto = mean_squared_error(test, pred_test)
                rmse_auto = np.sqrt(mse_auto)
                mae_auto = mean_absolute_error(test, pred_test)
                aic_auto = model_fit.aic
                bic_auto = model_fit.bic
                st.write(f"Erreur Quadratique Moyenne (MSE) : {mse_auto:.3f}")
                st.write(f"Racine de l'Erreur Quadratique Moyenne (RMSE) : {rmse_auto:.3f}")
                st.write(f"Erreur Absolue Moyenne (MAE) : {mae_auto:.3f}")
                st.write(f"Critère d'Information d'Akaike (AIC) : {aic_auto:.3f}")
                st.write(f"Critère d'Information Bayésien (BIC) : {bic_auto:.3f}")
                st.dataframe(forecast_future)                





def prediction_manuel(df):
    "Prévision manuel"
    if 'compute_manuel' not in st.session_state:
        st.session_state.compute_manuel = False

    if st.sidebar.button("Computer"):
        st.session_state.compute_manuel = True

    if st.session_state.compute_manuel:
        #on vérifie que le DataFrame est chargé et contient la colonne 'value'
        if df is None:
            st.write("Erreur : le DataFrame n'a pas été chargé.")
            return
        
        if 'value' not in df.columns:
            st.write("Erreur : la colonne 'value' est absente du DataFrame.")
            return

        #Test de stationatité
        if not test_stationarity(df['value']):
            # La série n'est pas stationnaire, on effectue n différenciation(s)
            n = 0
            df_diff = df.copy()
            while not test_stationarity(df_diff['value']):
                df_diff = df_diff.diff().dropna()
                n += 1

            st.write(f"Série non stationaire => {n} différenciations effectuée(s)")
            min, max = calcul_nbr_lags(df_diff)  #en 0 et 20% de la série
            user_p = st.slider("Nombre (p) pour valeurs passées", min_value=min, max_value=max, value=min, step=1)
            st.write(f"Le nombre p choisi est : {user_p}")
            user_p = int(user_p)
            user_q = st.slider("Nombre (q) pour moyenne mobile", min_value=min, max_value=max, value=min, step=1)
            st.write(f"Le nombre q choisi est : {user_q}")
            user_q = int(user_q)

            #On invite l'utilisateur à lancer le modèle
            if st.button("Exécuter"):

                with st.spinner("Entraînement <<manuel>> du modèle (différentié) en cours..."):
                    #PARTIE 1 : VALIDATION CLASSIQUE (80% train / 20% test)
                    #division des données
                    df_diff_size = len(df_diff)
                    train_size = int(df_diff_size * 0.8)
                    train = df_diff.iloc[:train_size]
                    test = df_diff.iloc[train_size:]
                    #modelisation
                    model = ARIMA(train, order=(user_p,0,user_q))
                    model_fit = model.fit()
                    #predictions différentiées
                    pred_train = model_fit.predict(start=train.index[0], end=train.index[-1], dynamic=False)
                    pred_test = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
                    #inverse de la différenciation sur le test et le train
                    nn = n
                    while nn > 0:
                        start_pos_train = df.index.get_loc(train.index[0])
                        last_train_value = df['value'].iloc[start_pos_train - 1]
                        start_pos_test = df.index.get_loc(test.index[0])
                        last_test_value = df['value'].iloc[start_pos_test - 1]
                        forecast_train = pred_train.cumsum() + last_train_value
                        forecast_test = pred_test.cumsum() + last_test_value
                        nn -= 1
                    
                    #ATTENTION les 2 séries (diff et pas) doivent avoir la meme taille => meme index
                    forecast_train.index = train.index
                    forecast_test.index = test.index

                    #PARTIE 2 : PRÉDICTIONS (20% avec 100% des données)
                    #ré-entraînement sur TOUTE la série différenciée
                    full_model = ARIMA(df_diff, order=(user_p,0,user_q))
                    full_model_fitted = full_model.fit()
                    n_future = int(len(df) * 0.2)  # 20% de la série originale
                    pred_future_diff = full_model_fitted.predict(start=len(df_diff), end=len(df_diff) + n_future - 1)
                    #inverse de la différenciation sur les valeurs futures
                    nnn = n
                    while nnn > 0:
                        last_future_value = df['value'].iloc[-1]
                        forecast_future = pred_future_diff.cumsum() + last_future_value
                        nnn -= 1
                    #Génération des valeurs futures (généerer les dates duture et conserver la fréquence des donées)
                    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=n_future,freq=df.index.inferred_freq)
                    forecast_future.index = future_dates

                    #AFFICHAGE DE TOUS LES RESULTATS
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                    ax.plot(forecast_train, label='Prédictions entrainements (80% Données réelles)', linestyle="--", color='orange')
                    ax.plot(forecast_test, label='Prédictions tests (20% Données réelles)', linestyle="--", color='red')
                    ax.plot(forecast_future, label='Prédictions futures (20% Valeurs futures)', linestyle="--", color='green')
                    ax.axvline(df.index[train_size], alpha=0.6, linestyle=':', color="gray", label='Train/Test split')
                    ax.axvline(df.index[-1], alpha=0.6, linestyle=':', color="black", label='Next data')
                    ax.set_title(f"Prédictions automatiques <<différentiées>> avec le modèle ARMA(p={user_p}, q={user_q})")
                    ax.legend()
                    st.pyplot(fig)

                    #st.write("Critères de performance (automatique)")
                    mse_auto = mean_squared_error(test, pred_test)
                    rmse_auto = np.sqrt(mse_auto)
                    mae_auto = mean_absolute_error(test, pred_test)
                    aic_auto = model_fit.aic
                    bic_auto = model_fit.bic
                    st.write(f"Erreur Quadratique Moyenne (MSE) : {mse_auto:.3f}")
                    st.write(f"Racine de l'Erreur Quadratique Moyenne (RMSE) : {rmse_auto:.3f}")
                    st.write(f"Erreur Absolue Moyenne (MAE) : {mae_auto:.3f}")
                    st.write(f"Critère d'Information d'Akaike (AIC) : {aic_auto:.3f}")
                    st.write(f"Critère d'Information Bayésien (BIC) : {bic_auto:.3f}")
                    st.dataframe(forecast_future)
        
        else:
            st.write("Série stationaire => PAS besoin de différenciation")
            min, max = calcul_nbr_lags(df)  #en 0 et 20% de la série
            user_p = st.slider("Nombre (p) pour valeurs passées", min_value=min, max_value=max, value=min, step=1)
            st.write(f"Le nombre p choisi est : {user_p}")
            user_p = int(user_p)
            user_q = st.slider("Nombre (q) pour moyenne mobile", min_value=min, max_value=max, value=min, step=1)
            st.write(f"Le nombre q choisi est : {user_q}")
            user_q = int(user_q)

            #On invite l'utilisateur à lancer le modèle
            if st.button("Exécuter"):

                with st.spinner("Entraînement <<lanuel>> du modèle (NON différentié) en cours..."):
                    #PARTIE 1 : VALIDATION CLASSIQUE (80% train / 20% test)
                    #division des données
                    df_size = len(df)
                    train_size = int(df_size * 0.8)
                    train = df.iloc[:train_size]
                    test = df.iloc[train_size:]
                    #modelisation
                    model = ARIMA(train, order=(user_p,0,user_q))
                    model_fit = model.fit()
                    #predictions différentiées
                    pred_train = model_fit.predict(start=train.index[0], end=train.index[-1], dynamic=False)
                    pred_test = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
                    #Pas d'inversion (car pas de différentiation), On utilise directement les prédictions
                    #train 
                    forecast_train = pred_train
                    #test
                    forecast_test = pred_test

                    #PARTIE 2 : PRÉDICTIONS (20% avec 100% des données)
                    #ré-entraînement sur TOUTE la série différenciée
                    full_model = ARIMA(df, order=(user_p,0,user_q))
                    full_model_fitted = full_model.fit()
                    #calcul des 20% FUTUR (basé sur la taille totale des données)
                    n_future = int(len(df) * 0.2)  # 20% de la série future
                    pred_future = full_model_fitted.predict(start=len(df), end=len(df) + n_future - 1)
                    forecast_future = pred_future

                    #AFFICHAGE DE TOUS LES RESULTATS
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                    ax.plot(forecast_train, label='Prédictions entrainement (80% Données réelles)', linestyle="--", color='orange')
                    ax.plot(forecast_test, label='Prédictions tests (20% Données réelles)', linestyle="--", color='red')
                    ax.plot(forecast_future, label='Prédictions futures (20% Valeurs futures)', linestyle="--", color='green')
                    ax.axvline(df.index[train_size], alpha=0.6, linestyle=':', color="gray", label='Train/Test split')
                    ax.axvline(df.index[-1], alpha=0.6, linestyle=':', color="black", label='Next data')
                    ax.set_title(f"Prédictions automatiques <<non différentiées>> avec le modèle ARMA(p={user_p}, q={user_q})")
                    ax.legend()
                    st.pyplot(fig)

                    #st.subheader("Critères de performance (automatique)")
                    mse_auto = mean_squared_error(test, pred_test)
                    rmse_auto = np.sqrt(mse_auto)
                    mae_auto = mean_absolute_error(test, pred_test)
                    aic_auto = model_fit.aic
                    bic_auto = model_fit.bic
                    st.write(f"Erreur Quadratique Moyenne (MSE) : {mse_auto:.3f}")
                    st.write(f"Racine de l'Erreur Quadratique Moyenne (RMSE) : {rmse_auto:.3f}")
                    st.write(f"Erreur Absolue Moyenne (MAE) : {mae_auto:.3f}")
                    st.write(f"Critère d'Information d'Akaike (AIC) : {aic_auto:.3f}")
                    st.write(f"Critère d'Information Bayésien (BIC) : {bic_auto:.3f}")
                    st.dataframe(forecast_future) 
            



def arma_model_plot(df, model):
    """Plot the ARMA model"""
    if st.sidebar.checkbox("Données & graphique"):
        display_one_series(df, 'Evolution de la série')
    
    type_mp = st.sidebar.selectbox(
            "Sélectionner une méthode de modélisation/prévision",
            ("auto", "manuel")
        )
    st.write(f"Modélisation/prévision => {model}, méthode => {type_mp}")

    if type_mp == "auto":
        prediction_auto(df)

    if type_mp == "manuel":
        prediction_manuel(df)



