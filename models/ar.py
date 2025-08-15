# AR model


# Import librairies
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.simplefilter('ignore', ValueWarning)


# Import global functions
from global_functions import calcul_nbr_lags, calculate_optimal_lags, test_stationarity




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
        #st.subheader("Test de stationarité")
        if not test_stationarity(df['value']):
            n = 0
            df_diff = df.copy()
            while not test_stationarity(df_diff['value']):
                df_diff = df_diff.diff().dropna()
                n += 1

            st.write(f"Série non stationaire => {n} différenciation effectuée(s)")

            lags_choice_auto = calculate_optimal_lags(df_diff)
            st.write(f"Le p calculé automatiquement est : {lags_choice_auto}")
            #st.subheader("Modeliser/Prédire (automatique) ")
            p_auto = lags_choice_auto

            with st.spinner("Entraînement <<auto>> du modèle (différentié) en cours..."):
                #PARTIE 1 : VALIDATION CLASSIQUE (80% train / 20% test)
                #division des données
                df_diff_size = len(df_diff)
                train_size = int(df_diff_size * 0.8)
                train = df_diff.iloc[:train_size]
                test = df_diff.iloc[train_size:]
                #modelisation
                model = AutoReg(train, lags=p_auto, old_names=False)
                model_fit = model.fit()
                #prédiction
                pred_train = model_fit.predict(start=train.index[p_auto], end=train.index[-1], dynamic=False)
                pred_test = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
                #inverse de la différenciation sur le test et le train
                #ATTENTION les 2 séries (diff et pas) doivent avoir la meme taille => meme index
                nn = n
                while nn > 0:
                    #train 
                    last_train_value = df['value'].iloc[p_auto-1]
                    forecast_train = pred_train.cumsum() + last_train_value
                    #test
                    last_test_value = df['value'].iloc[train_size]
                    forecast_test = pred_test.cumsum() + last_test_value
                    nn -= 1
                #ATTENTION les 2 séries (diff et pas) doivent avoir la meme taille => meme index
                forecast_train.index = train.index[p_auto:]
                forecast_test.index = test.index

                #PARTIE 2 : PRÉDICTION FUTURE (20% avec 100% des données)
                #ré-entraînement sur TOUTE la série différenciée
                full_model = AutoReg(df_diff, lags=p_auto, old_names=False)
                full_model_fit = full_model.fit()
                #calcul des 20% FUTUR (basé sur la taille totale des données)
                n_future = int(len(df) * 0.2)  # 20% de la série future
                pred_future_diff = full_model_fit.predict(start=len(df_diff), end=len(df_diff) + n_future - 1)
                #inverse la différenciation pour les valeurs futures
                nnn = n
                while nnn > 0:
                    last_value = df['value'].iloc[-1]  # Dernière valeur connue
                    forecast_future = pred_future_diff.cumsum() + last_value
                    nnn -= 1
                #Génération des valeurs futures (généerer les dates duture et conserver la fréquence des donées)
                future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=n_future,freq=df.index.inferred_freq)
                forecast_future.index = future_dates
                
                #AFFICHAGE DE TOUS LES RESULTATS
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                ax.plot(forecast_train, label='Prédictions entrainement (80% Données réelles)', linestyle="--", color='orange')
                ax.plot(forecast_test, label='Prédictions tests (20% Données réelles)', linestyle="--", color='red')
                ax.plot(forecast_future, label='Prédictions futures (20% Valeurs futures)', linestyle="--", color='green')
                ax.axvline(df.index[train_size], alpha=0.6, linestyle=':', color="gray", label='Train/Test split')
                ax.axvline(df.index[-1], alpha=0.6, linestyle=':', color="black", label='Next data')
                ax.set_title(f"Prédictions automatiques <<Différentiées>> avec le modèle AR(p={p_auto})")
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
            
        else:
            st.write("Série stationaire => PAS besoin de différenciation")
            #st.subheader("Graphe ACF et PACF")
            lags_choice_auto = calculate_optimal_lags(df)
            st.write(f"Le p calculé automatiquement est : {lags_choice_auto}")
            #st.subheader("Modeliser/Prédire (automatique) ")
            p_auto = lags_choice_auto

            with st.spinner("Entraînement <<auto>> du modèle (NON différentié) en cours..."):
                #PARTIE 1 : VALIDATION CLASSIQUE (80% train / 20% test)
                #division des données
                df_size = len(df)
                train_size = int(df_size * 0.8)
                train = df.iloc[:train_size]
                test = df.iloc[train_size:]
                #modelisation
                model = AutoReg(train, lags=p_auto, old_names=False)
                model_fit = model.fit()
                #prédiction
                pred_train = model_fit.predict(start=train.index[p_auto], end=train.index[-1], dynamic=False)
                pred_test = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
                #Pas d'inversion (car pas de différentiation), On utilise directement les prédictions
                #train
                forecast_train = pred_train
                #test
                forecast_test = pred_test
                
                #PARTIE 2 : PRÉDICTION FUTURE (20% avec 100% des données)
                #ré-entraînement sur TOUTE la série NON différenciée
                full_model = AutoReg(df, lags=p_auto, old_names=False)
                full_model_fit = full_model.fit()
                #calcul des 20% FUTUR (basé sur la taille totale des données)
                n_future = int(len(df) * 0.2)  # 20% de la série future
                pred_future = full_model_fit.predict(start=len(df), end=len(df) + n_future - 1)
                forecast_future = pred_future
                
                #AFFICHAGE DE TOUS LES RESULTATS
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                ax.plot(forecast_train, label='Prédictions entrainement (80% Données réelles)', linestyle="--", color='orange')
                ax.plot(forecast_test, label='Prédictions tests (20% Données réelles)', linestyle="--", color='red')
                ax.plot(forecast_future, label='Prédictions futures (20% Valeurs futures)', linestyle="--", color='green')
                ax.axvline(df.index[train_size], alpha=0.6, linestyle=':', color="gray", label='Train/Test split')
                ax.axvline(df.index[-1], alpha=0.6, linestyle=':', color="black", label='Next data')
                ax.set_title(f"Prédictions automatiques <<Non Différentiées>> avec le modèle AR(p={p_auto})")
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
        #st.subheader("Test de stationarité")
        if not test_stationarity(df['value']):
            # La série n'est pas stationnaire, on effectue n différenciation(s)
            n = 0
            df_diff = df.copy()
            while not test_stationarity(df_diff['value']):
                df_diff = df_diff.diff().dropna()
                n += 1

            st.write(f"Série non stationaire => {n} différenciation effectuée(s)")
            min, max = calcul_nbr_lags(df_diff)  #en 0 et 20% de la série
            user_lag = st.slider("Choisir le nombre de valeurs passées (p)", min_value=min, max_value=max, value=min, step=1)
            st.write(f"Le p manuel choisi est : {int(user_lag)}")
            #st.subheader("Modeliser/Prédire (manuellement) ")
            user_p = int(user_lag)

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
                    model = AutoReg(train, lags=user_p, old_names=False)
                    model_fit = model.fit()
                    #prédiction
                    pred_train = model_fit.predict(start=train.index[user_p], end=train.index[-1], dynamic=False)
                    pred_test = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
                    #inverse de la différenciation sur le test et le train
                    nn = n
                    while nn > 0:
                        #train
                        last_train_value = df['value'].iloc[user_p-1]
                        forecast_train = pred_train.cumsum() + last_train_value
                        #test
                        last_test_value = df['value'].iloc[train_size]
                        forecast_test = pred_test.cumsum() + last_test_value
                        nn -= 1
                    
                    #ATTENTION les 2 séries (diff et pas) doivent avoir la meme taille => meme index
                    forecast_train.index = train.index[user_p:]
                    forecast_test.index = test.index
                    
                    #PARTIE 2 : PRÉDICTION FUTURE (20% avec 100% des données)
                    #ré-entraînement sur TOUTE la série différenciée
                    full_model = AutoReg(df_diff, lags=user_p, old_names=False)
                    full_model_fit = full_model.fit()
                    #calcul des 20% FUTUR (basé sur la taille totale des données)
                    n_future = int(len(df) * 0.2)  # 20% de la série future
                    pred_future_diff = full_model_fit.predict(start=len(df_diff), end=len(df_diff) + n_future - 1)
                    #inverse la différenciation pour les valeurs futures
                    last_value = df['value'].iloc[-1]
                    forecast_future = pred_future_diff.cumsum() + last_value
                    #Génération des valeurs futures (généerer les dates dutures et conserver la fréquence des donées)
                    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=n_future,freq=df.index.inferred_freq)
                    forecast_future.index = future_dates

                    #AFFICHAGE DE TOUS LES RESULTATS
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                    ax.plot(forecast_train, label='Prédictions entrainement (80% Données réelles)', linestyle="--", color='orange')
                    ax.plot(forecast_test, label='Prédictions (20% Données réelles)', linestyle="--", color='red')
                    ax.plot(forecast_future, label='Prédictions futures (20% Valeurs futures)', linestyle="--", color='green')
                    ax.axvline(df.index[train_size], alpha=0.6, linestyle=':', color="gray", label='Train/Test split')
                    ax.axvline(df.index[-1], alpha=0.6, linestyle=':', color="black", label='Next data')
                    ax.set_title(f"Prédictions manuelles <<Différentié>> avec le modèle AR(p={user_p})")
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
            
        else:
            st.write("Série Stationaire => das besoin de Différenciation ")
            #st.subheader("Graphe ACF et PACF")
            min, max = calcul_nbr_lags(df)  #en 0 et 20% de la série
            user_lag = st.slider("Choisir le nombre de valeurs passées (p)", min_value=min, max_value=max, value=min, step=1)
            #st.subheader("Modeliser/Prédire (automatique) ")
            user_p = int(user_lag)
            st.write(f"Le p manuel choisi est : {int(user_lag)}")

            #On invite l'utilisateur à lancer le modèle
            if st.button("Exécuter"):

                with st.spinner("Entraînement <<manuel>> du modèle (NON différentié) en cours..."):
                    #PARTIE 1 : VALIDATION CLASSIQUE (80% train / 20% test)
                    #division des données
                    df_size = len(df)
                    train_size = int(df_size * 0.8)
                    train = df.iloc[:train_size]
                    test = df.iloc[train_size:]
                    #modelisation
                    model = AutoReg(train, lags=user_p, old_names=False)
                    model_fit = model.fit()
                    #prédiction
                    pred_train = model_fit.predict(start=train.index[user_p], end=train.index[-1], dynamic=False)
                    pred_test = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
                    #Pas d'inversion (car pas de différentiation), On utilise directement les prédictions
                    #train 
                    forecast_train = pred_train
                    #test
                    forecast_test = pred_test

                    #PARTIE 2 : PRÉDICTION FUTURE (20% avec 100% des données)
                    #ré-entraînement sur TOUTE la série NON différenciée
                    full_model = AutoReg(df, lags=user_p, old_names=False)
                    full_model_fit = full_model.fit()
                    #calcul des 20% FUTUR (basé sur la taille totale des données)
                    n_future = int(len(df) * 0.2)  # 20% de la série future
                    pred_future = full_model_fit.predict(start=len(df), end=len(df) + n_future - 1)
                    forecast_future = pred_future

                    #AFFICHAGE DE TOUS LES RESULTATS
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                    ax.plot(forecast_train, label='Prédictions entrainement (80% Données réelles)', linestyle="--", color='orange')
                    ax.plot(forecast_test, label='Prédictions tests (20% Données réelles)', linestyle="--", color='red')
                    ax.plot(forecast_future, label='Prédictions futures (20% Valeurs futures)', linestyle="--", color='green')
                    ax.axvline(df.index[train_size], alpha=0.6, linestyle=':', color="gray", label='Train/Test split')
                    ax.axvline(df.index[-1], alpha=0.6, linestyle=':', color="black", label='Train/Test split')
                    ax.set_title(f"Prédictions manuelles <<Non différentié>> avec le modèle AR(p={user_p})")
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





def ar_model_plot(df, model):
    """Plot the AR model"""
    if st.sidebar.checkbox("Données & graphique"):
        display_one_series(df, 'Evolution de la série')
    
    type_mp = st.sidebar.selectbox(
            "Sélectionner une méthode de modélisation/prévision",
            ("auto", "manuel")
        )
    st.subheader(f"Modélisation/prévision => {model}, méthode => {type_mp}")

    if type_mp == "auto":
        prediction_auto(df)

    if type_mp == "manuel":
        prediction_manuel(df)



