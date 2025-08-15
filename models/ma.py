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
from global_functions import calculate_optimal_q, calcul_nbr_lags




def display_one_series(df, tittle):
    "Displaytime series"
    col1, col2 = st.columns([1, 3])
    col1.subheader("Données")
    col1.write(df)
    col2.subheader("Graphique")
    fig = px.line(df, x=df.index, y="value", title=tittle)
    col2.plotly_chart(fig)




"""
1. Comportement du modèle MA(q) pur
Un modèle MA(q) (ARIMA(0,0,q)) ne dépend que des erreurs passées (résidus) et pas des valeurs passées de la série.

Hors de l’échantillon (test ou futur), le modèle n’a plus accès aux erreurs réelles (car il ne connaît pas les vraies valeurs futures).
    → Il suppose alors que les erreurs futures sont nulles, donc la prédiction devient une constante : 
    la moyenne des résidus pondérés par les coefficients du modèle.

Résultat : toutes les prédictions hors échantillon sont plates (constantes), ce qui explique la ligne horizontale sur ton graphique.
2. Ce n’est pas un bug de code, c’est la nature du modèle MA
Ce comportement est documenté dans la littérature sur les modèles MA et ARIMA.
Référence Statsmodels:
    "For MA models, forecasts beyond the end of the sample are equal to the mean of the process 
    (if a constant is included), because future errors are unknown and assumed tobe zero.


a) Le modèle MA(q)
MA(q) est un modèle à moyenne mobile pure, il ne peut pas extrapoler les dynamiques futures : 
il ne fait que "lisser" les erreurs passées. Quand tu fais des prévisions hors-échantillon (test), 
il n’a plus accès aux vraies erreurs (car elles sont inconnues), donc il prédit la moyenne ou une valeur constante.
b) Utilisation de .predict()
Sur la période d'entraînement, .predict() utilise les vraies valeurs pour calculer les erreurs et donc la prédiction colle aux données.
Sur la période test, les erreurs passées ne sont plus connues, donc le modèle se "fige" et prédit une valeur constante 
(souvent la moyenne ou la dernière valeur connue).
c) Ce n’est pas une erreur de code, mais une limite du modèle MA(q)
C’est une limitation structurelle du modèle MA(q) pour la prévision hors-échantillon

"""




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
        
        #ATTENTION pas besoin de tester la stationatité sur le modèle MA
        #st.subheader("Graphe ACF et PACF (automatique)")
        q_auto = calculate_optimal_q(df, "value")
        #q_auto = calculate_q(df, "value")
        st.write(f"Le q calculé automatiquement est : {q_auto}")

        with st.spinner("Entraînement <<auto>> du modèle automatique en cours..."):
            #PARTIE 1 : VALIDATION CLASSIQUE (80% train / 20% test)
            #division des données en 80% train et 20% test
            df_size = len(df)
            train_size = int(df_size * 0.8)
            train = df.iloc[:train_size]
            test = df.iloc[train_size:]
            #modelisation
            model = ARIMA(train, order=(0,0,q_auto))
            model_fit = model.fit()
            #prediction
            pred_train = model_fit.predict(start=train.index[0], end=train.index[-1], dynamic=False)
            pred_test = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
            #Pas d'inversion (car pas de différentiation), On utilise directement les prédictions
            #train
            pred_train.index = train.index
            forecast_train = pred_train
            #test
            pred_test.index = test.index
            forecast_test = pred_test

            #PARTIE 2 : PRÉDICTIONS (20% avec 100% des données)
            #ré-entraînement sur TOUTE la série différenciée
            full_model = ARIMA(df, order=(0,0,q_auto))
            full_model_fitted = full_model.fit()
            n_future = int(len(df) * 0.2)  # 20% de la série future
            forecast_steps = n_future
            #future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
            pred_future = full_model_fitted.forecast(steps=forecast_steps)
            future_forecast = pred_future

            #AFFICHAGE DE TOUS LES RESULTATS
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
            ax.plot(forecast_train, label='Prédictions entrainements (80% Données réelles)', linestyle="--", color='orange')
            ax.plot(forecast_test, label='Prédictions tests (20% Données réelles)', linestyle="--", color='red')
            ax.plot(future_forecast, label='Prédictions futures (20% Valeurs futures)', linestyle="--", color='green')
            ax.axvline(df.index[train_size], alpha=0.6, linestyle=':', color="gray", label='Train/Test split')
            ax.axvline(df.index[-1], alpha=0.6, linestyle=':', color="black", label='Next data')
            ax.set_title(f"Prédictions automatiques avec le modèle MA(q={q_auto})")
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
            st.dataframe(future_forecast)





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

        min, max = calcul_nbr_lags(df)     #en 0 et 20% de la série
        user_q = st.slider("Nombre de lags (q) pour moyenne mobile", min_value=min, max_value=max, value=min, step=1)
        st.write(f"Le nombre q  choisi est : {user_q}")
        user_q = int(user_q)

        #On invite l'utilisateur à lancer le modèle
        if st.button("Exécuter"):

            with st.spinner("Entraînement du modèle manuel en cours..."):
                #PARTIE 1 : VALIDATION CLASSIQUE (80% train / 20% test)
                #division des données en 80% train et 20% test
                df_size = len(df)
                train_size = int(df_size * 0.8)
                train = df.iloc[:train_size]
                test = df.iloc[train_size:]
                #modelisation MA(q)
                model = ARIMA(train, order=(0,0,user_q))
                model_fit = model.fit()
                #prediction
                pred_train = model_fit.predict(start=train.index[0], end=train.index[-1], dynamic=False)
                pred_test = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)
                #Pas d'inversion (car pas de différentiation), On utilise directement les prédictions
                #train
                pred_train.index = train.index
                forecast_train = pred_train
                #test
                pred_test.index = test.index
                forecast_test = pred_test

                #PARTIE 2 : PRÉDICTIONS (20% avec 100% des données)
                #ré-entraînement sur TOUTE la série différenciée
                full_model = ARIMA(df, order=(0,0,user_q))
                full_model_fitted = full_model.fit() 
                n_future = int(len(df) * 0.2)  # 20% de la série future
                forecast_steps = n_future
                pred_future = full_model_fitted.forecast(steps=forecast_steps)
                future_forecast = pred_future
                
                #AFFICHAGE DE TOUS LES RESULTATS
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                ax.plot(forecast_train, label='Prédictions entrainements (80% Données réelles)', linestyle="--", color='orange')
                ax.plot(forecast_test, label='Prédictions tests (20% Données réelles)', linestyle="--", color='red')
                ax.plot(future_forecast, label='Prédictions futures (20% Valeurs futures)', linestyle="--", color='green')
                ax.axvline(df.index[train_size], alpha=0.6, linestyle=':', color="gray", label='Train/Test split')
                ax.axvline(df.index[-1], alpha=0.6, linestyle=':', color="black", label='Next data')
                ax.set_title(f"Prédictions manuelles avec le modèle MA(q={user_q})")
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
                st.dataframe(future_forecast)





def ma_model_plot(df, model):
    """Plot the MA model"""
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



