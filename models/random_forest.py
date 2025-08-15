#random forest model

#import necessary libraries
import math
import pandas as pd
import numpy as np
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from pandas import DataFrame, concat
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Import global functions
from global_functions import test_stationarity




def display_one_series(df, tittle):
    "Displaytime series"
    col1, col2 = st.columns([1, 3])
    col1.subheader("Données")
    col1.write(df)
    col2.subheader("Graphique")
    fig = px.line(df, x=df.index, y="value", title=tittle)
    col2.plotly_chart(fig)




def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    "transform a time series dataset into a supervised learning dataset"
    df = DataFrame(data)
    #n_vars = 1 if type(data) is list else data.shape[1]
    cols = list()
	# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
	# put it all together
    agg = concat(cols, axis=1)
	# drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values





def train_test_split(data, n_test):
    "split a univariate dataset into train/test sets"
    return data[:-n_test, :], data[-n_test:, :]





def find_best_rf_params(X_train, y_train):
    """
    Recherche les meilleurs hyperparamètres pour un modèle Random Forest
    en utilisant GridSearchCV.
    """
    # Grille d'hyperparamètres à tester
    param_grid = {
        'n_estimators': [50, 75, 100, 200, 300],
        'max_depth': [5, 8, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['log2', 'sqrt'],
        'criterion': ['squared_error', 'absolute_error']
    }
    # Configuration du Grid Search
    grid_search = GridSearchCV(
        RandomForestRegressor(), 
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=1
        )
    # Entraînement de la recherche
    #grid_search.fit(X_train, y_train)
    grid_search.fit(X_train, y_train.ravel())
    # Retour des meilleurs paramètres
    return grid_search.best_params_





def prediction_auto(df):
    "Prediction automatique avec un Random_forest"
    if 'compute_auto' not in st.session_state:
        st.session_state.compute_auto = False

    if st.sidebar.button("Computer"):
        st.session_state.compute_auto = True

    if st.session_state.compute_auto:
        #définition des paramètres du modèle Random Forest ( automatiquement avec GridSarchCV)
        #par exemple, on peut définir le nombre d'arbres, la profondeur maximale, etc..

        #on vérifie que le DataFrame est chargé et contient la colonne 'value'
        if df is None:
            st.write("Erreur : le DataFrame n'a pas été chargé.")
            return
        
        if 'value' not in df.columns:
            st.write("Erreur : la colonne 'value' est absente du DataFrame.")
            return
        
        #on vérifie la stationarité de la série temporelle
        if not test_stationarity(df['value']):
            # La série n'est pas stationnaire, on effectue n différenciation(s)
            n = 0
            df_diff = df.copy()

            while not test_stationarity(df_diff['value']):
                df_diff = df_diff.diff().dropna()
                #print(f"Différenciation {n+1} effectuée")
                n += 1

            df_diff = df_diff.reset_index(drop=True)
            st.write(f"Série non stationaire => {n} différenciation(s) effectuée(s)")

            with st.spinner("Entraînement du modèle Random Forest <<automatique différentié>> en cours..."):
                # Préparation des données
                #print(df.head())
                #print(df.tail())

                #transformation des données pour le modèle Random Forest
                nbr_in = 6          #ATTENTION. a modifier pour des raisons de test (12)
                nbr_out = 1
                data = series_to_supervised(df_diff['value'], n_in=nbr_in, n_out=nbr_out, dropnan=True)
                #print(data.shape)

                #séparation des données en train et test
                n_test = int(len(df_diff) * 0.20)
                trainSet, testSet = train_test_split(data, n_test)

                #séparation des données en X et y
                train_X, trainy = trainSet[:, :-1], trainSet[:, -1:]
                test_X, testy = testSet[:, :-1], testSet[:, -1:]

                """
                #définition des paramètres du modèle Random Forest ( automatiquement avec GridSarchCV)
                #par exemple, on peut définir le nombre d'arbres, la profondeur maximale, etc..
                best_params = find_best_rf_params(train_X, trainy)
                #print(f"Meilleurs paramètres trouvés: {best_params}")
                # Entraînement du modèle Random Forest
                model = RandomForestRegressor(**best_params)
                model.fit(train_X, trainy)
                """

                # définition des paramètres <<différentiés>> du modèle manuellement (après avoir testé avec GridSearchCV)
                best_params = {
                    'criterion': 'squared_error',
                    'max_depth': 10,
                    'max_features': 'log2',
                    'min_samples_leaf': 1,
                    'min_samples_split': 5,
                    'n_estimators': 75
                }
                #print(f"Paramètres manuels utilisés : {best_params}")
                # Entraînement du modèle Random Forest
                model = RandomForestRegressor(**best_params)
                model.fit(train_X, trainy)

                # Prédictions sur les données d'entraînement et de test
                train_predictions = model.predict(train_X)
                test_predictions = model.predict(test_X)
                test_predictions_copy = test_predictions.copy()

                # Inversion de la différenciation pour le train et le test
                nn = n
                while nn > 0:
                    train_predictions = np.cumsum(train_predictions) + df['value'].iloc[nbr_in-1]
                    test_predictions = np.cumsum(test_predictions) + df['value'].iloc[nbr_in+len(train_predictions)-1]
                    nn -= 1

                # prediction future
                n_future = int(len(df_diff) * 0.20)               # 20% de la longueur de la série
                last_sequence = test_X[-1]                       # Dernière séquence connue (les données de test)
                future_preds = []
                input_seq = last_sequence.copy()
                for _ in range(n_future):
                    # Prédiction d'un pas
                    next_pred = model.predict(input_seq.reshape(1, -1))[0]
                    # ajout de la prédiction à la liste des prédictions futures
                    future_preds.append(next_pred)
                    # Mise à jour de la séquence d'entrée
                    input_seq = np.roll(input_seq, -1)
                    input_seq[-1] = next_pred  # insère la nouvelle prédiction à la fin (la valeur de debut est supprimée)
                # Inverser les différentiations pour revenir à l’échelle originale
                future_preds_final = future_preds.copy()
                nn = n
                while nn > 0:
                    last_known_value = df['value'].iloc[-1]
                    future_preds_final = np.cumsum(future_preds_final) + last_known_value
                    nn -= 1
                # Générer les dates futures (si index est une date)
                if isinstance(df.index, pd.DatetimeIndex):
                    last_date = df.index[-1]
                    freq = pd.infer_freq(df.index)
                    future_index = pd.date_range(start=last_date, periods=n_future+1, freq=freq)[1:]
                else:
                    future_index = np.arange(len(df), len(df) + n_future)

                # Affichage des résultats
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                ax.plot(df.index[nbr_in:nbr_in+len(train_predictions)], train_predictions, label='Prédictions entrainement (80% Données réelles)', linewidth=1, linestyle="--", color='orange')
                ax.plot(df.index[nbr_in+len(train_predictions):nbr_in+len(train_predictions)+len(test_predictions)], test_predictions, label='Prédictions test (20% Données réelles)', linewidth=1, linestyle="--", color='red')
                ax.plot(future_index, future_preds_final, label='Prédictions futures (20% Valeurs futures)', linewidth=1, linestyle="--",color='green')
                ax.axvline(df.index[nbr_in+len(train_predictions)], alpha=0.6, linestyle=':', color="gray", label='Train/Test split')
                ax.axvline(df.index[-1], alpha=0.6, linestyle=':', color="black", label='Next data')
                ax.set_title(f"Prédictions automatiques <<différentiées>> avec Random Forest")
                ax.legend()
                st.pyplot(fig)

                # Calcul des scores de performance
                #ATTENTION utiliser une copie de train_predictions avant inversion => test_predictions_copy
                MSE = mean_squared_error(testy, test_predictions_copy)
                MAE = mean_absolute_error(testy, test_predictions_copy)
                RSME = math.sqrt(MSE)
                st.write(f"Erreur Quadratique Moyenne (MSE) Train: {MSE:.3f}")
                st.write(f"Racine de l'Erreur Quadratique Moyenne (RMSE) : {RSME:.3f}")
                st.write(f"Erreur Absolue Moyenne (MAE) : {MAE:.3f}")
                df_future = pd.DataFrame({'Prévisions futures': future_preds_final}, index=future_index)
                st.dataframe(df_future)

                
        
        else:
            # La série est stationnaire, on n'effectue pas de différenciation
            st.write(f"Série stationaire pas besoin de différenciation(s)")

            with st.spinner("Entraînement du modèle Random Forest <<automatique non différentié>> en cours..."):
                # Préparation des données
                #print(df.head())
                #print(df.tail())

                #transformation des données pour le modèle Random Forest
                nbr_in = 6          #ATTENTION. a modifier pour des raisson de test (12)
                nbr_out = 1
                data = series_to_supervised(df['value'], n_in=nbr_in, n_out=nbr_out, dropnan=True)
                #print(data.shape)

                #séparation des données en train et test
                n_test = int(len(df) * 0.20)
                trainSet, testSet = train_test_split(data, n_test)

                #séparation des données en X et y
                train_X, trainy = trainSet[:, :-1], trainSet[:, -1:]
                test_X, testy = testSet[:, :-1], testSet[:, -1:]

                """
                #définition des paramètres <<non différentiés>> du modèle Random Forest ( automatiquement avec GridSarchCV)
                #par exemple, on peut définir le nombre d'arbres, la profondeur maximale, etc..
                best_params = find_best_rf_params(train_X, trainy)
                print(f"Meilleurs paramètres trouvés: {best_params}")

                # Entraînement du modèle Random Forest
                model = RandomForestRegressor(**best_params)
                model.fit(train_X, trainy)
                """

                # définition des paramètres <<non différentiés>> du modèle manuellement (après avoir testé avec GridSearchCV)
                best_params = {
                    'criterion': 'squared_error',
                    'max_depth': 20,
                    'max_features': 'sqrt',
                    'min_samples_leaf': 1,
                    'min_samples_split': 2,
                    'n_estimators': 200
                }
                #print(f"Paramètres manuels utilisés : {best_params}")
                # Entraînement du modèle Random Forest
                model = RandomForestRegressor(**best_params)
                model.fit(train_X, trainy)

                # Prédictions sur les données d'entraînement et de test
                train_predictions = model.predict(train_X)
                test_predictions = model.predict(test_X)
                test_predictions_copy = test_predictions.copy()

                # prediction future
                n_future = int(len(df) * 0.20)                   # 20% de la longueur de la série
                last_sequence = test_X[-1]                       # Dernière séquence connue (les données de test)
                future_preds = []
                input_seq = last_sequence.copy()
                for _ in range(n_future):
                    # Prédiction d'un pas
                    next_pred = model.predict(input_seq.reshape(1, -1))[0]
                    # ajout de la prédiction à la liste des prédictions futures
                    future_preds.append(next_pred)
                    # Mise à jour de la séquence d'entrée
                    input_seq = np.roll(input_seq, -1)
                    input_seq[-1] = next_pred  # insère la nouvelle prédiction à la fin (la valeur de debut est supprimée)
                #ATTENTION pas d'inversion car la série est stationnaire
                future_preds_final = future_preds.copy()
                # Générer les dates futures (si index est une date)
                if isinstance(df.index, pd.DatetimeIndex):
                    last_date = df.index[-1]
                    freq = pd.infer_freq(df.index)
                    future_index = pd.date_range(start=last_date, periods=n_future+1, freq=freq)[1:]
                else:
                    future_index = np.arange(len(df), len(df) + n_future)

                # Affichage des résultats
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                ax.plot(df.index[nbr_in:nbr_in+len(train_predictions)], train_predictions, label='Prédictions entrainement (80% Données réelles)', linewidth=1, linestyle="--", color='orange')
                ax.plot(df.index[nbr_in+len(train_predictions):nbr_in+len(train_predictions)+len(test_predictions)], test_predictions, label='Prédictions test (20% Données réelles)', linewidth=1, linestyle="--", color='red')
                ax.plot(future_index, future_preds_final, label='Prédictions futures (20% Valeurs futures)', linewidth=1, linestyle="--",color='green')
                ax.axvline(df.index[nbr_in+len(train_predictions)], alpha=0.6, linestyle=':', color="gray", label='Train/Test split')
                ax.axvline(df.index[-1], alpha=0.6, linestyle=':', color="black", label='Next data')
                ax.set_title(f"Prédictions automatiques non différentiées avec Random Forest")
                ax.legend()
                st.pyplot(fig)
                
                # Calcul des scores de performance
                MSE = mean_squared_error(testy, test_predictions_copy)
                MAE = mean_absolute_error(testy, test_predictions_copy)
                RSME = math.sqrt(MSE)
                st.write(f"Erreur Quadratique Moyenne (MSE) Train: {MSE:.3f}")
                st.write(f"Racine de l'Erreur Quadratique Moyenne (RMSE) : {RSME:.3f}")
                st.write(f"Erreur Absolue Moyenne (MAE) : {MAE:.3f}")
                df_future = pd.DataFrame({'Prévisions futures': future_preds_final}, index=future_index)
                st.dataframe(df_future)
          
          

            

def prediction_manuel(df):
    "Prediction manuel avec un Random Forest"
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
        
        #choix des paramètres du modèle Random Forest par l'utilisateur
        """
        param_grid = {
            'n_estimators': [50, 75, 100, 200, 300],
            'max_depth': [5, 8, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['log2', 'sqrt'],
            'criterion': ['squared_error', 'absolute_error']
        }
        """
        n_estimators = st.slider("Nombre d'arbres du modèle (n_estimators) :", min_value=50, max_value=300, value=100, step=25)
        st.write(f"Le n_estimators choisi est : {n_estimators}")
        max_depth = st.slider("Profondeur maximale d'une feuille (max_depth) :", min_value=5, max_value=20, value=10, step=1)
        st.write(f"La profondeur maximale choisie est : {max_depth}")
        min_samples_split = st.slider("Nombre minimum d'échantillons pour diviser un noeud (min_samples_split) :", min_value=2, max_value=20, value=5, step=1)
        st.write(f"Le nombre minimum d'échantillons pour diviser un noeud choisi est : {min_samples_split}")
        min_samples_leaf = st.slider("Nombre minimum d'échantillons dans une feuille (min_samples_leaf) :", min_value=1, max_value=4, value=2, step=1)
        st.write(f"Le nombre minimum d'échantillons dans une feuille choisi est : {min_samples_leaf}")
        max_features = st.selectbox("Nombre maximum de caractéristiques à considérer pour la meilleure séparation (max_features) :", 
                                    options=['log2', 'sqrt'], index=0)
        st.write(f"Le nombre maximum de caractéristiques à considérer pour la meilleure séparation choisi est : {max_features}")
        criterion = st.selectbox("Critère de qualité de la séparation (criterion)", 
                                 options=['squared_error', 'absolute_error'], index=0)
        st.write(f"Le critère de qualité de la séparation choisi est : {criterion}")
        
        #détermination de n_in, n_out et n_test
        n_in = st.slider("Nombre de pas en arrière utilisé pour prédire la valeur actuelle", min_value=4, max_value=12, value=4, step=1)
        st.write(f"Le n_in choisi est : {n_in}")
        n_out = 1       # Sortie à prédire

        #On invite l'utilisateur à lancer le modèle
        if st.button("Exécuter"):

            #on vérifie la stationarité de la série temporelle
            if not test_stationarity(df['value']):
                # La série n'est pas stationnaire, on effectue n différenciation(s)
                n = 0
                df_diff = df.copy()
                while not test_stationarity(df_diff['value']):
                    df_diff = df_diff.diff().dropna()
                    #print(f"Différenciation {n+1} effectuée")
                    n += 1

                df_diff = df_diff.reset_index(drop=True)
                st.write(f"Série non stationaire => {n} différenciation(s) effectuée(s)")

                with st.spinner("Entraînement du modèle Random Forest <<automatique différentié>> en cours..."):
                    # Préparation des données
                    #print(df.head())
                    #print(df.tail())

                    #transformation des données pour le modèle Random Forest
                    nbr_in = n_in
                    nbr_out = n_out
                    data = series_to_supervised(df_diff['value'], n_in=nbr_in, n_out=nbr_out, dropnan=True)
                    #print(data.shape)

                    #séparation des données en train et test
                    n_test = int(len(df_diff) * 0.20)
                    trainSet, testSet = train_test_split(data, n_test)

                    #séparation des données en X et y
                    train_X, trainy = trainSet[:, :-1], trainSet[:, -1:]
                    test_X, testy = testSet[:, :-1], testSet[:, -1:]

                    """
                    #définition des paramètres du modèle Random Forest ( automatiquement avec GridSarchCV)
                    #par exemple, on peut définir le nombre d'arbres, la profondeur maximale, etc..
                    best_params = find_best_rf_params(train_X, trainy)
                    #print(f"Meilleurs paramètres trouvés: {best_params}")
                    # Entraînement du modèle Random Forest
                    model = RandomForestRegressor(**best_params)
                    model.fit(train_X, trainy)
                    """
                    # définition des paramètres <<différentiés>> du modèle manuellement (après avoir testé avec GridSearchCV)
                    best_params = {
                        'criterion': criterion,
                        'max_depth': max_depth,
                        'max_features': max_features,
                        'min_samples_leaf': min_samples_leaf,
                        'min_samples_split': min_samples_split,
                        'n_estimators': n_estimators
                    }
                    #print(f"Paramètres manuels utilisés : {best_params}")
                    # Entraînement du modèle Random Forest
                    model = RandomForestRegressor(**best_params)
                    model.fit(train_X, trainy)

                    # Prédictions sur les données d'entraînement et de test
                    train_predictions = model.predict(train_X)
                    test_predictions = model.predict(test_X)
                    test_predictions_copy = test_predictions.copy()

                    # Inversion de la différenciation pour le train et le test
                    nn = n
                    while nn > 0:
                        train_predictions = np.cumsum(train_predictions) + df['value'].iloc[nbr_in-1]
                        test_predictions = np.cumsum(test_predictions) + df['value'].iloc[nbr_in+len(train_predictions)-1]
                        nn -= 1

                    # prediction future
                    n_future = int(len(df_diff) * 0.20)               # 20% de la longueur de la série
                    last_sequence = test_X[-1]                       # Dernière séquence connue (les données de test)
                    future_preds = []
                    input_seq = last_sequence.copy()
                    for _ in range(n_future):
                        # Prédiction d'un pas
                        next_pred = model.predict(input_seq.reshape(1, -1))[0]
                        # ajout de la prédiction à la liste des prédictions futures
                        future_preds.append(next_pred)
                        # Mise à jour de la séquence d'entrée
                        input_seq = np.roll(input_seq, -1)
                        input_seq[-1] = next_pred  # insère la nouvelle prédiction à la fin (la valeur de debut est supprimée)
                    # Inverser les différentiations pour revenir à l’échelle originale
                    future_preds_final = future_preds.copy()
                    nn = n
                    while nn > 0:
                        last_known_value = df['value'].iloc[-1]
                        future_preds_final = np.cumsum(future_preds_final) + last_known_value
                        nn -= 1
                    # Générer les dates futures (si index est une date)
                    if isinstance(df.index, pd.DatetimeIndex):
                        last_date = df.index[-1]
                        freq = pd.infer_freq(df.index)
                        future_index = pd.date_range(start=last_date, periods=n_future+1, freq=freq)[1:]
                    else:
                        future_index = np.arange(len(df), len(df) + n_future)

                    # Affichage des résultats
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                    ax.plot(df.index[nbr_in:nbr_in+len(train_predictions)], train_predictions, label='Prédictions entrainement (80% Données réelles)', linewidth=1, linestyle="--", color='orange')
                    ax.plot(df.index[nbr_in+len(train_predictions):nbr_in+len(train_predictions)+len(test_predictions)], test_predictions, label='Prédictions test (20% Données réelles)', linewidth=1, linestyle="--", color='red')
                    ax.plot(future_index, future_preds_final, label='Prédictions futures (20% Valeurs futures)', linewidth=1, linestyle="--",color='green')
                    ax.axvline(df.index[nbr_in+len(train_predictions)], alpha=0.6, linestyle=':', color="gray", label='Train/Test split')
                    ax.axvline(df.index[-1], alpha=0.6, linestyle=':', color="black", label='Next data')
                    ax.set_title(f"Prédictions automatiques <<différentiées>> avec Random Forest")
                    ax.legend()
                    st.pyplot(fig)

                    # Calcul des scores de performance
                    #ATTENTION utiliser une copie de train_predictions avant inversion => test_predictions_copy
                    MSE = mean_squared_error(testy, test_predictions_copy)
                    MAE = mean_absolute_error(testy, test_predictions_copy)
                    RSME = math.sqrt(MSE)
                    st.write(f"Erreur Quadratique Moyenne (MSE) Train: {MSE:.3f}")
                    st.write(f"Racine de l'Erreur Quadratique Moyenne (RMSE) : {RSME:.3f}")
                    st.write(f"Erreur Absolue Moyenne (MAE) : {MAE:.3f}")
                    df_future = pd.DataFrame({'Prévisions futures': future_preds_final}, index=future_index)
                    st.dataframe(df_future)
                    
            
            else:
                # La série est stationnaire, on n'effectue pas de différenciation
                st.write(f"Série stationaire pas besoin de différenciation(s)")

                with st.spinner("Entraînement du modèle Random Forest <<automatique non différentié>> en cours..."):
                    # Préparation des données
                    #print(df.head())
                    #print(df.tail())

                    #transformation des données pour le modèle Random Forest
                    nbr_in = n_in
                    nbr_out = n_out
                    data = series_to_supervised(df['value'], n_in=nbr_in, n_out=nbr_out, dropnan=True)
                    #print(data.shape)

                    #séparation des données en train et test
                    n_test = int(len(df) * 0.20)
                    trainSet, testSet = train_test_split(data, n_test)

                    #séparation des données en X et y
                    train_X, trainy = trainSet[:, :-1], trainSet[:, -1:]
                    test_X, testy = testSet[:, :-1], testSet[:, -1:]

                    """
                    #définition des paramètres <<non différentiés>> du modèle Random Forest ( automatiquement avec GridSarchCV)
                    #par exemple, on peut définir le nombre d'arbres, la profondeur maximale, etc..
                    best_params = find_best_rf_params(train_X, trainy)
                    print(f"Meilleurs paramètres trouvés: {best_params}")

                    # Entraînement du modèle Random Forest
                    model = RandomForestRegressor(**best_params)
                    model.fit(train_X, trainy)
                    """

                    # définition des paramètres <<non différentiés>> du modèle manuellement (après avoir testé avec GridSearchCV)
                    best_params = {
                        'criterion': criterion,
                        'max_depth': max_depth,
                        'max_features': max_features,
                        'min_samples_leaf': min_samples_leaf,
                        'min_samples_split': min_samples_split,
                        'n_estimators': n_estimators
                    }
                    #print(f"Paramètres manuels utilisés : {best_params}")
                    # Entraînement du modèle Random Forest
                    model = RandomForestRegressor(**best_params)
                    model.fit(train_X, trainy)

                    # Prédictions sur les données d'entraînement et de test
                    train_predictions = model.predict(train_X)
                    test_predictions = model.predict(test_X)
                    test_predictions_copy = test_predictions.copy()

                    # prediction future
                    n_future = int(len(df) * 0.20)                   # 20% de la longueur de la série
                    last_sequence = test_X[-1]                       # Dernière séquence connue (les données de test)
                    future_preds = []
                    input_seq = last_sequence.copy()
                    for _ in range(n_future):
                        # Prédiction d'un pas
                        next_pred = model.predict(input_seq.reshape(1, -1))[0]
                        # ajout de la prédiction à la liste des prédictions futures
                        future_preds.append(next_pred)
                        # Mise à jour de la séquence d'entrée
                        input_seq = np.roll(input_seq, -1)
                        input_seq[-1] = next_pred  # insère la nouvelle prédiction à la fin (la valeur de debut est supprimée)
                    #ATTENTION pas d'inversion car la série est stationnaire
                    future_preds_final = future_preds.copy()
                    # Générer les dates futures (si index est une date)
                    if isinstance(df.index, pd.DatetimeIndex):
                        last_date = df.index[-1]
                        freq = pd.infer_freq(df.index)
                        future_index = pd.date_range(start=last_date, periods=n_future+1, freq=freq)[1:]
                    else:
                        future_index = np.arange(len(df), len(df) + n_future)

                    # Affichage des résultats
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                    ax.plot(df.index[nbr_in:nbr_in+len(train_predictions)], train_predictions, label='Prédictions entrainement (80% Données réelles)', linewidth=1, linestyle="--", color='orange')
                    ax.plot(df.index[nbr_in+len(train_predictions):nbr_in+len(train_predictions)+len(test_predictions)], test_predictions, label='Prédictions test (20% Données réelles)', linewidth=1, linestyle="--", color='red')
                    ax.plot(future_index, future_preds_final, label='Prédictions futures (20% Valeurs futures)', linewidth=1, linestyle="--",color='green')
                    ax.axvline(df.index[nbr_in+len(train_predictions)], alpha=0.6, linestyle=':', color="gray", label='Train/Test split')
                    ax.axvline(df.index[-1], alpha=0.6, linestyle=':', color="black", label='Next data')
                    ax.set_title(f"Prédictions automatiques non différentiées avec Random Forest")
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Calcul des scores de performance
                    MSE = mean_squared_error(testy, test_predictions_copy)
                    MAE = mean_absolute_error(testy, test_predictions_copy)
                    RSME = math.sqrt(MSE)
                    st.write(f"Erreur Quadratique Moyenne (MSE) Train: {MSE:.3f}")
                    st.write(f"Racine de l'Erreur Quadratique Moyenne (RMSE) : {RSME:.3f}")
                    st.write(f"Erreur Absolue Moyenne (MAE) : {MAE:.3f}")
                    df_future = pd.DataFrame({'Prévisions futures': future_preds_final}, index=future_index)
                    st.dataframe(df_future) 
            
            



def random_forest_model_plot(df, model):
    """Plot the Random Forest model"""
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



