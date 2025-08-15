#lstm model

#import necessary libraries
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
#from keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasRegressor


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





def create_sequences(data, window_size):
    """Convert an array of values into a dataset matrix"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)





# Set the window size for the LSTM model => Number of previous time steps to consider for prediction
window_size = 6
# Set the number of features (1 for univariate time series)
num_features = 1

def create_model(units=50, optimizer='adam'):
    "Create LSTM model"
    model = Sequential()
    model.add(LSTM(units, activation='relu', return_sequences=True, input_shape=(window_size, num_features)))
    model.add(LSTM(units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.summary()
    return model





def inverse_differencing(pred_values, original_series, start_index, n_diffs):
    """
    Inverse la différenciation de manière correcte
    """
    result = pred_values.copy().flatten()
    
    for diff_level in range(n_diffs):
        # Récupérer la valeur de référence appropriée
        if start_index - diff_level - 1 >= 0:
            base_value = original_series.iloc[start_index - diff_level - 1]
        else:
            base_value = original_series.iloc[0]
        
        # Reconstruction cumulative
        result = np.cumsum(result) + base_value
    
    return result





def prediction_auto(df):
    "Prediction automatique avec un LSTM"
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

            with st.spinner("Entraînement <<auto>> du modèle LSTM <<automatique différentié>> en cours..."):
                # Préparation des données
                #print(df.info)
                df_diff
                X = df_diff['value']

                # Normalisation des données
                #RAPPEL les cellules LSTM utilisent des fonctions d'activation sigmoid ou tanh
                #On standardise les données pour qu'elle aient une moyenne à 0 et un écart-type 1
                X = X.values
                X = X.astype('float32')
                X = X.reshape(-1, 1)
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                #séparer les données de test et les données d'entrainement
                #ATTENTION on ne génère pas les groupes de test et train de facon aléatoire (comme avec split de sklearn)
                train_size = int(len(X) * 0.80)
                train, test = X[0:train_size,:], X[train_size:len(X),:]

                #Transformer les données en séquences (reshape into X=t and Y=t+1)
                c = 6
                X_train, y_train = create_sequences(train, window_size)
                X_test, y_test = create_sequences(test, window_size)

                # Reshape input to be [samples, time steps, features]
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                """
                # Find best parameters using grid search
                # Création du modèle LSTM
                model = KerasRegressor(model=create_model, verbose=0)
                # Paramètres à tester
                param_grid = {'model__units': [4, 20, 30, 50],
                            'model__optimizer': ['adam', 'rmsprop'],
                            'epochs': [10, 50, 80, 100],
                            'batch_size': [1, 10, 16, 32,  64]
                            }
                # Grid search pour trouver les meilleurs paramètres
                regressor = KerasRegressor(model=create_model, verbose=0)
                grid = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=3)
                grid.fit(X_train, y_train)
                best_params = grid.best_params_
                best_score = grid.best_score_
                print(f"Best parameters: {best_params}")
                print(f"Best score: {best_score}")
                """

                # On utilise les meilleurs paramètres trouvés par la recherche en grille
                # On ne prend que les paramètres pertinents pour la création du modèle
                window_size = 6
                optimizer = 'rmsprop'
                units = 4
                epochs = 100       
                batch_size = 1
                # Créer le modèle avec les meilleurs paramètres
                model = create_model(units=units, optimizer=optimizer)  
                # Entraînement du modèle
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)

                # Prédiction
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                #print(f"Train predictions: {train_pred}")

                # Inverser la normalisation du train et du test
                train_pred = scaler.inverse_transform(train_pred)
                y_train = scaler.inverse_transform(y_train)
                test_pred = scaler.inverse_transform(test_pred)
                y_test = scaler.inverse_transform(y_test)
                #print(len(train))     #difference de 1O avec le suivant
                #print(len(train_pred))

                # Calculer l'erreur quadratique moyenne (MSE) et de l'erreur aboslue moyenne (MAE)
                mse_error = mean_squared_error(y_test, test_pred)
                mae_error = mean_absolute_error(y_test, test_pred)

                # Inversion de la différenciation pour le train et le test
                train_pred = inverse_differencing(train_pred, df['value'], window_size, n)
                test_pred = inverse_differencing(test_pred, df['value'], train_size + window_size, n)

                # Déplacemont des prédictions pour aligner avec les données originales
                #préparation des arrays pour le graphique
                train_pred_plot = np.full(len(df), np.nan)
                test_pred_plot = np.full(len(df), np.nan)
                #ajustement des index pour l'affichage
                train_start_idx = window_size + n   # Index de début pour les prédictions d'entraînement
                test_start_idx = train_size + window_size + n    # Index de début pour les prédictions de test
                if train_start_idx + len(train_pred) <= len(df):
                    train_pred_plot[train_start_idx:train_start_idx + len(train_pred)] = train_pred

                if test_start_idx + len(test_pred) <= len(df):
                    test_pred_plot[test_start_idx:test_start_idx + len(test_pred)] = test_pred

                # Prévision du futur
                #n_forecast = 24  # Nombre de valeurs futures à prédire
                n_forecast = int(len(df) * 0.20)  # 20% de la longueur de la série
                # Prendre la dernière séquence connue (taille = window_size)
                last_sequence = X[-window_size:]
                current_sequence = last_sequence.copy()
                future_predictions = []
                for _ in range(n_forecast):
                    # Reshape pour le modèle
                    input_seq = current_sequence.reshape((1, window_size, 1))
                    # Prédiction
                    next_pred = model.predict(input_seq, verbose=0)
                    # Stocker la prédiction
                    future_predictions.append(next_pred[0, 0])
                    # Mettre à jour la séquence
                    current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
                # Inverser la normalisation sur les prévisions futures
                future_predictions = np.array(future_predictions).reshape(-1, 1)
                future_predictions = scaler.inverse_transform(future_predictions).flatten()
                # Inverser la différenciation pour les prévisions futures
                future_predictions = inverse_differencing(future_predictions, df['value'], len(df), n)

                # Préparation des index pour future_predictions
                if isinstance(df.index, pd.DatetimeIndex):
                    last_date = df.index[-1]
                    freq = pd.infer_freq(df.index)
                    future_index = pd.date_range(start=last_date, periods=n_forecast+1, freq=freq)[1:]
                else:
                    future_index = np.arange(len(df), len(df) + n_forecast)
                
                # Affichage des résultats
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df.index, df, label='Données réelles', linewidth=1, color='blue')
                ax.plot(df.index, train_pred_plot, label='Prédictions entrainements: (80% Données réelles)', linewidth=1, linestyle="--", color='orange')
                ax.plot(df.index, test_pred_plot, label='Prédictions tests: (20% Données réelles)', linewidth=1, linestyle="--", color='red')
                ax.plot(future_index, future_predictions, label='Prédictions futures: (20% Valeurs futures)', linewidth=1, linestyle="--", color='green')
                ax.set_title(f"Prédictions automatiques différentiées avec LSTM")
                ax.legend()
                st.pyplot(fig)

                # Critères de performance
                st.write(f"Erreur Quadratique Moyenne (MSE) Test : {mse_error:.3f}")
                rmse_auto = np.sqrt(mse_error)
                st.write(f"Racine de l'Erreur Quadratique Moyenne (RMSE) Test: {rmse_auto:.3f}")
                st.write(f"Erreur Absolu Moyenn (MAE) Test : {mae_error:.3f}")
                # prediction future
                #st.dataframe(future_predictions)
                df_future = pd.DataFrame({'Prévisions futures différentiées': future_predictions}, index=future_index)
                st.dataframe(df_future)

        else: 
            # La série est stationnaire, on peut l'utiliser directement
            st.write("Série stationaire => PAS besoin de différenciation")

            with st.spinner("Entraînement <<auto>> du modèle LSTM <<automatique non différentié>> en cours..."):
                # Préparation des données
                #print(df.info)
                df
                X = df['value']

                # Normalisation des données
                #RAPPEL les cellules LSTM utilise des fonctions d'activation sigmoid ou tanh
                #On standardise les données pour qu'elle aient une moyenne à 0 et un écart-type 1
                X = X.values
                X = X.astype('float32')
                X = X.reshape(-1, 1)
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                #séparer les données de test et les données d'entrainement
                #ATTENTION on ne génère pas les groupes de test et train de facon aléatoire (comme avec split de sklearn)
                train_size = int(len(X) * 0.80)
                train, test = X[0:train_size,:], X[train_size:len(X),:]

                #Transformer les données en séquences (reshape into X=t and Y=t+1)
                window_size = 6
                X_train, y_train = create_sequences(train, window_size)
                X_test, y_test = create_sequences(test, window_size)

                # Reshape input to be [samples, time steps, features]
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                """
                # Find best parameters using grid search
                # Création du modèle LSTM
                model = KerasRegressor(model=create_model, verbose=0)
                # Paramètres à tester
                param_grid = {'model__units': [4, 20, 30, 50],
                            'model__optimizer': ['adam', 'rmsprop'],
                            'epochs': [10, 50, 80, 100],
                            'batch_size': [1, 10, 16, 32,  64]
                            }
                # Grid search pour trouver les meilleurs paramètres
                regressor = KerasRegressor(model=create_model, verbose=0)
                grid = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=3)
                grid.fit(X_train, y_train)
                best_params = grid.best_params_
                best_score = grid.best_score_
                print(f"Best parameters: {best_params}")
                print(f"Best score: {best_score}")
                """

                # On utilise les meilleurs paramètres trouvés par la recherche en grille
                # On ne prend que les paramètres pertinents pour la création du modèle
                window_size = 6
                optimizer = 'rmsprop'
                units = 10
                epochs = 50
                batch_size = 1
                # Créer le modèle avec les meilleurs paramètres
                model = create_model(units=units, optimizer=optimizer)  
                # Entraînement du modèle
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)

                # Prédiction du train et du test
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                #print(f"Train predictions: {train_pred}")

                # Inverser la normalisation sur le train et le test
                train_pred = scaler.inverse_transform(train_pred)
                y_train = scaler.inverse_transform(y_train)
                test_pred = scaler.inverse_transform(test_pred)
                y_test = scaler.inverse_transform(y_test)
                #print(len(train))     #difference de 1O avec le suivant
                #print(len(train_pred))

                # Calculer l'erreur quadratique moyenne (MSE) et de l'erreur aboslue moyenne (MAE)
                mse_error = mean_squared_error(y_test, test_pred)
                mae_error = mean_absolute_error(y_test, test_pred)

                # Déplacemont des prédictions pour aligner avec les données originales
                train_pred_plot = np.empty_like(X)
                train_pred_plot[:, :] = np.nan
                train_pred_plot[window_size:window_size + len(train_pred), :] = train_pred
                test_pred_plot = np.empty_like(X)
                test_pred_plot[:, :] = np.nan
                test_pred_plot[train_size + window_size:train_size + window_size + len(test_pred), :] = test_pred

                # Prévision du futur
                # n_forecast = 24  # Nombre de valeurs futures à prédire
                n_forecast = int(len(df) * 0.20)  # 20% de la longueur de la série
                # Prendre la dernière séquence connue (taille = window_size)
                last_sequence = X[-window_size:]
                current_sequence = last_sequence.copy()
                future_predictions = []
                for _ in range(n_forecast):
                    # Reshape pour le modèle
                    input_seq = current_sequence.reshape((1, window_size, 1))
                    # Prédiction
                    next_pred = model.predict(input_seq, verbose=0)
                    # Stocker la prédiction
                    future_predictions.append(next_pred[0, 0])
                    # Mettre à jour la séquence
                    current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
                #print(f"Future predictions: {future_predictions}")
                # Inverser la normalisation
                future_predictions = np.array(future_predictions).reshape(-1, 1)
                future_predictions = scaler.inverse_transform(future_predictions).flatten()

                # Préparation des index pour future_predictions
                # Préparation des index pour future_predictions
                if isinstance(df.index, pd.DatetimeIndex):
                    last_date = df.index[-1]
                    freq = pd.infer_freq(df.index)
                    future_index = pd.date_range(start=last_date, periods=n_forecast+1, freq=freq)[1:]
                else:
                    future_index = np.arange(len(df), len(df) + n_forecast)

                # Affichage des résultats
                fig, ax = plt.subplots(figsize=(12, 5))
                #ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                ax.plot(df.index, scaler.inverse_transform(X), label='Données réelles', linewidth=1, linestyle="--", color='blue')
                ax.plot(df.index, train_pred_plot, label='Prédictions entrainements: (80% Données réelles)', linewidth=1, linestyle="--", color='orange')
                ax.plot(df.index, test_pred_plot, label='Prédictions tests: (20% Données réelles)', linewidth=1, linestyle="--", color='red')
                ax.plot(future_index, future_predictions, label='Prédictions futures: (20% Valeurs futures)', linewidth=1, linestyle="--", color='green')
                ax.set_title(f"Prédictions automatiques non différentiées avec LSTM")
                ax.legend()
                st.pyplot(fig)

                # Critères de performance
                st.write(f"Erreur Quadratique Moyenne (MSE) Test : {mse_error:.3f}")
                rmse_auto = np.sqrt(mse_error)
                st.write(f"Racine de l'Erreur Quadratique Moyenne (RMSE) Test: {rmse_auto:.3f}")
                st.write(f"Erreur Absolu Moyenn (MAE) Test : {mae_error:.3f}")
                # prediction future
                #st.dataframe(future_predictions)
                df_future = pd.DataFrame({'Prévisions futures non différentiées': future_predictions}, index=future_index)
                st.dataframe(df_future)
            





def prediction_manuel(df):
    "Prediction manuel avec un LSTM"
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
        
        #détermination des paramètres du modèle LSTM
        #window_size
        w_size = int(len(df) * 0.10)
        window_size = st.slider("Choisir la taille de la fenetre (window_size)", min_value=5, max_value=w_size, value=5, step=1)
        st.write(f"La taille de la fenetr choisi est : {window_size}")
        #units
        units = st.slider("Choisir le nombre d'unités par couche cachée (units)", min_value=4, max_value=100, value=4, step=2)
        st.write(f"Le nombre d'unités choisi est : {units}")
        #optimizer
        optimizer = st.selectbox("Sélectionner l'optimiseur", ("adam", "rmsprop", "sgd"))
        st.write(f"L'optimiseur choisi est : {optimizer}")
        #epochs
        epochs = st.slider("Choisir le nombre d'epochs (epochs)", min_value=10, max_value=1000, value=10, step=10)
        st.write(f"Le nombre d'epochs choisi est : {epochs}")
        #batch_size
        batch_size = st.slider("Choisir la taille du batch (batch_size)", min_value=1, max_value=128, value=1, step=1)
        st.write(f"La taille du batch choisi est : {batch_size}")
        #n_forecast     
        n_forecast = st.slider("Choisir le nombre de valeurs futures à prédire (n_forecast)", min_value=10, max_value=int(len(df) * 0.20), value=1, step=1)   
        st.write(f"Le nombre de valeurs futures à prédire est : {n_forecast}")

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

                with st.spinner("Entraînement <<manuel>> du modèle LSTM <<automatique différentié>> en cours..."):
                    # Préparation des données
                    #print(df.info)
                    df_diff
                    X = df_diff['value']

                    # Normalisation des données
                    #RAPPEL les cellules LSTM utilisent des fonctions d'activation sigmoid ou tanh
                    #On standardise les données pour qu'elle aient une moyenne à 0 et un écart-type 1
                    X = X.values
                    X = X.astype('float32')
                    X = X.reshape(-1, 1)
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)

                    #séparer les données de test et les données d'entrainement
                    #ATTENTION on ne génère pas les groupes de test et train de facon aléatoire (comme avec split de sklearn)
                    train_size = int(len(X) * 0.80)
                    train, test = X[0:train_size,:], X[train_size:len(X),:]

                    #Transformer les données en séquences (reshape into X=t and Y=t+1)
                    X_train, y_train = create_sequences(train, window_size)
                    X_test, y_test = create_sequences(test, window_size)

                    # Reshape input to be [samples, time steps, features]
                    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                    # On utilise les paramètres choisis par l'utilisateur pour la création du modèle
                    # Créer le modèle avec les meilleurs paramètres
                    model = create_model(units=units, optimizer=optimizer)  
                    # Entraînement du modèle
                    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)

                    # Prédiction
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    #print(f"Train predictions: {train_pred}")

                    # Inverser la normalisation du train et du test
                    train_pred = scaler.inverse_transform(train_pred)
                    y_train = scaler.inverse_transform(y_train)
                    test_pred = scaler.inverse_transform(test_pred)
                    y_test = scaler.inverse_transform(y_test)
                    #print(len(train))     #difference de 1O avec le suivant
                    #print(len(train_pred))

                    # Calculer l'erreur quadratique moyenne (MSE) et de l'erreur aboslue moyenne (MAE)
                    mse_error = mean_squared_error(y_test, test_pred)
                    mae_error = mean_absolute_error(y_test, test_pred)

                    # Inversion de la différenciation pour le train et le test
                    train_pred = inverse_differencing(train_pred, df['value'], window_size, n)
                    test_pred = inverse_differencing(test_pred, df['value'], train_size + window_size, n)

                    # Déplacemont des prédictions pour aligner avec les données originales
                    #préparation des arrays pour le graphique
                    train_pred_plot = np.full(len(df), np.nan)
                    test_pred_plot = np.full(len(df), np.nan)
                    #ajustement des index pour l'affichage
                    train_start_idx = window_size + n   # Index de début pour les prédictions d'entraînement
                    test_start_idx = train_size + window_size + n    # Index de début pour les prédictions de test
                    if train_start_idx + len(train_pred) <= len(df):
                        train_pred_plot[train_start_idx:train_start_idx + len(train_pred)] = train_pred

                    if test_start_idx + len(test_pred) <= len(df):
                        test_pred_plot[test_start_idx:test_start_idx + len(test_pred)] = test_pred

                    # Prévision du futur
                    #n_forecast = 24  # Nombre de valeurs futures à prédire
                    n_forecast = int(len(df) * 0.20)  # 20% de la longueur de la série
                    # Prendre la dernière séquence connue (taille = window_size)
                    last_sequence = X[-window_size:]
                    current_sequence = last_sequence.copy()
                    future_predictions = []
                    for _ in range(n_forecast):
                        # Reshape pour le modèle
                        input_seq = current_sequence.reshape((1, window_size, 1))
                        # Prédiction
                        next_pred = model.predict(input_seq, verbose=0)
                        # Stocker la prédiction
                        future_predictions.append(next_pred[0, 0])
                        # Mettre à jour la séquence
                        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
                    # Inverser la normalisation sur les prévisions futures
                    future_predictions = np.array(future_predictions).reshape(-1, 1)
                    future_predictions = scaler.inverse_transform(future_predictions).flatten()
                    # Inverser la différenciation pour les prévisions futures
                    future_predictions = inverse_differencing(future_predictions, df['value'], len(df), n)

                    # Préparation des index pour future_predictions
                    if isinstance(df.index, pd.DatetimeIndex):
                        last_date = df.index[-1]
                        freq = pd.infer_freq(df.index)
                        future_index = pd.date_range(start=last_date, periods=n_forecast+1, freq=freq)[1:]
                    else:
                        future_index = np.arange(len(df), len(df) + n_forecast)
                    
                    # Affichage des résultats
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df.index, df, label='Données réelles', linewidth=1, color='blue')
                    ax.plot(df.index, train_pred_plot, label='Prédictions entrainements: (80% Données réelles)', linewidth=1, linestyle="--", color='orange')
                    ax.plot(df.index, test_pred_plot, label='Prédictions tests: (20% Données réelles)', linewidth=1, linestyle="--", color='red')
                    ax.plot(future_index, future_predictions, label='Prédictions futures: (20% Valeurs futures)', linewidth=1, linestyle="--", color='green')
                    ax.set_title(f"Prédictions automatiques différentiées avec LSTM")
                    ax.legend()
                    st.pyplot(fig)

                    # Critères de performance
                    st.write(f"Erreur Quadratique Moyenne (MSE) Test : {mse_error:.3f}")
                    rmse_auto = np.sqrt(mse_error)
                    st.write(f"Racine de l'Erreur Quadratique Moyenne (RMSE) Test: {rmse_auto:.3f}")
                    st.write(f"Erreur Absolu Moyenn (MAE) Test : {mae_error:.3f}")
                    # prediction future
                    #st.dataframe(future_predictions)
                    df_future = pd.DataFrame({'Prévisions futures différentiées': future_predictions}, index=future_index)
                    st.dataframe(df_future)

            else: 
                # La série est stationnaire, on peut l'utiliser directement
                st.write("Série stationaire => PAS besoin de différenciation")

                with st.spinner("Entraînement <<manuel>> du modèle LSTM <<automatique non différentié>> en cours..."):
                    # Préparation des données
                    #print(df.info)
                    df
                    X = df['value']

                    # Normalisation des données
                    #RAPPEL les cellules LSTM utilise des fonctions d'activation sigmoid ou tanh
                    #On standardise les données pour qu'elle aient une moyenne à 0 et un écart-type 1
                    X = X.values
                    X = X.astype('float32')
                    X = X.reshape(-1, 1)
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)

                    #séparer les données de test et les données d'entrainement
                    #ATTENTION on ne génère pas les groupes de test et train de facon aléatoire (comme avec split de sklearn)
                    train_size = int(len(X) * 0.80)
                    train, test = X[0:train_size,:], X[train_size:len(X),:]

                    #Transformer les données en séquences (reshape into X=t and Y=t+1)
                    X_train, y_train = create_sequences(train, window_size)
                    X_test, y_test = create_sequences(test, window_size)

                    # Reshape input to be [samples, time steps, features]
                    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                    # On utilise les meilleurs paramètres entrés par l'utilisateur
                    # Créer le modèle avec les meilleurs paramètres
                    model = create_model(units=units, optimizer=optimizer)  
                    # Entraînement du modèle
                    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)

                    # Prédiction du train et du test
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    #print(f"Train predictions: {train_pred}")

                    # Inverser la normalisation sur le train et le test
                    train_pred = scaler.inverse_transform(train_pred)
                    y_train = scaler.inverse_transform(y_train)
                    test_pred = scaler.inverse_transform(test_pred)
                    y_test = scaler.inverse_transform(y_test)
                    #print(len(train))     #difference de 1O avec le suivant
                    #print(len(train_pred))

                    # Calculer l'erreur quadratique moyenne (MSE) et de l'erreur aboslue moyenne (MAE)
                    mse_error = mean_squared_error(y_test, test_pred)
                    mae_error = mean_absolute_error(y_test, test_pred)

                    # Déplacemont des prédictions pour aligner avec les données originales
                    train_pred_plot = np.empty_like(X)
                    train_pred_plot[:, :] = np.nan
                    train_pred_plot[window_size:window_size + len(train_pred), :] = train_pred
                    test_pred_plot = np.empty_like(X)
                    test_pred_plot[:, :] = np.nan
                    test_pred_plot[train_size + window_size:train_size + window_size + len(test_pred), :] = test_pred

                    # Prévision du futur
                    # n_forecast = 24  # Nombre de valeurs futures à prédire
                    n_forecast = int(len(df) * 0.20)  # 20% de la longueur de la série
                    # Prendre la dernière séquence connue (taille = window_size)
                    last_sequence = X[-window_size:]
                    current_sequence = last_sequence.copy()
                    future_predictions = []
                    for _ in range(n_forecast):
                        # Reshape pour le modèle
                        input_seq = current_sequence.reshape((1, window_size, 1))
                        # Prédiction
                        next_pred = model.predict(input_seq, verbose=0)
                        # Stocker la prédiction
                        future_predictions.append(next_pred[0, 0])
                        # Mettre à jour la séquence
                        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
                    #print(f"Future predictions: {future_predictions}")
                    # Inverser la normalisation
                    future_predictions = np.array(future_predictions).reshape(-1, 1)
                    future_predictions = scaler.inverse_transform(future_predictions).flatten()

                    # Préparation des index pour future_predictions
                    if isinstance(df.index, pd.DatetimeIndex):
                        last_date = df.index[-1]
                        freq = pd.infer_freq(df.index)
                        future_index = pd.date_range(start=last_date, periods=n_forecast+1, freq=freq)[1:]
                    else:
                        future_index = np.arange(len(df), len(df) + n_forecast)

                    # Affichage des résultats
                    fig, ax = plt.subplots(figsize=(12, 5))
                    #ax.plot(df['value'], label='Données réelles', linewidth=1, color='blue')
                    ax.plot(df.index, scaler.inverse_transform(X), label='Données réelles', linewidth=1, linestyle="--", color='blue')
                    ax.plot(df.index, train_pred_plot, label='Prédictions entrainements: (80% Données réelles)', linewidth=1, linestyle="--", color='orange')
                    ax.plot(df.index, test_pred_plot, label='Prédictions tests: (20% Données réelles)', linewidth=1, linestyle="--", color='red')
                    ax.plot(future_index, future_predictions, label='Prédictions futures: (20% Valeurs futures)', linewidth=1, linestyle="--", color='green')
                    ax.set_title(f"Prédictions automatiques non différentiées avec LSTM")
                    ax.legend()
                    st.pyplot(fig)

                    # Critères de performance
                    st.write(f"Erreur Quadratique Moyenne (MSE) Test : {mse_error:.3f}")
                    rmse_auto = np.sqrt(mse_error)
                    st.write(f"Racine de l'Erreur Quadratique Moyenne (RMSE) Test: {rmse_auto:.3f}")
                    st.write(f"Erreur Absolu Moyenn (MAE) Test : {mae_error:.3f}")
                    # prediction future
                    #st.dataframe(future_predictions)
                    df_future = pd.DataFrame({'Prévisions futures non différentiées': future_predictions}, index=future_index)
                    st.dataframe(df_future)
            


            



def lstm_model_plot(df, model):
    """Plot the LSTM model"""
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




