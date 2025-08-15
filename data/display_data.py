# Import librairies
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

from pandas import DataFrame, concat
from numpy import asarray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
#from keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasRegressor




def infos_series(json_info):
    """Display series information"""
    col1, col2 = st.columns([1, 3])
    with col2:
        st.subheader("Informations de la série")

        seriess = json_info.get("seriess", [])

        if seriess:
            df = pd.DataFrame(seriess)
            st.write("Données des séries :")
            st.dataframe(df)
            for serie in seriess:
                st.subheader(f"Série : {serie['id']}")
                st.write(f"**Titre :** {serie['title']}")
                st.write(f"**Période :** {serie['observation_start']} à {serie['observation_end']}")
                st.write(f"**Fréquence :** {serie['frequency']}")
                st.write(f"**Unités :** {serie['units']}")
                st.write(f"**Ajustement saisonnier :** {serie['seasonal_adjustment']}")
                st.write(f"**Dernière mise à jour :** {serie['last_updated']}")
                st.write(f"**Popularité :** {serie['popularity']}")
                st.write(f"**Notes :** {serie['notes']}")
        else:
            st.warning("Aucune série trouvée dans le JSON.")


        

def data_date_displays(df):
    """Display data between a interval"""
    date_min = df.index.min().date()
    date_max = df.index.max().date()

    start_date = st.sidebar.date_input(
        "Sélectionnez la date de début",
        value=date_min,
        min_value=date_min,
        max_value=date_max
    )

    end_date = st.sidebar.date_input(
        "Sélectionnez la date de fin",
        value=date_max,
        min_value=date_min,
        max_value=date_max
    )

    if start_date > end_date:
        st.error("⚠️ La date de début ne peut pas être après la date de fin.")

    else:
        df_filtered = df.loc[start_date:end_date]
        col1, col2 = st.columns([1, 3])
        col1.subheader("Données filtrées")
        col1.write(df_filtered)
        col2.subheader("Graphique de la série temporelle filtrée")
        fig = px.line(df_filtered, x=df_filtered.index, y="value", title="Graphique filtré")
        col2.plotly_chart(fig)




def option_type(df, type):
    """Select the type of data to display"""
    optionType = st.sidebar.selectbox(
        "Sélectionnez une option d'affichage",
        ("Moyenne", "Somme", "Fin de période")
    )

    if optionType == "Moyenne":
        df_resampled = df.resample(type).mean()

    if optionType == "Somme":
        df_resampled = df.resample(type).sum()

    if optionType == "Fin de période":
        df_resampled = df.resample(type).last()

    col1, col2 = st.columns([1, 3])

    col1.subheader("Données filtrées")
    col1.write(df_resampled)

    col2.subheader("Graphique de la série temporelle filtrée")
    fig = px.line(df_resampled, x=df_resampled.index, y="value", title="Données filtrées")
    col2.plotly_chart(fig)




def data_freq_displays(df):
    """Display data by frequency"""
    freq = pd.infer_freq(df.index)
    fb = "a"

    if freq is None:
        st.warning("⚠️ Impossible de déterminer la fréquence de la série temporelle.")

    else:
        if freq == "D" or "D" in freq:
            st.sidebar.write("Fréquence de base des données par JOUR")
            option = st.sidebar.selectbox(
                "Sélectionnez une fréquence d'affichage",
                ("Mensuelle", "Trimestrielle", "Semestrielle", "Annuelle")
            )

            if option == "Mensuelle":
                type = "M"
                option_type(df, type)

            if option == "Trimestrielle":
                type = "Q"
                option_type(df, type)

            if option == "Semestrielle":
                type = "2Q"
                option_type(df, type)

            if option == "Annuelle": 
                type = "A"
                option_type(df, type)
        
        elif freq == "M" or "M" in freq:
            st.sidebar.write("Fréquence de base des données par MOIS")
            option = st.sidebar.selectbox(
                "Sélectionnez une fréquence d'affichage",
                ("Trimestrielle", "Semestrielle", "Annuelle")
            )

            if option == "Trimestrielle":
                type = "Q"
                option_type(df, type)

            elif option == "Semestrielle":
                type = "2Q"
                option_type(df, type)

            elif option == "Annuelle": 
                type = "A"
                option_type(df, type)

        elif freq == "Q" or "Q" in freq:
            st.sidebar.write("Fréquence de base des données par TRIMESTRE")
            option = st.sidebar.selectbox(
                "Sélectionnez une fréquence d'affichage",
                ("Semestrielle", "Annuelle")
            )

            if option == "Semestrielle":
                type = "2Q"
                option_type(df, type)

            if option == "Annuelle":
                type = "A"
                option_type(df, type)

        elif freq == "A" or "A" in freq:
            st.sidebar.write("Fréquence de base des données par AN")

            col1, col2 = st.columns([1, 3])

            col1.subheader("Données annuelles")
            col1.write(df)

            col2.subheader("Graphique de la série temporelle")
            fig = px.line(df, x=df.index, y="value", title="Données annuelles")
            col2.plotly_chart(fig)
        
        else:
            st.warning("⚠️ Impossible de déterminer la fréquence de la série temporelle.")

            col1, col2 = st.columns([1, 3])

            col1.subheader("Données")
            col1.write(df)

            col2.subheader("Graphique de la série temporelle")
            fig = px.line(df, x=df.index, y="value", title="Données")
            col2.plotly_chart(fig)
        



def data_recession_displays(df):
    """Display recession data"""
    recession_periods = [
        ("1973-11-01", "1975-03-01"),
        ("1980-01-01", "1980-07-01"),
        ("1981-06-01", "1982-11-01"),
        ("1990-07-01", "1991-03-01"),
        ("2001-03-01", "2001-11-01"),
        ("2007-12-01", "2009-06-01"),
        ("2020-02-01", "2020-04-01")
    ]
    
    if df.index.name == 'date':
        df.reset_index(inplace=True)

    df['date'] = pd.to_datetime(df['date'])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['value'], mode='lines'))

    for start, end in recession_periods:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="blue", opacity=0.3,
            layer="below", line_width=0,
        )

    fig.update_layout(
        title="Série temporelle avec périodes de récession",
        xaxis_title="date",
        yaxis_title="value",
        legend_title="Légende",
        template="plotly_white"
    )

    st.plotly_chart(fig)



def add_decomposition(df):
    """Additional Seasonal decomposition of time series"""
    st.write("Additive Xt = Tt + St + Ɛt (Variation avec amplitude constante)")
    if 'value' not in df.columns:
        st.error("❌ La colonne 'value' est absente du DataFrame.")
    else:
        decomposition = seasonal_decompose(df['value'], model='additive', period=None)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        axs[0].plot(df['value'], label='Série originale')
        axs[0].legend(loc='best')
        axs[1].plot(trend, label='Tendance', color='orange')
        axs[1].legend(loc='best')
        axs[2].plot(seasonal, label='Saisonnalité', color='green')
        axs[2].legend(loc='best')
        axs[3].plot(residual, label='Résidus', color='red')
        axs[3].legend(loc='best')
        plt.tight_layout()
        st.pyplot(fig)



def mul_decomposition(df):
    """Miltiplicative Seasonal decomposition of time series"""
    st.subheader("Décomposition Multiplicative")
    st.write("Multiplicative Xt = Tt x St x Ɛt (Variation avec amplitude croissante au cours du temps)")
    if 'value' not in df.columns:
        st.error("❌ La colonne 'value' est absente du DataFrame.")
    elif (df['value'] <= 0).any():
        st.error("❌ La série contient des valeurs nulles ou négatives. La décomposition multiplicative n'est pas possible.")
    else:
        decomposition = seasonal_decompose(df['value'], model='multiplicative', period=None)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        axs[0].plot(df['value'], label='Série originale')
        axs[0].legend(loc='best')
        axs[1].plot(trend, label='Tendance', color='orange')
        axs[1].legend(loc='best')
        axs[2].plot(seasonal, label='Saisonnalité', color='green')
        axs[2].legend(loc='best')
        axs[3].plot(residual, label='Résidus', color='red')
        axs[3].legend(loc='best')
        plt.tight_layout()
        st.pyplot(fig)



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



def stationaity(df):
    """Stationnarity test"""
    if not test_stationarity(df['value']):
        st.warning("⚠️ La série n'est pas stationnaire. Essayez n différenciation(s) pour la rendre stationnaire.")
        # La série n'est pas stationnaire, on effectue n différenciation(s)
        n = 0
        df_diff = df.copy()
        while not test_stationarity(df_diff['value']):
            df_diff = df_diff.diff().dropna()
            n += 1
            print(n)

        st.write(f"Série non stationaire => {n} différenciation(s) effectuée(s)")
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            fig = px.line(df_diff, x=df_diff.index, y='value', title='Evolution de la serie temporelle différenciée')
            st.plotly_chart(fig)
        
    else:
        st.success("✅ La série est stationnaire.")
        st.write("Aucune différenciation n'est nécessaire.")



def calcul_nbr_lags(df):
    """Calculate and return the number of lags (min and max)"""
    min_lags = 1
    max_lags = max(40, int(len(df) * 0.20))
    return min_lags, max_lags



def acf_graph(df):
    """Autocorrelation Function (ACF) graph"""
    min, max = calcul_nbr_lags(df)  #en 1 et 20% de la série
    lag_acf = st.slider("Nombre de lags (acf) à afficher", min_value=min, max_value=max, value=min, step=1)
    if 'value' not in df.columns:
        st.error("❌ La colonne 'value' est absente du DataFrame.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_acf(df['value'], lags=lag_acf, ax=ax)
        ax.set_title('Fonction d\'Autocorrélation (ACF)')
        st.pyplot(fig)
    



def pacf_graph(df):
    """Partial Autocorrelation Function (PACF) graph"""
    min, max = calcul_nbr_lags(df)  #en 1 et 20% de la série
    lag_pacf = st.slider("Nombre de lags (pacf) à afficher", min_value=min, max_value=max, value=min, step=1)
    if 'value' not in df.columns:
        st.error("❌ La colonne 'value' est absente du DataFrame.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_pacf(df['value'], lags=lag_pacf, ax=ax)
        ax.set_title('Fonction d\'Autocorrélation Partielle (PACF)')
        st.pyplot(fig)
  



def train_test_split(data, n_test):
    "split a univariate dataset into train/test sets"
    return data[:-n_test, :], data[-n_test:, :]




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




def bestParamRf(X_train, y_train):
    """Find the best parameters for Random Forest model"""
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




def create_sequences(data, window_size):
    """Convert an array of values into a dataset matrix"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)




def create_model(units=50, optimizer='adam', window_size=6, num_features=1):
    "Create LSTM model"
    model = Sequential()
    model.add(LSTM(units, activation='relu', return_sequences=True, input_shape=(window_size, num_features)))
    model.add(LSTM(units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.summary()
    return model





def bestParamLstm(df):
    """Search best parameters for LSTM model"""
    # Taille de la fenêtre pour les séquences
    window_size = 6     #2e test avec 10

    # Préparation des données
    df
    X = df['value']

    # Normalisation des données
    #RAPPEL les cellule LSTM utilise des fonctions d'activation sigmoid ou tanh
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
    #window_size = 10
    X_train, y_train = create_sequences(train, window_size)
    X_test, y_test = create_sequences(test, window_size)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

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
    return best_params, best_score






def display_series(df, id_serie, json_info):
    """Display series data"""
    if st.sidebar.checkbox("Afficher les informations de la série"):
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            st.subheader(f"Informations de la série {id_serie}")
            infos_series(json_info)


    if st.sidebar.checkbox("Afficher le graphique de la série"):
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            fig = px.line(df, x=df.index, y='value', title='Evolution de la serie temporelle')
            st.plotly_chart(fig)


    if st.sidebar.checkbox("Afficher les 5 premières lignes"):
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            st.subheader(f"Voici les 5 premières lignes de la série temporelle {id_serie}")
            st.write(df.head())


    if st.sidebar.checkbox("Afficher les 5 dernières lignes"):
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            st.subheader(f"Voici les 5 dernières lignes de la série {id_serie}")
            st.write(df.tail())


    if st.sidebar.checkbox("Afficher toutes les données"):
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            st.subheader(f"Données entières de la série {id_serie}")
            st.write(df)


    if st.sidebar.checkbox("Afficher les statistiques de la série"):
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            st.subheader(f"Statistiques globales de la série {id_serie}")
            st.write(df.describe())


    if st.sidebar.checkbox("Afficher les données dans une intervalle"):
        data_date_displays(df)
    

    if st.sidebar.checkbox("Afficher la série par fréquence"):
        data_freq_displays(df)


    if st.sidebar.checkbox("Afficher les périodes de recession"):
        data_recession_displays(df)


    if st.sidebar.checkbox("Afficher la décomposition additive"):
        add_decomposition(df)
    

    if st.sidebar.checkbox("Afficher la décomposition multiplicative"):
        mul_decomposition(df)
    

    if st.sidebar.checkbox("Test de stationnarité"):
        stationaity(df)

    """
    if st.sidebar.checkbox("Graphe ACF sans différenciation"):
        acf_graph(df)

    if st.sidebar.checkbox("Graphe PACF sans différenciation"):
        pacf_graph(df)
    """


    if st.sidebar.checkbox("Graphe ACF/PACF sans différenciation"):
        acf_graph(df)
        pacf_graph(df)


    if st.sidebar.checkbox("Graphe ACF/PACF avec différenciation"):
        if not test_stationarity(df['value']):
            st.warning("⚠️ La série n'est pas stationnaire. Essayez n différenciation(s) pour la rendre stationnaire.")
            # La série n'est pas stationnaire, on effectue n différenciation(s)
            n = 0
            df_diff = df.copy()
            while not test_stationarity(df_diff):
                df_diff = df_diff.diff().dropna()
                n += 1

            st.write(f"Série non stationaire => {n} différenciations effectuée(s)")
            acf_graph(df_diff)
            pacf_graph(df_diff)

        else:
            st.success("✅ La série est stationnaire.")
            st.write("Aucune différenciation n'est nécessaire.")



    if st.sidebar.checkbox("Rechercher les meilleurs paramètres du modèle Random Forest"):
        if not test_stationarity(df['value']):
            st.warning("⚠️ La série n'est pas stationnaire. Essayez n différenciation(s) pour la rendre stationnaire.")
            # La série n'est pas stationnaire, on effectue n différenciation(s)
            n = 0
            df_diff = df.copy()

            while not test_stationarity(df_diff):
                df_diff = df_diff.diff().dropna()
                n += 1

            st.write(f"Série non stationaire => {n} différenciation(s) effectuée(s)")

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
            #test_X, testy = testSet[:, :-1], testSet[:, -1:]

            with st.spinner("Recherche des meilleurs paramètres en cours"):
                best_params = bestParamRf(train_X, trainy)
                st.write(f"Meilleurs paramètres pour le modèle Random Forest (différentié) : {best_params}")

        else:
            st.success("✅ La série est stationnaire.")
            st.write("Aucune différenciation n'est nécessaire.")

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
            #test_X, testy = testSet[:, :-1], testSet[:, -1:]
            
            with st.spinner("Recherche des meilleurs paramètres en cours"):
                best_params = bestParamRf(train_X, trainy)
                st.write(f"Meilleurs paramètres pour le modèle Random Forest (non différencié) : {best_params}")



    if st.sidebar.checkbox("Rechercher des meilleurs paramètres du modèle LSTM"):
        if not test_stationarity(df['value']):
            st.warning("⚠️ La série n'est pas stationnaire. Essayez n différenciation(s) pour la rendre stationnaire.")
            # La série n'est pas stationnaire, on effectue n différenciation(s)
            n = 0
            df_diff = df.copy()

            while not test_stationarity(df_diff):
                df_diff = df_diff.diff().dropna()
                n += 1

            st.write(f"Série non stationaire => {n} différenciation(s) effectuée(s)")

            with st.spinner("Recherche des meilleurs paramètres en cours"):
                best_scrore, best_params = bestParamLstm(df_diff)
                st.write(f"Meilleur score pour le modèle LSTM (différencié) : {best_scrore}")
                st.write(f"Meilleurs paramètres pour le modèleLSTM (différentié) : {best_params}")

        else:
            st.success("✅ La série est stationnaire.")
            st.write("Aucune différenciation n'est nécessaire.")
            
            with st.spinner("Recherche des meilleurs paramètres en cours"):
                best_scrore, best_params = bestParamLstm(df)
                st.write(f"Meilleur score pour le modèle LSTM (non différencié) : {best_scrore}")
                st.write(f"Meilleurs paramètres pour le modèle LSTM (non différencié) : {best_params}")
                

    print("Fin")