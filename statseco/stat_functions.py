# buils usual functions for stats-eco

#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import streamlit as st
import itertools
from scipy.stats import skew, kurtosis, jarque_bera, shapiro, normaltest, pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.stattools import durbin_watson

import warnings
warnings.filterwarnings('ignore')




def describe_series(serie):
    "Statistiques descriptives d'une série temporelle"
    jb, pval = jarque_bera(serie.dropna())   # jarque_bera renvoie deux valeurs
    stats = [
        serie.mean(),
        serie.median(),
        serie.min(),
        serie.max(),
        serie.std(),
        skew(serie),
        kurtosis(serie),      # par défaut excess kurtosis (= kurt - 3)
        jb,
        pval,
        serie.sum(),
        np.sum((serie - serie.mean())**2),
        serie.count()
    ]

    return stats





def normaliser_min_max(serie):
    """
    Normalise une série entre 0 et 1 avec la formule Min-Max
    Formule: (x - min) / (max - min)
    """
    serie_min = serie.min()
    serie_max = serie.max()
    
    # Éviter la division par zéro si toutes les valeurs sont identiques
    if serie_max == serie_min:
        return pd.Series([0.5] * len(serie), index=serie.index)
    
    serie_normalisee = (serie - serie_min) / (serie_max - serie_min)

    return serie_normalisee





def tester_normalite(serie, nom, alpha=0.05):
    """
    Fonction complète pour tester la normalité d'une variable (shapiro et Agostino-Pearson)
    Retourne la p-valeur et le resurlta du tes (True/False)
    """
    # Test de Shapiro-Wilk (recommandé pour n < 5000)
    if len(serie) <= 5000:
        st.write("Test de normalité de Shapiro-Wilk (n < 5000)")
        shapiro_stat, shapiro_p = shapiro(serie)

        if shapiro_p > alpha:
            result = True
        
        else:
            result = False

        return shapiro_stat, shapiro_p, result
    
    # Test de D'Agostino-Pearson (pour n> 5000)
    else:
        st.write("Test de normalité D'Agostino-Pearson (n > 5000)")
        dagostino_stat, dagostino_p = normaltest(serie)

        if dagostino_p > alpha:
            result = True
        
        else:
            result = False

        return dagostino_stat, dagostino_p, result




def test_correlation_pearson(serie1, serie2):
    """
    Calcule le coefficient de corrélation de Pearson et la p-value associée.
    Returns: tuple: (coefficient r, p-value)
    """
    r, pval = pearsonr(serie1.dropna(), serie2.dropna())

    return r, pval





def test_correlation_spearman(serie1, serie2):
    """
    Calcule le coefficient de corrélation de Spearman et la p-value associée.
    Args: tuple: (coefficient rho, p-value)
    """
    r_sp, pval = spearmanr(serie1.dropna(), serie2.dropna())

    return r_sp, pval






def granger_causality_analysis(serie1, serie2, max_lags=12):
    """
    Effectue un test de causalité de Granger bilatéral entre deux séries temporelles.
    Retourne les résultats pour chaque direction.
    """
    # Nettoyage : drop NA et aligner longueur
    serie1 = np.asarray(serie1).flatten()
    serie2 = np.asarray(serie2).flatten()

    # Result init
    best_pval = 1.0
    best_lag = 1
    best_pval_inv = 1.0
    best_lag_inv = 1
    causality_1to2 = False
    causality_2to1 = False

    # Test : serie1 → serie2
    try:
        data_granger = np.column_stack([serie2, serie1])  # Y, X (la cible en 1ère colonne)
        granger_results = grangercausalitytests(data_granger, maxlag=max_lags, verbose=False)
        best_lag = 1
        best_pval = 1.0

        for lag in range(1, min(max_lags + 1, len(granger_results) + 1)):
            if lag in granger_results:
                pval = granger_results[lag][0]['ssr_ftest'][1]
                if pval < best_pval:
                    best_pval = pval
                    best_lag = lag
        
        causality_1to2 = best_pval < 0.05
    
    except Exception as e:
        print("Erreur Granger 1to2 :", e)
        causality_1to2 = False

    # Test inverse : serie2 → serie1
    try:
        data_granger_inv = np.column_stack([serie1, serie2])
        granger_results_inv = grangercausalitytests(data_granger_inv, maxlag=max_lags, verbose=False)
        best_lag_inv = 1
        best_pval_inv = 1.0
        
        for lag in range(1, min(max_lags + 1, len(granger_results_inv) + 1)):
            if lag in granger_results_inv:
                pval = granger_results_inv[lag][0]['ssr_ftest'][1]
                if pval < best_pval_inv:
                    best_pval_inv = pval
                    best_lag_inv = lag
        
        causality_2to1 = best_pval_inv < 0.05
    
    except Exception as e:
        print("Erreur Granger 1to2 :", e)
        causality_2to1 = False

    return {
        'causality_1to2': causality_1to2,
        'pval_1to2': best_pval,
        'lag_1to2': best_lag,
        'causality_2to1': causality_2to1,
        'pval_2to1': best_pval_inv,
        'lag_2to1': best_lag_inv
    }





def model_var(df_1_2, maxlags=12, criterions=['aic', 'bic']):
    """
    Estime un modèle VAR, choisit le lag optimal, analyse les résultats et résidus.
    Retourne : - results : objet fit du modèle VAR sélectionné
    """

    # Sélection du lag optimal en fonction du critère (ex AIC)
    model = VAR(df_1_2)
    lag_order_results = model.select_order(maxlags)

    # Choix du lag optimal selon premier critère disponible
    for crit in criterions:
        selected_lag = getattr(lag_order_results, crit)
        if selected_lag is not None:
            break

    # Estimation du modèle VAR avec le lag choisi
    results = model.fit(selected_lag)

    # Durbin-Watson sur résidus :
    dw = durbin_watson(results.resid)
    dw_dict = {col: val for col, val in zip(df_1_2.columns, dw)}

    return {
        'lag_order_results': lag_order_results.summary().as_text(),
        'optimal_lag': selected_lag,
        'model_result': results.summary(),
        'model_result_raw': results,
        'model_residuals': results.resid,
        'durbin_watson': dw_dict
    }




def model_var_manuel(df_1_2, maxlags=12, criterion='aic'):
    """
    Estime un modèle VAR en choisissant automatiquement le lag optimal
    selon le critère choisi par l'utilisateur.
    
    Parameters:
        df_1_2 (DataFrame): Données time series multivariées
        maxlags (int): nombre maximum de retards à tester
        criterion (str): critère d'information à utiliser ('aic', 'bic', 'hqic', 'fpe')
    
    Returns:
        dict: résultats du modèle VAR et diagnostics
    """
    # 1. Sélection du lag optimal
    model = VAR(df_1_2)
    lag_order_results = model.select_order(maxlags)
    
    if not hasattr(lag_order_results, criterion):
        raise ValueError(f"Critère {criterion} non valide. Choisissez parmi: aic, bic, hqic, fpe.")
    
    selected_lag = getattr(lag_order_results, criterion)
    
    # 2. Estimation du modèle avec ce lag
    results = model.fit(selected_lag)
    
    # 3. Durbin-Watson sur chaque équation
    dw = durbin_watson(results.resid)
    dw_dict = {col: val for col, val in zip(df_1_2.columns, dw)}
    
    # 4. Retour des résultats
    return {
        'lag_order_results': lag_order_results.summary().as_text(),
        'optimal_lag': selected_lag,
        'criterion_used': criterion,
        'model_result': results.summary(),
        'model_result_raw': results,
        'model_residuals': results.resid,
        'durbin_watson': dw_dict
    }




def diagnostic_residus_var(results, df_columns):
    """Anlyse residuals:
        Q-Q plot
        ACF plot
    """

    resid = results['model_residuals']  # DataFrame résidus
    
    fig, axes = plt.subplots(len(df_columns), 3, figsize=(15, 4 * len(df_columns)))
    if len(df_columns) == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(df_columns):
        r = resid.iloc[:, i]
        
        # Histogramme
        axes[i,0].hist(r, bins=30, edgecolor='k')
        axes[i,0].set_title(f'Histogramme des résidus - {col}')
        
        # Q-Q plot
        sm.graphics.qqplot(r, line='s', ax=axes[i,1])
        axes[i,1].set_title(f'Q-Q plot (normalité) - {col}')
        
        # ACF plot
        plot_acf(r, ax=axes[i,2], lags=20)
        axes[i,2].set_title(f'Autocorrélation (ACF) des résidus - {col}')

        # Test de normalité Shapiro-Wilk
        alpha = 0.05
        stat, pvalue = stats.shapiro(r)
        st.write(f"Test Shapiro-Wilk résidus {col} : stat={stat}, pval={pvalue}")
        if  pvalue > alpha:
            st.success(f"✅ Le résidu de la colonne {col} de stat {stat} et de p-valeur {pvalue} suit une distribution Normale")

        else:
            st.warning(f"⚠️ Le résidu de la colonne {col} de stat {stat} et de p-valeur {pvalue} ne suit pas une distribution Normale")

    plt.tight_layout()
    st.pyplot(fig)




def test_homoscedasticite(residus, variables_exog=None):
    """
    Teste l'homoscédasticité des résidus avec le test de Breusch-Pagan.

    Arguments :
    - residus : pandas.Series ou DataFrame des résidus (ex : résultats.resid du VAR)
    - variables_exog : DataFrame ou array des variables explicatives pour l'équation auxiliaire (optionnel).
                       Si None, on utilisera une constante uniquement.

    Retour : - dict contenant la statistique du test et les p-values
    """

    # Si residus est DataFrame avec plusieurs colonnes (ex VAR résidus)
    if isinstance(residus, pd.DataFrame):
        results = {}

        for col in residus.columns:
            res = residus[col].dropna()
            if variables_exog is None:
                exog = sm.add_constant(pd.Series(range(len(res))))  # utilisation de l'index comme variable auxiliaire
            
            else:
                exog = variables_exog.loc[res.index]

            test = het_breuschpagan(res, exog)

            # test retourne : (LM_stat, LM_pvalue, F_stat, F_pvalue)
            results[col] = {
                'LM_stat': test[0],
                'LM_pvalue': test[1],
                'F_stat': test[2],
                'F_pvalue': test[3]
            }

        return results

    else:
        # residus est une Series ou array 1D
        res = residus.dropna() if hasattr(residus, 'dropna') else residus
        
        if variables_exog is None:
            exog = sm.add_constant(pd.Series(range(len(res))))  # index comme variable explicative
        
        else:
            exog = variables_exog.loc[res.index]

        test = het_breuschpagan(res, exog)
        
        return {
            'LM_stat': test[0],
            'LM_pvalue': test[1],
            'F_stat': test[2],
            'F_pvalue': test[3]
        }





def test_coint_granger(serie_1, serie_2, signif=0.05):
    """
    Réalise et interprète le test de cointégration d'Engle-Granger entre deux séries temporelles.
    Paramètres :
        - id_serie_1, id_serie_2 : noms ou identifiants des séries (affichage)
        - serie_1, serie_2 : pd.Series ou array-like des données (non différenciées, en niveau)
        - signif : seuil de signification (par défaut 5%)
    Retourne un dict avec les infos du test.
    """

    # Nettoyage : drop NA et aligner longueur
    serie_1 = np.asarray(serie_1).flatten()
    serie_2 = np.asarray(serie_2).flatten()

    # Régression en niveau (OLS)"
    X = sm.add_constant(serie_2)  # Régression de serie_1 sur serie_2
    model = sm.OLS(serie_1, X).fit()
    
    # Extraction des résidus et test de stationnarité (ADF) sur ces résidus
    resid = model.resid
    
    #Test de stationnarité (ADF) sur les résidus de la régression")
    res_adf_resid = adfuller(resid, maxlag=12, regression='c', autolag='AIC')
    stat_adf_resid, pval_adf_resid = res_adf_resid[0], res_adf_resid[1]
    
    cointegrated = False
    if pval_adf_resid < signif:
        cointegrated = True

    else:
        cointegrated = False
    
    #Résumé synthétique
    results = {
        'regression_summary': model.summary(),
        'residual_adf_stat': stat_adf_resid,
        'residual_adf_pval': pval_adf_resid,
        'cointegrated': cointegrated   
    }

    return results




def test_coint_granger_manuel(serie_1, serie_2, signif=0.05):
    """
    Réalise et interprète le test de cointégration d'Engle-Granger entre deux séries temporelles.
    Paramètres :
        - id_serie_1, id_serie_2 : noms ou identifiants des séries (affichage)
        - serie_1, serie_2 : pd.Series ou array-like des données (non différenciées, en niveau)
        - signif : seuil de signification (par défaut 5%)
    Retourne un dict avec les infos du test.
    """

    # Nettoyage : drop NA et aligner longueur
    serie_1 = np.asarray(serie_1).flatten()
    serie_2 = np.asarray(serie_2).flatten()

    # Régression en niveau (OLS)"
    X = sm.add_constant(serie_2)  # Régression de serie_1 sur serie_2
    model = sm.OLS(serie_1, X).fit()
    
    # Extraction des résidus et test de stationnarité (ADF) sur ces résidus
    resid = model.resid
    
    #Test de stationnarité (ADF) sur les résidus de la régression")
    user_maxlag_coint_grang = st.slider("Choisissez le lag maximum (coint-granger) :", min_value=1, max_value=24, value=12, step=1)
    st.write(f"La période d'étude (coint-granger) est : {user_maxlag_coint_grang}")
    user_trend_coint_grang = st.selectbox("Sélectionner la tendance du modèle (coint-granger):", ['c', 'ct', 'ctt', 'n'])
    st.write(f"La tendance sélectionnée est (coint-granger) : {user_trend_coint_grang}")
    user_crint_infor_coint_grang = st.selectbox("Sélectionner le critère d'information (coint-granger):", ['AIC', 'BIC', 't-stat'])
    st.write(f"Le critère d'information sélectionné est (coint-granger) : {user_crint_infor_coint_grang}")

    res_adf_resid = adfuller(resid, maxlag=user_maxlag_coint_grang, regression=user_trend_coint_grang, autolag=user_crint_infor_coint_grang)
    
    stat_adf_resid, pval_adf_resid = res_adf_resid[0], res_adf_resid[1]
    
    cointegrated = False
    if pval_adf_resid < signif:
        cointegrated = True

    else:
        cointegrated = False
    
    #Résumé synthétique
    results = {
        'regression_summary': model.summary(),
        'residual_adf_stat': stat_adf_resid,
        'residual_adf_pval': pval_adf_resid,
        'cointegrated': cointegrated   
    }

    return results





def determine_r(stats, crit):
    for i in range(len(stats)):
        if stats[i] < crit[i,1]:  # seuil 95%
            return i
        
    return len(stats)

def test_coint_johansen(serie_1, serie_2, det_order=0):
    """
    Réalise et interprète le test de cointégration de Johansen sur un DataFrame de séries I(1).
    Paramètres :
    - df : pd.DataFrame avec exactement 2 colonnes de séries temporelles en niveau (non différenciées)
    - det_order : spécifie la tendance dans le modèle Johansen
        -1 : pas de constante
         0 : constante sans tendance dans la cointégration
         1 : constante + tendance linéaire
    - k_ar_diff : nombre de différences retardées (lags) dans le modèle VAR du test (par défaut None = auto)
    - signif : seuil de signification (ex: 0.05)
    
    Affiche les résultats et retourne un dict résumant les conclusions.
    """
    #dtatframe commun (2 séries)
    df = pd.concat([serie_1, serie_2], axis=1)

    #vérification de la taille de al série
    if not isinstance(df, pd.DataFrame):
        st.error(" ⛔ Le paramètre df doit être un DataFrame pandas avec deux colonnes.")
        return
    
    if df.shape[1] != 2:
        st.error(" ⛔ Le DataFrame df doit contenir exactement deux colonnes.")
        return
    
    # Suppression des NA éventuels
    df_clean = df.dropna()
    
    #selection automatique de k_ar_diff
    # Estimation VAR sur données en niveau pour trouver le meilleur lag
    model_var = VAR(df_clean)
    lag_order_results = model_var.select_order(maxlags=12)  # maxlags selon ta préférence
    st.write(f"Le resultat du calcul du lag est:")
    st.write(lag_order_results)
    # Par exemple, choisir le lag minimisant l'AIC
    selected_lag = lag_order_results.aic
    # S’assurer que selected_lag est un entier
    if selected_lag is None:
        selected_lag = 1  # valeur par défaut raisonnable

    k_ar_diff = selected_lag
    st.write("La lag automatique est :", k_ar_diff)

    # Estimation du test Johansen
    result = coint_johansen(df_clean, det_order, k_ar_diff)

    #resultats du test de Joohansen
    st.write("Résumé rapide du test de Johansen")
    st.write("Statistiques Trace :", result.lr1)
    st.write("Valeurs critiques Trace (90%, 95%, 99%) :")
    st.write(result.cvt)
    st.write("Statistiques Valeur propre max :", result.lr2)
    st.write("Valeurs critiques Valeur propre max (90%, 95%, 99%) :")
    st.write(result.cvm)

    # Résultats clés
    trace_stat = result.lr1     # Statistiques trace
    crit_trace = result.cvt     # Valeurs critiques trace (90%, 95%, 99%)
    max_eig_stat = result.lr2   # Statistiques max eigenvalue
    crit_max_eig = result.cvm   # Valeurs critiques max eigenvalue
    
    k = df_clean.shape[1]

    st.write(f"Nombre de séries considérées dans le test de : {k}")
    st.write(f"Détermination du nombre de vecteurs de cointégration (0 <= r <= {k})")
    
    st.write("Test de trace")
    for i in range(k):
        st.write(f"r <= {i} : statistique = {trace_stat[i]} ; seuil 95% = {crit_trace[i,1]} ; "
                 f"{'⇒ Rejette H0' if trace_stat[i] > crit_trace[i,1] else '⇒ Ne rejette pas H0'}")

    st.write("Test de valeur propre maximale")
    for i in range(k):
        st.write(f"r <= {i} : statistique = {max_eig_stat[i]} ; seuil 95% = {crit_max_eig[i,1]} ; "
                 f"{'⇒ Rejette H0' if max_eig_stat[i] > crit_max_eig[i,1] else '⇒ Ne rejette pas H0'}")

    # Détermination nombre r de vecteurs de cointegration (plus petit r où H0 n'est pas rejetée)
    r_trace = determine_r(trace_stat, crit_trace)
    r_max_eig = determine_r(max_eig_stat, crit_max_eig)
    
    st.write(f"Nombre de vecteurs de cointégration estimé (trace test) : r = {r_trace}")
    st.write(f"Nombre de vecteurs de cointégration estimé (max eigenvalue test) : r = {r_max_eig}")

    cointegrated = (r_trace > 0 or r_max_eig > 0)
    
    return {
        "test_result": result,
        "num_cointegration_trace": r_trace,
        "num_cointegration_maxeig": r_max_eig,
        "cointegrated": cointegrated
    }





def test_coint_johansen_manuel(serie_1, serie_2):
    """
    Réalise et interprète le test de cointégration de Johansen sur un DataFrame de séries I(1).
    Paramètres :
    - df : pd.DataFrame avec exactement 2 colonnes de séries temporelles en niveau (non différenciées)
    - det_order : spécifie la tendance dans le modèle Johansen
        -1 : pas de constante
         0 : constante sans tendance dans la cointégration
         1 : constante + tendance linéaire
    - k_ar_diff : nombre de différences retardées (lags) dans le modèle VAR du test (par défaut None = auto)
    - signif : seuil de signification (ex: 0.05)
    
    Affiche les résultats et retourne un dict résumant les conclusions.
    """
    #dtatframe commun (2 séries)
    df = pd.concat([serie_1, serie_2], axis=1)

    #vérification de la taille de al série
    if not isinstance(df, pd.DataFrame):
        st.error(" ⛔ Le paramètre df doit être un DataFrame pandas avec deux colonnes.")
        return
    
    if df.shape[1] != 2:
        st.error(" ⛔ Le DataFrame df doit contenir exactement deux colonnes.")
        return
    
    # Suppression des NA éventuels
    df_clean = df.dropna()
    
    #selection automatique de k_ar_diff
    # Estimation VAR sur données en niveau pour trouver le meilleur lag
    model_var = VAR(df_clean)

    user_maxlag_coint_joh = st.slider("Choisissez le lag maximum (coint-joh) :", min_value=1, max_value=24, value=12, step=1)
    st.write(f"Le lag maximum (coint-granger) selectionné est : {user_maxlag_coint_joh}")
    user_trend_coint_joh = st.selectbox("Sélectionner la tendance du modèle (coint-joh):", ['c', 'ct', 'ctt', 'n'])
    st.write(f"La tendance sélectionnée est (coint-joh) : {user_trend_coint_joh}")

    lag_order_results = model_var.select_order(maxlags=user_maxlag_coint_joh, trend=user_trend_coint_joh)  # maxlags selon ta préférence

    st.write(f"Le resultat du calcul du lag est:")
    st.write(lag_order_results)

    # Par exemple, choisir le lag minimisant l'AIC
    selected_lag = lag_order_results.aic
    # S’assurer que selected_lag est un entier
    if selected_lag is None:
        selected_lag = 1  # valeur par défaut raisonnable

    k_ar_diff = selected_lag
    st.write("La lag automatique est :", k_ar_diff)

    # Estimation du test Johansen
    user_det_order = st.selectbox("Sélectionner la tendance du modèle johansen (coint-joh):", [0, -1, 1])
    st.write(f"La tendance sélectionnée est (coint-joh) : {user_det_order}")
    result = coint_johansen(df_clean, user_det_order, k_ar_diff)

    #resultats du test de Johansen
    st.write("Résumé rapide du test de Johansen")
    st.write("Statistiques Trace :", result.lr1)
    st.write("Valeurs critiques Trace (90%, 95%, 99%) :")
    st.write(result.cvt)
    st.write("Statistiques Valeur propre max :", result.lr2)
    st.write("Valeurs critiques Valeur propre max (90%, 95%, 99%) :")
    st.write(result.cvm)

    # Résultats clés
    trace_stat = result.lr1     # Statistiques trace
    crit_trace = result.cvt     # Valeurs critiques trace (90%, 95%, 99%)
    max_eig_stat = result.lr2   # Statistiques max eigenvalue
    crit_max_eig = result.cvm   # Valeurs critiques max eigenvalue
    
    k = df_clean.shape[1]

    st.write(f"Nombre de séries considérées dans le test de : {k}")
    st.write(f"Détermination du nombre de vecteurs de cointégration (0 <= r <= {k})")
    
    st.write("Test de trace")
    for i in range(k):
        st.write(f"r <= {i} : statistique = {trace_stat[i]} ; seuil 95% = {crit_trace[i,1]} ; "
                 f"{'⇒ Rejette H0' if trace_stat[i] > crit_trace[i,1] else '⇒ Ne rejette pas H0'}")

    st.write("Test de valeur propre maximale")
    for i in range(k):
        st.write(f"r <= {i} : statistique = {max_eig_stat[i]} ; seuil 95% = {crit_max_eig[i,1]} ; "
                 f"{'⇒ Rejette H0' if max_eig_stat[i] > crit_max_eig[i,1] else '⇒ Ne rejette pas H0'}")

    # Détermination nombre r de vecteurs de cointegration (plus petit r où H0 n'est pas rejetée)
    r_trace = determine_r(trace_stat, crit_trace)
    r_max_eig = determine_r(max_eig_stat, crit_max_eig)
    
    st.write(f"Nombre de vecteurs de cointégration estimé (trace test) : r = {r_trace}")
    st.write(f"Nombre de vecteurs de cointégration estimé (max eigenvalue test) : r = {r_max_eig}")

    cointegrated = (r_trace > 0 or r_max_eig > 0)
    
    return {
        "test_result": result,
        "num_cointegration_trace": r_trace,
        "num_cointegration_maxeig": r_max_eig,
        "cointegrated": cointegrated
    }

