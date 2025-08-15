#analyse bivarié 1

# Import necessary libraries
import gc
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ardl import ardl_select_order
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import normaltest
from scipy.stats import shapiro
from statsmodels.tsa.ardl import ARDL, UECM


#import local files
from stat_functions import describe_series, tester_normalite, test_correlation_pearson, test_correlation_spearman, granger_causality_analysis, model_var, model_var_manuel, diagnostic_residus_var, test_homoscedasticite, test_coint_granger, test_coint_granger_manuel, test_coint_johansen, test_coint_johansen_manuel




def qq_plot_simple(data, titre):
    """Méthode simple un qqplot avec scipy pour Streamlit"""
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    stats.probplot(data, dist="norm", plot=ax4)
    ax4.set_title(titre)
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)
    plt.close(fig4)




def interpretationDeLaNormalite(stat, pval, result, id_serie):
    if result:
        st.success(f"✅ La série {id_serie} de stat {stat} et de p-valeur {pval} suit une distribution Normale")
        
    else:
        st.warning(f"⚠️ La série {id_serie} de stat {stat} et de p-valeur {pval} ne suit pas une distribution Normale")
        



def interpretationPvalPearsonandSpearman(pval, id_serie_1, id_serie_2):
    if pval < 0.05:
        st.warning(f"⚠️ Correlation significative => Hypothèse nulle acceptée => pas de correlation entre {id_serie_1} et {id_serie_2}")

    else:
         st.success(f"✅Correlation PAS significative => Hypothèse nullle rejetée => Correlation entre {id_serie_1} et {id_serie_2}")
       



def interpretationRouRhoPearsonandSpearman(r, id_serie_1, id_serie_2):
    if r == 0:
        st.warning(f"⚠️ Aucune correlation linéaire entre {id_serie_1} et {id_serie_2}")

    elif r > 0:
        st.success(f"✅ Correlation linéaire positive entre {id_serie_1} et {id_serie_2}")

    else:
        st.success(f"✅ Correlation linéaire négative entre {id_serie_1} et {id_serie_2}")

    abs_r = abs(r)
    if abs_r < 0.2:
        st.warning(f"⚠️ Correlation linéaire négligeable entre {id_serie_1} et {id_serie_2}")
        
    elif 0.2 <= abs_r < 0.4:
        st.warning(f"⚠️ Correlation linéaire faible entre {id_serie_1} et {id_serie_2}")
            
    elif 0.4 <= abs_r < 0.7:
        st.warning(f"⚠️  Correlation linéaire modérée entre {id_serie_1} et {id_serie_2}")
            
    elif 0.7 <= abs_r < 0.9:
        st.success(f"✅ Correlation linéaire forte entre {id_serie_1} et {id_serie_2}")
        
    elif 0.9 <= abs_r < 1.0:
        st.success(f"✅ Correlation linéaire très forte entre {id_serie_1} et {id_serie_2}")

    elif abs_r == 1.0 or abs_r == -1.0:
        st.success(f"✅ Correlation linéaire parfaite entre {id_serie_1} et {id_serie_2}")
        
    else:
        st.warning(f"⚠️ Valeur de {abs_r} hors intervalle attendu")





def interpretationGranger(results_granger, id_serie_1, id_serie_2):
    if results_granger['causality_1to2'] and results_granger['causality_2to1']:
        st.success(f"✅ ✅ Causalité Bidirectionnelle entre {id_serie_1} et {id_serie_2}")

    elif results_granger['causality_1to2']:
        st.success(f"✅ Causalité Unidirectionnelle de {id_serie_1} vers {id_serie_2}")

    elif results_granger['causality_2to1']:
        st.success(f"✅ Causalité Unidirectionnelle de {id_serie_2} vers {id_serie_1}")

    else:
        st.warning(f"⚠️ Aucune causalité détectée entre {id_serie_1} et {id_serie_2}")




def interpretation_test_coint_granger(results_conit_granger, id_serie_1, id_serie_2):
    st.write(f"Résultats du modèle de regression OLS entre {id_serie_1} (y) et {id_serie_2} (x):")
    st.write(results_conit_granger['regression_summary'])
    #st.write(results_conit_granger["regression_summary".as_text()])
    st.write(f"Statistique ADF sur les résidus : {results_conit_granger['residual_adf_stat']}")
    st.write(f"P-valeur ADF sur les résidus : {results_conit_granger['residual_adf_pval']}")
    
    if results_conit_granger['cointegrated']:
        st.success(f"✅ Résidus stationnaires : les séries sont cointégrées (relation long terme confirmée).")

    else:
        st.warning("⚠️ Résidus non stationnaires : pas de preuve de cointégration (pas de relation long terme stable).")




def interpretation_test_johansen(results_johansen):
    
    if results_johansen['cointegrated']:
        st.success(f"✅ Les séries semblent cointégrées (relation stable à long terme détectée).")
    
    else:
        st.warning(f"⚠️ Pas de preuve robuste de cointégration entre les séries.")






def analyse_auto(df_1, df_2, id_serie_1, id_serie_2):
    "analyse bivariée auto"
    if 'compute_auto' not in st.session_state:
        st.session_state.compute_auto = False

    if st.sidebar.button("Computer"):
        st.session_state.compute_auto = True

    if st.session_state.compute_auto:
        #on vérifie que les 2 DataFrame ont été chargés et contienent la colonne 'value'
        if df_1 is None :
            st.write(" ⚠️ Erreur : le DataFrame 1 n'a pas été chargé.")
            return

        if 'value' not in df_1.columns:
            st.write(" ⚠️ Erreur : la colonne 'value' est absente du DataFrame 1.")
            return
        
        if df_2 is None :
            st.write(" ⚠️ Erreur : le DataFrame 2 n'a pas été chargé.")
            return

        if 'value' not in df_2.columns:
            st.write(" ⚠️ Erreur : la colonne 'value' est absente du DataFrame 2.")
            return

        # Supposons df_1 et df_2 ont "date" comme index
        df_1.reset_index(inplace=True)
        df_2.reset_index(inplace=True)

        # Assurez que la colonne 'date' soit bien en datetime64
        df_1['date'] = pd.to_datetime(df_1['date'])
        df_2['date'] = pd.to_datetime(df_2['date'])

        # Renommage les colonnes 
        df_1.rename(columns={'value': id_serie_1}, inplace=True)
        df_2.rename(columns={'value': id_serie_2}, inplace=True)

        #Merger les 2 séries sur la "date"
        df_1_2 = pd.merge(df_1, df_2, on='date', how='inner')

        #on vérifie que les 2 séries ont le même nombre d'observations
        print(f"Série 1 : {len(df_1)} observations")
        print(f"Série 2: {len(df_2)} observations") 
        print(f"Nombre d'observations total: {len(df_1_2)} observations")

        #Afficher les 2 séries (head, date debut et fin)
        print("Affichage (head, tail, date min et date max) des 2 séries:")
        print(df_1_2.head())
        print(df_1_2.tail())
        print(f"Date de début: {df_1_2['date'].min()}")
        print(f"Date de fin: {df_1_2['date'].max()}")


        #STEP 1: AFFICHAGE DES 2 SERIES (graphique (simple et normalisée) et nuage de points)
        #ATTENTION Se RAPPELER qui est x et qui est y.  (nuage des points)
        #Pour nous x = taux de chomage (serie2) et y = taux d'inflation (serie1)
        st.subheader("STEP 1 - Graphique et nuage des points des 2 séries temporelles")
        st.write("STEP 1 - 1 - Graphique simple des 2 séries temporelles")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(df_1_2['date'], df_1_2[id_serie_1], label=id_serie_1, color='blue')
        ax1.plot(df_1_2['date'], df_1_2[id_serie_2], label=id_serie_2, color='orange')
        ax1.set_title(f"Courbe des 2 séries temporelles {id_serie_1} et {id_serie_2}")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Valeurs numériques")
        ax1.legend()
        st.pyplot(fig1)

        st.write("STEP 1 - 2 - Graphique normalisée (Min-Max) des 2 séries temporelles")
        # Normalisation Min-Max
        df_1_2['norm_' + id_serie_1] = (df_1_2[id_serie_1] - df_1_2[id_serie_1].min()) / (df_1_2[id_serie_1].max() - df_1_2[id_serie_1].min())
        df_1_2['norm_' + id_serie_2] = (df_1_2[id_serie_2] - df_1_2[id_serie_2].min()) / (df_1_2[id_serie_2].max() - df_1_2[id_serie_2].min())
        # Affichage du graphique normalisé
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(df_1_2['date'], df_1_2['norm_' + id_serie_1], label=f'{id_serie_1} (normalisé)', color='blue')
        ax2.plot(df_1_2['date'], df_1_2['norm_' + id_serie_2], label=f'{id_serie_2} (normalisé)', color='orange')
        ax2.set_title(f"Courbes normalisées de {id_serie_1} et {id_serie_2}")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Valeurs numériques normalisées")
        ax2.legend()
        st.pyplot(fig2)

        st.write(f"STEP 1 - 3 - Nuage des points des 2 séries temporelles (format image) entre {id_serie_1} et {id_serie_2}")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.scatter(df_1_2[id_serie_2], df_1_2[id_serie_1], color='blue', alpha=0.7)
        ax3.set_title(f"Nuage de points {id_serie_1} vs {id_serie_2}")
        ax3.set_xlabel(f'{id_serie_2}')
        ax3.set_ylabel(f'{id_serie_1}')
        ax3.grid(True)
        st.pyplot(fig3)

        st.write(f"Nuage des points des 2 séries temporelles (format streamlit) entre {id_serie_1} et {id_serie_2}")
        fig4 = px.scatter(
            df_1_2, 
            x=id_serie_2,
            y=id_serie_1,  
            title=f"Nuage des points des 2 séries temporelles (format streamlit) entre {id_serie_1} et {id_serie_2}",
            labels={'id2': f'{id_serie_2}', 'id1': f'{id_serie_1}'},
            hover_data=['date']  # Afficher la date au survol
        )
        st.plotly_chart(fig4, use_container_width=True)




        #STEP 2: STATTISTIQUES DESCRIPTIVES DES 2 SERIES
        st.subheader("STEP 2 - Statistiques descriptives des 2 séries temporelles")
        #recupérer chaque série individuellement
        serie1 = df_1_2[['date', id_serie_1]]
        serie2 = df_1_2[['date', id_serie_2]]

        #définir les différents vvariables statistiques à calculer
        stat_names = [
            "Moyenne",
            "Médiane",
            "Minimum",
            "Maximum",
            "Std.Dev",
            "Skewness",
            "Kurtosis",
            "Jarque-Bera",
            "Probability",
            "Somme des observations",
            "Somme carrée des écarts à la moyenne",
            "Nombre d'observations"
        ]

        #définit le tableau
        tab_stats = pd.DataFrame({
            "Statistiques": stat_names,
            id_serie_1: describe_series(serie1[id_serie_1]),
            id_serie_2: describe_series(serie2[id_serie_2])
        })
        tab_stats = tab_stats.applymap(lambda x: f"{x}" if isinstance(x, float) else x)
        tab_stats.set_index("Statistiques", inplace=True)

        # affichage dans streamlit
        st.dataframe(tab_stats)




        #STEP 3: CORRELATION ET TESTS D'INDEPENDANCE
        st.subheader("STEP 3 - Corrélation et tests d'indépendance")

        #CORRELATION
        st.write(f"STEP 3 - 1 - Corrélation entre {id_serie_1} et {id_serie_2}")
        #Calcul de la covariance et de la correlation
        covariance = serie1[id_serie_1].cov(serie2[id_serie_2])
        correlation = serie1[id_serie_1].corr(serie2[id_serie_2])
        #affiche des valeurs
        st.write(f"La covariance entre {id_serie_1} et {id_serie_2}  est : {covariance}")
        st.write(f"Le coefficient de correlation entre {id_serie_1} et  {id_serie_2}  est : {correlation}")

        #interprétation 1
        if correlation == 0:
            st.warning(f"⚠️ La corrélation linéaire entre {id_serie_1} et {id_serie_2} observée est nulle")
        
        elif correlation == -1 or correlation == 1:
            st.success(f"✅ La corrélation linéaire entre {id_serie_1} et {id_serie_2} observée est parfaite")
        
        elif correlation > 0:
            st.success(f"✅ La liaison  entre {id_serie_1} et {id_serie_2} est positive")
        
        elif correlation < 0:
            st.warning(f"⚠️ La liaison entre {id_serie_1} et {id_serie_2} est négative")
        
        #interprétation 2
        if abs(correlation) < 0.1:
            st.warning(f"⚠️ La correlation linéaire entre {id_serie_1} et {id_serie_2} est Très faible")

        elif abs(correlation) < 0.3:
            st.warning(f"⚠️ La correlation linéaire entre {id_serie_1} et {id_serie_2} est Faible")

        elif abs(correlation) < 0.5:
            st.warning(f"⚠️ La correlation linéaire entre {id_serie_1} et {id_serie_2} est Modérée")

        elif abs(correlation) < 0.7:
            st.success(f"✅ La correlation linéaire entre {id_serie_1} et {id_serie_2} est Forte")

        else:
            st.success(f"✅ La correlation linéaire entre {id_serie_1} et {id_serie_2} est Très forte")

        #TEST DE NORMALITE
        st.write(f"STEP 3 - 2 - Normalité des séries temporelles {id_serie_1} et {id_serie_2}")
        #pas besoin de normalisé car il ne s'applique que sur chaque variable
        #serie1
        stat1, pval1, result1 = tester_normalite(serie1[id_serie_1], id_serie_1)
        interpretationDeLaNormalite(stat1, pval1, result1, id_serie_1)
        #serie2
        stat2, pval2, result2 = tester_normalite(serie2[id_serie_2], id_serie_2)
        interpretationDeLaNormalite(stat2, pval2, result2, id_serie_2)

        #QQPLOT DES 2 SERIES
        st.write(f" STEP 3 - 3 - Confirmation de la normalité (ou pas) avec le qqplot de  {id_serie_1} et de {id_serie_2}")
        #serie1
        qqplot1 = st.checkbox(f"Afficher le QQ-plot de la série {id_serie_1}")
        if qqplot1:
            titre_qq_1 = f"Q-Q Plot de {id_serie_1}"
            qq_plot_simple(serie1[id_serie_1], titre_qq_1)

        #serie2
        qqplot2 = st.checkbox(f"Afficher le QQ-plot de la série {id_serie_2}")
        if qqplot2:
            titre_qq_2 = f"Q-Q Plot de {id_serie_2}"
            qq_plot_simple(serie2[id_serie_2], titre_qq_2)
        
        #TEST D'INDEPENDANCE (de Pearson ou de Spearman)
        #ATTENTION les unités sont différentes, MAIS PAS BESOIN de les normaliser 
        # POURQUOI? Spearman fonctionne sur les rangs et Pearson est une mesure adimensionnelle (sans unité)
        st.write(f" STEP 3 - 4 - Test d'indépendance entre {id_serie_1} et de {id_serie_2}")
        if result1 and result2:
            st.write(f"Les 2 séries suivent une distribution Normale => Test de Pearson")
            r_p, pval_p = test_correlation_pearson(serie1[id_serie_1], serie2[id_serie_2])
            st.write(f"Le coefficient de correlation de Pearson est de r = {r_p}")
            interpretationRouRhoPearsonandSpearman(r_p, id_serie_1, id_serie_2)
            st.write(f"Le p-valeur de Pearson entre {id_serie_1} et {id_serie_2} est de p-valeur = {pval_p}")
            interpretationPvalPearsonandSpearman(pval_p, id_serie_1, id_serie_2)
            
        else:
            st.write(f"Une ou les 2 séries ne suivent/suit pas une distribution Normale => Test de Spearman")
            r_sp, pval_sp = test_correlation_spearman(serie1[id_serie_1], serie2[id_serie_2])
            st.write(f"Le coefficient de correlation de Spearman est de rho = {r_sp}")
            interpretationRouRhoPearsonandSpearman(r_sp, id_serie_1, id_serie_2)
            st.write(f"Le p-valeur de Spearman entre {id_serie_1} et {id_serie_2} est de p-valeur = {pval_sp}")
            interpretationPvalPearsonandSpearman(pval_sp, id_serie_1, id_serie_2)
        


        #STEP 4: TEST DE STATIONNARITÉ
        st.subheader("STEP 4 - Test de stationnarité des séries temporelles")
        
        #on tolère maximum 1 différentiation pour éviter de tourner à l'infini
        MAX_DIFF = 10

        #Test de Stationnarité pour la série 1
        serie1_diff = df_1.copy()
        n1 = 0
        adf_ok1 = False
        while n1 <= MAX_DIFF:
            res1 = adfuller(serie1_diff[id_serie_1].dropna())
            pval1 = res1[1]
            print(res1)
            print(serie1_diff)
            if pval1 < 0.05:
                if n1 == 0:
                    st.success(f"✅ La série {id_serie_1} (p-valeur {pval1}) est stationnaire de base.")
                else:
                    st.success(f"✅ La série {id_serie_1} est stationnaire après {n1} différenciation(s) (p-valeur {pval1}).")
                adf_ok1 = True
                break
            else:
                if n1 == 0:
                    st.warning(f"⚠️ La série {id_serie_1} n'est pas stationnaire, on différencie...")
                serie1_diff = serie1_diff.diff().dropna()
                n1 += 1
                print(n1)
        
        if not adf_ok1:
            st.error(f"⛔ La série {id_serie_1} n'est toujours pas stationnaire après {MAX_DIFF} différenciations !")
        

        #Test de stationnairté pour la série 2
        serie2_diff = df_2.copy()
        n2 = 0
        adf_ok2 = False
        while n2 <= MAX_DIFF:
            res2 = adfuller(serie2_diff[id_serie_2].dropna())
            pval2 = res2[1]
            if pval2 < 0.05:
                if n2 == 0:
                    st.success(f"✅ La série {id_serie_2} (p-valeur {pval2}) est stationnaire de base.")
                else:
                    st.success(f"✅ La série {id_serie_2} est stationnaire après {n2} différenciation(s) (p-valeur {pval2}).")
                adf_ok2 = True
                break
            else:
                if n2 == 0:
                    st.warning(f"⚠️ La série {id_serie_2} n'est pas stationnaire, on différencie...")
                serie2_diff = serie2_diff.diff().dropna()
                n2 += 1
        
        if not adf_ok2:
            st.error(f"⛔ La série {id_serie_2} n'est toujours pas stationnaire après {MAX_DIFF} différenciations !")
        
        
        #ATTENTION => RAPPEL : Pour le RAPPORT, faire le résumé des tests de stationnarité sous forme de tableau



        #STEP 5 TEST DE CO-INTÉGRATION ET MODELES
        #ATTENTION pour le rapport, Se RAPPELER qui est x et qui est y
        #Pour nous x = taux de chomage (serie2) et y = taux d'inflation (serie1)
        st.subheader("STEP 5 - Test(s) de co-intégration et/ou Estimation de(s) modèle(s)")

        #ATTENTION: Toujours vérifier le type de données utilisé en entrée => IDENTIQUE EN TYPE (surtout cas I(1) et I(0))

        #ATTENTION le test de cointégration ne marche pas avec des séries > I(2)

        ######CAS I(2) et I(0) ou I(1) ou encore I(2)######
        if n1 > 1 or n2 >1 :
            st.error(f"⛔ Une des séries est différentiée PLUS D'UNE FOIS => Oupsssssssss!!!!!!!!!!!")
            return


        ######CAS I(0) et I(0)######
        elif n1 == 0 and n2 == 0:
            st.info(f"Cas ou la série 1 {id_serie_1} => I(0) et la série 2 {id_serie_2} => I(0).")

            #Rappel (Test de stationnarité)
            st.info(f"STEP 5 - 0 - (RAPPEL) Test de stationnarité de {id_serie_1} et de {id_serie_2}")
            st.success(f"✅ La série {id_serie_1} est stationnaire de base.")
            st.success(f"✅ La série {id_serie_2} est stationnaire de base.")

            #Relation à COURT TERME
            #Test(s) de corrélation (Pearson ou Spearman)
            st.info(f"STEP 5 - 1 - Test de corrélation (Pearson ou Spearman) entre {id_serie_1} et de {id_serie_2}")
            st.write("Voire les résultats ci-dessus => (STEP 3 - 4)")

            #Test de causalité de Granger
            st.info(f"STEP 5 - 2 - Test de causalité de Granger entre {id_serie_1} et de {id_serie_2}")
            results_granger = granger_causality_analysis(df_1_2[id_serie_1], df_1_2[id_serie_2])
            st.write(results_granger)
            interpretationGranger(results_granger, id_serie_1, id_serie_2)

            #Modèle VAR
            st.info(f"STEP 5 - 3 - Modèle VAR entre {id_serie_1} et de {id_serie_2}")
            #ATTENTION les données doivent etre dans un meme dataframe
            results_var = model_var(df_1_2[[id_serie_1, id_serie_2]])
            st.write("Résultats du modèle VAR :")
            st.write(results_var)   #c'est l'ensemble des résultat du modèle VAR

            #ATTENTION interpretation des résulatats d'analyse
            model_result_raw = results_var['model_result_raw']  # objet VARResults
            params = model_result_raw.params  # DataFrame coefficients
            pvalues = model_result_raw.pvalues  # DataFrame p-values
            for eq_name in params.columns:
                st.write(f"Équation pour variable dépendante : {eq_name}")
                for var_name in params.index:
                    coef = params.loc[var_name, eq_name]
                    pval = pvalues.loc[var_name, eq_name]

                    if var_name == 'const':
                        st.write(f"=>Terme constant : coef={coef}, p-value={pval}")
                    
                    elif pval < 0.05:
                        signe = 'positif' if coef > 0 else 'négatif'
                        st.write(f"=> ✅ Effet {signe} et significatif de '{var_name}' sur '{eq_name}' (coef={coef}, p-value={pval}).")
                    
                    elif pval < 0.10:
                        st.write(f"=> ⚠️ Effet faiblement significatif de '{var_name}' sur '{eq_name}' (coef={coef}, p-value={pval}).")
                    
                    else:
                        st.write(f"=> ⛔ Pas d’effet significatif de '{var_name}' sur '{eq_name}' (p-value={pval}).")

            st.info(f"STEP 5 - 4 - Analyse des résidus du modèle VAR")
            #Moyenne et écart-type des résidus
            st.write(f"Résidus - Moyennes: {results_var['model_residuals'].mean().values}")
            st.write(f"Résidus - Écarts-types: {results_var['model_residuals'].std().values}")
            #Graphiques des résidus + test de Normalité
            resid_var_plot = st.checkbox(f"Afficher les graphiques (histo, qq-plot, pacf) des résidus")
            if resid_var_plot:
                last_two_cols = df_1_2.columns[-2:]
                diagnostic_residus_var(results_var, last_two_cols)
            
            #Homoscédasticité des résidus
            st.write("Tests d'homoscédasticité de Breusch-Pagan :")
            bp_test_results = test_homoscedasticite(results_var['model_residuals'])
            for var, testres in bp_test_results.items():
                st.write(f"Variable {var} : Breusch-Pagan LM stat={testres['LM_stat']}, p-value={testres['LM_pvalue']}")
                if testres['LM_pvalue'] < 0.05:
                    st.warning(f"⚠️ Rejet de l'hypothèse d'homoscédasticité (Variance non constante)")
                
                else:
                    st.success(f"✅ Pas de preuve suffisante contre l'homoscédasticité (Variance constante)")
            
            #Test de durbin_watson
            st.write("Test de durbin_watson :")
            dw_dict_var = results_var['durbin_watson']
            for var, dw_val in dw_dict_var.items():
                if dw_val < 1.5:
                    st.warning(f"⚠️ Résidus de {var} présentent une autocorrélation positive (DW={dw_val})")
                
                elif dw_val > 2.5:
                    st.warning(f"⚠️ Résidus de {var} présentent une autocorrélation négative (DW={dw_val})")
                
                else:
                    st.success(f"✅ Résidus de {var} sans autocorrélation significative (DW={dw_val})")
            #ATTENTION => Peut-etre essayer une regression lineaire 
            #Relation à LONG TERME
            #La cointégration s’intéresse essentiellement à des séries non stationnaires (I(1))



        ######CAS I(1) et I(0)######
        elif n1 == 1 and n2 == 0:
            st.info(f"Cas ou la série {id_serie_1} => I(1) et la série {id_serie_2} => I(0).")

            #Rappel (Test de stationnarité)
            st.info(f"STEP 5 - 0 (RAPPEL) - Test de stationnarité de {id_serie_1} et de {id_serie_2}")
            st.warning(f"⚠️ La série {id_serie_1} n'est pas stationnaire, mais I(1).")
            st.success(f"✅ La série {id_serie_2} est stationnaire de base.")

            #Sélection automatique ordre optimal ARDL. (RAPPEL => Y=id_serie_1 et X=id_serie_2)
            st.info("STEP 5 - 1 - Sélection automatique ordre optimal ARDL")
            model_selection = ardl_select_order(
                endog=df_1_2[id_serie_1],   #y
                maxlag=12,
                exog=df_1_2[[id_serie_2]],   #les x (d'ou le double crochet => ici, on a un seul) 
                maxorder=12,     
                ic='aic',
                trend="c",
                seasonal=True,
                period=12
            )
            st.write(f"L'ordre de modèle ARDL (p,q) optimal :", model_selection.model.ardl_order)
            st.write(f"Le décalage autoregressif (AR) :", model_selection.model.ar_lags)
            st.write(f"Le décalage de la variable exogène :", model_selection.model.dl_lags)
            st.write("Résultats du modèle ADRL optimal :")
            st.text(model_selection.model.fit().summary().as_text())            
            
            #Bounds Test (Pesaran et al.) (RAPPEL => Y=id_serie_1 et X=id_serie_2)
            st.info("STEP 5 - 2 - Bounds Test (Pesaran et al.) + interpretation")
            order = model_selection.model.ardl_order
            if isinstance(order, tuple) and len(order) == 1:
                st.error(f"⛔ ATTENTION: il manque 1 des lag exigés pour la suite.")
                return

            elif isinstance(order, (tuple, list)) and len(order) == 2:
                lag_p, lag_q = order

            else:
                st.error(f"⛔ Pas d'ordre retourné dans la sélection automatique.")  
                return    

            st.write(f"Lags endogènes automatiques => {lag_p} & lags exogènes automatiques => {lag_q}")
            model_ardl = ARDL(
                endog=df_1_2[id_serie_1],     #y
                lags=lag_p,
                exog=df_1_2[[id_serie_2]],    #les x (d'ou le double crochet => ici, on a un seul) 
                order=lag_q,                  #ce sont les lags des x
                trend="c",                   #voir la doc pour les # cas possibles
                seasonal=True,
                period=12
            ).fit()
            uecm_model = UECM.from_ardl(model_ardl.model)
            uecm_results = uecm_model.fit()
            st.text("Résultats du modèle test de Pesaran :")
            st.text("Modèle ADRL:")
            st.text(model_ardl.summary().as_text())
            st.text("Modèle UECM (dérivé de ADRL):")
            st.text(uecm_results.summary().as_text())
            st.text("Modèle UECM sur la co-intégration:")
            st.text(uecm_results.ci_summary().as_text())
            #_ = uecm_model.fit().ci_resids.plot(title="Erreur de cointégration")
            #uecm_results.ci_resids.plot(title="Erreur de cointégration")
            st.write("Graphique de l'erreur de cointégration :")
            fig6 = uecm_results.ci_resids.plot(title="Erreur de cointégration")
            st.pyplot(fig6.figure)
            st.write("Interpretation du test de Pesaran :")
            #trend est  "c"    =>>>>> case = 3
            bounds_results = uecm_results.bounds_test(case=3, cov_type='nonrobust', use_t=True, asymptotic=True)
            #st.write(bounds_results.crit_vals)
            stat_F = bounds_results.stat                #Statistique F
            st.write(f"La statistique F est de :")
            st.write(stat_F)
            crit_bounds = bounds_results.crit_vals         #Bornes critiques
            st.write(f"Les bornes critiques du test sont :")
            st.write(crit_bounds)
            for level in [90.0, 95.0, 99.0]:
                lower = crit_bounds.loc[level, 'lower']
                upper = crit_bounds.loc[level, 'upper']
                st.write(f"Bornes au niveau {int(level)}% : inférieure = {lower}, supérieure = {upper}")
            # Récupérer les bornes critiques à 5%
            lower_bound_5pct = crit_bounds.loc[95.0, 'lower']
            upper_bound_5pct = crit_bounds.loc[95.0, 'upper']
            st.write(f"La borne inférieure à 5% est : {lower_bound_5pct}")
            st.write(f"La borne supérieure à 5% est : {upper_bound_5pct}")
            co_integration = False
            if stat_F > upper_bound_5pct:            #on rejette l’hypothèse nulle d’absence de cointégration, donc il y a cointégration
                st.write(f"La statistique F ({stat_F}) est supérieure à la borne supérieure ({upper_bound_5pct})")
                st.success(f"✅ Cointégration détectée (relation de long terme) entre {id_serie_1} et {id_serie_2}")
                co_integration = True

            elif stat_F < lower_bound_5pct:           #on ne rejette pas l’hypothèse nulle, donc pas de cointégration.
                st.write(f"La statistique F ({stat_F}) est inférieure à la borne inférieure ({upper_bound_5pct})")  
                st.warning(f"⚠️ Pas de cointégration entre {id_serie_1} et {id_serie_2}")
                co_integration = False
            
            else:
                st.error(f"⛔ Test non conclusif (zone indéterminée car {lower_bound_5pct} < {stat_F} < {upper_bound_5pct})")
                co_integration = False
                #return
            
            #ATTENTION (1er cas => si co-inégration)
            if co_integration:
                st.info("STEP 5 - 3 - Modélisation UECM complète + interpretation (si cointégration)")
                #Modélisation ARDL/ECM : effet de long terme et vitesse d’ajustement
                # Coef vitesse d'ajustement (coefficient sur y.L1)
                speed_adj = uecm_results.params.get(f"{id_serie_1}.L1", None)
                if speed_adj is not None:
                    st.success(f"✅ Vitesse d’ajustement (coefficient sur {id_serie_1}.L1): {speed_adj}")
                    if speed_adj < 0:
                        st.success(f"✅ Cela signifie que la variable corrigera efficacement les déséquilibres à long terme")
                    
                    else:
                        st.warning("⚠️ Attention, la vitesse d'ajustement est positive, ce qui peut indiquer un problème")
                
                else:
                    st.warning("⚠️ Coefficient vitesse d'ajustement non trouvé.")

                # Effet long terme = - coef exogène niveau / coef vitesse d'ajustement
                x_level = uecm_results.params.get(f"{id_serie_2}.L0") or uecm_results.params.get(f"D.{id_serie_2}.L0")
                if x_level is not None and speed_adj is not None and speed_adj != 0:
                    long_term_effect = -x_level / speed_adj
                    pval_x = uecm_results.pvalues.get(f"{id_serie_2}.L0") or uecm_results.pvalues.get(f"D.{id_serie_2}.L0")
                    st.write(f"L'effet de long terme estimé de {id_serie_2} sur {id_serie_1} est {long_term_effect}")
                    if pval_x is not None and pval_x < 0.05:
                        st.success(f"✅ Cet effet est statistiquement significatif")
                    
                    else:
                        st.warning("⚠️ Cet effet n'est pas significatif")
                
                else:
                    st.error(f"⛔ Impossible de calculer l'effet de long terme (manque de coefficients ou vitesse nulle).")

                # Analyse des résidus
                st.info("STEP 5 - 4 - Analyse des résidus du modèle UECM complet)")
                if stat_F > crit_bounds.iloc[1,1]:
                    resid_adrl_ucem = uecm_results.resid

                else:
                    resid_adrl_ucem = uecm_results.resid

                # Graphiques des résidus + Normalité
                resid_ardl_ucem_plot = st.checkbox(f"Afficher les graphiques (histo, qq-plot, pacf) des résidus")
                if resid_ardl_ucem_plot:
                    # Plots diagnostics résidus
                    fig, axes_ucem = plt.subplots(1, 3, figsize=(15, 4))
                    axes_ucem[0].hist(resid_adrl_ucem, bins=30, edgecolor='k')
                    axes_ucem[0].set_title('Histogramme des résidus')
                    sm.qqplot(resid_adrl_ucem, line='s', ax=axes_ucem[1])
                    axes_ucem[1].set_title('QQ-plot des résidus')
                    sm.graphics.tsa.plot_acf(resid_adrl_ucem, lags=20, ax=axes_ucem[2])
                    axes_ucem[2].set_title('ACF des résidus')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Diagnostic normalité (Shapiro-Wilk)
                adrl_ucem_stat, adrl_ucem_pval = shapiro(resid_adrl_ucem)
                alpha_ucem = 0.05
                if adrl_ucem_pval > alpha_ucem:
                    st.success(f"✅ Test Shapiro-Wilk: le résidu de stat {adrl_ucem_stat} et de p-valeur {adrl_ucem_pval} suit une distribution Normale")

                else:
                    st.warning(f"⚠️ Test Shapiro-Wilk: le résidu de stat {adrl_ucem_stat} et de p-valeur {adrl_ucem_pval} ne suit pas une distribution Normale")

                # Autocorrélation (Durbin-Watson)
                dw_val_adrl_ucem = durbin_watson(resid_adrl_ucem)
                st.write(f"Valeur de Durbin-Watson : {dw_val_adrl_ucem}")
                if dw_val_adrl_ucem < 1.5:
                    st.warning(f"⚠️ Présomption d'autocorrélation positive des résidus (DW < 1.5).")

                elif dw_val_adrl_ucem > 2.5:
                    st.warning(f"⚠️ Présomption d'autocorrélation négative des résidus (DW > 2.5).")

                else:
                    st.success(f"✅ Pas d'autocorrélation significative des résidus (DW ≈ 2).")

                # Hétéroscédasticité (Breusch-Pagan)
                exog = uecm_results.model.exog
                if exog.shape[1] < 2:
                    st.warning("⚠️ Le test de Breusch-Pagan requiert au moins deux variables explicatives, dont une constante. Test non effectué ici.")
                
                else:
                    bp_stat_adrl_ucem, bp_pval_adrl_ucem, _, _ = het_breuschpagan(resid_adrl_ucem, uecm_results.model.exog)
                    st.write(f"Test de Breusch-Pagan de de stat : {bp_stat_adrl_ucem} et de p-valeur : {bp_pval_adrl_ucem}")
                    if bp_pval_adrl_ucem < 0.05:
                        st.warning(f"⚠️ Rejet de l'hypothèse d'homoscédasticité (hétéroscédasticité détectée).")
                    
                    else:
                        st.success(f"✅ Pas de preuve d'hétéroscédasticité (variance des résidus constante).")

                    # Durbin-Watson + Breusch-Pagan
                    if (adrl_ucem_pval > 0.05 and 1.5 <= dw_val_adrl_ucem <= 2.5 and bp_pval_adrl_ucem > 0.05):
                        st.success("✅ Modèle sans défaut structurel majeur des résidus : diagnostics OK.")


            #ATTENTION (2e cas => pas de co-intégration)
            else:
                st.info("STEP 5 - 3 - Pas de modelisation UECM complète (pas de cointégration).")
                #on va estimer un modèle ADRL en différences ()
                st.info(f"Modelisation ADRL sur les différences (.diff) entre {id_serie_1} et {id_serie_2}")
                #création de données différentiés
                df_1_2_copy = df_1_2.copy()
                df_1_2_copy[id_serie_1] = df_1_2_copy[id_serie_1].diff()
                df_1_2_copy[id_serie_2] = df_1_2_copy[id_serie_2].diff()
                df_1_2_copy_diff = df_1_2_copy.dropna(subset=[id_serie_1, id_serie_2])
                #sélection automatique différentié des ordres p et q
                model_selection_diff = ardl_select_order(
                    endog=df_1_2_copy_diff[id_serie_1],   #y
                    maxlag=12,
                    exog=df_1_2_copy_diff[[id_serie_2]],   #les x (d'ou le double crochet => ici, on a un seul) 
                    maxorder=12,     
                    ic='aic',
                    trend="c",
                    seasonal=True,
                    period=12
                )          
                order_diff = model_selection_diff.model.ardl_order
                if isinstance(order_diff, tuple) and len(order_diff) == 1:
                    st.error(f"⛔ ATTENTION: il manque 1 des lag exigé pour la suite.")
                    return

                elif isinstance(order_diff, (tuple, list)) and len(order_diff) == 2:
                    lag_p_diff, lag_q_diff = order_diff

                else:
                    st.error(f"⛔ Pas de la retourné dans la sélection automatique.")  
                    return    

                st.write(f"Lags différentiés endogènes: {lag_p_diff} & lags différentiés exogènes: {lag_q_diff}")
                #modelisation ADRL et estimation
                model_ardl_diff = ARDL(
                    endog=df_1_2_copy_diff[id_serie_1],     #y
                    lags=lag_p_diff,
                    exog=df_1_2_copy_diff[[id_serie_2]],    #les x (d'ou le double crochet => ici, on a un seul) 
                    order=lag_q_diff,                  #ce sont les lags des x
                    trend="c",                   #voir la doc pour les # cas possibles
                    seasonal=True,
                    period=12
                )
                results_model_ardl_diff = model_ardl_diff.fit()
                st.text("Résultats du modèle ADRL en différences :")
                st.text(results_model_ardl_diff.summary().as_text())
                # Interpretation du résultat du modèle
                st.write("Aucune relation de long terme détectée. Interprétation des effets à court terme du modèle ADRL en différences")
                for param, pval in results_model_ardl_diff.pvalues.items():
                    if pval < 0.05:
                        coef_val = results_model_ardl_diff.params[param]
                        sign = "positif" if coef_val > 0 else "négatif"
                        st.write(f"Effet significatif de {param} avec coefficient {coef_val} ({sign})")

                # Analyse des résidus
                st.info("STEP 5 - 4 - Analyse des résidus du modèle ADRL sur les différences")
                resid_diff = results_model_ardl_diff.resid
                #affichage des graphiques
                if st.checkbox("Afficher graphiques diagnostics (histogramme, qq-plot, ACF) des résidus"):
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                    # Histogramme des résidus
                    axes[0].hist(resid_diff, bins=30, edgecolor='k')
                    axes[0].set_title("Histogramme des résidus")
                    # QQ plot
                    sm.qqplot(resid_diff, line='s', ax=axes[1])
                    axes[1].set_title("QQ-plot des résidus")
                    # Autocorrélation (ACF)
                    sm.graphics.tsa.plot_acf(resid_diff, lags=20, ax=axes[2])
                    axes[2].set_title("Fonction d'autocorrélation (ACF)")
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Test de normalité des résidus (Shapiro-Wilk)
                stat_shapiro_diff, pval_shapiro_diff = shapiro(resid_diff)
                alpha_diff = 0.05
                if pval_shapiro_diff > alpha_diff:
                    st.success(f"✅ Test Shapiro-Wilk: les résidus de stat {stat_shapiro_diff} et de p-valeur {pval_shapiro_diff} suivent une distribution Normale")
                
                else:
                    st.warning(f"⚠️ Test Shapiro-Wilk: les résidus de stat {stat_shapiro_diff} et de p-valeur {pval_shapiro_diff} ne suivent pas une distribution Normale")
               
                # Test d'autocorrélation des résidus (Durbin-Watson)
                dw_stat_difff = durbin_watson(resid_diff)
                st.write(f"Valeur de Durbin-Watson : {dw_stat_difff}")
                if dw_stat_difff < 1.5:
                    st.warning("⚠️ Possible autocorrélation positive des résidus (DW < 1.5).")

                elif dw_stat_difff > 2.5:
                    st.warning("⚠️ Possible autocorrélation négative des résidus (DW > 2.5).")

                else:
                    st.success("✅ Résidus sans autocorrélation significative (DW proche de 2).")

                # Test d'hétéroscédasticité (Breusch-Pagan)
                exog_diff = results_model_ardl_diff.model.exog
                # Vérification que matrice exog est valide pour le test
                if exog_diff.shape[1] < 2:
                    st.warning("⚠️ Le test de Breusch-Pagan requiert au moins deux variables explicatives, dont une constante. Test non effectué ici.")
                
                else:
                    bp_stat_diff, bp_pval_diff, _, _ = het_breuschpagan(resid_diff, exog_diff)
                    st.write(f"Test Breusch-Pagan (hétéroscédasticité) de stat = {bp_stat_diff} et de p-valeur = {bp_pval_diff}")

                    if bp_pval_diff < alpha_diff:
                        st.warning("⚠️ Hétéroscédasticité détectée (rejet homoscédasticité).")

                    else:
                        st.success("✅ Pas d'hétéroscédasticité détectée (variance des résidus constante).")

                    # Bilan global des diagnostics
                    if pval_shapiro_diff > alpha_diff and 1.5 <= dw_stat_difff <= 2.5 and (exog_diff.shape[1] < 2 or bp_pval_diff > alpha_diff):
                        st.success("✅ Diagnostics résiduels OK : modèle sans défauts majeurs détectés.")
                    
                    else:
                        st.info("⚠️ Attention : certains diagnostics suggèrent des problèmes sur les résidus.")





        ######CAS I(0) et I(1)######
        elif n1 == 0 and n2 == 1:
            st.write(f"Cas ou la série {id_serie_1} => I(0) et la série {id_serie_2} => I(1).")
            st.write("C'est l'inverse du cas I(1) et I(0), Merci d'inverser l'odre des ID des séries")





        ######CAS I(1) et I(1)######
        elif n1 == 1 and n2 == 1:
            st.info(f"Cas ou la série {id_serie_1} => I(1) et la série {id_serie_2} => I(1).")

            #Rappel (Test de stationnarité)
            st.info(f"STEP 5 - 0 (RAPPEL) - Test de stationnarité de {id_serie_1} et de {id_serie_2}")
            st.warning(f"⚠️ La série {id_serie_1} n'est pas stationnaire, mais I(1).")
            st.warning(f"⚠️ La série {id_serie_2} n'est pas stationnaire, mais I(1).")

            #Test de co-intégration d’Engle-Granger
            st.info(f"#STEP 5 - 1 - Test de cointégration d'Engle-Granger entre `{id_serie_1}` et `{id_serie_2}`")
            results_coint_Granger = test_coint_granger(df_1_2[id_serie_1], df_1_2[id_serie_2])
            interpretation_test_coint_granger(results_coint_Granger, id_serie_1, id_serie_2)

            #Test de cointégaration (Johansen)
            st.info(f"#STEP 5 - 1 - Test de cointégration de Johansen entre `{id_serie_1}` et `{id_serie_2}`")
            results_johansen = test_coint_johansen(df_1_2[id_serie_1], df_1_2[id_serie_2])
            interpretation_test_johansen(results_johansen)

            #ATTENTION: ICI ON CHOISI JOHANSEN CAR IL EST PLUS PERFORMANT PAR RAPPORT A engler
            #ATTENTION (1er cas => co-intégration)
            if results_johansen['cointegrated']:
                st.info(f"STEP 5 - 2 - Modèle VECM `{id_serie_1}` et `{id_serie_2}`")
                df_vecm = df_1_2[[id_serie_1, id_serie_2]].dropna()
                # Choix automatique du nombre de lags :
                order_result_vecm = select_order(df_vecm, maxlags=12, deterministic="co")
                st.write("Le resultat du calcul du lag est :")
                st.write(order_result_vecm)
                lags_auto_vecm = order_result_vecm.selected_orders['aic']
                if lags_auto_vecm is None:
                    lags_auto_vecm = 1              # sécurité si indéterminé
                    st.write(f"Le lag fixé est : {lags_auto_vecm}")
                
                else:
                    st.write(f"Le lag déterminé automatiquement est : {lags_auto_vecm}")

                # Déterminer le rang de cointégration (déjà déterminé avec Johansen)
                rank_result = select_coint_rank(df_vecm, det_order=0, k_ar_diff=lags_auto_vecm, method='trace', signif=0.05)
                st.write(f"Le resultat du calcul du rank est :")
                st.write(rank_result)
                rank_vecm = rank_result.rank
                if rank_vecm is None:
                    rank_vecm = results_johansen['num_cointegration_trace']  #généralement 1
                    st.write(f"Le rank issu du test de cointégration est : {rank_vecm}")
                
                else:
                    st.write(f"Le rank déterminé automatiquement est : {rank_vecm}")

                # Estimation du VECM
                vecm = VECM(df_vecm, k_ar_diff=lags_auto_vecm, coint_rank=rank_vecm, deterministic="co")
                vecm_results = vecm.fit()
                st.write("Résultats du modèle VECM :")
                st.write(vecm_results.summary())

                #interpretation des résultats
                alpha = vecm_results.alpha        # matrice (variables × vecteurs cointégration)
                beta = vecm_results.beta          # matrice (vecteurs cointégration × variables)
                n_vars = alpha.shape[0]
                n_coint = alpha.shape[1]
                st.write("Relations de cointégration (vecteurs beta)")
                for i in range(n_coint):
                    coefs = []
                    for j in range(beta.shape[1]):
                        name = df_vecm.columns[j]
                        coefs.append(f"{beta[i,j]}×{name}")
                    st.write(f"Relation de cointégration #{i+1} : " + " + ".join(coefs))

                # Interprétation des coefficients alpha (correction d’erreur)
                st.write("Coefficients alpha (vitesse de correction d’erreur)")
                for i in range(n_vars):
                    name = df_vecm.columns[i]
                    alphas = alpha[i, :]
                    # Evaluation arbitraire : présence si magnitude > 0.01
                    reacts = np.any(np.abs(alphas) > 1e-2)
                    st.write(f"{name} : alpha = {alphas}")
                    if reacts:
                        st.write(f"  → {name} réagit aux déséquilibres (ajustement vers l'équilibre)")
                    
                    else:
                        st.write(f"  → {name} ne réagit pas (ou faiblement) aux déséquilibres")
            
                # Analyse des résidus
                st.info("STEP 5 - 3 - Analyse des résidus du modèle VECM")
                resid_vecm = pd.DataFrame(vecm_results.resid, columns=[f"Var{i}" for i in range(n_vars)])
                threshold = 0.05
                # Test de Ljung-Box (correction de l'erreur)
                st.write("Test de Ljung-Box (Autocorrélation)")
                for i, col in enumerate(resid_vecm.columns):
                    # CORRECTION: Extraire chaque colonne individuellement (1D)
                    residuals_col = resid_vecm.iloc[:, i]  # ou resid_vecm[col]
                    try:
                        lb_test = acorr_ljungbox(residuals_col, lags=[10], return_df=True)
                        pval_auto = float(lb_test.iloc[0]['lb_pvalue'])
                        auto_msg = "✅ Pas d'autocorrélation détectée" if pval_auto > threshold else "❌ Autocorrélation significative détectée"
                        st.write(f"**{col}** : {auto_msg} (p-value = {pval_auto})")
        
                    except Exception as e:
                        st.write(f"**{col}** : Erreur dans le test Ljung-Box - {str(e)}")

                #Test de normalité (D'Agostino-Pearson
                st.write("Test de normalité (D'Agostino-Pearson)")
                for i, col in enumerate(resid_vecm.columns):
                    try:
                        residuals_col = resid_vecm.iloc[:, i]
                        stat, pval_norm = normaltest(residuals_col)
                        norm_msg = "✅ Résidus compatibles avec la normalité" if pval_norm > threshold else "❌ Résidus non normaux"
                        st.write(f"**{col}** : {norm_msg} (p-value = {pval_norm})")
                        
                    except Exception as e:
                        st.write(f"**{col}** : Erreur dans le test de normalité - {str(e)}")

                #Test de Durbin-Watson 
                st.write("Test de Durbin-Watson")
                for i, col in enumerate(resid_vecm.columns):
                    try:
                        residuals_col = resid_vecm.iloc[:, i]
                        dw_stat = durbin_watson(residuals_col)
                        # Interprétation simple
                        if 1.5 <= dw_stat <= 2.5:
                            dw_msg = "✅ Pas d'autocorrélation d'ordre 1 détectée"

                        elif dw_stat < 1.5:
                            dw_msg = "❌ Autocorrélation positive détectée"

                        else:  # dw_stat > 2.5
                            dw_msg = "❌ Autocorrélation négative détectée"
                        
                        # Affichage (même format que vos autres tests)
                        st.write(f"**{col}** : {dw_msg} (DW = {dw_stat})")
                        
                    except Exception as e:
                        st.write(f"**{col}** : Erreur dans le test Durbin-Watson - {str(e)}")

                # Graphiques des résidus
                if st.checkbox("Afficher les graphiques des résidus"):
                    fig7, axes = plt.subplots(n_vars, 2, figsize=(12, 4*n_vars))
                    if n_vars == 1:
                        axes = axes.reshape(1, -1)
                    
                    for i, col in enumerate(resid_vecm.columns):
                        residuals_col = resid_vecm.iloc[:, i]
                        # Graphique temporel des résidus
                        axes[i, 0].plot(residuals_col)
                        axes[i, 0].set_title(f'Résidus - {col}')
                        axes[i, 0].set_ylabel('Résidus')
                        axes[i, 0].grid(True)
                        # Q-Q plot pour normalité
                        from scipy import stats
                        stats.probplot(residuals_col, dist="norm", plot=axes[i, 1])
                        axes[i, 1].set_title(f'Q-Q Plot - {col}')
                    plt.tight_layout()
                    st.pyplot(fig7)


            #ATTENTION (2e cas => pas de co-intégration)
            else:
                st.info(f"STEP 5 - 2 - Modèle ADRL sur les différences (.diff) entre `{id_serie_1}` et `{id_serie_2}`")
                #création de données différentiés
                df_1_2_copy = df_1_2.copy()
                df_1_2_copy[id_serie_1] = df_1_2_copy[id_serie_1].diff()
                df_1_2_copy[id_serie_2] = df_1_2_copy[id_serie_2].diff()
                df_1_2_copy_diff = df_1_2_copy.dropna(subset=[id_serie_1, id_serie_2])
                #sélection automatique différentié des ordres p et q
                model_selection_diff = ardl_select_order(
                    endog=df_1_2_copy_diff[id_serie_1],   #y
                    maxlag=12,
                    exog=df_1_2_copy_diff[[id_serie_2]],   #les x (d'ou le double crochet => ici, on a un seul) 
                    maxorder=12,     
                    ic='aic',
                    trend="c",
                    seasonal=True,
                    period=12
                )          
                order_diff = model_selection_diff.model.ardl_order
                if isinstance(order_diff, tuple) and len(order_diff) == 1:
                    st.error(f"⛔ ATTENTION: il manque 1 des lag exigé pour la suite.")
                    return

                elif isinstance(order_diff, (tuple, list)) and len(order_diff) == 2:
                    lag_p_diff, lag_q_diff = order_diff

                else:
                    st.error(f"⛔ Pas de la retourné dans la sélection automatique.")  
                    return    

                st.write(f"Lags différentiés endogènes: {lag_p_diff} & lags différentiés exogènes: {lag_q_diff}")
                #modelisation ADRL et estimation
                model_ardl_diff = ARDL(
                    endog=df_1_2_copy_diff[id_serie_1],     #y
                    lags=lag_p_diff,
                    exog=df_1_2_copy_diff[[id_serie_2]],    #les x (d'ou le double crochet => ici, on a un seul) 
                    order=lag_q_diff,                  #ce sont les lags des x
                    trend="c",                   #voir la doc pour les # cas possibles
                    seasonal=True,
                    period=12
                )
                results_model_ardl_diff = model_ardl_diff.fit()
                st.text("Résultat du modèle ADRL en différences :")
                st.text(results_model_ardl_diff.summary().as_text())
                #interpretation
                st.write("Aucune relation de long terme détectée. Interprétation des effets à court terme du modèle ADRL en différences")
                for param, pval in results_model_ardl_diff.pvalues.items():
                    if pval < 0.05:
                        coef_val = results_model_ardl_diff.params[param]
                        sign = "positif" if coef_val > 0 else "négatif"
                        st.write(f"Effet significatif de {param} avec coefficient {coef_val} ({sign})")

                #analyse des résidus
                st.write("STEP 5 - 3 - Analyse des résidus du modèle ADRL sur les différences")
                resid_diff = results_model_ardl_diff.resid
                #affichage des graphiques
                if st.checkbox("Afficher graphiques diagnostics (histog., qq-plot, ACF) des résidus"):
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

                    # Histogramme des résidus
                    axes[0].hist(resid_diff, bins=30, edgecolor='k')
                    axes[0].set_title("Histogramme des résidus")

                    # QQ plot
                    sm.qqplot(resid_diff, line='s', ax=axes[1])
                    axes[1].set_title("QQ-plot des résidus")

                    # Autocorrélation (ACF)
                    sm.graphics.tsa.plot_acf(resid_diff, lags=20, ax=axes[2])
                    axes[2].set_title("Fonction d'autocorrélation (ACF)")

                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Test de normalité des résidus (Shapiro-Wilk)
                stat_shapiro_diff, pval_shapiro_diff = shapiro(resid_diff)
                alpha_diff = 0.05
                if pval_shapiro_diff > alpha_diff:
                    st.success(f"✅ Test Shapiro-Wilk: les résidus de stat {stat_shapiro_diff} et de p-valeur {pval_shapiro_diff} suivent une distribution Normale")
                
                else:
                    st.warning(f"⚠️ Test Shapiro-Wilk: les résidus de stat {stat_shapiro_diff} et de p-valeur {pval_shapiro_diff} ne suivent pas une distribution Normale")
               
                # Test d'autocorrélation des résidus (Durbin-Watson)
                dw_stat_difff = durbin_watson(resid_diff)
                st.write(f"Valeur de Durbin-Watson : {dw_stat_difff}")
                if dw_stat_difff < 1.5:
                    st.warning("⚠️ Possible autocorrélation positive des résidus (DW < 1.5).")

                elif dw_stat_difff > 2.5:
                    st.warning("⚠️ Possible autocorrélation négative des résidus (DW > 2.5).")

                else:
                    st.success("✅ Résidus sans autocorrélation significative (DW proche de 2).")

                # Test d'hétéroscédasticité (Breusch-Pagan)
                exog_diff = results_model_ardl_diff.model.exog
                # Vérification que matrice exog est valide pour le test
                if exog_diff.shape[1] < 2:
                    st.warning("⚠️ Le test de Breusch-Pagan requiert au moins deux variables explicatives, dont une constante. Test non effectué ici.")
                
                else:
                    bp_stat_diff, bp_pval_diff, _, _ = het_breuschpagan(resid_diff, exog_diff)
                    st.write(f"Test Breusch-Pagan (hétéroscédasticité) de stat = {bp_stat_diff} et de p-valeur = {bp_pval_diff}")

                    if bp_pval_diff < alpha_diff:
                        st.warning("⚠️ Hétéroscédasticité détectée (rejet homoscédasticité).")

                    else:
                        st.success("✅ Pas d'hétéroscédasticité détectée (variance des résidus constante).")

                    # Bilan global des diagnostics
                    if pval_shapiro_diff > alpha_diff and 1.5 <= dw_stat_difff <= 2.5 and (exog_diff.shape[1] < 2 or bp_pval_diff > alpha_diff):
                        st.success("✅ Diagnostics résiduels OK : modèle sans défauts majeurs détectés.")
                    
                    else:
                        st.info("⚠️ Attention : certains diagnostics suggèrent des problèmes sur les résidus.")










def analyse_manuel(df_1_orig, df_2_orig, id_serie_1, id_serie_2):
    "analyse bivariée manuel"
    if 'compute_auto' not in st.session_state:
        st.session_state.compute_auto = False

    if st.sidebar.button("Computer"):
        st.session_state.compute_auto = True

    if st.session_state.compute_auto:
        #on vérifie que les 2 DataFrame ont été chargés et contienent la colonne 'value'
        if df_1_orig is None :
            st.write(" ⚠️ Erreur : le DataFrame 1 n'a pas été chargé.")
            return

        if 'value' not in df_1_orig.columns:
            st.write(" ⚠️ Erreur : la colonne 'value' est absente du DataFrame 1.")
            return
        
        if df_2_orig is None :
            st.write(" ⚠️ Erreur : le DataFrame 2 n'a pas été chargé.")
            return

        if 'value' not in df_2_orig.columns:
            st.write(" ⚠️ Erreur : la colonne 'value' est absente du DataFrame 2.")
            return

        """STEP 0: donner la possibilité à l'utilisateur de choisir une intervalle de date à évaluer
            Par défaut, les dates max sont utilisées ().
            L'user peut choisir une date MIN et un date MAX et l'ancer l'exécution
            #On invite l'utilisateur à lancer le modèle
                if st.button("Exécuter"):
                    #Tous le code s'exécuter ici
        """

        #selection de l'intervalle de date pour la dataframe df_1
        date_min_1 = df_1_orig.index.min().date()
        date_max_1 = df_1_orig.index.max().date()
        start_date_1 = st.sidebar.date_input(
        "Sélectionnez la date de début pour la série 1",
            value=date_min_1,
            min_value=date_min_1,
            max_value=date_max_1
        )
        end_date_1 = st.sidebar.date_input(
            "Sélectionnez la date de fin pour la série 1",
            value=date_max_1,
            min_value=date_min_1,
            max_value=date_max_1
        )

        #selction de l'intervalle de date pour la dataframe df_2
        date_min_2 = df_2_orig.index.min().date()
        date_max_2 = df_2_orig.index.max().date()
        start_date_2 = st.sidebar.date_input(
        "Sélectionnez la date de début pour la série 2",
            value=date_min_2,
            min_value=date_min_2,
            max_value=date_max_2
        )
        end_date_2 = st.sidebar.date_input(
            "Sélectionnez la date de fin pour la série 2",
            value=date_max_2,
            min_value=date_min_2,
            max_value=date_max_2
        )

        #Rechercher les séries temporelles + session_state
        if st.sidebar.button("Rechercher les séries 1 & 2"):

            if start_date_1 > end_date_1 or start_date_2 > end_date_2:
                st.error("⛔ La date de début (de la série 1 ou/et de la série 2) ne peut pas être après la date de fin.")

            else:
                st.session_state.df_1 = df_1_orig.loc[start_date_1:end_date_1].copy()
                st.session_state.df_2 = df_2_orig.loc[start_date_2:end_date_2].copy()
                st.session_state.series_found = True
                st.success(f"✅ Séries temporelles trouvées et chargées pour les dates sélectionnées : {start_date_1} à {end_date_1} pour la série 1 et {start_date_2} à {end_date_2} pour la série 2.")

        #Initialiser les dataframes au chargement
        if 'series_found' not in st.session_state:
            st.session_state.series_found = False

        if 'df_1' not in st.session_state:
            st.session_state.df_1 = pd.DataFrame()

        if 'df_2' not in st.session_state:
            st.session_state.df_2 = pd.DataFrame()


        #Analyser les séries temporelles
        if st.session_state.series_found:
            df_1 = st.session_state.df_1.copy()
            df_2 = st.session_state.df_2.copy()

            #ATTENTION: on vérifie que les 2 DataFrame ont été chargés + non vides
            if df_1 is None or df_2 is None:
                st.error(" ⛔ Erreur : les DataFrame 1 et/ou 2 n'ont pas été chargés.")
                return
        
            elif df_1.empty or df_2.empty:
                st.error(" ⛔ Erreur : les DataFrame 1 et/ou 2 sont vides.")
                return
            
            else:
                # Supposons df_1 et df_2 ont "date" comme index
                df_1.reset_index(inplace=True)
                df_2.reset_index(inplace=True)

                # Assurez que la colonne 'date' soit bien en datetime64
                df_1['date'] = pd.to_datetime(df_1['date'])
                df_2['date'] = pd.to_datetime(df_2['date'])

                # Renommage les colonnes 
                df_1.rename(columns={'value': id_serie_1}, inplace=True)
                df_2.rename(columns={'value': id_serie_2}, inplace=True)

                #Merger les 2 séries sur la "date"
                df_1_2 = pd.merge(df_1, df_2, on='date', how='inner')

                #on vérifie que les 2 séries ont le même nombre d'observations
                print(f"Série 1 : {len(df_1)} observations")
                print(f"Série 2: {len(df_2)} observations") 
                print(f"Nombre d'observations total: {len(df_1_2)} observations")

                #Afficher les 2 séries (head, date debut et fin)
                print("Affichage (head, tail, date min et date max) des 2 séries:")
                print(df_1_2.head())
                print(df_1_2.tail())
                print(f"Date de début: {df_1_2['date'].min()}")
                print(f"Date de fin: {df_1_2['date'].max()}")


                #STEP 1: AFFICHAGE DES 2 SERIES (graphique (simple et normalisée) et nuage de points)
                #ATTENTION Se RAPPELER qui est x et qui est y.  (nuage des points)
                #Pour nous x = taux de chomage (serie2) et y = taux d'inflation (serie1)
                st.subheader("STEP 1 - Graphique et nuage des points des 2 séries temporelles (manuel)")
                st.write("STEP 1 - 1 - Graphique simple des 2 séries temporelles (manuel)")
                fig1, ax1 = plt.subplots(figsize=(12, 5))
                ax1.plot(df_1_2['date'], df_1_2[id_serie_1], label=id_serie_1, color='blue')
                ax1.plot(df_1_2['date'], df_1_2[id_serie_2], label=id_serie_2, color='orange')
                ax1.set_title(f"Courbe des 2 séries temporelles {id_serie_1} et {id_serie_2}")
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Valeurs numériques")
                ax1.legend()
                st.pyplot(fig1)

                st.write("STEP 1 - 2 - Graphique normalisée (Min-Max) des 2 séries temporelles (manuel)")
                # Normalisation Min-Max
                df_1_2['norm_' + id_serie_1] = (df_1_2[id_serie_1] - df_1_2[id_serie_1].min()) / (df_1_2[id_serie_1].max() - df_1_2[id_serie_1].min())
                df_1_2['norm_' + id_serie_2] = (df_1_2[id_serie_2] - df_1_2[id_serie_2].min()) / (df_1_2[id_serie_2].max() - df_1_2[id_serie_2].min())
                # Affichage du graphique normalisé
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                ax2.plot(df_1_2['date'], df_1_2['norm_' + id_serie_1], label=f'{id_serie_1} (normalisé)', color='blue')
                ax2.plot(df_1_2['date'], df_1_2['norm_' + id_serie_2], label=f'{id_serie_2} (normalisé)', color='orange')
                ax2.set_title(f"Courbes normalisées de {id_serie_1} et {id_serie_2}")
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Valeurs numériques normalisées")
                ax2.legend()
                st.pyplot(fig2)

                st.write(f"STEP 1 - 3 - Nuage des points des 2 séries temporelles (format image) entre {id_serie_1} et {id_serie_2} (manuel)")
                fig3, ax3 = plt.subplots(figsize=(12, 5))
                ax3.scatter(df_1_2[id_serie_2], df_1_2[id_serie_1], color='blue', alpha=0.7)
                ax3.set_title(f"Nuage de points {id_serie_1} vs {id_serie_2}")
                ax3.set_xlabel(f'{id_serie_2}')
                ax3.set_ylabel(f'{id_serie_1}')
                ax3.grid(True)
                st.pyplot(fig3)

                st.write(f"Nuage des points des 2 séries temporelles (format streamlit) entre {id_serie_1} et {id_serie_2} (manuel)")
                fig4 = px.scatter(
                    df_1_2, 
                    x=id_serie_2,
                    y=id_serie_1,  
                    title=f"Nuage des points des 2 séries temporelles (format streamlit) entre {id_serie_1} et {id_serie_2}",
                    labels={'id2': f'{id_serie_2}', 'id1': f'{id_serie_1}'},
                    hover_data=['date']  # Afficher la date au survol
                )
                st.plotly_chart(fig4, use_container_width=True)



                #STEP 2: STATTISTIQUES DESCRIPTIVES DES 2 SERIES
                st.subheader("STEP 2 - Statistiques descriptives des 2 séries temporelles")
                #recupérer chaque série individuellement
                serie1 = df_1_2[['date', id_serie_1]]
                serie2 = df_1_2[['date', id_serie_2]]

                #définir les différents vvariables statistiques à calculer
                stat_names = [
                    "Moyenne",
                    "Médiane",
                    "Minimum",
                    "Maximum",
                    "Std.Dev",
                    "Skewness",
                    "Kurtosis",
                    "Jarque-Bera",
                    "Probability",
                    "Somme des observations",
                    "Somme carrée des écarts à la moyenne",
                    "Nombre d'observations"
                ]

                #définit le tableau
                tab_stats = pd.DataFrame({
                    "Statistiques": stat_names,
                    id_serie_1: describe_series(serie1[id_serie_1]),
                    id_serie_2: describe_series(serie2[id_serie_2])
                })
                tab_stats = tab_stats.applymap(lambda x: f"{x}" if isinstance(x, float) else x)
                tab_stats.set_index("Statistiques", inplace=True)

                # affichage dans streamlit
                st.dataframe(tab_stats)



                #STEP 3: CORRELATION ET TESTS D'INDEPENDANCE
                st.subheader("STEP 3 - Corrélation et tests d'indépendance")

                #CORRELATION
                st.write(f"STEP 3 - 1 - Corrélation entre {id_serie_1} et {id_serie_2}")
                #Calcul de la covariance et de la correlation
                covariance = serie1[id_serie_1].cov(serie2[id_serie_2])
                correlation = serie1[id_serie_1].corr(serie2[id_serie_2])
                #affiche des valeurs
                st.write(f"La covariance entre {id_serie_1} et {id_serie_2}  est : {covariance}")
                st.write(f"Le coefficient de correlation entre {id_serie_1} et  {id_serie_2}  est : {correlation}")

                #interprétation 1
                if correlation == 0:
                    st.warning(f"⚠️ La corrélation linéaire entre {id_serie_1} et {id_serie_2} observée est nulle")
                
                elif correlation == -1 or correlation == 1:
                    st.success(f"✅ La corrélation linéaire entre {id_serie_1} et {id_serie_2} observée est parfaite")
                
                elif correlation > 0:
                    st.success(f"✅ La liaison  entre {id_serie_1} et {id_serie_2} est positive")
                
                elif correlation < 0:
                    st.warning(f"⚠️ La liaison entre {id_serie_1} et {id_serie_2} est négative")
                
                #interprétation 2
                if abs(correlation) < 0.1:
                    st.warning(f"⚠️ La correlation linéaire entre {id_serie_1} et {id_serie_2} est Très faible")

                elif abs(correlation) < 0.3:
                    st.warning(f"⚠️ La correlation linéaire entre {id_serie_1} et {id_serie_2} est Faible")

                elif abs(correlation) < 0.5:
                    st.warning(f"⚠️ La correlation linéaire entre {id_serie_1} et {id_serie_2} est Modérée")

                elif abs(correlation) < 0.7:
                    st.success(f"✅ La correlation linéaire entre {id_serie_1} et {id_serie_2} est Forte")

                else:
                    st.success(f"✅ La correlation linéaire entre {id_serie_1} et {id_serie_2} est Très forte")

                #TEST DE NORMALITE
                st.write(f"STEP 3 - 2 - Normalité des séries temporelles {id_serie_1} et {id_serie_2}")
                #pas besoin de normalisé car il ne s'applique que sur chaque variable
                #serie1
                stat1, pval1, result1 = tester_normalite(serie1[id_serie_1], id_serie_1)
                interpretationDeLaNormalite(stat1, pval1, result1, id_serie_1)
                #serie2
                stat2, pval2, result2 = tester_normalite(serie2[id_serie_2], id_serie_2)
                interpretationDeLaNormalite(stat2, pval2, result2, id_serie_2)

                #QQPLOT DES 2 SERIES
                st.write(f" STEP 3 - 3 - Confirmation de la normalité (ou pas) avec le qqplot de  {id_serie_1} et de {id_serie_2}")
                #serie1
                qqplot1 = st.checkbox(f"Afficher le QQ-plot de la série {id_serie_1}")
                if qqplot1:
                    titre_qq_1 = f"Q-Q Plot de {id_serie_1}"
                    qq_plot_simple(serie1[id_serie_1], titre_qq_1)

                #serie2
                qqplot2 = st.checkbox(f"Afficher le QQ-plot de la série {id_serie_2}")
                if qqplot2:
                    titre_qq_2 = f"Q-Q Plot de {id_serie_2}"
                    qq_plot_simple(serie2[id_serie_2], titre_qq_2)
                    
                #TEST D'INDEPENDANCE (de Pearson ou de Spearman)
                #ATTENTION les unités sont différentes, MAIS PAS BESOIN de les normaliser 
                # POURQUOI? Spearman fonctionne sur les rangs et Pearson est une mesure adimensionnelle (sans unité)
                st.write(f" STEP 3 - 4 - Test d'indépendance entre {id_serie_1} et de {id_serie_2}")
                if result1 and result2:
                    st.write(f"Les 2 séries suivent une distribution Normale => Test de Pearson")
                    r_p, pval_p = test_correlation_pearson(serie1[id_serie_1], serie2[id_serie_2])
                    st.write(f"Le coefficient de correlation de Pearson est de r = {r_p}")
                    interpretationRouRhoPearsonandSpearman(r_p, id_serie_1, id_serie_2)
                    st.write(f"Le p-valeur de Pearson entre {id_serie_1} et {id_serie_2} est de p-valeur = {pval_p}")
                    interpretationPvalPearsonandSpearman(pval_p, id_serie_1, id_serie_2)
                    
                else:
                    st.write(f"Une ou les 2 séries ne suivent/suit pas une distribution Normale => Test de Spearman")
                    r_sp, pval_sp = test_correlation_spearman(serie1[id_serie_1], serie2[id_serie_2])
                    st.write(f"Le coefficient de correlation de Spearman est de rho = {r_sp}")
                    interpretationRouRhoPearsonandSpearman(r_sp, id_serie_1, id_serie_2)
                    st.write(f"Le p-valeur de Spearman entre {id_serie_1} et {id_serie_2} est de p-valeur = {pval_sp}")
                    interpretationPvalPearsonandSpearman(pval_sp, id_serie_1, id_serie_2)
                


                #STEP 4: TEST DE STATIONNARITÉ
                st.subheader("STEP 4 - Test de stationnarité des séries temporelles")
                
                #on tolère maximum 1 différentiation pour éviter de tourner à l'infini
                MAX_DIFF = 10

                #Test de Stationnarité pour la série 1
                serie1_diff = df_1.copy()
                n1 = 0
                adf_ok1 = False
                while n1 <= MAX_DIFF:
                    res1 = adfuller(serie1_diff[id_serie_1].dropna())
                    pval1 = res1[1]
                    print(res1)
                    print(serie1_diff)
                    if pval1 < 0.05:
                        if n1 == 0:
                            st.success(f"✅ La série {id_serie_1} (p-valeur {pval1}) est stationnaire de base.")
                        else:
                            st.success(f"✅ La série {id_serie_1} est stationnaire après {n1} différenciation(s) (p-valeur {pval1}).")
                        adf_ok1 = True
                        break
                    else:
                        if n1 == 0:
                            st.warning(f"⚠️ La série {id_serie_1} n'est pas stationnaire, on différencie...")
                        serie1_diff = serie1_diff.diff().dropna()
                        n1 += 1
                        print(n1)
                
                if not adf_ok1:
                    st.error(f"⛔ La série {id_serie_1} n'est toujours pas stationnaire après {MAX_DIFF} différenciations !")
                
                #Test de stationnairté pour la série 2
                serie2_diff = df_2.copy()
                n2 = 0
                adf_ok2 = False
                while n2 <= MAX_DIFF:
                    res2 = adfuller(serie2_diff[id_serie_2].dropna())
                    pval2 = res2[1]
                    if pval2 < 0.05:
                        if n2 == 0:
                            st.success(f"✅ La série {id_serie_2} (p-valeur {pval2}) est stationnaire de base.")
                        else:
                            st.success(f"✅ La série {id_serie_2} est stationnaire après {n2} différenciation(s) (p-valeur {pval2}).")
                        adf_ok2 = True
                        break
                    else:
                        if n2 == 0:
                            st.warning(f"⚠️ La série {id_serie_2} n'est pas stationnaire, on différencie...")
                        serie2_diff = serie2_diff.diff().dropna()
                        n2 += 1
                
                if not adf_ok2:
                    st.error(f"⛔ La série {id_serie_2} n'est toujours pas stationnaire après {MAX_DIFF} différenciations !")
                
                

                #STEP 5 TEST DE CO-INTÉGRATION ET MODELES
                #ATTENTION pour le rapport, Se RAPPELER qui est x et qui est y
                #Pour nous x = taux de chomage (serie2) et y = taux d'inflation (serie1)
                st.subheader("STEP 5 - Test(s) de co-intégration et/ou Estimation de(s) modèle(s)")

                #ATTENTION: Toujours vérifier le type de données utilisé en entrée => IDENTIQUE EN TYPE (surtout cas I(1) et I(0))

                #ATTENTION le test de cointégration ne marche pas avec des séries > I(2)

                ######CAS I(2) et I(0) ou I(1) ou encore I(2)######
                if n1 > 1 or n2 >1 :
                    st.error(f"⛔ Une des séries est différentiée PLUS D'UNE FOIS => Oupsssssssss!!!!!!!!!!!")
                    return


                ######CAS I(0) et I(0)######
                elif n1 == 0 and n2 == 0:
                    st.info(f"Cas ou la série 1 {id_serie_1} => I(0) et la série 2 {id_serie_2} => I(0).")

                    #Rappel (Test de stationnarité)
                    st.info(f"STEP 5 - 0 - (RAPPEL) Test de stationnarité de {id_serie_1} et de {id_serie_2}")
                    st.success(f"✅ La série {id_serie_1} est stationnaire de base.")
                    st.success(f"✅ La série {id_serie_2} est stationnaire de base.")

                    #Relation à COURT TERME
                    #Test(s) de corrélation (Pearson ou Spearman)
                    st.info(f"STEP 5 - 1 - Test de corrélation (Pearson ou Spearman) entre {id_serie_1} et de {id_serie_2}")
                    st.write("Voire les résultats ci-dessus => (STEP 3 - 4)")

                    #Test de causalité de Granger
                    st.info(f"STEP 5 - 2 - Test de causalité de Granger entre {id_serie_1} et de {id_serie_2}")
                    results_granger = granger_causality_analysis(df_1_2[id_serie_1], df_1_2[id_serie_2])
                    st.write(results_granger)
                    interpretationGranger(results_granger, id_serie_1, id_serie_2)

                    #Modèle VAR
                    st.info(f"STEP 5 - 3 - Modèle VAR entre {id_serie_1} et de {id_serie_2}")
                    #ATTENTION les données doivent etre dans un meme dataframe
                    user_maxlags = st.slider("Choisissez le nombre maximal de lags à tester (maxlags) :", min_value=1, max_value=24, value=12, step=1)
                    st.write(f"Le nombre maximum de lags à tester choisi est : {user_maxlags}")
                    user_criterion = st.selectbox("Sélectionner un critère d'information :", ['aic', 'bic', 'fpe', 'hqic'])
                    st.write(f"Le critère de'information selectionné est : {user_criterion}")
                    results_var = model_var_manuel(df_1_2[[id_serie_1, id_serie_2]], maxlags=user_maxlags, criterion=user_criterion)
                    st.write("Résultats du modèle VAR :")
                    st.write(results_var)   #c'est l'ensemble des résultats du modèle VAR

                    #ATTENTION interpretation des résulatats d'analyse
                    model_result_raw = results_var['model_result_raw']  # objet VARResults
                    params = model_result_raw.params  # DataFrame coefficients
                    pvalues = model_result_raw.pvalues  # DataFrame p-values
                    for eq_name in params.columns:
                        st.write(f"Équation pour variable dépendante : {eq_name}")
                        for var_name in params.index:
                            coef = params.loc[var_name, eq_name]
                            pval = pvalues.loc[var_name, eq_name]

                            if var_name == 'const':
                                st.write(f"=>Terme constant : coef={coef}, p-value={pval}")
                            
                            elif pval < 0.05:
                                signe = 'positif' if coef > 0 else 'négatif'
                                st.write(f"=> ✅ Effet {signe} et significatif de '{var_name}' sur '{eq_name}' (coef={coef}, p-value={pval}).")
                            
                            elif pval < 0.10:
                                st.write(f"=> ⚠️ Effet faiblement significatif de '{var_name}' sur '{eq_name}' (coef={coef}, p-value={pval}).")
                            
                            else:
                                st.write(f"=> ⛔ Pas d’effet significatif de '{var_name}' sur '{eq_name}' (p-value={pval}).")

                    st.info(f"STEP 5 - 4 - Analyse des résidus du modèle VAR")
                    #Moyenne et écart-type des résidus
                    st.write(f"Résidus - Moyennes: {results_var['model_residuals'].mean().values}")
                    st.write(f"Résidus - Écarts-types: {results_var['model_residuals'].std().values}")
                    #Graphiques des résidus + test de Normalité
                    resid_var_plot = st.checkbox(f"Afficher les graphiques (histo, qq-plot, pacf) des résidus")
                    if resid_var_plot:
                        last_two_cols = df_1_2.columns[-2:]
                        diagnostic_residus_var(results_var, last_two_cols)
                    
                    #Homoscédasticité des résidus
                    st.write("Tests d'homoscédasticité de Breusch-Pagan :")
                    bp_test_results = test_homoscedasticite(results_var['model_residuals'])
                    for var, testres in bp_test_results.items():
                        st.write(f"Variable {var} : Breusch-Pagan LM stat={testres['LM_stat']}, p-value={testres['LM_pvalue']}")
                        if testres['LM_pvalue'] < 0.05:
                            st.warning(f"⚠️ Rejet de l'hypothèse d'homoscédasticité (Variance non constante)")
                        
                        else:
                            st.success(f"✅ Pas de preuve suffisante contre l'homoscédasticité (Variance constante)")
                    
                    #Test de durbin_watson
                    st.write("Test de durbin_watson :")
                    dw_dict_var = results_var['durbin_watson']
                    for var, dw_val in dw_dict_var.items():
                        if dw_val < 1.5:
                            st.warning(f"⚠️ Résidus de {var} présentent une autocorrélation positive (DW={dw_val})")
                        
                        elif dw_val > 2.5:
                            st.warning(f"⚠️ Résidus de {var} présentent une autocorrélation négative (DW={dw_val})")
                        
                        else:
                            st.success(f"✅ Résidus de {var} sans autocorrélation significative (DW={dw_val})")
                    #ATTENTION => Peut-etre essayer une regression lineaire 
                    #Relation à LONG TERME
                    #La cointégration s’intéresse essentiellement à des séries non stationnaires (I(1))



                ######CAS I(1) et I(0)######
                elif n1 == 1 and n2 == 0:
                    st.info(f"Cas ou la série {id_serie_1} => I(1) et la série {id_serie_2} => I(0).")

                    #Rappel (Test de stationnarité)
                    st.info(f"STEP 5 - 0 (RAPPEL) - Test de stationnarité de {id_serie_1} et de {id_serie_2}")
                    st.warning(f"⚠️ La série {id_serie_1} n'est pas stationnaire, mais I(1).")
                    st.success(f"✅ La série {id_serie_2} est stationnaire de base.")

                    #Sélection automatique ordre optimal ARDL. (RAPPEL => Y=id_serie_1 et X=id_serie_2)
                    st.info("STEP 5 - 1 - Sélection automatique ordre optimal ARDL")
                    user_maxlag = st.slider("Choisissez le lag maximal pour la variable endogène (select-adrl) :", min_value=1, max_value=24, value=12, step=1)
                    st.write(f"Le lag maximal (select-adrl) à tester choisi est : {user_maxlag}")
                    user_maxorder = st.slider("Choisissez le lag maximal pour la variable exogène (select-adrl) :", min_value=1, max_value=24, value=12, step=1)
                    st.write(f"Le lag maximal (select-adrl) à tester choisi est : {user_maxorder}")
                    user_ic = st.selectbox("Sélectionner le critère d'information (select-adrl) :", ['aic', 'bic', 'hqic'])
                    st.write(f"La critère d'information (select-adrl) sélectionné est : {user_ic}")
                    user_trend_select = st.selectbox("Sélectionner la tendance du modèle ((select-adrl)) :", ['c', 'n', 't', 'ct'])
                    st.write(f"La tendance sélectionnée (select-adrl) est : {user_trend_select}") 
                    user_period_select = st.slider("Choisissez la période d'étude (select-adrl) :", min_value=1, max_value=24, value=12, step=1)
                    st.write(f"La période d'étude (select-adrl) choisie est : {user_period_select}")
                    model_selection = ardl_select_order(
                        endog=df_1_2[id_serie_1],   #y
                        maxlag=user_maxlag,
                        exog=df_1_2[[id_serie_2]],   #les x (d'ou le double crochet => ici, on a un seul) 
                        maxorder=user_maxorder,     
                        ic=user_ic,
                        trend=user_trend_select,
                        seasonal=True,
                        period=user_period_select
                    )
                    st.write(f"L'ordre de modèle ARDL (p,q) optimal :", model_selection.model.ardl_order)
                    st.write(f"Le décalage autoregressif (AR) :", model_selection.model.ar_lags)
                    st.write(f"Le décalage de la variable exogène :", model_selection.model.dl_lags)
                    st.write("Résultats du modèle ADRL optimal :")
                    st.text(model_selection.model.fit().summary().as_text())            
                    
                    #Bounds Test (Pesaran et al.) (RAPPEL => Y=id_serie_1 et X=id_serie_2)
                    st.info("STEP 5 - 2 - Bounds Test (Pesaran et al.) + interpretation")
                    order = model_selection.model.ardl_order
                    if isinstance(order, tuple) and len(order) == 1:
                        st.error(f"⛔ ATTENTION: il manque 1 des lag exigés pour la suite.")
                        return

                    elif isinstance(order, (tuple, list)) and len(order) == 2:
                        lag_p, lag_q = order

                    else:
                        st.error(f"⛔ Pas d'ordre retourné dans la sélection automatique.")  
                        return    

                    st.write(f"Lags endogènes automatiques => {lag_p} & lags exogènes automatiques => {lag_q}")
                    user_trend_adrl = st.selectbox("Sélectionner la tendance du modèle (adrl):", ['c', 'n', 't', 'ct'])
                    st.write(f"La tendance sélectionnée est (adrl) : {user_trend_adrl}")
                    user_period_adrl = st.slider("Choisissez la période d'étude (adrl) :", min_value=1, max_value=24, value=12, step=1)
                    st.write(f"La période d'étude est : {user_period_adrl}")
                    model_ardl = ARDL(
                        endog=df_1_2[id_serie_1],     #y
                        lags=lag_p,
                        exog=df_1_2[[id_serie_2]],    #les x (d'ou le double crochet => ici, on a un seul) 
                        order=lag_q,                  #ce sont les lags des x
                        trend=user_trend_adrl,        #voir la doc pour les # cas possibles
                        seasonal=True,
                        period=user_period_adrl
                    ).fit()
                    uecm_model = UECM.from_ardl(model_ardl.model)
                    uecm_results = uecm_model.fit()
                    st.text("Résultats du modèle test de Pesaran :")
                    st.text("Modèle ADRL:")
                    st.text(model_ardl.summary().as_text())
                    st.text("Modèle UECM (dérivé de ADRL):")
                    st.text(uecm_results.summary().as_text())
                    st.text("Modèle UECM sur la co-intégration:")
                    st.text(uecm_results.ci_summary().as_text())
                    #_ = uecm_model.fit().ci_resids.plot(title="Erreur de cointégration")
                    #uecm_results.ci_resids.plot(title="Erreur de cointégration")
                    st.write("Graphique de l'erreur de cointégration :")
                    fig6 = uecm_results.ci_resids.plot(title="Erreur de cointégration")
                    st.pyplot(fig6.figure)
                    st.write("Interpretation du test de Pesaran :")
                    #trend est  "c"    =>>>>> case = 3
                    user_case = st.selectbox("Sélectionner le cas pour le test PSS :", [3, 1, 2, 4, 5])
                    st.write(f"Le cas sélectionnée est : {user_case}")
                    user_cov_type = st.selectbox("Sélectionner l'estimateur à utiliser' :", ['nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC'])
                    st.write(f"L'estimateur de covariance est : {user_cov_type}")
                    bounds_results = uecm_results.bounds_test(case=user_case, cov_type=user_cov_type, use_t=True, asymptotic=True)
                    #st.write(bounds_results.crit_vals)
                    stat_F = bounds_results.stat                #Statistique F
                    st.write(f"La statistique F est de :")
                    st.write(stat_F)
                    crit_bounds = bounds_results.crit_vals         #Bornes critiques
                    st.write(f"Les bornes critiques du test sont :")
                    st.write(crit_bounds)
                    for level in [90.0, 95.0, 99.0]:
                        lower = crit_bounds.loc[level, 'lower']
                        upper = crit_bounds.loc[level, 'upper']
                        st.write(f"Bornes au niveau {int(level)}% : inférieure = {lower}, supérieure = {upper}")
                    # Récupérer les bornes critiques à 5%
                    lower_bound_5pct = crit_bounds.loc[95.0, 'lower']
                    upper_bound_5pct = crit_bounds.loc[95.0, 'upper']
                    st.write(f"La borne inférieure à 5% est : {lower_bound_5pct}")
                    st.write(f"La borne supérieure à 5% est : {upper_bound_5pct}")
                    co_integration = False
                    if stat_F > upper_bound_5pct:            #on rejette l’hypothèse nulle d’absence de cointégration, donc il y a cointégration
                        st.write(f"La statistique F ({stat_F}) est supérieure à la borne supérieure ({upper_bound_5pct})")
                        st.success(f"✅ Cointégration détectée (relation de long terme) entre {id_serie_1} et {id_serie_2}")
                        co_integration = True

                    elif stat_F < lower_bound_5pct:           #on ne rejette pas l’hypothèse nulle, donc pas de cointégration.
                        st.write(f"La statistique F ({stat_F}) est inférieure à la borne inférieure ({lower_bound_5pct})")  
                        st.warning(f"⚠️ Pas de cointégration entre {id_serie_1} et {id_serie_2}")
                        co_integration = False
                    
                    else:
                        st.error(f"⛔ Test non conclusif (zone indéterminée car {lower_bound_5pct} < {stat_F} < {upper_bound_5pct})")
                        co_integration = False
                        #return
                    
                    #ATTENTION (1er cas => si co-inégration)
                    if co_integration:
                        st.info("STEP 5 - 3 - Modélisation UECM complète + interpretation (si cointégration)")
                        #Modélisation ARDL/ECM : effet de long terme et vitesse d’ajustement
                        # Coef vitesse d'ajustement (coefficient sur y.L1)
                        speed_adj = uecm_results.params.get(f"{id_serie_1}.L1", None)
                        if speed_adj is not None:
                            st.success(f"✅ Vitesse d’ajustement (coefficient sur {id_serie_1}.L1): {speed_adj}")
                            if speed_adj < 0:
                                st.success(f"✅ Cela signifie que la variable corrigera efficacement les déséquilibres à long terme")
                            
                            else:
                                st.warning("⚠️ Attention, la vitesse d'ajustement est positive, ce qui peut indiquer un problème")
                        
                        else:
                            st.warning("⚠️ Coefficient vitesse d'ajustement non trouvé.")

                        # Effet long terme = - coef exogène niveau / coef vitesse d'ajustement
                        x_level = uecm_results.params.get(f"{id_serie_2}.L0") or uecm_results.params.get(f"D.{id_serie_2}.L0")
                        if x_level is not None and speed_adj is not None and speed_adj != 0:
                            long_term_effect = -x_level / speed_adj
                            pval_x = uecm_results.pvalues.get(f"{id_serie_2}.L0") or uecm_results.pvalues.get(f"D.{id_serie_2}.L0")
                            st.write(f"L'effet de long terme estimé de {id_serie_2} sur {id_serie_1} est {long_term_effect}")
                            if pval_x is not None and pval_x < 0.05:
                                st.success(f"✅ Cet effet est statistiquement significatif")
                            
                            else:
                                st.warning("⚠️ Cet effet n'est pas significatif")
                        
                        else:
                            st.error(f"⛔ Impossible de calculer l'effet de long terme (manque de coefficients ou vitesse nulle).")

                        # Analyse des résidus
                        st.info("STEP 5 - 4 - Analyse des résidus du modèle UECM complet)")
                        if stat_F > crit_bounds.iloc[1,1]:
                            resid_adrl_ucem = uecm_results.resid

                        else:
                            resid_adrl_ucem = uecm_results.resid

                        # Graphiques des résidus + Normalité
                        resid_ardl_ucem_plot = st.checkbox(f"Afficher les graphiques (histo, qq-plot, pacf) des résidus")
                        if resid_ardl_ucem_plot:
                            # Plots diagnostics résidus
                            fig, axes_ucem = plt.subplots(1, 3, figsize=(15, 4))
                            axes_ucem[0].hist(resid_adrl_ucem, bins=30, edgecolor='k')
                            axes_ucem[0].set_title('Histogramme des résidus')
                            sm.qqplot(resid_adrl_ucem, line='s', ax=axes_ucem[1])
                            axes_ucem[1].set_title('QQ-plot des résidus')
                            sm.graphics.tsa.plot_acf(resid_adrl_ucem, lags=20, ax=axes_ucem[2])
                            axes_ucem[2].set_title('ACF des résidus')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Diagnostic normalité (Shapiro-Wilk)
                        adrl_ucem_stat, adrl_ucem_pval = shapiro(resid_adrl_ucem)
                        alpha_ucem = 0.05
                        if adrl_ucem_pval > alpha_ucem:
                            st.success(f"✅ Test Shapiro-Wilk: le résidu de stat {adrl_ucem_stat} et de p-valeur {adrl_ucem_pval} suit une distribution Normale")

                        else:
                            st.warning(f"⚠️ Test Shapiro-Wilk: le résidu de stat {adrl_ucem_stat} et de p-valeur {adrl_ucem_pval} ne suit pas une distribution Normale")

                        # Autocorrélation (Durbin-Watson)
                        dw_val_adrl_ucem = durbin_watson(resid_adrl_ucem)
                        st.write(f"Valeur de Durbin-Watson : {dw_val_adrl_ucem}")
                        if dw_val_adrl_ucem < 1.5:
                            st.warning(f"⚠️ Présomption d'autocorrélation positive des résidus (DW < 1.5).")

                        elif dw_val_adrl_ucem > 2.5:
                            st.warning(f"⚠️ Présomption d'autocorrélation négative des résidus (DW > 2.5).")

                        else:
                            st.success(f"✅ Pas d'autocorrélation significative des résidus (DW ≈ 2).")

                        # Hétéroscédasticité (Breusch-Pagan)
                        exog = uecm_results.model.exog
                        if exog.shape[1] < 2:
                            st.warning("⚠️ Le test de Breusch-Pagan requiert au moins deux variables explicatives, dont une constante. Test non effectué ici.")
                        
                        else:
                            bp_stat_adrl_ucem, bp_pval_adrl_ucem, _, _ = het_breuschpagan(resid_adrl_ucem, uecm_results.model.exog)
                            st.write(f"Test de Breusch-Pagan de de stat : {bp_stat_adrl_ucem} et de p-valeur : {bp_pval_adrl_ucem}")
                            if bp_pval_adrl_ucem < 0.05:
                                st.warning(f"⚠️ Rejet de l'hypothèse d'homoscédasticité (hétéroscédasticité détectée).")
                            
                            else:
                                st.success(f"✅ Pas de preuve d'hétéroscédasticité (variance des résidus constante).")

                            # Durbin-Watson + Breusch-Pagan
                            if (adrl_ucem_pval > 0.05 and 1.5 <= dw_val_adrl_ucem <= 2.5 and bp_pval_adrl_ucem > 0.05):
                                st.success("✅ Modèle sans défaut structurel majeur des résidus : diagnostics OK.")


                    #ATTENTION (2e cas => pas de co-intégration)
                    else:
                        st.info("STEP 5 - 3 - Pas de modelisation UECM complète (pas de cointégration).")
                        #on va estimer un modèle ADRL en différences ()
                        st.info(f"Modelisation ADRL sur les différences (.diff) entre {id_serie_1} et {id_serie_2}")
                        #création de données différentiés
                        df_1_2_copy = df_1_2.copy()
                        df_1_2_copy[id_serie_1] = df_1_2_copy[id_serie_1].diff()
                        df_1_2_copy[id_serie_2] = df_1_2_copy[id_serie_2].diff()
                        df_1_2_copy_diff = df_1_2_copy.dropna(subset=[id_serie_1, id_serie_2])
                        #sélection automatique différentié des ordres p et q
                        user_maxlag_diff = st.slider("Choisissez le lag maximal pour la variable endogène (diff):", min_value=1, max_value=24, value=12, step=1)
                        st.write(f"Le lag maximal (diff) à tester choisi est  : {user_maxlag_diff}")
                        user_maxorder_diff = st.slider("Choisissez le lag maximal pour la variable exogène (diff):", min_value=1, max_value=24, value=12, step=1)
                        st.write(f"Le lag maximal (diff) à tester choisi est : {user_maxorder_diff}")
                        user_ic_diff = st.selectbox("Sélectionner le critère d'information (diff) :", ['aic', 'bic', 'hqic'])
                        st.write(f"La critère d'information (diff) sélectionné est : {user_ic_diff}")
                        user_trend_select_diff = st.selectbox("Sélectionner la tendance (diff) du modèle (select) :", ['c', 'n', 't', 'ct'])
                        st.write(f"La tendance sélectionnée (diff) est (select) : {user_trend_select_diff}")
                        user_period_select_diff = st.slider("Choisissez la période d'étude (diff):", min_value=1, max_value=24, value=12, step=1)
                        st.write(f"La période d'étude (diff) est : {user_period_select_diff}")
                        model_selection_diff = ardl_select_order(
                            endog=df_1_2_copy_diff[id_serie_1],   #y
                            maxlag=user_maxlag_diff,
                            exog=df_1_2_copy_diff[[id_serie_2]],   #les x (d'ou le double crochet => ici, on a un seul) 
                            maxorder=user_maxorder_diff,     
                            ic=user_ic_diff,
                            trend=user_trend_select_diff,
                            seasonal=True,
                            period=user_period_select_diff
                        )          
                        order_diff = model_selection_diff.model.ardl_order
                        if isinstance(order_diff, tuple) and len(order_diff) == 1:
                            st.error(f"⛔ ATTENTION: il manque 1 des lag exigé pour la suite.")
                            return

                        elif isinstance(order_diff, (tuple, list)) and len(order_diff) == 2:
                            lag_p_diff, lag_q_diff = order_diff

                        else:
                            st.error(f"⛔ Pas de la retourné dans la sélection automatique.")  
                            return    

                        st.write(f"Lags différentiés endogènes: {lag_p_diff} & lags différentiés exogènes: {lag_q_diff}")
                        #modelisation ADRL et estimation
                        st.write(f"Lags endogènes automatiques => {lag_p} & lags exogènes automatiques => {lag_q}")
                        user_trend_adrl_diff = st.selectbox("Sélectionner la tendance (diff) du modèle (adrl):", ['c', 'n', 't', 'ct'])
                        st.write(f"La tendance sélectionnée est (adrl) : {user_trend_adrl_diff}")
                        user_period_adrl_diff = st.slider("Choisissez la période d'étude (diff) :", min_value=1, max_value=24, value=12, step=1)
                        st.write(f"La période d'étude est : {user_period_adrl_diff}")
                        model_ardl_diff = ARDL(
                            endog=df_1_2_copy_diff[id_serie_1],     #y
                            lags=lag_p_diff,
                            exog=df_1_2_copy_diff[[id_serie_2]],    #les x (d'ou le double crochet => ici, on a un seul) 
                            order=lag_q_diff,                  #ce sont les lags des x
                            trend=user_trend_adrl_diff,                   #voir la doc pour les # cas possibles
                            seasonal=True,
                            period=user_period_adrl_diff
                        )
                        results_model_ardl_diff = model_ardl_diff.fit()
                        st.text("Résultats du modèle ADRL en différences :")
                        st.text(results_model_ardl_diff.summary().as_text())
                        # Interpretation du résultat du modèle
                        st.write("Aucune relation de long terme détectée. Interprétation des effets à court terme du modèle ADRL en différences")
                        for param, pval in results_model_ardl_diff.pvalues.items():
                            if pval < 0.05:
                                coef_val = results_model_ardl_diff.params[param]
                                sign = "positif" if coef_val > 0 else "négatif"
                                st.write(f"Effet significatif de {param} avec coefficient {coef_val} ({sign})")

                        # Analyse des résidus
                        st.info("STEP 5 - 4 - Analyse des résidus du modèle ADRL sur les différences")
                        resid_diff = results_model_ardl_diff.resid
                        #affichage des graphiques
                        if st.checkbox("Afficher graphiques diagnostics (histogramme, qq-plot, ACF) des résidus"):
                            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                            # Histogramme des résidus
                            axes[0].hist(resid_diff, bins=30, edgecolor='k')
                            axes[0].set_title("Histogramme des résidus")
                            # QQ plot
                            sm.qqplot(resid_diff, line='s', ax=axes[1])
                            axes[1].set_title("QQ-plot des résidus")
                            # Autocorrélation (ACF)
                            sm.graphics.tsa.plot_acf(resid_diff, lags=20, ax=axes[2])
                            axes[2].set_title("Fonction d'autocorrélation (ACF)")
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Test de normalité des résidus (Shapiro-Wilk)
                        stat_shapiro_diff, pval_shapiro_diff = shapiro(resid_diff)
                        alpha_diff = 0.05
                        if pval_shapiro_diff > alpha_diff:
                            st.success(f"✅ Test Shapiro-Wilk: les résidus de stat {stat_shapiro_diff} et de p-valeur {pval_shapiro_diff} suivent une distribution Normale")
                        
                        else:
                            st.warning(f"⚠️ Test Shapiro-Wilk: les résidus de stat {stat_shapiro_diff} et de p-valeur {pval_shapiro_diff} ne suivent pas une distribution Normale")
                    
                        # Test d'autocorrélation des résidus (Durbin-Watson)
                        dw_stat_difff = durbin_watson(resid_diff)
                        st.write(f"Valeur de Durbin-Watson : {dw_stat_difff}")
                        if dw_stat_difff < 1.5:
                            st.warning("⚠️ Possible autocorrélation positive des résidus (DW < 1.5).")

                        elif dw_stat_difff > 2.5:
                            st.warning("⚠️ Possible autocorrélation négative des résidus (DW > 2.5).")

                        else:
                            st.success("✅ Résidus sans autocorrélation significative (DW proche de 2).")

                        # Test d'hétéroscédasticité (Breusch-Pagan)
                        exog_diff = results_model_ardl_diff.model.exog
                        # Vérification que matrice exog est valide pour le test
                        if exog_diff.shape[1] < 2:
                            st.warning("⚠️ Le test de Breusch-Pagan requiert au moins deux variables explicatives, dont une constante. Test non effectué ici.")
                        
                        else:
                            bp_stat_diff, bp_pval_diff, _, _ = het_breuschpagan(resid_diff, exog_diff)
                            st.write(f"Test Breusch-Pagan (hétéroscédasticité) de stat = {bp_stat_diff} et de p-valeur = {bp_pval_diff}")

                            if bp_pval_diff < alpha_diff:
                                st.warning("⚠️ Hétéroscédasticité détectée (rejet homoscédasticité).")

                            else:
                                st.success("✅ Pas d'hétéroscédasticité détectée (variance des résidus constante).")

                            # Bilan global des diagnostics
                            if pval_shapiro_diff > alpha_diff and 1.5 <= dw_stat_difff <= 2.5 and (exog_diff.shape[1] < 2 or bp_pval_diff > alpha_diff):
                                st.success("✅ Diagnostics résiduels OK : modèle sans défauts majeurs détectés.")
                            
                            else:
                                st.info("⚠️ Attention : certains diagnostics suggèrent des problèmes sur les résidus.")




                ######CAS I(0) et I(1)######
                elif n1 == 0 and n2 == 1:
                    st.write(f"Cas ou la série {id_serie_1} => I(0) et la série {id_serie_2} => I(1).")
                    st.write("C'est l'inverse du cas I(1) et I(0), Merci d'inverser l'odre des ID des séries")

                

                ######CAS I(1) et I(1)######
                elif n1 == 1 and n2 == 1:
                    st.info(f"Cas ou la série {id_serie_1} => I(1) et la série {id_serie_2} => I(1).")

                    #Rappel (Test de stationnarité)
                    st.info(f"STEP 5 - 0 (RAPPEL) - Test de stationnarité de {id_serie_1} et de {id_serie_2}")
                    st.warning(f"⚠️ La série {id_serie_1} n'est pas stationnaire, mais I(1).")
                    st.warning(f"⚠️ La série {id_serie_2} n'est pas stationnaire, mais I(1).")

                    #Test de co-intégration d’Engle-Granger
                    st.info(f"#STEP 5 - 1 - Test de cointégration d'Engle-Granger entre `{id_serie_1}` et `{id_serie_2}`")
                    results_coint_Granger = test_coint_granger_manuel(df_1_2[id_serie_1], df_1_2[id_serie_2])
                    interpretation_test_coint_granger(results_coint_Granger, id_serie_1, id_serie_2)

                    #Test de cointégaration (Johansen)
                    st.info(f"#STEP 5 - 1 - Test de cointégration de Johansen entre `{id_serie_1}` et `{id_serie_2}`")
                    results_johansen = test_coint_johansen_manuel(df_1_2[id_serie_1], df_1_2[id_serie_2])
                    interpretation_test_johansen(results_johansen)

                    #ATTENTION: ICI ON CHOISI JOHANSEN CAR IL EST PLUS PERFORMANT PAR RAPPORT A engler
                    st.info(f"#NB: C'est le test de cointégration de Johansen entre qui sera utilisé pour la suite de l'analyse car il est plus performant que celui d'Engle-Granger.")

                    #ATTENTION (1er cas => co-intégration)
                    if results_johansen['cointegrated']:
                        st.info(f"STEP 5 - 2 - Modèle VECM `{id_serie_1}` et `{id_serie_2}`")
                        df_vecm = df_1_2[[id_serie_1, id_serie_2]].dropna()

                        # Choix automatique du nombre de lags :
                        user_maxlag_select_vecm = st.slider("Choisissez le lag maximum (select vecm) :", min_value=1, max_value=24, value=12, step=1)
                        st.write(f"La lag maximum (select vecm) sélectionné est : {user_maxlag_select_vecm}")
                        user_deter_select_vecm = st.selectbox("Sélectionner le type de terme déterministe (select vecm):", ['co', 'n', 'ci', 'lo', 'li'])
                        st.write(f"Le type de terme déterministe (select vecm) selectionné est : {user_deter_select_vecm}")
                        order_result_vecm = select_order(df_vecm, maxlags=user_maxlag_select_vecm, deterministic=user_deter_select_vecm)
                        st.write("Le resultat du calcul du lag est :")
                        st.write(order_result_vecm)
                        lags_auto_vecm = order_result_vecm.selected_orders['aic']
                        if lags_auto_vecm is None:
                            lags_auto_vecm = 1              # sécurité si indéterminé
                            st.write(f"Le lag fixé est : {lags_auto_vecm}")
                        
                        else:
                            st.write(f"Le lag déterminé automatiquement est : {lags_auto_vecm}")

                        # Déterminer le rang de cointégration (déjà déterminé avec Johansen)
                        user_det_order = st.selectbox("Sélectionner la tendance du modèle johansen (vecm-coint):", [0, -1, 1])
                        st.write(f"La tendance sélectionnée est (vecm-coint) : {user_det_order}")
                        user_test_stat = st.selectbox("Sélectionner le type de statistique à tester (vecm-coint) :", ['trace', 'maxeig'])
                        st.write(f"Le type de statistique à tester (vecm-coint) : {user_test_stat}")
                        rank_result = select_coint_rank(df_vecm, det_order=user_det_order, k_ar_diff=lags_auto_vecm, method=user_test_stat, signif=0.05)
                        st.write(f"Le resultat du calcul du rank est :")
                        st.write(rank_result)
                        rank_vecm = rank_result.rank
                        if rank_vecm is None:
                            rank_vecm = results_johansen['num_cointegration_trace']  #généralement 1
                            st.write(f"Le rank issu du test de cointégration est : {rank_vecm}")
                        
                        else:
                            st.write(f"Le rank déterminé automatiquement est : {rank_vecm}")

                        # Estimation du VECM
                        user_deter_model_vecm = st.selectbox("Sélectionner le type de terme déterministe (model vecm):", ['co', 'n', 'ci', 'lo', 'li'])
                        st.write(f"Le type de terme déterministe (model vecm) selectionné est : {user_deter_model_vecm}")
                        vecm = VECM(df_vecm, k_ar_diff=lags_auto_vecm, coint_rank=rank_vecm, deterministic=user_deter_model_vecm)
                        vecm_results = vecm.fit()
                        st.write("Résultats du modèle VECM :")
                        st.write(vecm_results.summary())

                        #interpretation des résultats
                        alpha = vecm_results.alpha        # matrice (variables × vecteurs cointégration)
                        beta = vecm_results.beta          # matrice (vecteurs cointégration × variables)
                        n_vars = alpha.shape[0]
                        n_coint = alpha.shape[1]
                        st.write("Relations de cointégration (vecteurs beta)")
                        for i in range(n_coint):
                            coefs = []
                            for j in range(beta.shape[1]):
                                name = df_vecm.columns[j]
                                coefs.append(f"{beta[i,j]}×{name}")
                            st.write(f"Relation de cointégration #{i+1} : " + " + ".join(coefs))

                        # Interprétation des coefficients alpha (correction d’erreur)
                        st.write("Coefficients alpha (vitesse de correction d’erreur)")
                        for i in range(n_vars):
                            name = df_vecm.columns[i]
                            alphas = alpha[i, :]
                            # Evaluation arbitraire : présence si magnitude > 0.01
                            reacts = np.any(np.abs(alphas) > 1e-2)
                            st.write(f"{name} : alpha = {alphas}")
                            if reacts:
                                st.write(f"  → {name} réagit aux déséquilibres (ajustement vers l'équilibre)")
                            
                            else:
                                st.write(f"  → {name} ne réagit pas (ou faiblement) aux déséquilibres")
                    
                        # Analyse des résidus
                        st.info("STEP 5 - 3 - Analyse des résidus du modèle VECM")
                        resid_vecm = pd.DataFrame(vecm_results.resid, columns=[f"Var{i}" for i in range(n_vars)])
                        threshold = 0.05
                        # Test de Ljung-Box (correction de l'erreur)
                        st.write("Test de Ljung-Box (Autocorrélation)")
                        for i, col in enumerate(resid_vecm.columns):
                            # CORRECTION: Extraire chaque colonne individuellement (1D)
                            residuals_col = resid_vecm.iloc[:, i]  # ou resid_vecm[col]
                            try:
                                lb_test = acorr_ljungbox(residuals_col, lags=[10], return_df=True)
                                pval_auto = float(lb_test.iloc[0]['lb_pvalue'])
                                auto_msg = "✅ Pas d'autocorrélation détectée" if pval_auto > threshold else "❌ Autocorrélation significative détectée"
                                st.write(f"**{col}** : {auto_msg} (p-value = {pval_auto})")
                
                            except Exception as e:
                                st.write(f"**{col}** : Erreur dans le test Ljung-Box - {str(e)}")

                        #Test de normalité (D'Agostino-Pearson
                        st.write("Test de normalité (D'Agostino-Pearson)")
                        for i, col in enumerate(resid_vecm.columns):
                            try:
                                residuals_col = resid_vecm.iloc[:, i]
                                stat, pval_norm = normaltest(residuals_col)
                                norm_msg = "✅ Résidus compatibles avec la normalité" if pval_norm > threshold else "❌ Résidus non normaux"
                                st.write(f"**{col}** : {norm_msg} (p-value = {pval_norm})")
                                
                            except Exception as e:
                                st.write(f"**{col}** : Erreur dans le test de normalité - {str(e)}")

                        #Test de Durbin-Watson 
                        st.write("Test de Durbin-Watson")
                        for i, col in enumerate(resid_vecm.columns):
                            try:
                                residuals_col = resid_vecm.iloc[:, i]
                                dw_stat = durbin_watson(residuals_col)
                                # Interprétation simple
                                if 1.5 <= dw_stat <= 2.5:
                                    dw_msg = "✅ Pas d'autocorrélation d'ordre 1 détectée"

                                elif dw_stat < 1.5:
                                    dw_msg = "❌ Autocorrélation positive détectée"

                                else:  # dw_stat > 2.5
                                    dw_msg = "❌ Autocorrélation négative détectée"
                                
                                # Affichage (même format que vos autres tests)
                                st.write(f"**{col}** : {dw_msg} (DW = {dw_stat})")
                                
                            except Exception as e:
                                st.write(f"**{col}** : Erreur dans le test Durbin-Watson - {str(e)}")

                        # Graphiques des résidus
                        if st.checkbox("Afficher les graphiques des résidus"):
                            fig7, axes = plt.subplots(n_vars, 2, figsize=(12, 4*n_vars))
                            if n_vars == 1:
                                axes = axes.reshape(1, -1)
                            
                            for i, col in enumerate(resid_vecm.columns):
                                residuals_col = resid_vecm.iloc[:, i]
                                # Graphique temporel des résidus
                                axes[i, 0].plot(residuals_col)
                                axes[i, 0].set_title(f'Résidus - {col}')
                                axes[i, 0].set_ylabel('Résidus')
                                axes[i, 0].grid(True)
                                # Q-Q plot pour normalité
                                from scipy import stats
                                stats.probplot(residuals_col, dist="norm", plot=axes[i, 1])
                                axes[i, 1].set_title(f'Q-Q Plot - {col}')
                            plt.tight_layout()
                            st.pyplot(fig7)


                    #ATTENTION (2e cas => pas de co-intégration)
                    else:
                        st.info(f"STEP 5 - 2 - Modèle ADRL sur les différences (.diff) entre `{id_serie_1}` et `{id_serie_2}`")
                        #création de données différentiés
                        df_1_2_copy = df_1_2.copy()
                        df_1_2_copy[id_serie_1] = df_1_2_copy[id_serie_1].diff()
                        df_1_2_copy[id_serie_2] = df_1_2_copy[id_serie_2].diff()
                        df_1_2_copy_diff = df_1_2_copy.dropna(subset=[id_serie_1, id_serie_2])
                        #sélection automatique différentié des ordres p et q
                        user_maxlag_diff_1 = st.slider("Choisissez le lag maximal pour la variable endogène (diff1):", min_value=1, max_value=24, value=12, step=1)
                        st.write(f"Le lag maximal (diff1) à tester choisi est  : {user_maxlag_diff_1}")
                        user_maxorder_diff_1 = st.slider("Choisissez le lag maximal pour la variable exogène (diff1):", min_value=1, max_value=24, value=12, step=1)
                        st.write(f"Le lag maximal (diff1) à tester choisi est : {user_maxorder_diff_1}")
                        user_ic_diff_1 = st.selectbox("Sélectionner le critère d'information (diff1) :", ['aic', 'bic', 'hqic'])
                        st.write(f"La critère d'information (diff1) sélectionné est : {user_ic_diff_1}")
                        user_trend_select_diff_1 = st.selectbox("Sélectionner la tendance (diff1) du modèle :", ['c', 'n', 't', 'ct'])
                        st.write(f"La tendance sélectionnée (diff1) est (select) : {user_trend_select_diff_1}")
                        user_period_select_diff_1 = st.slider("Choisissez la période d'étude (diff1):", min_value=1, max_value=24, value=12, step=1)
                        st.write(f"La période d'étude (diff1) est : {user_period_select_diff_1}")
                        model_selection_diff = ardl_select_order(
                            endog=df_1_2_copy_diff[id_serie_1],   #y
                            maxlag=user_maxlag_diff_1,
                            exog=df_1_2_copy_diff[[id_serie_2]],   #les x (d'ou le double crochet => ici, on a un seul) 
                            maxorder=user_maxorder_diff_1,     
                            ic=user_ic_diff_1,
                            trend=user_trend_select_diff_1,
                            seasonal=True,
                            period=user_period_select_diff_1
                        )          
                        order_diff = model_selection_diff.model.ardl_order
                        if isinstance(order_diff, tuple) and len(order_diff) == 1:
                            st.error(f"⛔ ATTENTION: il manque 1 des lag exigé pour la suite.")
                            return

                        elif isinstance(order_diff, (tuple, list)) and len(order_diff) == 2:
                            lag_p_diff, lag_q_diff = order_diff

                        else:
                            st.error(f"⛔ Pas de la retourné dans la sélection automatique.")  
                            return    

                        st.write(f"Lags différentiés endogènes: {lag_p_diff} & lags différentiés exogènes: {lag_q_diff}")
                        #modelisation ADRL et estimation
                        user_trend_adrl_diff_1 = st.selectbox("Sélectionner la tendance du modèle (diff1):", ['c', 'n', 't', 'ct'])
                        st.write(f"La tendance sélectionnée est (diff1) : {user_trend_adrl_diff_1}")
                        user_period_adrl_diff1 = st.slider("Choisissez la période d'étude (diff1) :", min_value=1, max_value=24, value=12, step=1)
                        st.write(f"La période d'étude (diff1) est : {user_period_adrl_diff1}")
                        model_ardl_diff = ARDL(
                            endog=df_1_2_copy_diff[id_serie_1],     #y
                            lags=lag_p_diff,
                            exog=df_1_2_copy_diff[[id_serie_2]],    #les x (d'ou le double crochet => ici, on a un seul) 
                            order=lag_q_diff,                       #ce sont les lags des x
                            trend=user_trend_adrl_diff_1,           #voir la doc pour les # cas possibles
                            seasonal=True,
                            period=user_period_adrl_diff1
                        )
                        results_model_ardl_diff = model_ardl_diff.fit()
                        st.text("Résultat du modèle ADRL en différences :")
                        st.text(results_model_ardl_diff.summary().as_text())
                        #interpretation
                        st.write("Aucune relation de long terme détectée. Interprétation des effets à court terme du modèle ADRL en différences")
                        for param, pval in results_model_ardl_diff.pvalues.items():
                            if pval < 0.05:
                                coef_val = results_model_ardl_diff.params[param]
                                sign = "positif" if coef_val > 0 else "négatif"
                                st.write(f"Effet significatif de {param} avec coefficient {coef_val} ({sign})")

                        #analyse des résidus
                        st.write("STEP 5 - 3 - Analyse des résidus du modèle ADRL sur les différences")
                        resid_diff = results_model_ardl_diff.resid
                        #affichage des graphiques
                        if st.checkbox("Afficher graphiques diagnostics (histog., qq-plot, ACF) des résidus"):
                            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

                            # Histogramme des résidus
                            axes[0].hist(resid_diff, bins=30, edgecolor='k')
                            axes[0].set_title("Histogramme des résidus")

                            # QQ plot
                            sm.qqplot(resid_diff, line='s', ax=axes[1])
                            axes[1].set_title("QQ-plot des résidus")

                            # Autocorrélation (ACF)
                            sm.graphics.tsa.plot_acf(resid_diff, lags=20, ax=axes[2])
                            axes[2].set_title("Fonction d'autocorrélation (ACF)")

                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Test de normalité des résidus (Shapiro-Wilk)
                        stat_shapiro_diff, pval_shapiro_diff = shapiro(resid_diff)
                        alpha_diff = 0.05
                        if pval_shapiro_diff > alpha_diff:
                            st.success(f"✅ Test Shapiro-Wilk: les résidus de stat {stat_shapiro_diff} et de p-valeur {pval_shapiro_diff} suivent une distribution Normale")
                        
                        else:
                            st.warning(f"⚠️ Test Shapiro-Wilk: les résidus de stat {stat_shapiro_diff} et de p-valeur {pval_shapiro_diff} ne suivent pas une distribution Normale")
                    
                        # Test d'autocorrélation des résidus (Durbin-Watson)
                        dw_stat_difff = durbin_watson(resid_diff)
                        st.write(f"Valeur de Durbin-Watson : {dw_stat_difff}")
                        if dw_stat_difff < 1.5:
                            st.warning("⚠️ Possible autocorrélation positive des résidus (DW < 1.5).")

                        elif dw_stat_difff > 2.5:
                            st.warning("⚠️ Possible autocorrélation négative des résidus (DW > 2.5).")

                        else:
                            st.success("✅ Résidus sans autocorrélation significative (DW proche de 2).")

                        # Test d'hétéroscédasticité (Breusch-Pagan)
                        exog_diff = results_model_ardl_diff.model.exog
                        # Vérification que matrice exog est valide pour le test
                        if exog_diff.shape[1] < 2:
                            st.warning("⚠️ Le test de Breusch-Pagan requiert au moins deux variables explicatives, dont une constante. Test non effectué ici.")
                        
                        else:
                            bp_stat_diff, bp_pval_diff, _, _ = het_breuschpagan(resid_diff, exog_diff)
                            st.write(f"Test Breusch-Pagan (hétéroscédasticité) de stat = {bp_stat_diff} et de p-valeur = {bp_pval_diff}")

                            if bp_pval_diff < alpha_diff:
                                st.warning("⚠️ Hétéroscédasticité détectée (rejet homoscédasticité).")

                            else:
                                st.success("✅ Pas d'hétéroscédasticité détectée (variance des résidus constante).")

                            # Bilan global des diagnostics
                            if pval_shapiro_diff > alpha_diff and 1.5 <= dw_stat_difff <= 2.5 and (exog_diff.shape[1] < 2 or bp_pval_diff > alpha_diff):
                                st.success("✅ Diagnostics résiduels OK : modèle sans défauts majeurs détectés.")
                            
                            else:
                                st.info("⚠️ Attention : certains diagnostics suggèrent des problèmes sur les résidus.")



        else:
            st.warning("⚠️ Veuillez sélectionner une date de début et de fin pour chaque série et cliquer sur le bouton <<Rechercher les séries>>")
            st.stop()





def bivaraite_plot(df_1, df_2, id_serie_1, id_serie_2):
    "analyse bivariée de 2 série temprelles"
    type_mp = st.sidebar.selectbox(
            "Sélectionner une méthode de modélisation/prévision",
            ("auto", "manuel")
        )
    st.subheader(f"Analyse bivariée des séries {id_serie_1} et {id_serie_2} en mode {type_mp}")

    if type_mp == "auto":
        analyse_auto(df_1, df_2, id_serie_1, id_serie_2)

    if type_mp == "manuel":
        analyse_manuel(df_1, df_2, id_serie_1, id_serie_2)




