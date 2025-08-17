# -*- coding: utf-8 -*-

#Interface Streamlit (Frontend)

# Import librairies
import os
import sys
import streamlit as st
from streamlit_option_menu import option_menu



# Import data files
data_path = os.getcwd()
final_data_path = os.path.join(data_path, "data")
sys.path.insert(0, final_data_path)
from get_data import get_data_serie, get_data_info
from display_data import display_series



# Import model files
model_path = os.getcwd()
final_model_path = os.path.join(model_path, "models")
sys.path.insert(0, final_model_path)
from ar import ar_model_plot
from ma import ma_model_plot
from arma import arma_model_plot
from arima import arima_model_plot
from random_forest import random_forest_model_plot
from lstm import lstm_model_plot




# Import statseco files
statseco_path = os.getcwd()
final_statseco_path = os.path.join(statseco_path, "statseco")
sys.path.insert(0, final_statseco_path)
from bivar import bivaraite_plot




def menu():
    """Configuration du menu principal"""
    #barre de menu à gauche
    st.sidebar.header("Menu principal")

    #titre de l'application (au centre)
    st.subheader("Visualisation prévision macroéconomique (Fred API)", anchor=False, divider=True)

    #menu de gauche
    pages = ["Données", "Analyse univariée", "Analyse bivariée"]
    
    if "menu_selection" not in st.session_state:
        st.session_state.menu_selection = pages[0]  # première option par défaut

    with st.sidebar:
        for page in pages:
            if st.button(page):
                st.session_state.menu_selection = page
    
    return st.session_state.menu_selection




def main():
    """Main function"""
    #menu principal
    selected = menu()

    #affichage des données
    if selected == "Données":
        st.subheader("Données => affichage et analyse des données d'une série temporelle")
        id_serie = st.sidebar.text_input("Entrer l'ID d'une série à visualiser. Exemples : GNPCA, CPIAUCSL, EXJPUS, CORESTICKM159SFRBATL, ect.")
        
        if 'show_data_display' not in st.session_state:
            st.session_state.show_data_display = False

        if st.sidebar.button("Afficher et analyser la série"):
            st.session_state.show_data_display = True

        if st.session_state.show_data_display:
            df = get_data_serie(id_serie)
            df_info = get_data_info(id_serie)

            if df is not None:
                display_series(df, id_serie, df_info)
            else:
                st.warning("⚠️ Merci d'entrer l'ID d'une série.")
        
    #analyse univariée
    elif selected == "Analyse univariée":
        st.subheader("Analyse univariée => prévision d'une série temporelle")

        id_serie = st.sidebar.text_input("Entrez l'ID de la série à modeliser/prédire: GNPCA, CPIAUCSL, EXJPUS, CORESTICKM159SFRBATL, ect.")
        model = st.sidebar.selectbox(
            "Sélectionner un modèle",
            ("AR", "MA", "ARMA", "ARIMA", "RandomForest", "LSTM")
        )

        df = get_data_serie(id_serie)
        if df is None:
            st.warning("⚠️ Merci d'entrer l'ID d'une série.")
        
        else:
            pass

        if model == "AR":
            ar_model_plot(df, model)

        if model == "MA":
            ma_model_plot(df, model)

        if model == "ARMA":
            arma_model_plot(df, model)

        if model == "ARIMA":
            arima_model_plot(df, model)
        
        if model == "RandomForest":
            random_forest_model_plot(df, model)

        if model == "LSTM":
            lstm_model_plot(df, model)
        

    #analyse bivariée
    elif selected == "Analyse bivariée":
        st.subheader("Analyse bivariée => analyse de la relation entre 2 séries temporelles")

        id_serie_1 = st.sidebar.text_input("Entrez l'ID de la série 1.")
        id_serie_2 = st.sidebar.text_input("Entrez l'ID de la série 2.")

        df_1 = get_data_serie(id_serie_1)
        df_2 = get_data_serie(id_serie_2)

        if df_1 is None:
            st.warning("⚠️ Merci d'entrer l'ID d'une série 1.")

        else:
            pass
        
        if df_2 is None:
            st.warning("⚠️ Merci d'entrer l'ID d'une série 2.")
        
        else:
            pass
        
        bivaraite_plot(df_1, df_2, id_serie_1, id_serie_2)
            




#main
if __name__ == '__main__':
    main()


