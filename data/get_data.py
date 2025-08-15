#fonction to get data
# Import librairies
import os
import sys
import streamlit as st
import plotly.express as px


# Import backend files
path = os.getcwd()
final_path = os.path.join(path, "backend")
sys.path.insert(0, final_path)
from fred_api import get_observation_data, get_info


# Cache data serie info
@st.cache_data
def get_data_info(id_serie):
    """Get dand return data from the FRED API"""
    df = get_info(id_serie)
    return df


# Cache data serie observation
@st.cache_data
def get_data_serie(id_serie):
    """Get and return data from the FRED API"""
    infos = get_observation_data(id_serie)
    return infos

