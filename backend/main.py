# -*- coding: utf-8 -*-

# Main code to test backend and models

# Import modules
import sys
import json

# Import fred_api functions
from fred_api import get_observation_data, get_info

# Import models

# Import fred_api.py file
sys.path.insert(0, '/Users/tasipiju/Documents/ucl-master-60-cour/code-memoire/code/models')
#from arima import arima_models, test_stationarity



if __name__ == '__main__':
    
    """
    # Get category details (result is json)
    # Define the category_id
    cat_id = 125
    cat_details = get_category_details(cat_id)
    if cat_details is None:
        print(' OUPPPSSSS, No data returned.')
        exit()
    else:
        print(" Data retrieved successfully.")
        print(cat_details)
    """

    """
    # Get category children (result is json)
    # Define the parent_id
    parent_id = 13
    cat_chidren = get_category_children(parent_id)
    if cat_chidren is None:
        print(' OUPPPSSSS, No data returned.')
        exit()
    else:
        print(" Data retrieved successfully.")
        print(cat_chidren)
    """

    """
    #Get series of a category (result is dataframe)
    # Define the category_id
    cat_id = 25
    series = get_series_from_category(cat_id)
    if series is None:
        print(' OUPPPSSSS, No data returned.')
        exit()
    else:
        print(" Data retrieved successfully.")
        print(series.shape)
        print(series)
    """

    """
    #Get series of a category (result is dataframe)
    # Define the serie_id
    serie_id = "EXJPUS"
    categories = get_categories_from_serie(serie_id)
    if categories is None:
        print(' OUPPPSSSS, No data returned.')
        exit()
    else:
        print(" Data retrieved successfully.")
        print(categories)
    """

    """
    # Get observation data  + plot time series
    # Define the serie_id GPDI EXJPUS GDP CE16OV
    serie_id = "GPDI"
    #serie_id = "CPIAUCSL"
    obs_data = get_observation_data(serie_id)
    if obs_data is None:
        print(' OUPPPSSSS, No data returned.')
        exit()
    else:
        print(" Data retrieved successfully.")
        #print(obs_data.head(14))
        #print("")
        print(obs_data.tail(14))
        #test_stationarity(obs_data["value"])
        #arima_models(obs_data)
    """

    # Get observation data  + plot time series
    # Define the serie_id GPDI EXJPUS GDP CE16OV CPIAUCSL
    serie_id = "CPIAUCSL"
    data_info = get_info(serie_id)
    if data_info is None:
        print(' OUPPPSSSS, No data returned.')
        exit()
    else:
        print(" Data retrieved successfully.")
        print(json.dumps(data_info, indent=1))