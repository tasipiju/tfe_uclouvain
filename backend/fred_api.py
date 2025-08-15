# -*- coding: utf-8 -*-

# Import modules
import requests
import pandas as pd

# Import API key
from fred_key import fred_key

# Import or assign API key
api_key = fred_key

# Define the FRED API endpoint
base_url = 'https://api.stlouisfed.org/fred/'



#Define functions to get different type of data from the FRED API
def get_category_details(cat_id):
    '''Get category details from the FRED API.'''
    # Assign endpoint
    cat_endpoint = 'category'

    # Assign params
    cat_params = {
        'api_key': api_key,
        'file_type': 'json',
        'category_id': cat_id
    }

    # Make request to FRED API
    response = requests.get(base_url + cat_endpoint, params=cat_params)

    # Response (cat_data) is json file
    if response.status_code == 200:
        cat_data = response.json()
        return cat_data
        
    else:
        print('Failed to retrieve data. Status code:', response.status_code)
        return None




def get_category_children(parent_id):
    '''Get categories children from the FRED API.'''
    # Assign endpoint
    child_endpoint = 'category/children'

    # Assign params
    child_params = {
        'api_key': api_key,
        'file_type': 'json',
        'category_id': parent_id
    }

    # Make request to FRED API
    response = requests.get(base_url + child_endpoint, params=child_params)

    # Response (child_data) is json file
    if response.status_code == 200:
        child_data = response.json()
        return child_data
    
    else:
        print('Failed to retrieve data. Status code:', response.status_code)
        return None

   


def get_series_from_category(cat_id):
    '''Get series from a category from the FRED API.'''
    # Assign endpoint
    cat_series_endpoint = 'category/series'

    # Assign params
    cat_series_params = {
        'api_key': api_key,
        'file_type': 'json',
        'category_id': cat_id,
        'limit': 1000,
        'order_by': 'popularity',
        'sort_order': 'asc'
    }

    # Make request to FRED API
    response = requests.get(base_url + cat_series_endpoint, params=cat_series_params)

    # Response (cat_srs_data) is json file
    if response.status_code == 200:
        cat_srs_data = response.json()
        cat_srs = pd.DataFrame(cat_srs_data['seriess'])
        return cat_srs
    
    else:
        print('Failed to retrieve data. Status code:', response.status_code)
        return None




def get_categories_from_serie(serie_id):
    '''Get categories from a serie from the FRED API.'''
    # Assign endpoint
    srs_cat_endpoint = 'series/categories'

    # Assign params
    srs_cat_params = {
        'api_key': api_key,
        'file_type': 'json',
        'series_id': serie_id
    }

    # Make request to FRED API
    response = requests.get(base_url + srs_cat_endpoint, params=srs_cat_params)

    # Response (srs_cat_data) is json file
    if response.status_code == 200:
        srs_cat_data = response.json()
        return srs_cat_data
    
    else:
        print('Failed to retrieve data. Status code:', response.status_code)
        return None




def get_observation_data(series):
    '''Get and return observation data from the FRED API. exemple of series = CPIAUCSL, GPDI, INDPRO, PCE, UNRATE'''
    # Assign endpoint
    obs_endpoint = 'series/observations'

    # Assign parameters
    series_id = series
    start_date = '1776-07-04'
    end_date = '9999-12-31'

    # Observation parameters
    obs_params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date
    }

    # Make request to FRED API
    response = requests.get(base_url + obs_endpoint, params=obs_params)

    # Format data and convert to DataFrame
    if response.status_code == 200:
        res_data = response.json()
        obs_data = pd.DataFrame(res_data['observations'])
        obs_data['date'] = pd.to_datetime(obs_data['date'])
        obs_data.set_index('date', inplace=True)

        # Convert value to float or 0
        def safe_convert(val):
            try:
                return float(val)
            except ValueError:
                return 0.0
        
        obs_data['value'] = obs_data['value'].apply(safe_convert)
        obs_data['value'] = pd.to_numeric(obs_data['value'], errors='coerce')
        df = obs_data.drop(['realtime_start', 'realtime_end'], axis=1)
        return df

    else:
        print('Failed to retrieve data. Status code:', response.status_code)
        return None



def get_info(series):
    '''Get and return information data about series from the FRED API.'''
    # Assign endpoint
    info_endpoint = 'series'

    # Assign parameters
    series_id = series

    # Info parameters
    info_params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json'
    }

    # Make request to FRED API
    response = requests.get(base_url + info_endpoint, params=info_params)

    # Format data and convert to DataFrame
    if response.status_code == 200:
        res_data = response.json()
        return res_data

    else:
        print('Failed to retrieve data. Status code:', response.status_code)
        return None
    
