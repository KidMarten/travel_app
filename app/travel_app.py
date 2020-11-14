import streamlit as st
import pandas as pd
import numpy as np
from utils import conf, frontend, geo
from ml.search import query
from PIL import Image

frontend.set_max_width(1200)

# Wallpaper at the top of the page
wallpaper = Image.open('static_data/zanzibar.jpg')
wallpaper = wallpaper.resize((1280, 500)) 
st.image(wallpaper, use_column_width=True)

# App title
st.title('SmartTravel')
st.write('')

# Request block
request = st.text_input(
    label='Enter your request',
    value=conf.Config.example_query
)
st.write('')
st.subheader('User request')
st.write(request)

# Filter on number of search results
num_results = st.sidebar.slider("Number of search results", min_value=3, max_value=20, value=10)
 
# Decision box of recommendation model 
model_name = st.sidebar.selectbox(
    'Chose the search algorithm',
    ('Tf-Idf', 'LDA', 'Doc2Vec')
)

# Request Dataframe
result = query(request, num_results, model_name)
result['URL'] = result['URL'].apply(frontend.make_clickable_link)

# Result Table placeholder
st.subheader('Search result')
result_table = st.empty()

# Map displaying search results
display_map = st.sidebar.radio('Display Map', ('Yes', 'No'), index=1)
if display_map == 'Yes':
    st.write('')
    st.subheader('Map')
    geo_df = geo.get_coords(result['Name'])
    st.map(geo_df)

# Fill the result table placeholder
result = result.to_html(escape=False)
result_table.write(result, unsafe_allow_html=True)