import streamlit as st
import pandas as pd
import numpy as np
from utils import conf, frontend, geo
from ml.search import query
import pydeck as pdk

frontend.set_max_width(1300)

st.title('SmartTravel')
st.write('')

# Request block
request = st.text_input(
    label='Enter your request',
    value=conf.Config.example_query
)
st.write('')
st.write('User request')
st.write(request)

# Filter on number of search results
num_results = st.sidebar.slider("Number of search results", min_value=3, max_value=20, value=5)

# Decision box of recommendation model 
model_name = st.selectbox(
    'Chose the search algorithm',
    ('Tf-Idf', 'LDA')
)

# Request Dataframe
result = query(request, num_results, model_name)
result['URL'] = result['URL'].apply(frontend.make_clickable_link)

# Result Table placeholder
result_table = st.empty()

# Map displaying search results
st.write('')
st.subheader('Map')
geo_df = geo.get_coords(result['Name'])
st.map(geo_df)

result = result.to_html(escape=False)
result_table.write(result, unsafe_allow_html=True)