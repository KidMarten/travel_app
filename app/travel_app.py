import streamlit as st
import pandas as pd
import numpy as np
from utils import conf, frontend, geo
from ml.search import query
import pydeck as pdk

frontend.set_max_width(1500)

st.title('SmartTravel')

# Request block
request = st.text_input(
    label='Enter your request',
    value=conf.Config.example_query
)
st.write('User request')
st.write(request)

# Result Table
result = query(request, 5)
result['URL'] = result['URL'].apply(frontend.make_clickable_link)
result = result.to_html(escape=False)
st.write(result, unsafe_allow_html=True)

st.subheader('Map')