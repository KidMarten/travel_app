import streamlit as st


def make_clickable_link(link):
    # Tuen a link in a table clickable
    return f'<a target="_blank" href="{link}">{link}</a>'


def set_max_width(pixels):
    # Set custom layout size
    max_width_str = f"max-width: {pixels}px;"

    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
