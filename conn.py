import psycopg2
from psycopg2 import OperationalError
import streamlit as st

def get_connection():
    try:
        return psycopg2.connect(
            st.secrets["DATABASE_URL"],
            sslmode="require"
        )
    except OperationalError as e:
        st.error(f"❌ Database connection failed: {e}")
        return None
