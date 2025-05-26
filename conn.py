import psycopg2
from psycopg2 import OperationalError
import streamlit as st

def get_connection():
    try:
        return psycopg2.connect(
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            host=st.secrets["DB_HOST"],
            port=st.secrets["DB_PORT"],
            dbname=st.secrets["DB_NAME"],
            sslmode="require",
            options="-c tcp_user_timeout=2000",       # optional tuning
            # bind to a specific local IPv6 address/interface:
            source_address=('2606:4700:110:8468:10d3:a5f3:d1fd:1793', 0)
        )

    except OperationalError as e:
        st.error(f"❌ Database connection failed: {e}")
        return None
