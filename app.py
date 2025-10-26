import streamlit as st
st.title("✈️ Flight Delay Prototype App")
st.write("Hello from Streamlit + PySpark setup!")

# quick PySpark test
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("hello").getOrCreate()
    st.success("PySpark initialized successfully ✅")
except Exception as e:
    st.error(f"PySpark error: {e}")
