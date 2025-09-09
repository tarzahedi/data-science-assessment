import streamlit as st
import pandas as pd
import os

# Try to load the CSV
csv_path = "resources/export/top3.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    has_data = True
else:
    df = pd.DataFrame()  # empty fallback
    has_data = False

# Streamlit app
st.title("Search Similarity")

if has_data:
    # Selectbox with placeholder
    user_id = st.selectbox(
        "Enter an ID to search:",
        list(df["rfq_id"].unique()),
        index=None,                  # no default selection
        placeholder="Select an ID"   # this disappears when clicked
    )

    if st.button("Search"):
        if user_id is None:
            st.warning("Please select a valid ID.")
        else:
            result = df[df["rfq_id"] == user_id].reset_index(drop=True)

            if not result.empty:
                st.success("ID found!")
                st.dataframe(result)
            else:
                st.error(f"ID {user_id} is not available in the dataset.")
else:
    st.error(f"CSV file not found at: {csv_path}")
    st.info("Please make sure the file exists and try again.")
