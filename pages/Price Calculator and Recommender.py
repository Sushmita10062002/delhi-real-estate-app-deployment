import  streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


st.set_page_config(page_title = "Price Calculator and Recommender")

st.title("Price Predictor")
with open('./pickle file/df.pkl', 'rb') as file:
    df = joblib.load(file)
with open('./pickle file/df_v6.pkl', 'rb') as file:
    df_v6 = joblib.load(file)
with open('./pickle file/compressed_pipeline1.pkl', 'rb') as file:
    loaded_pipeline = joblib.load(file)
with open('./pickle file/df_locality.pkl', 'rb') as file:
    df_locality = joblib.load(file)
with open('./pickle file/df_w.pkl', 'rb') as file:
    df_w = joblib.load(file)
with open('./pickle file/pipeline_encode.pkl', 'rb') as file:
    pipeline_encode = joblib.load(file)



def recommend(data, columns):
    rec_one_df = pd.DataFrame(data, columns=columns)
    # return rec_one_df
    df_encoded_one_df = pipeline_encode.transform(rec_one_df)
    df_locality_one_df = df_encoded_one_df[:, 0:150]
    cs_loc = cosine_similarity(df_locality, df_locality_one_df)
    s_loc_df = pd.DataFrame(cs_loc, index = df_v6["PROP_ID"])
    s_loc_df.rename(columns={0: "s_loc_score"}, inplace=True)
    cs = cosine_similarity(df_w, df_encoded_one_df[:, 151:])
    s_df = pd.DataFrame(cs, index = df_v6["PROP_ID"])
    s_df.rename(columns={0: "s_df_score"}, inplace=True)
    df = pd.merge(s_df, s_loc_df, on="PROP_ID", how="inner")
    df["similarity_score"] = 0.6 * df["s_loc_score"] + 0.4 * df["s_df_score"]
    df = df["similarity_score"]
    l1 = df.sort_values(ascending=False)[0:5].index.values.tolist()
    return df_v6[df_v6["PROP_ID"].isin(l1)]


def property_description(row_number):
    row = recommended_df.iloc[row_number]
    st.subheader(row["PROP_NAME"], divider='rainbow')
    st.markdown(f'<div style="color: grey; white-space: pre-line;">{row["DESCRIPTION"]}</div>', unsafe_allow_html=True)
    st.text("Property type: {}".format(row["PROPERTY_TYPE"]))
    st.text("{},{}".format(row["LOCALITY_NAME"], row["CITY_NAME"]))
    st.text("{} bedrooms, {} bathrooms and {} balcony".format(row["BEDROOM_NUM"], row["BATHROOM_NUM"], row["BALCONY_NUM"]))
    st.text("Builtup Area (sqft): {}".format(row["CALCULATED_AREA_SQFT"]))
    st.text("Furnish type: {}".format(row["FURNISH"]))
    st.text("Property age: {}".format(row["AGE_POSSESSION"]))
    st.text("Poperty Price in Crores: {}".format(row["PRICE_CR"]))

    url = "https://99acres.com/{}".format(row["PROP_ID"])
    st.markdown(f'<a href="{url}" target="_blank">{"Buy Property"}</a>', unsafe_allow_html=True)


st.markdown("#### Enter your inputs")


property_type = st.selectbox('Property Type', ['flat', 'house'])
city_name = st.selectbox('City Name', sorted(df["CITY_NAME"].unique().tolist()))
locality = st.selectbox('Locality', sorted(df[df["CITY_NAME"] == city_name]["LOCALITY_NAME"].unique().tolist()))
owntype = st.selectbox('Owntype', sorted(df["OWNTYPE"].unique().tolist()))
bedrooms = st.selectbox('Number of Bedrooms', sorted(df["BEDROOM_NUM"].unique().tolist()))
bathrooms = st.selectbox('Number of Bathrooms', sorted(df["BATHROOM_NUM"].unique().tolist()))
balcony = st.selectbox('Number of Balcony', sorted(df["BALCONY_NUM"].unique().tolist()))
calculated_area_sqft = float(st.number_input('Built Up Area'))
furnishing_type = st.selectbox('Furnishing Type', sorted(df["FURNISH"].unique().tolist()))
age_possession = st.selectbox('Property Age', sorted(df["AGE_POSSESSION"].unique().tolist()))
amenities = st.selectbox('Luxury Category', sorted(df["AMENITIES_CLUSTER"].unique().tolist()))
floor = st.selectbox('Floor Category', sorted(df["FLOOR_CATEGORY"].unique().tolist()))


if st.button('Predict'):
    data = [[property_type, city_name, locality, owntype, bedrooms, bathrooms, balcony, furnishing_type, calculated_area_sqft,
             age_possession, amenities, floor]]
    columns = ["PROPERTY_TYPE", "CITY_NAME", "LOCALITY_NAME", "OWNTYPE", "BEDROOM_NUM", "BATHROOM_NUM", "BALCONY_NUM",
               "FURNISH", "CALCULATED_AREA_SQFT", "AGE_POSSESSION", "AMENITIES_CLUSTER", "FLOOR_CATEGORY"]

    input_df = pd.DataFrame(data, columns = columns)

    # predict
    price = np.expm1(loaded_pipeline.predict(input_df))[0]
    low = price - 0.17
    high = price + 0.17
    st.markdown("### The Price of the flat is between {}Cr and {}Cr".format(round(low, 2), round(high, 2)))


    st.markdown("---")

    st.title('Recommendation')
    rec_data = [[property_type, locality, age_possession, bedrooms, bathrooms, calculated_area_sqft]]
    rec_columns = [['PROPERTY_TYPE', 'LOCALITY_NAME', 'AGE_POSSESSION', 'BEDROOM_NUM', 'BATHROOM_NUM', 'CALCULATED_AREA_SQFT']]
    recommended_df = recommend(rec_data, rec_columns)


    for i in range(0,5):
        property_description(i)



