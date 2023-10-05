import streamlit as st

st.set_page_config(
    page_title="Real Estate App",
    page_icon="🏠",
)

st.title("🏠 Explore Delhi's Real Estate Market")

st.sidebar.success("Select a demo above.")

# Description of your app
st.write("Our app harnesses the power of machine learning to simplify the real estate experience in Delhi. Whether you're a potential buyer or simply curious about property values, our app has you covered.")

# Key Features
st.write("📊 Key Features:")
st.write("1. **Price Calculator**: Input property details such as location, property type, bedrooms, bathrooms, balcony, area, age, furnish type, amenities you want and our app will provide you with a range of prices based on market data.")
st.write("2. **Property Recommendations**: Looking for your dream home? Our recommendation system suggests properties tailored to your preferences, based on locality, property type, bedroom number, bathroom number, area and age, making your search easier than ever.")
