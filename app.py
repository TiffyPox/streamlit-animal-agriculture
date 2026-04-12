# Import the necessary libraries
import streamlit as st
import pandas as pd
import joblib

# Setting the page title and icon
st.set_page_config(
    page_title="Animal Agriculture",
    page_icon="earth_africa"
)

# Load my model
model = joblib.load('best_model.pkl')

# Load my dataset
@st.cache_data
def get_global_data():
    df = pd.read_csv("final_dataset_new.csv")

    # Get the recent stats
    recent_stats = df['Year'].max()
    global_numbers = df[df['Year'] == recent_stats].sum(numeric_only=True)

    # Get the number of countries
    countries_count = df['Country'].nunique()
    return global_numbers, df, countries_count

global_stats, master_df, n_countries = get_global_data()

# { The Future (Predictions) Section }
st.title("🌍 The Future.")
st.markdown("""Using a Random Forest model and datasets provided by the **FAO** and **Our World in Data**, this application makes
predictions on the Earth's future CO² emissions in the next 50 years from animal agriculture.
\nThis model is trained using extensive data from 173 countries around the world. The data includes: the total population,
the number of land animals slaughtered for meat, the resources used for animal agriculture (Land and Water), as well as the greenhouse gases emitted.
""")

# 'Years' slider for user input
st.write("")
st.markdown("Adjust the sliders below to see how global emissions change over time.")
st.write("")
years_ahead = st.select_slider(
    "**Choose how many years in the future (from 2023)**",
    options=[0, 5, 10, 15, 20, 30, 50],
    value=10
)

# The user's chosen target year
target_year = 2023 + years_ahead

# 'Meat Consumption' slider for user input
meat_label = st.select_slider(
    "**Global Meat Consumption**",
    options=["No Change", "25% Less Meat", "50% Less Meat", "75% Less Meat", "100% Less Meat (Vegan)"],
    value="No Change"
)

# Map the meat consumption values
reduction_map = {"No Change": 1.0, "25% Less Meat": 0.75, "50% Less Meat": 0.50, "75% Less Meat": 0.25, "100% Less Meat (Vegan)": 0.0}
meat_multiplier = reduction_map[meat_label]

# Future predictions Logic
# The recent stats
baseline_avg_input = pd.DataFrame([[
    global_stats['Population'] / n_countries,
    global_stats['Land_Use(ha)'] / n_countries,
    global_stats['Water_Use(m3)'] / n_countries,
    global_stats['Animals_Slaughtered'] / n_countries
]], columns=['Population', 'Land_Use(ha)', 'Water_Use(m3)', 'Animals_Slaughtered'])

# The stats the user has changed
shifted_avg_input = pd.DataFrame([[
    global_stats['Population'] / n_countries,
    (global_stats['Land_Use(ha)'] / n_countries) * (0.8 + (0.2 * meat_multiplier)),
    global_stats['Water_Use(m3)'] / n_countries,
    (global_stats['Animals_Slaughtered'] / n_countries) * meat_multiplier
]], columns=['Population', 'Land_Use(ha)', 'Water_Use(m3)', 'Animals_Slaughtered'])

# Use the model to predict the new values
base_p = model.predict(baseline_avg_input)[0]
shift_p = model.predict(shifted_avg_input)[0]
diet_impact_ratio = shift_p / base_p
future_trend = (1.005 ** years_ahead)
f_co2 = global_stats['CO2_Emissions'] * future_trend * diet_impact_ratio
f_pop = global_stats['Population'] * ((1.011) ** years_ahead)
f_animals = (global_stats['Animals_Slaughtered'] * ((1.008) ** years_ahead)) * meat_multiplier

# Display the results based on the values the user picked
st.write("")
st.markdown(f"## Global Data for {target_year}")
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Population", f"{f_pop/1e9:.2f} Billion")
col2.metric("Animals Slaughtered", f"{f_animals/1e9:.2f} Billion")
col3.metric(
    "Predicted CO² Emissions",
    f"{f_co2:,.0f} kt",
    delta=f"{f_co2 - global_stats['CO2_Emissions']:,.0f} kt vs 2023",
    delta_color="inverse"
)

# { The Present Data Section }
st.markdown("---")
st.title("🔥 The Present.")

# Emissions information
st.write(f"In 2023, the world produced **{global_stats['CO2_Emissions']:,.0f} kt** of CO²-equivalent emissions.")
st.subheader("Global CO² Emissions (1961 - 2023)")
animals_billion = global_stats['Animals_Slaughtered'] / 1e9
global_history = master_df.groupby('Year')['CO2_Emissions'].sum().reset_index()
global_history['Year'] = global_history['Year'].astype(str)
st.line_chart(data=global_history, x='Year', y='CO2_Emissions')

# Animals Slaughtered information
st.write(f"Every year, over **{animals_billion:.2f} Billion** land animals are slaughtered for the meat industry alone.")
st.subheader("Global Animals Slaughtered (1961 - 2023)")
animal_history = master_df.groupby('Year')['Animals_Slaughtered'].sum().reset_index()
animal_history['Year'] = animal_history['Year'].astype(str)
animal_history['Animals (Billions)'] = animal_history['Animals_Slaughtered'] / 1e9
st.line_chart(data=animal_history, x='Year', y='Animals (Billions)')

# Show the Full Dataset
st.markdown("---")
st.title("🔍 Browse the full Dataset")
st.dataframe(master_df.style.format({
    'Population': "{:,.0f}",
    'Land_Use(ha)': "{:,.0f}",
    'Water_Use(m3)': "{:,.2f}",
    'CO2_Emissions': "{:,.2f}"
}))

# Credits
st.markdown("---")
st.caption("Developed for Wrexham University by Tiffany Davies | https://github.com/TiffyPox")
