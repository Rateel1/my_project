import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import folium
import os
from streamlit_folium import st_folium
from PIL import Image

# إعداد الصفحة
st.set_page_config(page_title="لوحة المعلومات العقارية", layout="wide", initial_sidebar_state="expanded")

# تحميل النموذج
@st.cache_resource
def load_model():
    return joblib.load("lgbm.joblib")

model = load_model()

# تحميل قائمة الأحياء من ملف Excel
@st.cache_data
def load_districts():
    try:
        df = pd.read_excel("district_centers.xlsx", engine="openpyxl")
        return sorted(df["district"].dropna().unique())
    except Exception as e:
        st.error(f"فشل في تحميل الأحياء: {e}")
        return []

districts_list = load_districts()

# تنسيق CSS
st.markdown("""
<style>
.stApp { background-color: #f8f9fa; }
.stButton>button {
    color: #ffffff;
    background-color: #e63946;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
}
.metric-box {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    text-align: center;
    margin-bottom: 10px;
}
.metric-label { font-size: 18px; color: #555; }
.metric-value { font-size: 30px; font-weight: bold; color: #e63946; }
</style>
""", unsafe_allow_html=True)

# الشريط الجانبي للتنقل
st.sidebar.header("انتقل")
selected_page = st.sidebar.radio("Go to", ["نظرة عامة", "رؤى السوق  العقاري", "التنبؤ"])

if selected_page == "نظرة عامة":
    st.title("🏠 لوحة المعلومات العقارية")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-box'><div class='metric-label'>Waterfront Properties</div><div class='metric-value'>163</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-box'><div class='metric-label'>Total Bedrooms</div><div class='metric-value'>22K</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-box'><div class='metric-label'>Renovated Properties</div><div class='metric-value'>10K</div></div>", unsafe_allow_html=True)

    st.subheader("📊 عدد الغرف")
    bedrooms_data = pd.DataFrame({
        "Bedrooms": ["3 Bedroom", "4 Bedrooms", "5 Bedrooms", "6 Bedrooms", "7 Bedrooms"],
        "Count": [274, 2760, 9824, 6882, 1601]
    })
    fig_bedrooms = px.bar(bedrooms_data, x="Bedrooms", y="Count", color="Bedrooms", title="عدد الغرف في العقار")
    st.plotly_chart(fig_bedrooms)

elif selected_page == "رؤى السوق  العقاري":
    st.title("📈 رؤى السوق  العقاري")

    DEALS_FILES = {"2022": "selected2022_a.csv", "2023": "selected2023_a.csv", "2024": "selected2024_a.csv"}
    TOTAL_COST_FILE = "deals_total.csv"

    @st.cache_data
    def load_deals_data():
        dataframes = []
        for year, file in DEALS_FILES.items():
            if os.path.exists(file):
                df = pd.read_csv(file)
                df["Year"] = int(year)
                dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True) if dataframes else None

    @st.cache_data
    def load_total_cost_data():
        if os.path.exists(TOTAL_COST_FILE):
            df = pd.read_csv(TOTAL_COST_FILE)
            first_col = df.columns[0]
            df = df.melt(id_vars=[first_col], var_name="Year", value_name="Total Cost")
            df.rename(columns={first_col: "District"}, inplace=True)
            df["Year"] = df["Year"].astype(int)
            return df
        return None

    df_deals = load_deals_data()
    df_cost = load_total_cost_data()

    if df_deals is not None and df_cost is not None:
        st.subheader("📊 عدد الصفقات حسب الحي")
        deals_per_district = df_deals.groupby(["District"])["Deal Count"].sum().reset_index()
        fig_deals = px.bar(deals_per_district, x="District", y="Deal Count", color="District", title="Deals per District")
        st.plotly_chart(fig_deals)

        st.subheader("💰 التكلفة الكلية للصفقات حسب الحي")
        cost_per_district = df_cost.groupby(["District"])["Total Cost"].sum().reset_index()
        fig_cost = px.bar(cost_per_district, x="District", y="Total Cost", color="District", title="Total Cost of Deals")
        st.plotly_chart(fig_cost)
    else:
        st.error("❌ لم يتم العثور على ملفات البيانات!")

elif selected_page == "التنبؤ":
    st.title("🔮 الأسعار التنبؤية")
    with st.form("house_details_form"):
        district = st.selectbox("District", districts_list)
        beds = st.slider("Bedrooms", 3, 7, 3)
        area = st.number_input("Area (sqm)", 150, 12000, 150)
        age = st.number_input("Age of Property", 0, 36, 5)
        street_width = st.selectbox("Street Width (m)", [10, 12, 15, 18, 20, 25])
        submitted = st.form_submit_button("🔎 Predict Price")
        if submitted:
            new_record = {'district': district, 'beds': beds, 'area': area, 'age': age, 'street_width': street_width}
            input_df = pd.DataFrame([new_record])
            predicted_price = model.predict(input_df)[0]
            st.success(f"🏷️ Estimated Price: ${predicted_price:,.2f}")

st.markdown("---")


# Bottom section: Visualization
st.header("📊 رؤى")
# Second Row: Feature Importance, Deals Count, Deals Cost

# --- 📊 Feature Importance Section ---
FEATURE_IMPORTANCE_FILE = "feature importance.csv"  # Ensure file name matches your actual file

@st.cache_data
def load_feature_importance_data():
    """Loads feature importance data from CSV."""
    if not os.path.exists(FEATURE_IMPORTANCE_FILE):
        st.error(f"⚠️ Missing file: {FEATURE_IMPORTANCE_FILE}")
        return None

    try:
        df = pd.read_csv(FEATURE_IMPORTANCE_FILE)

        # ✅ Check column names to avoid KeyError
        expected_columns = {"الخاصية", "تأثيرها على السعر"}
        if not expected_columns.issubset(df.columns):
            missing_cols = expected_columns - set(df.columns)
            st.error(f"⚠️ CSV file is missing required columns: {missing_cols}")
            return None

        return df

    except Exception as e:
        st.error(f"⚠️ Error reading {FEATURE_IMPORTANCE_FILE}: {e}")
        return None


df_features = load_feature_importance_data()
col3, col4, col5 = st.columns([1, 1, 1])


with col3:
    if df_features is not None and all(col in df_features.columns for col in ["الخاصية", "تأثيرها على السعر"]):
        fig_features = px.bar(
            df_features,
            x="تأثيرها على السعر",
            y="الخاصية",
            orientation="h",
            title="Feature Importance",
            color="تأثيرها على السعر"
        )
        st.plotly_chart(fig_features)
    else:
        st.error("تحقق من أسماء الأعمدة: 'الخاصية' و 'تأثيرها على السعر' غير موجودة في df_features")

    
# File paths for CSV files
DEALS_FILES = {
    "2022": "selected2022_a.csv",
    "2023": "selected2023_a.csv",
    "2024": "selected2024_a.csv"
}
TOTAL_COST_FILE = "deals_total.csv"

# ✅ Load & Transform "Total Cost of Deals" CSV
@st.cache_data
def load_total_cost_data():
    if os.path.exists(TOTAL_COST_FILE):
        try:
            df = pd.read_csv(TOTAL_COST_FILE)
            first_col = df.columns[0]
            df = df.melt(id_vars=[first_col], var_name="Year", value_name="Total Cost")
            df.rename(columns={first_col: "District"}, inplace=True)
            df["Year"] = df["Year"].astype(int)
            return df
        except Exception as e:
            st.error(f"⚠️ Error reading {TOTAL_COST_FILE}: {e}")
            return None
    else:
        st.warning(f"⚠️ Missing file: {TOTAL_COST_FILE}")
        return None

# ✅ Load & Transform "Number of Deals" Data from Multiple CSV Files
@st.cache_data
def load_deals_data():
    dataframes = []
    for year, file in DEALS_FILES.items():
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                df["Year"] = int(year)
                dataframes.append(df)
            except Exception as e:
                st.error(f"⚠️ Error reading {file}: {e}")
        else:
            st.warning(f"⚠️ Missing file: {file}")
    return pd.concat(dataframes, ignore_index=True) if dataframes else None

# ✅ Load Data
df_deals = load_deals_data()
df_cost = load_total_cost_data()

if df_deals is not None and df_cost is not None:
   

    # ✅ Sidebar Filters
    valid_years = [year for year in sorted(df_deals["Year"].unique()) if year in [2022, 2023, 2024]]
    selected_year = st.sidebar.selectbox("📅 Select Year", ["All"] + valid_years)
    sort_by = st.sidebar.radio("📊 Sort By", ["Deal Count", "Total Cost"])

    # ✅ Filter Data Based on Selected Year
    if selected_year != "All":
        df_deals_filtered = df_deals[df_deals["Year"] == int(selected_year)]
        df_cost_filtered = df_cost[df_cost["Year"] == int(selected_year)]
    else:
        df_deals_filtered = df_deals
        df_cost_filtered = df_cost

   
with col4:
    st.subheader("📊 عدد الصفقات حسب الحي")
    deals_per_district = df_deals_filtered.groupby(["District"])["Deal Count"].sum().reset_index()
    
    # ✅ Sort districts by total Deal Count in descending order
    deals_per_district = deals_per_district.sort_values(by="Deal Count", ascending=False)
    
    fig_deals = px.bar(
        df_deals_filtered, x="District", y="Deal Count", color="Year",
        #barmode="group", title="Number of Deals per District per Year",
        category_orders={"District": deals_per_district["District"].tolist()}  # Sorting reflected in plot
    )
    fig_deals.update_layout(coloraxis_colorbar=dict(tickvals=[2022, 2023, 2024], ticktext=["2022", "2023", "2024"]))  # ✅ Only show 2022, 2023, 2024
    st.plotly_chart(fig_deals)

   
with col5:
    st.subheader("💰 التكلفة الكلية للصفقات")

    if df_cost_filtered is not None:
        cost_per_district = df_cost_filtered.groupby(["District"])["Total Cost"].sum().reset_index()

        # ✅ Sort districts by total Total Cost in descending order
        cost_per_district = cost_per_district.sort_values(by="Total Cost", ascending=False)

        fig_cost = px.bar(
            df_cost_filtered, x="District", y="Total Cost", color="Year",
            #barmode="stack", title="Total Cost of Deals per District per Year",
            category_orders={"District": cost_per_district["District"].tolist()}  # Sorting reflected in plot
        )
        fig_cost.update_layout(coloraxis_colorbar=dict(tickvals=[2022, 2023, 2024], ticktext=["2022", "2023", "2024"]))  # ✅ Only show 2022, 2023, 2024
        st.plotly_chart(fig_cost)
    
    else:
        st.error("❌ Data files not found! Please ensure the files are correctly stored in the predefined locations.")



# Footer
st.markdown("---")
