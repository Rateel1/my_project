import streamlit as st
import joblib
import xgboost
import os

import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from PIL import Image
import numpy as np

# قراءة ملف مراكز الأحياء من نفس مجلد التطبيق
district_centers = pd.read_excel("district_centers.xlsx")

st.set_page_config(page_title="لوحة المعلومات العقارية", layout="wide", initial_sidebar_state="collapsed")

# Centered title using markdown and HTML
st.markdown(
    "<h1 style='text-align: center; direction: rtl;'> 🏠لوحة  المعلومات  العقارية</h1>",
    unsafe_allow_html=True
)

# Custom CSS for styling
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.stButton>button {
    color: #ffffff;
    background-color: #4CAF50;
    border-radius: 5px;
}
.stMetricLabel {
    font-size: 20px;
}
.stMetricValue {
    font-size: 40px;
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
# Load the model
def load_model():
    return joblib.load("selected_xgb_modelafter.joblib")

model = load_model()

# Load the columns used during training
model_columns = joblib.load("model_columnsXGB.pkl")

# ✅ Prediction function
def predict_price(new_record):
    model = joblib.load("selected_xgb_modelafter.joblib")
    model_columns = joblib.load("xgb_model_featuresafter.pkl")

    # Convert to DataFrame
    new_record_df = pd.DataFrame([new_record])

    # One-hot encode if needed
    new_record_df = pd.get_dummies(new_record_df)

    # Add missing columns
    for col in model_columns:
        if col not in new_record_df:
            new_record_df[col] = 0
    new_record_df = new_record_df[model_columns]

    # Ensure correct dtype
    new_record_df = new_record_df.astype(float)

    # Predict
    log_price = model.predict(new_record_df)[0]
    return np.expm1(log_price)  # if log target

# أول صف: الخريطة + النموذج
col1, col2 = st.columns([1, 2])

with col1:
    with st.container():
        st.subheader("📍 اختر الموقع")

        # الإحداثيات الافتراضية - الرياض
        riyadh_lat, riyadh_lng = 24.7136, 46.6753
        if 'location_lat' not in st.session_state:
            st.session_state['location_lat'] = riyadh_lat
        if 'location_lng' not in st.session_state:
            st.session_state['location_lng'] = riyadh_lng

        # إنشاء الخريطة
        m = folium.Map(
            location=[st.session_state['location_lat'], st.session_state['location_lng']],
            zoom_start=12,
            tiles="CartoDB positron",
            control_scale=True
        )

        from folium.plugins import MeasureControl, MousePosition
        m.add_child(MeasureControl(primary_length_unit='kilometers'))
        m.add_child(MousePosition(position='bottomright'))

        marker = folium.Marker(
            location=[st.session_state['location_lat'], st.session_state['location_lng']],
            draggable=True,
            icon=folium.Icon(color="red", icon="map-marker")
        )
        marker.add_to(m)

        map_data = st_folium(m, width=700, height=450)

        if map_data.get('last_clicked'):
            st.session_state['location_lat'] = map_data['last_clicked']['lat']
            st.session_state['location_lng'] = map_data['last_clicked']['lng']

        st.success(f"📌 الموقع المحدد: {st.session_state['location_lat']:.4f}, {st.session_state['location_lng']:.4f}")

with col2:
    st.subheader("🏠 أدخل تفاصيل المنزل لتقدير قيمته السوقية")

    with st.form("house_details_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            beds = st.slider("عدد غرف النوم 🛏️", 3, 7, 3)
            livings = st.slider("عدد غرف المعيشة 🛋️", 1, 7, 1)
            wc = st.slider("عدد دورات المياه 🚽", 2, 5, 2)
            area = st.number_input("المساحة (متر مربع) 📏", 150.0, 600.0, 150.0)

        with col_b:
            street_width = st.selectbox("عرض الشارع (متر) 🛣️", [10, 12, 15, 18, 20, 25], index=2)
            age = st.number_input("عمر العقار 🗓️", 0, 5, 1)
            street_direction = st.selectbox("نوع الواجهة 🧭", [
                "واجهة شمالية", "واجهة شرقية", "واجهة غربية", "واجهة جنوبية",
                "واجهة شمالية شرقية", "واجهة جنوبية شرقية", "واجهة جنوبية غربية", "واجهة شمالية غربية",
                "الفلة تقع على ثلاثة شوارع", "الفلة تقع على أربعة شوارع"
            ])
            ketchen = st.selectbox("المطبخ مجهز🍳؟ ", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
            furnished = st.selectbox("الفلة مؤثثة🪑؟", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")

        # استخدام البيانات من ملف district_centers بدلاً من القائمة اليدوية
        district_list = list(zip(district_centers['district_id'], district_centers['district_name'], district_centers['city_name']))
        selected_district = st.selectbox("اختر الحي 🏙️", district_list, format_func=lambda x: f"{x[1]} ({x[2]})")

        district = selected_district[1]

        submitted = st.form_submit_button("🔮 حساب القيمة التقديرية")

        if submitted:
            with st.spinner('جاري الحساب...'):
                new_record = {
                    'beds': beds, 'livings': livings, 'wc': wc, 'area': area,
                    'street_width': street_width, 'age': age, 'street_direction': street_direction,
                    'ketchen': ketchen, 'furnished': furnished,
                    'location.lat': st.session_state['location_lat'],
                    'location.lng': st.session_state['location_lng'],
                    'district': district
                }
                predicted_price = predict_price(new_record)
            st.success('تمت عملية التوقع بنجاح!')
            st.metric(label="السعر التقريبي", value=f"ريال {predicted_price:,.2f}")

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
