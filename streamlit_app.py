import streamlit as st
import joblib
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from PIL import Image
from folium.plugins import MeasureControl, MousePosition
from math import radians, sin, cos, sqrt, atan2
import os
import plotly.express as px

# إعداد الصفحة
st.set_page_config(page_title="لوحة المعلومات العقارية", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <h1 style='text-align: center; font-size: 4rem; margin-top: 0;'> لوحة المعلومات العقارية 🏠</h1>
""", unsafe_allow_html=True)

st.markdown("""

<style>

html, body, [data-testid="stAppViewContainer"] {
    direction: rtl;
    text-align: right;
}

h2, h3, h4, h5, h6{
    text-align: right;
    font-size:2rem !important;
}

section[data-testid="stSidebar"] {
    direction: rtl;
    text-align: right;
}

.stNumberInput input {
    font-size: 1.6rem !important;
}



[data-testid="stForm"] label {
    font-size: 2rem !important;
    font-weight: bold !important;
    display: block;
    
    text-align: right;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    font-size: 2rem !important;
    font-weight: bold !important;
    text-align: right;
      
}
/* Selectbox height */
div[data-baseweb="select"] {
    min-height: 70px !important;
}

/* Selected value area (visible text when closed) */
div[data-baseweb="select"] > div {
    min-height: 70px !important;
    display: flex;
    align-items: center !important;
    font-size: 1.8rem !important;
}

/* Fix font size for selected item text */
div[data-baseweb="select"] div[role="combobox"] {
    font-size: 1.8rem !important;
}

/* Font size for dropdown menu options */
div[data-baseweb="menu"] div[role="option"] {
    font-size: 2rem !important;
}


</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
div.stForm button {
    font-size: 2.4rem !important;
    font-weight: bold !important;
    background-color:#c0c0c0 !important;
    color: black !important;
    border-radius: 8px !important;
    padding: 0.4em 1.2em !important;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Apply font size to all elements inside the form */
div[data-testid="stForm"] * {
    font-size: 1.8rem !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("selected_xgb_modelafter.joblib")

@st.cache_resource
def load_model_columns():
    return joblib.load("xgb_model_featuresafter.pkl")

model = load_model()
model_columns = load_model_columns()

def predict_price(new_record):
    new_record_df = pd.DataFrame([new_record])
    new_record_df = pd.get_dummies(new_record_df)
    for col in model_columns:
        if col not in new_record_df:
            new_record_df[col] = 0
    new_record_df = new_record_df[model_columns].astype(float)
    log_price = model.predict(new_record_df)[0]
    return np.expm1(log_price)

def haversine_distance(lat1, lng1, lat2, lng2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

district_centers = pd.read_excel("district_centers.xlsx").dropna(subset=['district'])

riyadh_lat, riyadh_lng = 24.7136, 46.6753
st.session_state.setdefault('location_lat', float(riyadh_lat))
st.session_state.setdefault('location_lng', float(riyadh_lng))
st.session_state.setdefault('location_manually_set', False)
st.session_state.setdefault('selected_district', district_centers.iloc[0]['district'])


col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<h1 style='font-size:2.4rem;'>📍 اختر الموقع</h1>", unsafe_allow_html=True)

    if st.button("🔁 إعادة تعيين الموقع"):
        st.session_state['location_manually_set'] = False
        selected_row = district_centers[district_centers['district'] == st.session_state['selected_district']].iloc[0]
        st.session_state['location_lat'] = selected_row['location.lat']
        st.session_state['location_lng'] = selected_row['location.lng']

    m = folium.Map(location=[st.session_state['location_lat'], st.session_state['location_lng']], zoom_start=12, tiles="CartoDB positron", control_scale=True)
    m.add_child(MeasureControl(primary_length_unit='kilometers'))
    m.add_child(MousePosition(position='bottomright'))

    marker = folium.Marker(location=[st.session_state['location_lat'], st.session_state['location_lng']],
                           draggable=True, icon=folium.Icon(color="red", icon="map-marker"))
    marker.add_to(m)

    map_data = st_folium(m, width=700, height=450)
    if map_data.get('last_clicked'):
        st.session_state['location_lat'] = map_data['last_clicked']['lat']
        st.session_state['location_lng'] = map_data['last_clicked']['lng']
        st.session_state['location_manually_set'] = True
        distances = district_centers.apply(
            lambda row: haversine_distance(st.session_state['location_lat'], st.session_state['location_lng'],
                                           row['location.lat'], row['location.lng']), axis=1)
        st.session_state['selected_district'] = district_centers.loc[distances.idxmin(), 'district']

    st.success(f"📌 الموقع المحدد: {st.session_state['location_lat']:.4f}, {st.session_state['location_lng']:.4f}")

with col2:
   
    st.markdown("<h1 style='font-size:2.4rem;'>🏠 أدخل تفاصيل المنزل لتقدير قيمته السوقية</h1>", unsafe_allow_html=True)

    with st.form("house_details_form"):
        col_a, col_b = st.columns(2)
   
        with col_a:
           
            st.markdown("<label style='font-size:1rem; font-weight:bold;'>عدد غرف المعيشة 🛋️</label>", unsafe_allow_html=True)
            livings = st.selectbox("", list(range(1, 8)), key="livings")
           
            st.markdown("<label style='font-size:1rem; font-weight:bold;'>المساحة (متر مربع) 📏</label>", unsafe_allow_html=True)
            area = st.number_input("", 150.0, 600.0, 150.0, key="area")
            st.markdown("<label style='font-size:1.8rem;'>اختر الحي 🏙️</label>", unsafe_allow_html=True)
            district = st.selectbox("", district_centers['district'].unique().tolist(),
                                index=district_centers['district'].tolist().index(st.session_state['selected_district']),
                                key="district")

       

        with col_b:
            st.markdown("<label style='font-size:1rem; font-weight:bold;'>عرض الشارع (متر) 🛣️</label>", unsafe_allow_html=True)
            street_width = st.selectbox("", [10, 12, 15, 18, 20, 25], key="street_width")
            st.markdown("<label style='font-size:1rem; font-weight:bold;'>عمر العقار 🗓️</label>", unsafe_allow_html=True)
            age = st.selectbox("", list(range(0, 6)), key="age")
            st.markdown("<label style='font-size:1rem; font-weight:bold;'>نوع الواجهة 🧭 </label>", unsafe_allow_html=True)
            street_direction = st.selectbox("", [
                "واجهة شمالية", "واجهة شرقية", "واجهة غربية", "واجهة جنوبية",
                "واجهة شمالية شرقية", "واجهة جنوبية شرقية", "واجهة جنوبية غربية", "واجهة شمالية غربية"
              
            ], key="street_direction")
           

        if not st.session_state['location_manually_set']:
            row = district_centers[district_centers['district'] == district].iloc[0]
            st.session_state['location_lat'] = row['location.lat']
            st.session_state['location_lng'] = row['location.lng']
        st.session_state['selected_district'] = district
        
       
        submitted = st.form_submit_button(" حساب القيمة التقديرية 🔮")

           
        if submitted:
            with st.spinner('جاري الحساب...'):
                input_data = {
                     'livings': livings, 'area': area,
                    'street_width': street_width, 'age': age, 'street_direction': street_direction,
                   
                    'location.lat': st.session_state['location_lat'],
                    'location.lng': st.session_state['location_lng'],
                    'district': district
                }
                price = predict_price(input_data)
                st.success("تمت عملية التوقع بنجاح!")
                st.metric("السعر التقريبي", f"ريال {price:,.2f}")

    


st.markdown("<h1 style='font-size:2.4rem;'>📊 الرؤى واتجاهات السوق العقاري</h1>", unsafe_allow_html=True)

# --- 📊 Feature Importance Section ---
FEATURE_IMPORTANCE_FILE = "feature importance.csv"  

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
    st.subheader("📊 تأثير الخصائص على السعر")
    if df_features is not None and all(col in df_features.columns for col in ["الخاصية", "تأثيرها على السعر"]):
  
        fig_features = px.bar(
            df_features,
            x="تأثيرها على السعر",
            y="الخاصية",
            orientation="h",
            color="تأثيرها على السعر",
            height=400  # تقليل الارتفاع
        )
        fig_features.update_layout(
            margin=dict(l=100, r=20, t=40, b=40),  # ضبط الهوامش
            yaxis=dict(
                tickfont=dict(size=14),
                title=dict(text="الخاصية", standoff=60, font=dict(size=20))
            ),
            xaxis=dict(
                title=dict(text="تأثيرها على السعر", font=dict(size=20))
            )
        )

        st.plotly_chart(fig_features, use_container_width=True)
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
    
    # تجميع عدد الصفقات حسب الحي
    deals_per_district = df_deals_filtered.groupby(["District"])["Deal Count"].sum().reset_index()
    deals_per_district = deals_per_district.sort_values(by="Deal Count", ascending=False)

    # رسم المخطط
    fig_deals = px.bar(
        df_deals_filtered,
        x="District",
        y="Deal Count",
        color="Year",
        category_orders={"District": deals_per_district["District"].tolist()},
        height=400  # تقليل الارتفاع لتناسق العرض
    )

    # تنسيق الرسم البياني
    fig_deals.update_layout(
        margin=dict(l=60, r=20, t=40, b=40),
        xaxis=dict(
            title=dict(
                text="الحي",standoff=70,
                font=dict(size=20)
            ),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title=dict(
                text="عدد الصفقات",  # ✅ عنوان المحور Y بالعربية
                standoff=60,
                font=dict(size=20)
            ),
            tickfont=dict(size=14)
        ),
        coloraxis_colorbar=dict(
            title="السنة",  # ✅ تعريب شريط الألوان
            tickvals=[2022, 2023, 2024],
            ticktext=["2022", "2023", "2024"]
        )
    )

    # عرض المخطط في Streamlit
    st.plotly_chart(fig_deals, use_container_width=True)
   

with col5:
    st.subheader("💰 التكلفة الكلية للصفقات")

    if df_cost_filtered is not None:
        # تجميع التكلفة حسب الحي
        cost_per_district = df_cost_filtered.groupby(["District"])["Total Cost"].sum().reset_index()
        cost_per_district = cost_per_district.sort_values(by="Total Cost", ascending=False)

        # رسم المخطط
        fig_cost = px.bar(
            df_cost_filtered,
            x="District",
            y="Total Cost",
            color="Year",
            category_orders={"District": cost_per_district["District"].tolist()},
            height=400  # تقليل الارتفاع لتناسق العرض
        )

        # تنسيق الرسم البياني
        fig_cost.update_layout(
            margin=dict(l=60, r=20, t=40, b=40),
            xaxis=dict(
                title=dict(
                    text="الحي", standoff=70,
                    font=dict(size=20)
                ),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title=dict(
                    text="التكلفة الكلية",
                    standoff=60,
                    font=dict(size=20)
                ),
                tickfont=dict(size=14)
            ),
            coloraxis_colorbar=dict(
                title="السنة",
                tickvals=[2022, 2023, 2024],
                ticktext=["2022", "2023", "2024"]
            )
        )

        # عرض المخطط في Streamlit
        st.plotly_chart(fig_cost, use_container_width=True)
    
    else:
        st.error("❌ البيانات غير متوفرة. الرجاء التأكد من توفر الملفات في المسارات المحددة.")


# Footer
st.markdown("---")
