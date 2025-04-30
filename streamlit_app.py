import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from PIL import Image


st.set_page_config(page_title="لوحة المعلومات العقارية ", layout="wide", initial_sidebar_state="collapsed")

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
def load_model():
    return joblib.load("lgbm.joblib")

model = load_model()

# Relevant features for prediction
relevant_features = [
    'beds', 'livings', 'wc', 'area', 'street_width', 'age', 'street_direction', 'ketchen',
    'furnished', 'location.lat', 'location.lng', 'city_id', 'district_id'
]

# Prediction function
def predict_price(new_record):
    new_record_df = pd.DataFrame([new_record])
    new_record_df = pd.get_dummies(new_record_df, drop_first=True)
    new_record_df = new_record_df.reindex(columns=relevant_features, fill_value=0)
    return model.predict(new_record_df)[0]



# Layout with 3 columns and 2 rows


# First Row: Map, Specification & Prediction
col1, col2 = st.columns([1, 2])



with col1:
    st.subheader("📍 اختر الموقع")
    
    # Set default location to Riyadh, Saudi Arabia
    riyadh_lat, riyadh_lng = 24.7136, 46.6753
    if 'location_lat' not in st.session_state:
        st.session_state['location_lat'] = riyadh_lat
    if 'location_lng' not in st.session_state:
        st.session_state['location_lng'] = riyadh_lng
    
    # Folium map centered and restricted to Riyadh
    m = folium.Map(
        location=[riyadh_lat, riyadh_lng], 
        zoom_start=11, 
        max_bounds=True,
        min_lat=24.4, max_lat=25.0,  # Approximate bounding box for Riyadh
        min_lon=46.4, max_lon=47.0
    )
    
    marker = folium.Marker(
        location=[st.session_state['location_lat'], st.session_state['location_lng']],
        draggable=True
    )
    marker.add_to(m)
    
    map_data = st_folium(m, width=700, height=400)
    
    if map_data['last_clicked']:
        st.session_state['location_lat'] = map_data['last_clicked']['lat']
        st.session_state['location_lng'] = map_data['last_clicked']['lng']
    
    st.write(f"الموقع المحدد: {st.session_state['location_lat']:.4f}, {st.session_state['location_lng']:.4f}")

# Column 2: Input Form
with col2:
    st.subheader("🏠 أدخل تفاصيل المنزل لتقدير قيمته السوقية")

    # Create a form for house details
    with st.form("house_details_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            beds = st.slider("عدد غرف النوم 🛏️", 3, 7, 3)
            livings = st.slider("عدد غرف المعيشة 🛋️", 1, 7, 1)
            wc = st.slider("عدد دورات المياه 🚽", 2, 5, 2)
            area = st.number_input("المساحة (متر مربع) 📏", 150.0, 12000.0, 150.0)

        with col_b:
            street_width = st.selectbox("عرض الشارع (متر) 🛣️", [10, 12, 15, 18, 20, 25], index=2)
            age = st.number_input("عمر العقار 🗓️", 0, 36, 5)
            street_direction = st.selectbox("نوع الواجهة 🧭", [
                "واجهة شمالية", "واجهة شرقية", "واجهة غربية", "واجهة جنوبية",
                "واجهة شمالية شرقية", "واجهة جنوبية شرقية", "واجهة جنوبية غربية", "واجهة شمالية غربية",
                "الفلة تقع على ثلاثة شوارع", "الفلة تقع على أربعة شوارع"
            ])
            ketchen = st.selectbox("وجود المطبخ 🍳", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
            furnished = st.selectbox("الفلة مؤثثة 🪑", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")

        # District selection
        city_name_to_id = {'الرياض': 66}
        district_data = [
           (470, 'حي السويدي', 'الرياض'),
    (692, 'حي عتيقة', 'الرياض'),
        (474, 'حي الشرفية', 'الرياض'),
        (606, 'حي النسيم الغربي', 'الرياض'),
       (590, 'حي المونسية', 'الرياض'),
   
    (448, 'حي الرماية', 'الرياض'),
        (514, 'حي الغدير', 'الرياض'),
    (718, 'حي نمار', 'الرياض'),
       (622, 'حي الوادي', 'الرياض'),
    (616, 'حي النور', 'الرياض'),
       (542, 'حي المربع', 'الرياض'),
       (416, 'حي الحمراء', 'الرياض'),
    
    (504, 'حي العقيق', 'الرياض'),
   
    (604, 'حي النسيم الشرقي', 'الرياض'),
        (568, 'حي الملز', 'الرياض'),
    
    (548, 'حي المروج', 'الرياض'),
       (658, 'حي جامعة الملك سعود', 'الرياض'),
   
    (424, 'حي الدار البيضاء', 'الرياض'),
        (492, 'حي الضباط', 'الرياض'),
       
    (688, 'حي ظهرة لبن', 'الرياض'),
        (532, 'حي القدس', 'الرياض'),
    
    (496, 'حي العريجاء', 'الرياض'),
        (418, 'حي الخالدية', 'الرياض'),
    (436, 'حي الرائد', 'الرياض'),
        (612, 'حي النموذجية', 'الرياض'),
        (410, 'حي الجنادرية', 'الرياض'),
    (682, 'حي طويق', 'الرياض'),
        (510, 'حي العمل', 'الرياض'),
        (706, 'حي قرطبة', 'الرياض'),
    (588, 'حي المهدية', 'الرياض'),
    (690, 'حي ظهرة نمار', 'الرياض'),
    (536, 'حي القيروان', 'الرياض'),
        (644, 'حي ام سليم', 'الرياض'),
    (516, 'حي الغنامية', 'الرياض'),
        (716, 'حي منفوحة الجديدة', 'الرياض'),
        (530, 'حي القادسية', 'الرياض'),
        (554, 'حي المصفاة', 'الرياض'),
       (398, 'حي البديعة', 'الرياض'),
       (582, 'حي المنار', 'الرياض'),
    (662, 'حي جرير', 'الرياض'),
        (566, 'حي المغرزات', 'الرياض'),
    (526, 'حي الفيحاء', 'الرياض'),
       (714, 'حي منفوحة', 'الرياض'),
        (586, 'حي المنصورية', 'الرياض'),
        (442, 'حي الرحمانية', 'الرياض'),
        (392, 'حي الازدهار', 'الرياض'),
    (574, 'حي الملك عبدالله', 'الرياض'),
    (684, 'حي طيبة', 'الرياض'),
   
    (488, 'حي الصفا', 'الرياض'),
    (708, 'حي لبن', 'الرياض'),
        (646, 'حي أحد', 'الرياض'),
        (520, 'حي الفاروق', 'الرياض'),
        (698, 'حي عكاظ', 'الرياض'),
    (614, 'حي النهضة', 'الرياض'),
       (572, 'حي الملك عبدالعزيز', 'الرياض'),
       (578, 'حي الملك فيصل', 'الرياض'),
       (458, 'حي الزهرة', 'الرياض'),
  
    (712, 'حي مطار الملك خالد الدولي', 'الرياض'),
  
    (672, 'حي سلطانة', 'الرياض'),
    (618, 'حي الهدا', 'الرياض'),
    (524, 'حي الفوطة', 'الرياض'),
    (518, 'حي الفاخرية', 'الرياض'),
    (506, 'حي العليا', 'الرياض'),
        (674, 'حي شبرا', 'الرياض'),
    (444, 'حي الرفيعة', 'الرياض'),
       (404, 'حي التعاون', 'الرياض'),
       (592, 'حي المؤتمرات', 'الرياض'),
    
    (450, 'حي الروابي', 'الرياض'),
    (422, 'حي الخليج', 'الرياض'),
    (632, 'حي الياسمين', 'الرياض'),
       (558, 'حي المعذر', 'الرياض'),
    (624, 'حي الورود', 'الرياض'),
       (546, 'حي المروة', 'الرياض'),
    (478, 'حي الشفا', 'الرياض'),
    (396, 'حي الاندلس', 'الرياض'),
    (494, 'حي العارض', 'الرياض'),
        (598, 'حي الندى', 'الرياض'),
       (446, 'حي الرمال', 'الرياض'),
        (676, 'حي صلاح الدين', 'الرياض'),
       (502, 'حي العزيزية', 'الرياض'),
    (602, 'حي النزهة', 'الرياض'),
       (570, 'حي الملقا', 'الرياض'),
    (620, 'حي الواحة', 'الرياض'),
    
    (680, 'حي ضاحية نمار', 'الرياض'),
    (420, 'حي الخزامى', 'الرياض'),
       (456, 'حي الزهراء', 'الرياض'),
    (710, 'حي مركز الملك عبدالله للدراسات والبحوث', 'الرياض'),
       (428, 'حي الدريهمية', 'الرياض'),
       (464, 'حي السلام', 'الرياض'),
        (564, 'حي المعيزلة', 'الرياض'),
    (522, 'حي الفلاح', 'الرياض'),
        (560, 'حي المعذر الشمالي', 'الرياض'),
       (538, 'حي المحمدية', 'الرياض'),
    
    (528, 'حي الفيصلية', 'الرياض'),
        (550, 'حي المشاعل', 'الرياض'),
        (694, 'حي عرقة', 'الرياض'),
        (668, 'حي ديراب', 'الرياض'),
    (460, 'حي السعادة', 'الرياض'),
        (696, 'حي عريض', 'الرياض'),
       (652, 'حي ثليم', 'الرياض'),
    (686, 'حي ظهرة البديعة', 'الرياض'),
       
    (580, 'حي المناخ', 'الرياض'),
    
    (656, 'حي جامعة الأميرة نورة', 'الرياض'),
    (400, 'حي البرية', 'الرياض'),
    (640, 'حي ام الحمام الغربي', 'الرياض'),
    (552, 'حي المصانع', 'الرياض'),
        (704, 'حي غرناطة', 'الرياض'),
        (452, 'حي الروضة', 'الرياض'),
        (600, 'حي النرجس', 'الرياض'),
    (654, 'حي جامعة الامام محمد بن سعود الاسلامية', 'الرياض'),
    (394, 'حي الاسكان', 'الرياض'),
    (482, 'حي الشهداء', 'الرياض'),
    (576, 'حي الملك فهد', 'الرياض'),
       (486, 'حي الصحافة', 'الرياض'),
    (666, 'حي خشم العان', 'الرياض'),
        (626, 'حي الوزارات', 'الرياض'),
        (440, 'حي الربيع', 'الرياض'),
    (650, 'حي بنبان', 'الرياض'),
       (466, 'حي السلي', 'الرياض'),
   
    (584, 'حي المنصورة', 'الرياض'),
    (438, 'حي الربوة', 'الرياض'),
    
    (408, 'حي الجزيرة', 'الرياض'),
        (608, 'حي النظيم', 'الرياض'),
    (596, 'حي النخيل', 'الرياض'),
    (390, 'حي اشبيلية', 'الرياض'),
    (412, 'حي الحائر', 'الرياض'),
    (1326, 'حي مدينة العمال', 'الدمام'),
    (406, 'حي الجرادية', 'الرياض'),
        (610, 'حي النفل', 'الرياض'),
    (476, 'حي الشرق', 'الرياض'),
    (648, 'حي بدر', 'الرياض'),
        (544, 'حي المرسلات', 'الرياض'),
    (472, 'حي السويدي الغربي', 'الرياض'),
    
    (636, 'حي اليمامة', 'الرياض'),
    (540, 'حي المدينة الصناعية الجديدة', 'الرياض'),
    (454, 'حي الريان', 'الرياض'),
       (634, 'حي اليرموك', 'الرياض'),
    (556, 'حي المصيف', 'الرياض'),
  
    (660, 'حي جبرة', 'الرياض'),
    (468, 'حي السليمانية', 'الرياض'),
    (630, 'حي الوشام', 'الرياض'),
        (498, 'حي العريجاء الغربية', 'الرياض'),
    (490, 'حي الصناعية', 'الرياض'),
    (500, 'حي العريجاء الوسطى', 'الرياض'),
    (700, 'حي عليشة', 'الرياض'),
    (702, 'حي غبيرة', 'الرياض'),
    (638, 'حي ام الحمام الشرقي', 'الرياض'),
       (664, 'حي حطين', 'الرياض'),
   
    (414, 'حي الحزم', 'الرياض'),
        ]

        selected_district = st.selectbox("اختر الحي 🏙️", district_data, format_func=lambda x: f"{x[1]} ({x[2]})")
        district_id = selected_district[0]
        city_id = city_name_to_id[selected_district[2]]

        # ✅ Submit button inside the form
        submitted = st.form_submit_button("🔮 حساب القيمة التقديرية")

        if submitted:
            with st.spinner('جاري الحساب...'):
                new_record = {
                    'beds': beds, 'livings': livings, 'wc': wc, 'area': area,
                    'street_width': street_width, 'age': age, 'street_direction': street_direction,
                    'ketchen': ketchen, 'furnished': furnished,
                    'location.lat': st.session_state['location_lat'],
                    'location.lng': st.session_state['location_lng'],
                    'city_id': city_id, 'district_id': district_id
                }
                predicted_price = predict_price(new_record)
            st.success('تمت عملية التوقع بنجاح!')
            st.metric(label="السعر التقريبي", value=f"ريال {predicted_price:,.2f}")
# Bottom section: Visualization
st.header("📊 رؤى")
# Second Row: Feature Importance, Deals Count, Deals Cost
import os
import pandas as pd
import streamlit as st
import plotly.express as px

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
        expected_columns = {"Feature", "Importance"}
        if not expected_columns.issubset(df.columns):
            missing_cols = expected_columns - set(df.columns)
            st.error(f"⚠️ CSV file is missing required columns: {missing_cols}")
            return None

        return df
    except Exception as e:
        st.error(f"⚠️ Error reading {FEATURE_IMPORTANCE_FILE}: {e}")
        return None

df_features = load_feature_importance_data()

# --- 📊 Feature Importance Section ---
col3, col4, col5 = st.columns([1, 1, 1]) 

with col3:
    if df_features is not None:
        st.subheader("📊 تأثير الخصائص على السعر")

        # ✅ Plot feature importance (assuming correct columns exist)
        fig_features = px.bar(
            df_features, x="Importance", y="Feature", orientation="h",
            title="Feature Importance", color="Importance"
        )
        st.plotly_chart(fig_features)
    else:
        st.error("❌ Feature importance data not found!")



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

    # --- 📊 Number of Deals per District ---
with col4:
    st.subheader("📊 عدد الصفقات حسب الحي")
    deals_per_district = df_deals_filtered.groupby(["District"])["Deal Count"].sum().reset_index()
    
    # ✅ Sort districts by total Deal Count in descending order
    deals_per_district = deals_per_district.sort_values(by="Deal Count", ascending=False)
    
    fig_deals = px.bar(
        df_deals_filtered, x="District", y="Deal Count", color="Year",
        barmode="group", title="Number of Deals per District per Year",
        category_orders={"District": deals_per_district["District"].tolist()}  # Sorting reflected in plot
    )
    fig_deals.update_layout(coloraxis_colorbar=dict(tickvals=[2022, 2023, 2024], ticktext=["2022", "2023", "2024"]))  # ✅ Only show 2022, 2023, 2024
    st.plotly_chart(fig_deals)

   # --- 💰 Total Cost of Deals per District ---
with col5:
    st.subheader("💰 التكلفة الكلية للصفقات")

    if df_cost_filtered is not None:
        cost_per_district = df_cost_filtered.groupby(["District"])["Total Cost"].sum().reset_index()

        # ✅ Sort districts by total Total Cost in descending order
        cost_per_district = cost_per_district.sort_values(by="Total Cost", ascending=False)

        fig_cost = px.bar(
            df_cost_filtered, x="District", y="Total Cost", color="Year",
            barmode="stack", title="Total Cost of Deals per District per Year",
            category_orders={"District": cost_per_district["District"].tolist()}  # Sorting reflected in plot
        )
        fig_cost.update_layout(coloraxis_colorbar=dict(tickvals=[2022, 2023, 2024], ticktext=["2022", "2023", "2024"]))  # ✅ Only show 2022, 2023, 2024
        st.plotly_chart(fig_cost)
    
    else:
        st.error("❌ Data files not found! Please ensure the files are correctly stored in the predefined locations.")



# Footer
st.markdown("---")
