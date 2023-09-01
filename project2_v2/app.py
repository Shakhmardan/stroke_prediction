import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pickle

st.set_page_config(
    page_title="stroke prediction app",
    page_icon="🤕",
    layout="wide",
    initial_sidebar_state="expanded"
)


# global variables
df_predicted = pd.DataFrame()
uploaded_files = None


# session state init
if 'df_input' not in st.session_state or uploaded_files is None:
    st.session_state['df_input'] = pd.DataFrame()

if 'df_predicted' not in st.session_state:
    st.session_state['df_predicted'] = pd.DataFrame()

if 'tab_selected' not in st.session_state:
    st.session_state['tab_selected'] = None

def reset_session_state():
    st.session_state['df_input'] = pd.DataFrame()
    st.session_state['df_predicted'] = pd.DataFrame()



# ml section start
numerical = ['age', 'avg_glucose_level', 'bmi'] 
categorical = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# logistic regression model
model_file_path = 'C:/Users/shakh/Desktop/project2_v2/models/lr_model_stroke_prediction.sav'
model = pickle.load(open(model_file_path, 'rb'))
# encoding model DictVectorizer
encoding_model_file_path = 'C:/Users/shakh/Desktop/project2_v2/models/encoding_model.sav'
encoding_model = pickle.load(open(encoding_model_file_path, 'rb'))

@st.cache_data
def predict_stroke(df_input, treshold):
    scaler = MinMaxScaler()
    
    df_orig = df_input.copy()
    df_input[numerical] = scaler.fit_transform(df_input[numerical])

    df_input['gender'] = df_input['gender'].map({'Male': 0, 'Female': 1})
    df_input['ever_married'] = df_input['ever_married'].map({'Yes': 1, 'No': 0})

    dicts_df = df_input[categorical+numerical].to_dict(orient='records')
    X = encoding_model.transform(dicts_df)
    X[np.isnan(X)] = 0
    y_pred = model.predict_proba(X)[:, 1]
    stroke_decision = (y_pred >= treshold)
    df_orig['stroke_predicted'] = stroke_decision
    df_orig['stroke_predicted_probability'] = y_pred

    return df_orig
# ml section end


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


# sidebar section start
with st.sidebar:
    st.title('input data')
    tab1, tab2 = st.tabs(['file data', 'input data'])
    with tab1:
        uploaded_files = st.file_uploader("Choose a CSV file", type=['csv', 'xlsx'], on_change=reset_session_state) 
        if uploaded_files is not None:
            treshold = st.slider('treshold', 0.0, 1.0, 0.5, 0.01)
            prediction_btn = st.button('predict', use_container_width=True)
            st.session_state['df_input'] = pd.read_csv(uploaded_files)
            if prediction_btn:
                st.session_state['df_predicted'] = predict_stroke(st.session_state['df_input'], treshold)
                st.session_state['tab_selected'] = 'tab1'

    with tab2:
        patient_id = st.text_input('patient id', '00000')
        gender = st.selectbox('gender', ('Male', 'Female'))
        age = st.number_input('age', min_value=0, max_value=150)
        hypertension = st.selectbox('hypertension', (1, 0))
        heart_disease = st.selectbox('heart disease', (1, 0))
        ever_married = st.selectbox('ever married', ('Yes', 'No'))
        work_type = st.selectbox('work type', ('Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked'))
        Residence_type = st.selectbox('residence type', ('Urban', 'Rural'))
        avg_glucose_level = st.number_input('glucose level')
        bmi = st.number_input('bmi')
        smoking_status = st.selectbox('smoking status', ('never smoked', 'Unknown', 'formerly smoked', 'smokes'))

        if patient_id is not None:
            treshold = st.slider('treshold', 0.0, 1.0, 0.5, 0.01, key='slider2')
            prediction_btn_tab2 = st.button('predict', use_container_width=True, key='btn_tab2')
            if prediction_btn_tab2:
                st.session_state['tab_selected'] = 'tab2'
                st.session_state['df_input'] = pd.DataFrame({
                    'id': patient_id,
                    'gender': gender,
                    'age': age,
                    'hypertension': hypertension,
                    'heart_disease': heart_disease,
                    'ever_married': ever_married,
                    'work_type': work_type,
                    'Residence_type': Residence_type,
                    'avg_glucose_level': avg_glucose_level,
                    'bmi': bmi,
                    'smoking_status': smoking_status
                }, index=[0])
                st.session_state['df_predicted'] = predict_stroke(st.session_state['df_input'], treshold)




# sidebar section end



# main part
st.image('https://www.nme.com/wp-content/uploads/2023/01/kim-joo-hun-newjeans-ador-hybe-030123.jpg', width=400)
st.title('stroke prediction')

with st.expander("project description"):
    st.write(
        "hellour✧(≧∇≦)ﾉ"
    )

if len(st.session_state['df_input']) > 0:
    st.subheader('file data')
    st.write(st.session_state['df_input'])

if len(st.session_state['df_predicted']) > 0 and st.session_state['tab_selected'] == 'tab1':
    st.subheader('results')
    st.write(st.session_state['df_predicted'])

    res_all_csv = convert_df(st.session_state['df_predicted'])
    st.download_button(
        label="download prediction",
        data=res_all_csv,
        file_name='df-stroke-prediction.csv',
        mime='text/csv',
    )
    risk_patients = st.session_state['df_predicted'][st.session_state['df_predicted']['stroke_predicted'] == 1]

    if len(risk_patients) > 0:
        st.subheader('patiends with high risk')
        st.write(risk_patients)


if len(st.session_state['df_predicted']) > 0 and st.session_state['tab_selected'] == 'tab2':
    st.subheader('results')
    st.write(st.session_state['df_predicted'])

    res_all_csv = convert_df(st.session_state['df_predicted'])
    st.download_button(
        label="download prediction",
        data=res_all_csv,
        file_name='df-stroke-prediction.csv',
        mime='text/csv',
    )

    if st.session_state['df_predicted']['stroke_predicted'][0] == 0:
        st.subheader(f':green[no] stroke with probability -> {st.session_state["df_predicted"]["stroke_predicted_probability"][0] * 100:.2f} %')
        with st.expander("check it!"):
            st.write(
                "btw, check that patient. i'm not sure about this app heh"
            )
    else:
        st.subheader(f':red[yes🤕] stroke with probability -> {st.session_state["df_predicted"]["stroke_predicted_probability"] * 100:.2f} %')
        with st.expander("check it!"):
            st.write(
                "btw, check that patient. i'm not sure about this app heh"
            )

    
    

