import pickle
import streamlit as st

model = pickle.load(open('UAS_ML1_211351016_Ammar.sav'))

st.title('Estimasi Tipe Kaca')

RI = st.number_input ("Input Refractive Index 1,51115-1,53393", min_value=1.51115, max_value=1.53393, value=None, step=0.00001, format="%.5f", placeholder="Input Refractive Index...")
Na = st.number_input("Input Sodium 10,73-17,38", min_value=10.73, max_value=17.38, value=None, step=0.00001, format="%.2f", placeholder="Input Sodium...")
Mg = st.number_input("Input Magnesium 0-4,49", min_value=0.0, max_value=4.49, value=None, step=0.00001, format="%.2f", placeholder="Input Magnesium...")
Al = st.number_input("Input Alumunium 0,29-3,5", min_value=0.29, max_value=3.5, value=None, step=0.00001, format="%.2f", placeholder="Input Alumunium...")
Si = st.number_input("Input Silicon 69,8-75,41", min_value=69.8, max_value=75.41, value=None, step=0.00001, format="%.2f", placeholder="Input Silicon...")
K = st.number_input("Input Potassium 0-6,21", min_value=0.0, max_value=6.21, value=None, step=0.00001, format="%.2f", placeholder="Input Potassium...")
Ca = st.number_input("Input Calcium 5,43-16,19", min_value=5.43, max_value=16.19, value=None, step=0.00001, format="%.2f", placeholder="Input Calcium...")
Ba = st.number_input("Input Barium 0-3,15", min_value=0.0, max_value=3.15, value=None, step=0.00001, format="%.2f", placeholder="Input Barium...")
Fe = st.number_input("Input Iron 0-0,51", min_value=0.0, max_value=0.51, value=None, step=0.00001, format="%.2f", placeholder="Input Iron...")

predict = ''

if st.button('Cek Tipe Kaca'):
    predict = model.predict(
        [[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]]
    )
    st.write ('Estimasi Type Kaca : ', predict)