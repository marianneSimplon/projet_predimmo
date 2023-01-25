# streamlit run app.py
import os
import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt

## Load MODEL ###
MODEL_VERSION = 'knn.pkl'
MODEL_PATH = os.path.join(os.getcwd(), '../my_pickles',
                          MODEL_VERSION)  # path vers le modèle
with open(MODEL_PATH, 'rb') as handle:
    MODEL = pickle.load(handle)

## Load SCALER ###
SCALER_VERSION = 'minmaxscaler.pkl'
SCALER_PATH = os.path.join(os.getcwd(), '../my_pickles',
                           SCALER_VERSION)  # path vers le modèle
with open(SCALER_PATH, 'rb') as handle:
    SCALER = pickle.load(handle)

## Load ENCODER ###
ENCODER_VERSION = 'onehotencoder.pkl'
ENCODER_PATH = os.path.join(os.getcwd(), '../my_pickles',
                            ENCODER_VERSION)  # path vers le modèle
with open(ENCODER_PATH, 'rb') as handle:
    ENCODER = pickle.load(handle)

### RECUPERATION DES INPUTS ###


def inputs_to_df(longitude, latitude, population, median_income, ocean_proximity):
    data = {'longitude': [float(longitude)], 'latitude': [float(latitude)], 'population': [
        population], 'median_income': [float(median_income)], 'ocean_proximity': [ocean_proximity]}
    my_inputs_dataframe = pd.DataFrame(data)
    return my_inputs_dataframe


def preprocess_inputs(input_dataframe):
    non_cat = pd.DataFrame(input_dataframe[input_dataframe.columns[0:4]])
    cat = pd.DataFrame(input_dataframe[input_dataframe.columns[-1]])
    scale_data = pd.DataFrame(SCALER.transform(non_cat), columns=[
                              "longitude", "latitude", "population", "median_income"])
    encode_data = pd.DataFrame(pd.DataFrame.sparse.from_spmatrix(
        ENCODER.transform(cat)))
    encode_data.rename({0: 'H_OCEAN', 1: 'INLAND', 2: 'ISLAND',
                       3: 'NEAR_BAY', 4: 'NEAR_OCEAN'}, inplace=True, axis=1)
    encode_data = encode_data.drop(
        ["H_OCEAN", "INLAND", "ISLAND", "NEAR_OCEAN"], axis=1)
    preprocess_data = pd.concat([scale_data, encode_data], axis=1)
    return preprocess_data


def predict(preprocessed_input):
    pred = MODEL.predict(pd.DataFrame(preprocessed_input))
    return pred[0]


#############################################################################################
######################################### STREAMLIT #########################################
#############################################################################################

### SETTINGS ###
st.set_page_config(
    page_title="PREDIMMO - Quartiers californiens",
    page_icon=Image.open('./assets/imgs/logo_predimmo.PNG')
)


### BODY ###

col1, col2 = st.columns([1, 3])
with col1:
    logo = Image.open('./assets/imgs/logo_predimmo.PNG')
    st.image(logo)

with col2:
    st.title("Prédiction du prix médian d'un quartier de logements en Californie")


image = Image.open('./assets/imgs/san-francisco-210230_960_720.jpg')
st.image(image, caption='Maisons de San Francisco')

# EXPANDER
with st.expander("Rappels sur le prix médian des logements de quartiers californiens"):
    df = pd.read_csv(
        r'./data/traindata_ori.csv', delimiter=',', decimal='.')
    median_house_value = df['median_house_value']
    fig, ax = plt.subplots()
    ax.hist(median_house_value, bins=20)
    ax.set_title(
        "Distribution du prix médian d'un quartier de logements en Californie")
    ax.set_xlabel("Prix médian d'un quartier")
    st.pyplot(fig)

st.subheader(
    "Veuillez entrer les caractéristiques du quartier de logement qui vous intéresse :")

longitude = st.text_input('Longitude')

latitude = st.text_input('Latitude')

population = st.number_input('Nombre total de résidents', step=1)

median_income = st.text_input('Revenu médian des ménages')

ocean_proximity = st.selectbox(
    "Proximité de l'océan (Sélectionnez une option ci-dessous)",
    ('INLAND', '<1H OCEAN', 'NEAR BAY', 'NEAR OCEAN', 'ISLAND'))

if st.button(label="Estimer le prix médian"):
    try:
        datas = inputs_to_df(longitude, latitude, population,
                             median_income, ocean_proximity)
        if datas.isnull().values.any() == False:
            preprocess_datas = preprocess_inputs(datas)
            my_pred = predict(preprocess_datas)
            st.write('Prix médian estimé du quartier de logements :')
            st.write(my_pred)
        else:
            st.write(
                "Veuillez remplir tous les champs du formulaire s'il vous plait")
    # except Exception as e:
    except Exception:
        st.write("Une erreur s'est produite.")
        # st.write(e)
        # print(e)
