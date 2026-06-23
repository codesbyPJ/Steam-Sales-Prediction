import streamlit as st
import numpy as np 
import pickle as pi
from scipy.sparse import hstack, csr_matrix

#Modell, Vectorizer und den Scaler laden
with open("model.pkl","rb") as f: my_model = pi.load(f) #"rb" bedeutet read binary also es öffnet die datei im lesemodus als binär, f ist das Dateiobjekt
with open("tfidf.pkl","rb") as f: my_tfidf = pi.load(f)
with open("scaler.pkl","rb") as f: my_scaler = pi.load(f)
st.set_page_config(page_title="Steam Sales Predictions", page_icon= "🎮") #emoji from https://emojipedia.org/video-game 

st.title("Steam Sales Prediction 🎮") #Titel auf der Seite
st.subheader("Wie viele Reviews wird ihr Spiel bekommen") #Selbsterklärend ein Untertitel
st.divider () #eine Trennlinie

my_Description = st.text_area("Spielbeschreibung", placeholder="Bitte beschreibe dein Spiel") #Das Textfeld für die Spielbeschreibung
my_Tags = st.text_input("Tags", placeholder= "Bitte tragen sie hier komma getrennt die Tags ihres Spiels ein bsp: Indie, Farming, singleplayer") #Textfeld für die Tags
my_Genres = st.text_input ("Genre", placeholder="Bitte nennen sie hier das Genre ihres Spiels bsp: Farming, Indie")
my_Price = st.slider("Preis", min_value=0.99, max_value=99.99, value =19.99, step=0.50) #Das ist ein Schieberegler für den Preis, min value ist der minimals möglich auszuwählende Preis der liegt bei steam bei 0.99€ und $, für Max price gibt es so ein limit nicht deswegen ist der mal auf 100 gesetzt, value ist der default wert für den Schieberegler, steps sind die schritte in denen der preis angepasst werden kann
my_Achievements = st.number_input("Anzahl Achievements", min_value=0, value =0) #Wie viele Achievements hat das game
my_Windows = st.checkbox("Windows", value=True) #Checkbox dafür ob das game auf windows läuft, auf default = ja
my_Linux = st.checkbox("Linux", value=False) #Checkbox dafür ob das game auf Linux läuft, standard auf false weil weniger common
my_Mac = st.checkbox("Mac", value=False) #Checkbox dafür ob das game auf Mac, ebenfalls standard auf false

if st.button("Vorhersage Starten"): #Interessanterweise wird in Streamlit der button hier erstellt und muss nicht vorher definiert werden 
    if not my_Description or not my_Tags: #failsave für wenn beschreibung bzw tags leer
        st.warning("Bitte Beschreibung und Tags angeben") #warnung
    else:
        st.info("Modell noch nicht geladen - bitte warten") #Placeholder bis das modell fertig ist.