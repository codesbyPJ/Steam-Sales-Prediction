import streamlit as st
import numpy as np 
import pickle as pi
import xgboost as xgb
import torch
from torch import nn
from transformers import DistilBertTokenizerFast, DistilBertModel
from scipy.sparse import hstack, csr_matrix

#Modell, Vectorizer und den Scaler laden
my_model = xgb.XGBRegressor()
my_model.load_model("model.json")
with open("tfidf.pkl","rb") as f: my_tfidf = pi.load(f)
with open("scaler.pkl","rb") as f: my_scaler = pi.load(f)
with open("ridge.pkl","rb") as f: my_ridge = pi.load(f) 
with open("rf.pkl","rb") as f: my_rf = pi.load(f)
with open("scaler_bert.pkl","rb") as f: my_scaler_bert = pi.load(f) #eigener Scaler, weil DistilBERT separat trainiert wurde

#DistilBERT braucht dieselbe Modell-Klasse wie beim Training, sonst passt der gespeicherte Zustand nicht
class DistilBertRegression(nn.Module):
    def __init__(self, num_numeric_features):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.regressor = nn.Sequential(
            nn.Linear(768 + num_numeric_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, numeric_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_output.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_output, numeric_features], dim=1)
        return self.regressor(combined).squeeze(1)

#DistilBERT + Tokenizer nur einmal laden (nicht bei jedem Klick neu) - das lädt beim ersten Start
#automatisch die Basisgewichte von distilbert-base-uncased herunter, dafür wird Internet gebraucht
@st.cache_resource
def load_bert():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertRegression(num_numeric_features=8)
    model.load_state_dict(torch.load("distilbert_model.pt", map_location="cpu"))
    model.eval()
    return tokenizer, model

my_bert_tokenizer, my_bert_model = load_bert()

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
        #Text wird kombiniert und durch tfidf gejagt
        my_text = f"{my_Description} {my_Tags} {my_Genres}"
        my_text_features = my_tfidf.transform([my_text])
        #Numeric Features zusammenstellen
        my_num_features = np.array([[
            my_Price,
            int(my_Achievements >0),
            len([t for t in my_Tags.split(",") if t.strip()]), #Die Zeile war pain da hats 3 versuche gebraucht die richtig zu machen, dies zählt die anzahl der tags und splittet den string nach jedem Komma und filtert leere Einträge heraus und zählt wie viele übrig bleiben
            len(my_Description),
            int(my_Windows),
            int(my_Mac),
            int(my_Linux),
            2026 #Das ist das release Jahr als feature das modell wurde mit einem release jahr mittrainiert also muss die app auch ein jahr mitgeben, 2026 ist jetzt gerade also deswegen
            ]])

        # Scaler anwenden und die Features zusammenführen
        my_num_scaled = my_scaler.transform(my_num_features)
        my_input = hstack([my_text_features, csr_matrix(my_num_scaled)])

        
        
        my_pred_xgb = int(np.expm1(my_model.predict(my_input)[0]))
        my_pred_ridge = int(np.expm1(my_ridge.predict(my_input)[0]))
        my_pred_rf = int(np.expm1(my_rf.predict(my_input)[0]))

        #DistilBERT braucht eigenes Textformat (wie beim Training) und eigenen Scaler
        my_bert_text = f"Tags: {my_Tags} Genres: {my_Genres} Categories:  Description: {my_Description}"
        my_num_scaled_bert = my_scaler_bert.transform(my_num_features)

        my_encoded = my_bert_tokenizer(
            my_bert_text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        with torch.no_grad():
            my_pred_log_bert = my_bert_model(
                input_ids=my_encoded["input_ids"],
                attention_mask=my_encoded["attention_mask"],
                numeric_features=torch.tensor(my_num_scaled_bert, dtype=torch.float32)
            )
        my_pred_bert = int(np.expm1(my_pred_log_bert.item()))

        st.divider()
        st.subheader("Ergebnisse")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("XGBoost", f"{my_pred_xgb:,} Reviews")
        col2.metric("Ridge Regression", f"{my_pred_ridge:,} Reviews")
        col3.metric("Random Forest", f"{my_pred_rf:,} Reviews")
        col4.metric("DistilBERT", f"{my_pred_bert:,} Reviews")
