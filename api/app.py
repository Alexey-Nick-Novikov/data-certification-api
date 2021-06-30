
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}

@app.get("/predict")
def predict(original_title,
            title,
            release_date,
            duration_min,
            description,
            budget,
            original_language,
            status,
            number_of_awards_won,
            number_of_nominations,
            has_collection,
            all_genres,
            top_countries,
            number_of_top_productions,
            available_in_english):
    
    # Building a df for trainer.py
    X = pd.DataFrame(dict(
        original_title=[str(original_title)],
        title=[str(title)],
        release_date=[str(release_date)],
        duration_min=[float(duration_min)],
        description=[str(description)],
        budget=[float(budget)],
        original_language=[str(original_language)],
        status=[str(status)],
        number_of_awards_won=[int(number_of_awards_won)],
        number_of_nominations=[int(number_of_nominations)],
        has_collection=[int(has_collection)],
        all_genres=[str(all_genres)],
        top_countries=[str(top_countries)],
        number_of_top_productions=[float(number_of_top_productions)],
        available_in_english=[bool(available_in_english)]
    ))
    
    pipeline = joblib.load('model.joblib')
    
    prediction = pipeline.predict(X)[0]
    
    return {'title' : title,
            'popularity': prediction}

# Implement a /predict endpoint
