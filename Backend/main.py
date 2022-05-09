from typing import Optional
from fastapi import FastAPI
from joblib import load
from DataModel import DataModel
from Preparation import prepare
import pandas as pd
import sklearn
from sklearn.metrics import r2_score
import math

app = FastAPI()
model = load("assets/modeloP1E2.joblib")

@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.get("/integrantes")
def intgrantes():
   return "Juan Felipe Peña Criado (201426463), Juan Diego González Gómez (201911031) y Sergio Ramírez Vélez (201714577)"

@app.post("/predict")
def make_predictions(row: DataModel):
   df = pd.DataFrame(row.dict(), columns=row.dict().keys(), index=[0])
   df.columns = row.columns()

   try:
      new_df = prepare(df)
      result = model.predict(new_df)
      return result.tolist()[0]
   except:
      return -1
