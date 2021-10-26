from fastapi import FastAPI, File, UploadFile
from typing import Optional
import pandas as pd
import json
import joblib

model = joblib.load('../model/baseline_model.joblib')

def convertBytesToString(bytes):
    data = bytes.decode('utf-8').splitlines()
    df = pd.DataFrame(data)
    return parse_csv(df)

def parse_csv(df):
    return json.loads(df.to_json(orient='records'))

app = FastAPI()

@app.get("/")
def getName():
    return {"name" : 'Handy Text Classifier'}

@app.post("/predict_csv/")
async def parsecsv(file: UploadFile = File(...)):
    contents = await file.read()
    json_string = convertBytesToString(contents)
    return {'file contents' : json_string}, 200

@app.post("/predict_json/")
def 
    return parse_csv(prediction), 200


