from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import json
import joblib
import uvicorn
import sys
from io import StringIO
import os
from sklearn.metrics import roc_auc_score, accuracy_score
sys.path.append(os.path.join(os.path.dirname(
    os.path.dirname(__file__)), 'helpers'))
import sklearn_wrappers
# Creating the data model for data validation


class Text(BaseModel):
    WEB_TEXT: str
    TITLE: str
    DESCRIPTION: str
    KEYWORDS: str


model = joblib.load('model/baseline_model.joblib')


def get_metrics(model, data):
    error_message = "No data with non-empty true labels provided"
    if 'TARGET' not in data.columns:
        return error_message, error_message
    else:
        data['TARGET'] = data['TARGET'].map({'restaurant': 'restaurant',
                                             'retsaurant': 'restaurant',
                                             'accommodation': 'accommodation',
                                             'acommodation': 'accommodation',
                                             })
        data = data[data['TARGET'].notnull()]
        if data.shape[0] == 0:
            return error_message, error_message
        data['BINARY_TARGET'] = (data['TARGET'] != 'restaurant')*1
        data['PREDICTIONS'] = model.predict_proba(data)

        return roc_auc_score(data.BINARY_TARGET, data.PREDICTIONS), \
            accuracy_score(data.BINARY_TARGET,
                           (data.PREDICTIONS >= model['logreg'].predict_threshold_)*1)


app = FastAPI()


@app.get("/")
def getName():
    return {"name": 'Handy Text Classifier'}


@app.post("/predict_csv/")
async def parsecsv(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        contents = await file.read()
        input_df = pd.read_csv(
            StringIO(str(contents, 'utf-8')), encoding='utf-8')
        try:
            input_df[['WEB_TEXT', 'TITLE', 'DESCRIPTION', 'KEYWORDS']].shape
        except KeyError:
            raise HTTPException(status_code=400,
                            detail=\
                                '''Please, check that you have all of the following columns: 
                                'WEB_TEXT', 'TITLE', 'DESCRIPTION', 'KEYWORDS'
                                ''')

        if input_df.shape[0] == 0:
            raise HTTPException(status_code=400,
                            detail="Please, provide non-empty file with text data")
                            
        predictions = model['logreg'].predict_as_name(
            model['union'].transform(input_df))
        roc_auc, accuracy = get_metrics(model, input_df)
        return {'roc_auc_score': roc_auc,
                'accuracy': accuracy,
                'predictions': predictions
                }
    else:
        raise HTTPException(status_code=400,
                            detail="Invalid file format. Only CSV Files accepted")


@app.post("/predict_json/")
async def predict_json(text: Text):
    input_df = pd.DataFrame([text.dict()])
    if len(input_df.columns) - len(input_df.dropna(axis=1, how='all').columns) != 0 or \
            input_df.loc[:, (input_df == '').all()].shape[1] != 0:
        raise HTTPException(status_code=400,
                            detail="At least one attribute in JSON should be non-empty string")

    return model['logreg'].predict_as_name(model['union'].transform(input_df))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
