import numpy as np
import pandas as pd
import logging, pickle, io
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from io import StringIO
from sklearn.linear_model import Ridge


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def predict(features) -> float:
    logging.debug(f'Predicting price for features: {features}')
    with open('ridge.pkl', 'rb') as file:
        model = pickle.load(file)
        pred = model.predict(features)
    return np.exp(pred)

def preprocess_items(df_test: pd.DataFrame) -> pd.DataFrame:
    df_test['mileage'].apply(lambda x: str(x).split()[-1]).value_counts()
    df_test['mileage'] = df_test['mileage'].apply(lambda x: float(str(x)[:-6])*1.40 \
                                              if str(x).endswith('km/kg') else x)
    df_test['mileage'] = df_test['mileage'].apply(lambda x: str(x).split()[0]).astype(float)
    try:
        df_test['mileage'] = df_test['mileage'].fillna(df_test['mileage'].median())
    except:
        pass
    df_test['engine'].apply(lambda x: str(x).split()[-1]).value_counts()
    df_test['engine'] = df_test['engine'].apply(lambda x: str(x).split()[0]).astype(float)
    try:
        df_test['engine'] = df_test['engine'].fillna(df_test['engine'].median())
    except:
        pass
    df_test['max_power'] = df_test['max_power'].apply(lambda x: str(x).split()[0])
    df_test['max_power'] = df_test['max_power'].apply(lambda x: str(x).replace('bhp', 'nan'))
    df_test['max_power'] = df_test['max_power'].astype(float)
    try:
        df_test['max_power'] = df_test['max_power'].fillna(df_test['max_power'].median())
    except:
        pass
    df_test['tor'] = df_test['torque'].apply(lambda x: str(x).split()[0])
    df_test['t'] = df_test['tor'].apply(lambda x: float(str(x)[:-2]) if str(x).lower()[-2:] == 'nm'
                      else float(110) if str(x) == '110(11.2)@'
                      else float(x) * 9.80665 if str(x)[-1] != '@'
                      else float(str(x)[:-4]) * 9.80665 if str(x)[-3:] == 'gm@'
                      else float(str(x)[:-3]) if str(x).lower()[-2:] == 'm@'
                      else float(str(x)[:-1]) * 9.80665 if str(x) != 'nan' else x)
    try:
        df_test['t'] = df_test['t'].fillna(df_test['t'].median())
    except:
        pass
    df_test['max_torque_rpm'] = df_test['torque'].apply(lambda x: x if str(x) != '400Nm' else None)
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: str(x).replace('(kgm@ rpm)', ''))
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: str(x).replace('kgm at ', ''))
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: str(x).replace('Nm@', ''))
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: str(x).replace('rpm', ''))
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: str(x).replace('nm@', ''))
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: str(x).replace('@', ''))
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: str(x).replace(',', ''))
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: str(x).split()[1:])
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: ' '.join(x))
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: str(x).replace('~', '-'))
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: str(x).replace('/', ''))
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: str(x).replace('21800', '2180'))
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: str(x).split('+-')[0])
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: 3000 if str(x) == 'KGM at 3000 RPM'
           else 1900-2750 if str(x) == 'KGM at 1900-2750 RPM' else str(x).split()[-1] if x else x)
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: str(x).split('-')[-1] if x else x)
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].replace('', None)
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].apply(lambda x: int(x) if x else x)
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].fillna(df_test['max_torque_rpm'].median())
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].astype(np.int64)
    
    df_test['torque'] = df_test['t']
    try:
        df_test['seats'] = df_test['seats'].fillna(df_test['seats'].median())
    except:
        pass
    df_test = df_test[['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
           'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque',
           'seats', 'max_torque_rpm']]
    df_test['engine'] = df_test['engine'].astype(np.int64)
    df_test['seats'] = df_test['seats'].astype(np.int64)
    df_test['brend'] = df_test['name'].apply(lambda x: x.split()[:2])
    df_test['brend'] = df_test['brend'].apply(lambda x: ' '.join(x))
    df_test = df_test[['year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
            'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque',
            'seats', 'max_torque_rpm', 'brend']]
    X_test_cat = df_test.drop(columns = ['selling_price'], axis = 1)
    X_test = df_test._get_numeric_data().drop(columns = ['selling_price'], axis = 1)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    cols_num = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm']
    cols_cat = ['fuel', 'seller_type', 'transmission', 'owner', 'seats', 'brend']
    X_test_norm = pd.DataFrame(data = scaler.transform(X_test), columns = cols_num)
    with open('onehot_encoder.pkl', 'rb') as file:
        enc = pickle.load(file)
    codes = enc.transform(X_test_cat[cols_cat]).toarray()
    feature_names = enc.get_feature_names_out(cols_cat)
    X_test_ = pd.concat([X_test_norm.loc[:, ~X_test_norm.columns.isin(cols_cat)],
        pd.DataFrame(codes, columns=feature_names).astype(int)], axis=1)
    return X_test_

@app.get("/")
async def root():
    return {
        "Name": "Car price prediction"
    }

@app.post("/predict_item")
async def predict_item(item: Item) -> dict:
    logging.info(f'Predicting price for item: {item}')
    item_dict = item.model_dump()
    df = pd.DataFrame([item_dict])
    df = preprocess_items(df)
    prediction = predict(df)
    logging.info(f'Predicted price: {prediction}')
    return {"predicted_price": prediction[0]}

@app.post("/predict_items")
async def predict_items(upload_file: UploadFile = File(...)) -> dict:
    logging.info(f'Received file: {upload_file.filename}')
    
    try:
        df_test = pd.read_csv(upload_file.file)
        logging.info(f'CSV file read successfully with columns: {df_test.columns.tolist()}')
    except Exception as e:
        logging.error(f'Error reading CSV file: {e}')
        raise HTTPException(status_code=400, detail=f"Invalid file: {e}")
    
    expected_columns = ["name", "year", "selling_price", "km_driven", "fuel", "seller_type",
                "transmission", "owner", "mileage", "engine", "max_power", "torque", "seats"]
    
    if not all(column in df_test.columns for column in expected_columns):
        logging.error('CSV file missing required columns.')
        raise HTTPException(status_code=400, detail="CSV file missing required columns.")
    
    X_test_ = preprocess_items(df_test) 
    X_test_['predicted_price'] = predict(X_test_)

    logging.info('Predictions made for all items in file.')

    output = StringIO()
    X_test_.to_csv(output, index=False)
    X_test_.to_csv('result.csv', index=False)
    result_csv = output.getvalue()
    output.close()
    
    logging.info('CSV file with predictions created.')
    return {"file_content": result_csv}