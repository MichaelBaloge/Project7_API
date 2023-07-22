from flask import Flask, request, jsonify
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
import lime
from lime.lime_tabular import LimeTabularExplainer

app = Flask(__name__)

train = pd.read_csv('new_train.csv')
test = pd.read_csv('application_test.csv')
pipeline = joblib.load('model.joblib')

features = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
       'WEEKDAY_APPR_PROCESS_START', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
       'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
       'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_MOBIL', 'FLAG_EMP_PHONE',
       'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
       'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
       'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
       'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'EXT_SOURCE_2',
       'EXT_SOURCE_3', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
       'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
       'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

explainer = LimeTabularExplainer(pipeline[0].transform(train[features]), 
                                                       feature_names=features, 
                                                       class_names=['0', '1'], 
                                                       verbose=True)

@app.route('/', methods=['GET'])
def ok():
    return 'ok'


@app.route('/predict', methods=['POST'])
def predict():
    # Get the row number from the request
    id_number = request.json['data']

    # Fetch the data for the specified row number
    # Replace this with your own code to fetch the data for the specified row
    data = test.loc[test['SK_ID_CURR']==id_number][features]

    # Perform prediction using the ref home credit model
    threshold = 0.5157
    prediction_proba = pipeline.predict_proba(data)[:, 1]  
    prediction = (prediction_proba >= threshold).astype(int)

    # Generate local feature importance with LIME
    transform_data = pipeline[0].transform(data)
    lime_explanation = explainer.explain_instance(transform_data[0], pipeline[1].predict_proba, num_features=20)

    # Extract feature importance values
    feature_importance = {}
    for feature, importance in lime_explanation.as_list():
        feature_importance[feature] = importance

    # Determine the gauge based on the prediction value
    gauge = 'Credit pouvant être accepté' if prediction == 0 else 'Credit refusé'

    # Prepare the response
    response = {
        'prediction': prediction.tolist(),
        'prob' : prediction_proba.tolist(),
        'feature_importance': feature_importance,
        'gauge': gauge,
        'transform_data' : transform_data.tolist()
    }

    return jsonify(response)


# Route pour mettre à jour les données du client
@app.route('/update', methods=['POST'])
def update_client_data():
    # Récupérer les nouvelles données du client depuis la requête POST
    updated_data = request.json['data']
    client_id = updated_data['client_id']

    update = test.copy()
    # Mettre à jour les données du client dans le DataFrame test
    for feat in updated_data.keys():
        update.loc[update['SK_ID_CURR'] == client_id, feat] = updated_data[feat]

    # Utilisez le DataFrame "update" au lieu de "data" pour effectuer la prédiction
    data = update.loc[update['SK_ID_CURR'] == client_id][features]
    
    threshold = 0.5157
    prediction_proba = pipeline.predict_proba(data)[:, 1]
    prediction = (prediction_proba >= threshold).astype(int)

    # Generate local feature importance with LIME
    transform_data = pipeline[0].transform(data)
    lime_explanation = explainer.explain_instance(transform_data[0], pipeline[1].predict_proba, num_features=20)

    # Extract feature importance values
    feature_importance = {}
    for feature, importance in lime_explanation.as_list():
        feature_importance[feature] = importance

    # Determine the gauge based on the prediction value
    gauge = 'Credit pouvant être accepté' if prediction == 0 else 'Credit refusé'

    # Prepare the response
    response = {
        'prediction': prediction.tolist(),
        'prob': prediction_proba.tolist(),
        'feature_importance': feature_importance,
        'gauge': gauge,
        'transform_data': transform_data.tolist()
    }

    return jsonify(response)



if __name__ == '__main__':
    app.run()
