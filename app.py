from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
import lime
from lime.lime_tabular import LimeTabularExplainer

# Initialisation de l'API
app = Flask(__name__)


### Chargement des données ###

# Import des features utilisées dans la modélisation
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

# Import des données et du modèle
train = pd.read_csv('new_train.csv')
test = pd.read_csv('application_test.csv')
pipeline = joblib.load('model.joblib')

# Séparation des colonnes numériques et catégorielles pour les graphiques
num = test.select_dtypes(exclude='object').columns
cat = test.select_dtypes(include='object').columns
num_col = list(set(num).intersection(features))
cat_col = list(set(cat).intersection(features))

# Explainer Lime pour l'explicabilité locale
explainer = LimeTabularExplainer(pipeline[0].transform(train[features]), 
                                                       feature_names=features, 
                                                       class_names=['0', '1'], 
                                                       verbose=True)

# Affichage par défaut sur l'url de l'API pour premier contrôle
@app.route('/', methods=['GET'])
def ok():
    return 'ok'

# Route pour obtenir les id des clients sous forme de dataframe
@app.route('/reflist', methods=['GET'])
def client_ids():
    return jsonify(test[['SK_ID_CURR']].to_json(orient = 'split'))

# Route pour obtenir les features par type sous forme de listes
@app.route('/features', methods=['GET'])
def features_list():
    response = {
        'num' : num_col,
        'cat' : cat_col
    }
    return jsonify(response)

# Route pour obtenir l'image de l'explicabilité globale
@app.route('/shap', methods=['GET'])
def explain_img():
    return send_file('glob_exp.png')

# Route pour obtenir les informations du client sélectionné sous forme de dataframe
@app.route('/clientinfo', methods=['POST'])
def client_info():
    # obtention de l'id du client sélectionné
    id_number = request.json['data']
    return jsonify(test.loc[test['SK_ID_CURR']==id_number, features + ['SK_ID_CURR']].to_json(orient = 'split'))

# Route pour obtenir les informations de prédictions et d'explicabilité pour le client sélectionné
@app.route('/predict', methods=['POST'])
def predict():
    id_number = request.json['data']
    data = test[test['SK_ID_CURR']==id_number][features]

    # Prédiction
    threshold = 0.5157
    prediction_proba = pipeline.predict_proba(data)[:, 1]  
    prediction = (prediction_proba >= threshold).astype(int)

    # Explicabilité locale avec Lime
    transform_data = pipeline[0].transform(data)
    lime_explanation = explainer.explain_instance(transform_data[0], pipeline[1].predict_proba, num_features=20)

    # Extraction des features importances
    feature_importance = {}
    for feature, importance in lime_explanation.as_list():
        feature_importance[feature] = importance

    # Information d'aide à la décision
    gauge = 'Credit pouvant être accepté' if prediction == 0 else 'Credit refusé'
        
    # Réponse
    response = {
        'prediction': prediction.tolist(),
        'prob' : prediction_proba.tolist(),
        'feature_importance': feature_importance,
        'gauge': gauge
    }

    return jsonify(response)

# Route pour obtenir les informations générales des clients de référence afin de tracer les graphs correspondants (liste de features voulues)
@app.route('/featureinfo', methods=['POST'])
def feature_info():
    feat = request.json['data']
    return jsonify(train[feat + ['proba', 'TARGET']].to_json(orient = 'split'))

# Route pour mettre à jour les données du client
@app.route('/update', methods=['POST'])
def update_client_data():
    # Récupérer les nouvelles données du client (dictionnaire de features)
    updated_data = request.json['data']
    client_id = updated_data['client_id']

    update = test.copy()
    # Mettre à jour les données du client dans le DataFrame test
    for feat in updated_data.keys():
        update.loc[update['SK_ID_CURR'] == client_id, feat] = updated_data[feat]

    # Utilisez le DataFrame "update" au lieu de "data" pour effectuer la prédiction comme pour la route /predict
    data = update.loc[update['SK_ID_CURR'] == client_id][features]
    
    threshold = 0.5157
    prediction_proba = pipeline.predict_proba(data)[:, 1]
    prediction = (prediction_proba >= threshold).astype(int)

    transform_data = pipeline[0].transform(data)
    lime_explanation = explainer.explain_instance(transform_data[0], pipeline[1].predict_proba, num_features=20)

    feature_importance = {}
    for feature, importance in lime_explanation.as_list():
        feature_importance[feature] = importance

    gauge = 'Credit pouvant être accepté' if prediction == 0 else 'Credit refusé'

    response = {
        'prediction': prediction.tolist(),
        'prob': prediction_proba.tolist(),
        'feature_importance': feature_importance,
        'gauge': gauge
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run()
