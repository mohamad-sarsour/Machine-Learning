import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import os

pd.options.mode.chained_assignment = None 

#CHANGES:
#    outliers removal range reduced to lower and higher 10% instead of 20%.
#    pcrResult5 replaced BloodType.
#    pcrResult5 and pcrResult16 were exempted from outliers selection.
#    MinMaxNormalization was replaced with StandardScale Normalization.


#split label into more meaningful categories
def categorize_encode_label(df):
    label = df['TestResultsCode']
    df.drop(columns=['TestResultsCode'])

    df['disease_type'] = label.apply(lambda x: x.split('_')[0].replace('not','not_detected'))
    df['risk_type'] = label.apply(lambda x: 'at_risk' if '_atRisk' in x else 'not_at_risk')
    df['spreader_type'] = label.apply(lambda x: 'spreader' if '_Spreader_' in x else 'not_spreader')

    return df

#explained in pdf
def apply_remove_outliers(train, val, test, to_predict):
    Q1 = train.quantile(0.1)
    Q3 = train.quantile(0.9)
    IQR = Q3 - Q1
    
    tr = pd.DataFrame()
    va = pd.DataFrame()
    te = pd.DataFrame()
    pr = pd.DataFrame()
    
    tr['pcrResult16'] = train['pcrResult16']
    tr['pcrResult5']  = train['pcrResult5'] 
    va['pcrResult16'] = val['pcrResult16']
    va['pcrResult5']  = val['pcrResult5']
    te['pcrResult16'] = test['pcrResult16']
    te['pcrResult5']  = test['pcrResult5']
    pr['pcrResult16'] = to_predict['pcrResult16']
    pr['pcrResult5']  = to_predict['pcrResult5']
    
    trmask = (train < (Q1 - 1.5 * IQR)) | (train > (Q3 + 1.5 * IQR))
    
    #drop outliers based on training data
    train[trmask] = np.nan
    val[trmask] = np.nan
    test[trmask] = np.nan
    to_predict[trmask] = np.nan
    
    train['pcrResult16'] = tr['pcrResult16']
    train['pcrResult5'] = tr['pcrResult5']
    val['pcrResult16'] = va['pcrResult16']
    val['pcrResult5'] = va['pcrResult5']
    test['pcrResult16'] = te['pcrResult16']
    test['pcrResult5'] = te['pcrResult5']
    to_predict['pcrResult16'] = pr['pcrResult16']
    to_predict['pcrResult5']  = pr['pcrResult5']
    
    #using dictionary to convert specific columns 
    convert_dict = {'Address': 'string', 
                    'BloodType': 'category',
                    'CurrentLocation' : 'string',
                    'DateOfPCRTest' : 'string',
                    'Job' : 'string',
                    'SelfDeclarationOfIllnessForm' : 'string',
                    'Sex' : 'category',
                    'TestResultsCode': 'string'
               } 
    
    
    train = train.astype(convert_dict) 
    val = val.astype(convert_dict) 
    test = test.astype(convert_dict) 
    to_predict = to_predict.astype(convert_dict)
    
    return train, val, test, to_predict

#split location feature into numeric coordinates
def apply_replace_location(df):
    df['CurrentLocation_X'] = df['CurrentLocation'].apply(lambda x: float(x.split('\'')[1]) if not pd.isnull(x) else np.nan)
    df['CurrentLocation_Y'] = df['CurrentLocation'].apply(lambda x: float(x.split('\'')[3]) if not pd.isnull(x) else np.nan)
    df = df.drop(columns=['CurrentLocation'], axis=1)
    return df

#explained in pdf
def apply_empty_data_filter(df):
    to_drop = ['AvgTimeOnSocialMedia', 'AvgTimeOnStuding', 'Job', 'Address', 'PatientID']
    df = df.drop(columns=df.columns.intersection(to_drop), axis=1)
    return df

def apply_remove_correlated(df):
    to_drop = ['DisciplineScore', 'StepsPerYear', 'AvgHouseholdExpenseOnPresents',
     'NrCousins', 'pcrResult11', 'pcrResult8', 'pcrResult10', 'TimeOnSocialActivities', 'pcrResult15']
    df = df.drop(columns=to_drop)
    return df

def encode_helper(col, value):
    if(pd.isnull(value)):
        return np.nan
    if(col in value.split(';')):
        return 1
    return 0

def encode_categories(df):
    types = pd.unique(df['SelfDeclarationOfIllnessForm'].apply(lambda x : str(x).replace(' ','')).str.split(';',expand=True).stack())
    types = types[types != 'nan']
    
    for col in types:
        if len(col) > 5:
            col = col.replace('nose', ' nose')
            df['SDOIF_'+col] = df['SelfDeclarationOfIllnessForm'].apply(lambda x: encode_helper(col, x))
    
    df = df.drop(columns=['SelfDeclarationOfIllnessForm'], axis=1)
    
    cat_variables = df[['Sex', 'BloodType']]
    cat_dummies = pd.get_dummies(cat_variables, drop_first=True)

    df = df.drop(['Sex', 'BloodType'], axis=1)
    df = pd.concat([df, cat_dummies], axis=1)
    return df

def apply_replace_date(df):
    df['MonthOfPCRTest'] = df['DateOfPCRTest'].apply(lambda x: float(x.split('-')[1]) if not pd.isnull(x) else np.nan)
    df['DayOfPCRTest'] = df['DateOfPCRTest'].apply(lambda x: float(x.split('-')[2]) if not pd.isnull(x) else np.nan)
    df = df.drop(columns=['DateOfPCRTest'], axis=1)
    return df

def apply_filter(df, data_only = False):
    if data_only == False:
        df = categorize_encode_label(df)
    else:
        df = df.drop(columns=['TestResultsCode'])
    df = apply_replace_location(df)
    df = apply_empty_data_filter(df)
    df = apply_remove_correlated(df)
    df = encode_categories(df)
    df = apply_replace_date(df)
    return df

def apply_selection(df):
    #SBS wrapping with logistic_regression and k-fold with 2 splits
    #yielded 72% accuracy on diseases, 76% accuracy on risk and 80% on spread
    final = ['AgeGroup', 'AvgHouseholdExpenseOnSocialGames', 'AvgHouseholdExpenseParkingTicketsPerYear',
             'AvgMinSportsPerDay', 'HappinessScore', 'pcrResult1', 'pcrResult12', 'pcrResult13', 'pcrResult5',
             'pcrResult16', 'pcrResult2', 'pcrResult3', 'pcrResult4', 'pcrResult9', 'pcrResult14', 'CurrentLocation_X',
             'CurrentLocation_Y', 'SDOIF_Diarrhea', 'SDOIF_Shortness_of_breath', 'SDOIF_Congestion_or_runny nose',
             'SDOIF_Headache', 'SDOIF_Muscle_or_body_aches', 'SDOIF_Chills', 'SDOIF_New_loss_of_taste_or_smell',
             'SDOIF_Sore_throat', 'Sex_M']
    
    df = df[final]
    return df

def extract_mapped(categorial_label):
    disease_dict = {
                'not_detected' : 0,
                'cold': 1, 
                'flue': 2,
                'cmv' : 3,
                'measles' : 4,
                'covid' : 5
               } 

    spreader_dict = {
             'spreader' : 0,
             'not_spreader' : 1
            }

    risk_dict = {
             'at_risk' : 0,
             'not_at_risk' : 1
            }
    
    y_disease = categorial_label.disease_type.apply(lambda x: disease_dict[x])
    y_spreader = categorial_label.spreader_type.apply(lambda x: spreader_dict[x])
    y_risk = categorial_label.risk_type.apply(lambda x: risk_dict[x])
    
    return y_disease, y_spreader, y_risk

    
def transform_train_validate_test_prediction_data(train, validate, test, to_predict):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=7)
    
    label_cols = ['disease_type', 'risk_type', 'spreader_type', 'TestResultsCode']
    label_ret = ['disease_type', 'risk_type', 'spreader_type']
    tr, va, te, pr = apply_remove_outliers(train, validate, test, to_predict)
    
    tr = apply_filter(tr)
    va = apply_filter(va)
    te = apply_filter(te)
    pr = apply_filter(pr, True)
    
    train_x = tr[[x for x in tr.columns if x not in label_cols]]
    validate_x = va[[x for x in va.columns if x not in label_cols]]
    test_x = te[[x for x in te.columns if x not in label_cols]]
    predict_x = pr[[x for x in te.columns if x not in label_cols]]
    
    #fit scaler and imputer on training data only
    train_x = pd.DataFrame(imputer.fit_transform(train_x), columns = train_x.columns)
    train_x = pd.DataFrame(scaler.fit_transform(train_x), columns = train_x.columns)
    
    #transform test data using training's fitting
    validate_x = pd.DataFrame(imputer.transform(validate_x), columns = validate_x.columns)
    validate_x = pd.DataFrame(scaler.transform(validate_x), columns = validate_x.columns)
    test_x = pd.DataFrame(imputer.transform(test_x), columns = test_x.columns)
    test_x = pd.DataFrame(scaler.transform(test_x), columns = test_x.columns)
    predict_x = pd.DataFrame(imputer.transform(predict_x), columns = predict_x.columns)
    predict_x = pd.DataFrame(scaler.transform(predict_x), columns = predict_x.columns)
    
    #apply wrappers
    train_x = apply_selection(train_x)
    validate_x = apply_selection(validate_x)
    test_x = apply_selection(test_x)
    predict_x = apply_selection(predict_x)
    
    return train_x, tr[label_ret], validate_x, va[label_ret], test_x, te[label_ret], predict_x

def save_to_file(df, filepath):
    df.to_excel(filepath.replace(".csv", ".xlsx"))

################# HW3 functions ################
def get_dataframes(data_filepath, to_predict_filepath):
    #read and split file
    df = pd.read_csv(data_filepath, header=0)
    train, test = train_test_split(df, test_size=0.25, shuffle=True, stratify=df['TestResultsCode'])
    test, validation = train_test_split(test, test_size=15/25, shuffle=True, stratify=test['TestResultsCode'])
    
    to_predict_data = pd.read_csv(to_predict_filepath, header=0)
    return train, validation, test, to_predict_data

def transform_data(train, validation, test, to_predict_data, save_tr, save_va, save_te):
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    #save splits unmodified for future referencing
    save_to_file(train, 'VIRGIN_' + save_tr)
    save_to_file(validation, 'VIRGIN_' + save_va)
    save_to_file(test, 'VIRGIN_' + save_te)
    
    #save original labels
    tr_y = train['TestResultsCode']
    va_y = validation['TestResultsCode']
    te_y = test['TestResultsCode']
    
    #transform datasets
    x_train, y_train, x_validate, y_validate, x_test, y_test, x_predict = transform_train_validate_test_prediction_data(train, validation, test, to_predict_data)    
    
    #save to excel files
    save_to_file(pd.concat([x_train, tr_y], axis=1), save_tr)
    save_to_file(pd.concat([x_validate, va_y], axis=1), save_va)
    save_to_file(pd.concat([x_test, te_y], axis=1), save_te)
    
    return x_train, y_train, x_validate, y_validate, x_test, y_test, x_predict

def predict_label(data_path, to_predict_path, prediction_save_file, show_test_results=False):
    train, validate, test, to_predict = get_dataframes(data_path, to_predict_path)
    x_train, y_train, x_validate, y_validate, x_test, y_test, x_predict = transform_data(train, validate, test, to_predict,
                                                                                         'train.xlsx', 'validate.xlsx', 'test.xlsx')
    #validate is no longer needed at this phase, append it to data
    tr_y_disease, tr_y_spreader, tr_y_risk = extract_mapped(y_train)
    te_y_disease, te_y_spreader, te_y_risk = extract_mapped(y_test)
    
    #chosen model
    rfc = RandomForestClassifier(n_jobs=-1, n_estimators = 800, min_samples_split = 2,
                                       min_samples_leaf = 2, max_features = 'sqrt', max_depth = 52, bootstrap =False)
    
    if show_test_results == True:
        #show results on test set
        pred_disease = predict_results(x_train, tr_y_disease, rfc, x_test, k_folds=8)
        pred_spread = predict_results(x_train, tr_y_spreader, rfc, x_test, k_folds=8)
        pred_risk = predict_results(x_train, tr_y_risk, rfc, x_test, k_folds=8)
        predicted_label = build_label_from_mapped(pred_disease, pred_spread, pred_risk)
        
        test_label = build_label_from_mapped(te_y_disease, te_y_spreader, te_y_risk)
        
        print("test disease type prediction accuracy:", metrics.accuracy_score(pred_disease, te_y_disease))
        print("test spread type prediction accuracy:", metrics.accuracy_score(pred_spread, te_y_spreader))
        print("test risk type prediction accuracy:", metrics.accuracy_score(pred_risk, te_y_risk))
        print("test final combined label prediction accuracy:", calculate_accuracy(test_label, predicted_label))
    
    #predict and print data on prediction data
    pred_disease = predict_results(x_train, tr_y_disease, rfc, x_predict, k_folds=8)
    pred_spread = predict_results(x_train, tr_y_spreader, rfc, x_predict, k_folds=8)
    pred_risk = predict_results(x_train, tr_y_risk, rfc, x_predict, k_folds=8)
    
    predicted_label = build_label_from_mapped(pred_disease, pred_spread, pred_risk)
    to_save_df = pd.read_csv(prediction_save_file, header=0)
    to_save_df.TestResultsCode = predicted_label.TestResultsCode
    save_to_file(to_save_df, prediction_save_file)
    
    return x_train, y_train, x_validate, y_validate, x_test, y_test, x_predict
    
def get_key(val, my_dict):
        for key, value in my_dict.items():
             if val == value:
                 return key
        return 'none'

def get_full_label(disease, spread, risk):
    disease_dict = {
                'not_detected' : 0,
                'cold': 1, 
                'flue': 2,
                'cmv' : 3,
                'measles' : 4,
                'covid' : 5
               } 

    spreader_dict = {
             'Spreader' : 0,
             'NotSpreader' : 1
            }

    risk_dict = {
             'atRisk' : 0,
             'NotatRisk' : 1
            }
    
    word_disease = get_key(disease, disease_dict)
    word_spreader = get_key(spread, spreader_dict)
    word_risk = get_key(risk, risk_dict)
    
    return word_disease+'_'+word_spreader+'_'+word_risk

def build_label_from_mapped(disease, spread, risk):
    label = pd.DataFrame(columns=['TestResultsCode'])
    try: #got pd frame
        for i in range(0, len(disease)):
            label.loc[i] = get_full_label(disease.loc[i], spread.loc[i], risk.loc[i])
    except: #got np array
        for i in range(0, len(disease)):
            label.loc[i] = get_full_label(disease[i], spread[i], risk[i])
            
    return label

def calculate_accuracy(label1, label2):
    total = len(label1)
    match = len(label1)
    for i in range(0, total):
        if label1.TestResultsCode[i] != label2.TestResultsCode[i]:
            match -= 1
    
    return float(match) / float(total)

#################################### HW 3 BONUS ####################################
    ## part A ##

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import metrics

def predict_results(train_x, train_y, classifier, pred_x, k_folds):
    cross_val_scores = cross_val_score(classifier, train_x, train_y,
                                       cv=StratifiedKFold(n_splits=k_folds, random_state=13, shuffle=True), scoring='accuracy')
    
    classifier.fit(train_x, train_y)
    prediction = classifier.predict(pred_x)
        
    return prediction

def choose_classifier(train_x, train_y, pred_x, pred_y, classifiers, k_folds=2):
    best_accuracy = 0
    best_classifier = 0
    best_prediction = 0
    
    for classifier in classifiers:
        prediction = predict_results(train_x, train_y, classifier, pred_x, k_folds)
        accuracy = metrics.accuracy_score(prediction, pred_y)
        
        if(accuracy > best_accuracy):
            best_accuracy = accuracy
            best_classifier = classifier
            best_prediction = prediction
        
    
    return best_classifier, best_prediction, best_accuracy
    
    
 ## part B ##   
 def choose_classifiers_non_single_model_approach(x_train, y_train x_test,classifiers):
    ## choose the best model for disease type 
    best_clf_disease, pred_disease, acc_disease = choose_classifier(x_train, tr_y_disease, x_test, test_y_disease, classifiers, k_folds=10)
    ## choose the best model for spreaderness
    best_clf_spread, pred_spread, acc_spread = choose_classifier(x_train, tr_y_spreader, x_test, test_y_spreader, classifiers, k_folds=10)
    ## choose the best model for riskness
    best_clf_risk, pred_risk, acc_risk = choose_classifier(x_train, tr_y_risk, x_test, test_y_risk, classifiers, k_folds=10)
    ## return a dictionary that maps the taks and the best model for it 
    dict={ 'disease type' : best_clf_disease ,
           'spreaderness' : best_clf_spread  ,
           'riskness'     : best_clf_risk
          }
     return dict      