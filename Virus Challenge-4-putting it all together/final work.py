import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import os

pd.options.mode.chained_assignment = None 

#split label into more meaningful categories
def categorize_encode_label(df):
    label = df['TestResultsCode']
    df.drop(columns=['TestResultsCode'])

    df['disease_type'] = label.apply(lambda x: x.split('_')[0].replace('not','not_detected'))
    df['risk_type'] = label.apply(lambda x: 'at_risk' if '_atRisk' in x else 'not_at_risk')
    df['spreader_type'] = label.apply(lambda x: 'spreader' if '_Spreader_' in x else 'not_spreader')

    return df

def apply_remove_outliers(train, val, test, to_predict):
    Q1 = train.quantile(0.15)
    Q3 = train.quantile(0.85)
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
    vamask = (val < (Q1 - 1.5 * IQR)) | (val > (Q3 + 1.5 * IQR))
    temask = (test < (Q1 - 1.5 * IQR)) | (test > (Q3 + 1.5 * IQR))
    prmask = (to_predict < (Q1 - 1.5 * IQR)) | (to_predict > (Q3 + 1.5 * IQR))
    
    #drop outliers based on training data
    train[trmask] = np.nan
    val[vamask] = np.nan
    test[temask] = np.nan
    to_predict[prmask] = np.nan
    
    train['pcrResult16'] = tr['pcrResult16']
    train['pcrResult5'] = tr['pcrResult5']
    val['pcrResult16'] = va['pcrResult16']
    val['pcrResult5'] = va['pcrResult5']
    test['pcrResult16'] = te['pcrResult16']
    test['pcrResult5'] = te['pcrResult5']
    to_predict['pcrResult16'] = pr['pcrResult16']
    to_predict['pcrResult5']  = pr['pcrResult5']
        
    return train, val, test, to_predict

def apply_selection(df):
    final = [ 'AgeGroup', 'AvgMinSportsPerDay', 'AvgHouseholdExpenseParkingTicketsPerYear', 
             'DisciplineScore', 'TimeOnSocialActivities', 'StepsPerYear',
             'SyndromeClass' ,'pcrResult1', 'pcrResult4', 'pcrResult5',
             'pcrResult12', 'pcrResult14', 'pcrResult16']
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

def transform_train_validate_test_data(train, validate, test, pred_df):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=13)
    
    label_ret = ['disease_type', 'risk_type', 'spreader_type']
    
    #encode labels
    tr_lbl = categorize_encode_label(train)[label_ret].copy(deep=True)
    va_lbl = categorize_encode_label(validate)[label_ret].copy(deep=True)
    te_lbl = categorize_encode_label(test)[label_ret].copy(deep=True)
    
    #select best features and remove outliers
    tr = apply_selection(train)
    va = apply_selection(validate)
    te = apply_selection(test)
    pr = apply_selection(pred_df)
    
    tr, va, te, pr = apply_remove_outliers(tr, va, te, pr)
    
    #fit scaler and imputer on training data only
    tr = pd.DataFrame(imputer.fit_transform(tr), columns = tr.columns)
    tr = pd.DataFrame(scaler.fit_transform(tr), columns = tr.columns)
    
    #transform test data using training's fitting
    va = pd.DataFrame(imputer.transform(va), columns = va.columns)
    va = pd.DataFrame(scaler.transform(va), columns = va.columns)
    te = pd.DataFrame(imputer.transform(te), columns = te.columns)
    te = pd.DataFrame(scaler.transform(te), columns = te.columns)
    
    #transform prediction data
    pr = pd.DataFrame(imputer.transform(pr), columns = pr.columns)
    pr = pd.DataFrame(scaler.transform(pr), columns = pr.columns)
    
    return tr, tr_lbl, va, va_lbl, te, te_lbl, pr;

def save_to_file(df, filepath):
    df.to_csv(filepath, index=False)
        
def read_df(filepath):
    df = pd.read_csv(filepath, header=0)
    return df

def read_split_transform_data(filepath, to_predict):
    #read and split file
    df = pd.read_csv(filepath, header=0)
    train, test = train_test_split(df, test_size=0.2, shuffle=False)
    test, validation = train_test_split(test, test_size=0.5, shuffle=False)
    
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    predict = read_df(to_predict)
    pred_ids = predict['PatientID'].copy(deep= True)
    
    #transform datasets
    x_train, y_train, x_validate, y_validate, x_test, y_test, pred = transform_train_validate_test_data(train, validation, test, predict)
         
    return x_train, y_train, x_validate, y_validate, x_test, y_test, pred, pred_ids
    
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
    
    return [word_disease, word_spreader, word_risk]

def build_label_from_mapped(disease, spread, risk):
    label = pd.DataFrame(columns=['Virus', 'Spreader', 'Risk'])
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

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import metrics

def fit_predict_results(train_x, train_y, classifier, pred_x, k_folds):
    cross_val_scores = cross_val_score(classifier, train_x, train_y,
                                       cv=StratifiedKFold(n_splits=k_folds, random_state=13, shuffle=True), scoring='accuracy')
    
    classifier.fit(train_x, train_y)
    prediction = classifier.predict(pred_x)
        
    return classifier, prediction

def predict_results(clf, pred_x):    
    return clf.predict(pred_x)

def choose_classifier(train_x, train_y, pred_x, pred_y, classifiers, k_folds=2):
    best_accuracy = 0
    best_classifier = 0
    best_prediction = 0
    
    for classifier in classifiers:
        clf, prediction = fit_predict_results(train_x, train_y, classifier, pred_x, k_folds)
        accuracy = metrics.accuracy_score(prediction, pred_y)
        
        if(accuracy > best_accuracy):
            best_accuracy = accuracy
            best_classifier = clf
            best_prediction = prediction
        
    
    return best_classifier, best_prediction, best_accuracy

def train_predict_export(path, to_predict_path, save_path, show_results=False):
    import copy

    x_train, y_train, x_validate, y_validate, x_test, y_test, x_predict, pred_ids = read_split_transform_data(path, to_predict_path)
    
    tr_y_disease, tr_y_spreader, tr_y_risk = extract_mapped(y_train)
    va_y_disease, va_y_spreader, va_y_risk = extract_mapped(y_validate)
    te_y_disease, te_y_spreader, te_y_risk = extract_mapped(y_test)
    
    classifiers = []
    classifiers.append(GradientBoostingClassifier(learning_rate= 0.1,
                        max_depth= 100,
                        max_features= 'sqrt',
                        min_samples_leaf= 4,
                        min_samples_split= 5,
                        n_estimators= 1266))
    classifiers.append(LogisticRegression(n_jobs=-1, random_state=0))
    classifiers.append(RandomForestClassifier(
                        max_depth= 72,
                        max_features= 'sqrt',
                        min_samples_leaf= 3,
                        min_samples_split= 4,
                        n_estimators= 1383, n_jobs = -1))
    classifiers.append(GradientBoostingClassifier(learning_rate= 0.1,
                        max_depth= 83,
                        max_features= 'sqrt',
                        min_samples_leaf= 1,
                        min_samples_split= 5,
                        n_estimators= 1080,
                        random_state = 13))
    
    #need deep copies to reduce required training amount
    classifiers2 = copy.deepcopy(classifiers)
    classifiers3 = copy.deepcopy(classifiers)
    
    #choose best classifiers for each task based on validation set
    best_classifier_disease, _, dis = choose_classifier(
        x_train, tr_y_disease, x_validate, va_y_disease, classifiers, 9)
    best_classifier_spreader, _, spr = choose_classifier(
        x_train, tr_y_spreader, x_validate, va_y_spreader, classifiers2, 9)
    best_classifier_risk, _, rsk = choose_classifier(
        x_train, tr_y_risk, x_validate, va_y_risk, classifiers3, 9)
    
    print("best classifier for disease type (", dis, '):\n', best_classifier_disease)
    print("\nbest classifier for spreader type (", spr, '):\n', best_classifier_spreader)
    print("\nbest classifier for risk type (", rsk, '):\n', best_classifier_risk)

    prediction_disease = predict_results(best_classifier_disease, x_predict)
    prediction_spread = predict_results(best_classifier_spreader, x_predict)
    prediction_risk = predict_results(best_classifier_risk, x_predict)
    
    #print results on test set
    if show_results == True:
        pred_test_disease = predict_results(best_classifier_disease, x_test)
        pred_test_spread = predict_results(best_classifier_spreader, x_test)
        pred_test_risk = predict_results(best_classifier_risk, x_test)
        print("\ntest disease type prediction accuracy:", metrics.accuracy_score(pred_test_disease, te_y_disease))
        print("test spread type prediction accuracy:", metrics.accuracy_score(pred_test_spread, te_y_spreader))
        print("test risk type prediction accuracy:", metrics.accuracy_score(pred_test_risk, te_y_risk))

    #save prediction to file
    ids = pd.DataFrame()
    ids['PatientID'] = pred_ids
    result = build_label_from_mapped(prediction_disease, prediction_spread,
                                     prediction_risk)
    
    final = pd.concat([ids, result], axis=1)
    save_to_file(final, save_path)

    return final

def sample_entry:
    working_dir = r'C:\Users\Maroon\Desktop\semester 5\236756 intro to ML\hw5'

    path = os.path.join(working_dir, 'virus_hw5.csv')
    to_predict_path = os.path.join(working_dir, 'virus_hw5_test.csv')
    prediction_save_file = os.path.join(working_dir, 'predicted.csv')

    result = train_predict_export('virus_hw5.csv', 'virus_hw5_test.csv', 'predicted.csv', show_results=True)