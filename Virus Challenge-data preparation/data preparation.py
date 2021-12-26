import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest, f_classif, SelectKBest
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
import os

pd.options.mode.chained_assignment = None 



#sample code to run:
#   working_dir = r'C:\Users\Maroon\Desktop\semester 5\236756 intro to ML\hw2'
#   path = os.path.join(working_dir, 'virus_hw2.csv')
#   x_train, y_train, x_validate, y_validate, x_test, y_test = read_split_transform_save_data(path,
#                                                        'train.xlsx', 'validate.xlsx', 'test.xlsx')

#packages
#matplotlib	3.3.1
#mlxtend	0.17.3
#numpy		1.19.1
#openpyxl	3.0.5
#pandas		1.1.3
#scikit-learn	0.23.2
#scipy		1.5.0
#seaborn	0.11.0

######################## FUNCTIONS ########################
#split label into more meaningful categories
def categorize_encode_label(df):
    label = df['TestResultsCode']
    df.drop(columns=['TestResultsCode'])
    
    #these were used for some pearsons correlation checking and later dropped
    #df['Spreader'] = label.apply(lambda x: 1 if '_Spreader_' in x else 0)
    #df['At_Risk'] = label.apply(lambda x: 1 if '_atRisk' in x else 0)
    #df['not_detected'] = label.apply(lambda x: 1 if 'not_detected' in x else 0)
    #df['cold'] = label.apply(lambda x: 1 if 'cold' in x else 0)
    #df['flue'] = label.apply(lambda x: 1 if 'flue' in x else 0)
    #df['covid'] = label.apply(lambda x: 1 if 'covid' in x else 0)
    #df['cmv'] = label.apply(lambda x: 1 if 'cmv' in x else 0)
    #df['measles'] = label.apply(lambda x: 1 if 'measles' in x else 0)

    df['disease_type'] = label.apply(lambda x: x.split('_')[0].replace('not','not_detected'))
    df['risk_type'] = label.apply(lambda x: 'at_risk' if '_atRisk' in x else 'not_at_risk')
    df['spreader_type'] = label.apply(lambda x: 'spreader' if '_Spreader_' in x else 'not_spreader')

    return df

#explained in pdf
def apply_remove_outliers(train, val, test):
    Q1 = train.quantile(0.20)
    Q3 = train.quantile(0.80)
    IQR = Q3 - Q1
    trmask = (train < (Q1 - 1.5 * IQR)) | (train > (Q3 + 1.5 * IQR))
    vamask = (train < (Q1 - 1.5 * IQR)) | (train > (Q3 + 1.5 * IQR))
    temask = (train < (Q1 - 1.5 * IQR)) | (train > (Q3 + 1.5 * IQR))
    train[trmask] = np.nan
    val[vamask] = np.nan
    test[temask] = np.nan
    
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
    
    return train, val, test

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

#this function was originally used to choose columns to drop,
#a few instances were tested.
def drop_correlated(df, pairs):
    to_drop = []
    v = ''
    for pair in pairs:
        if np.random.uniform(0,1,1)[0] > .5:
            v = pair[0]
        else:
            v = pair[1]
            
        if v not in to_drop:
            to_drop.append(v)
    df = df.drop(columns=to_drop)    
    return df

def apply_remove_correlated(df):
    to_drop = ['DisciplineScore', 'StepsPerYear', 'AvgHouseholdExpenseOnPresents',
     'NrCousins', 'pcrResult11', 'pcrResult8', 'TimeOnSocialActivities', 'pcrResult15']
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

def apply_filter(df):
    df = categorize_encode_label(df)
    df = apply_replace_location(df)
    df = apply_empty_data_filter(df)
    df = apply_remove_correlated(df)
    df = encode_categories(df)
    df = apply_replace_date(df)
    return df

def SelectKBest_Wrapper(x_train, labels):
    selected_features = [] 
    for label in labels:
        selector = SelectKBest(f_classif, k=35)
        selector.fit(x_train, label)
        selected_features.append(list(selector.scores_))
    
    mask = selector.get_support() 
    new_features = [] 

    for bool, feature in zip(mask, x_train.columns):
        if bool:
            new_features.append(feature)
        
    return new_features

def SFS_BackwardsLR(x_train, label):
    sbs = SFS(LogisticRegression(max_iter=500, n_jobs=-1, penalty = 'l2', C = 3, random_state = 0),
         k_features=25,
         forward=False,
         floating=False,
         cv=0)

    sbs.fit(x_train, label)
    return sbs.k_feature_names_

def SFS_BackwardsLR_2Kfold(x_train, label):
    sbs = SFS(LogisticRegression(max_iter=300, n_jobs=-1, penalty='l2', random_state = 42),
         k_features=24,
         forward=False,
         floating=False,
         cv=KFold(n_splits=2, random_state=1, shuffle=True))
    
    sbs.fit(x_train, label)
    return sbs.k_feature_names_

def apply_selection(df):
    #these features were chosen after running data cleaning and then following
    #up with the two wrappers above. Both were tested with different tuning,
    #the results below are the best two, we chose the highest overall accuracy,
    #at around:
    #72% accuracy on diseases classification
    #76% accuracy on risk classification
    #80% accuracy on spread classification
    
    #initial SelectKBest chi2 wrapping (with KNN, k = 7 imputation)
    #yielded 67% accuracy on diseases label, 71-74% in both risk/spread labels.
    _filter =  ['AvgHouseholdExpenseOnSocialGames', 'AvgHouseholdExpenseParkingTicketsPerYear',
                'BMI', 'HappinessScore', 'StepsPerYear', 'SyndromeClass', 'pcrResult1', 'pcrResult12',
                'pcrResult13', 'pcrResult14', 'pcrResult16', 'pcrResult2', 'pcrResult5', 'pcrResult7',
                'SDOIF_Sore_throat', 'SDOIF_Nausea_or_vomiting',
                'SDOIF_Congestion_or_runny nose', 'SDOIF_New_loss_of_taste_or_smell', 'SDOIF_Headache',
                'SDOIF_Fatigue', 'SDOIF_Shortness_of_breath', 'SDOIF_Muscle_or_body_aches',
                'SDOIF_Diarrhea', 'SDOIF_Chills', 'SDOIF_Skin_redness', 'CurrentLocation_Y', 'Sex_M',
                'BloodType_A-', 'BloodType_AB+', 'BloodType_AB-', 'BloodType_B+', 'BloodType_B-',
                'BloodType_O+', 'BloodType_O-']
    
    #SBS wrapping with logistic_regression
    #yielded 69% accuracy on diseases, 79% accuracy on risk and 80% on spread
    filter2 = ['AgeGroup', 'AvgHouseholdExpenseOnSocialGames', 'AvgHouseholdExpenseParkingTicketsPerYear', 'BMI',
               'HappinessScore', 'AvgMinSportsPerDay', 'SyndromeClass', 'pcrResult1',
               'pcrResult12', 'pcrResult13', 'pcrResult14', 'pcrResult16', 'pcrResult2',
               'pcrResult5', 'pcrResult7', 'SDOIF_New_loss_of_taste_or_smell', 'SDOIF_Shortness_of_breath',
               'SDOIF_Muscle_or_body_aches', 'SDOIF_Skin_redness', 'Sex_M',
               'BloodType_A-', 'BloodType_AB+', 'BloodType_B+', 'BloodType_B-', 'BloodType_O+', 'BloodType_O-']
    
    #SBS wrapping with logistic_regression and k-fold with 2 splits
    #yielded 72% accuracy on diseases, 76% accuracy on risk and 80% on spread
    final = ['AgeGroup', 'AvgHouseholdExpenseOnSocialGames', 'AvgHouseholdExpenseParkingTicketsPerYear', 'AvgMinSportsPerDay', 'HappinessScore', 'pcrResult1', 'pcrResult12', 'pcrResult13', 'pcrResult14', 'pcrResult16', 'pcrResult2', 'pcrResult3', 'pcrResult4', 'pcrResult9', 'CurrentLocation_X', 'CurrentLocation_Y', 'SDOIF_Diarrhea', 'SDOIF_Shortness_of_breath', 'SDOIF_Congestion_or_runny nose', 'SDOIF_Headache', 'SDOIF_Muscle_or_body_aches', 'SDOIF_Chills', 'SDOIF_New_loss_of_taste_or_smell', 'SDOIF_Sore_throat', 'Sex_M', 'BloodType_A-', 'BloodType_AB-']
    
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


def transform_train_validate_test_data(train, validate, test):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=7)
    
    label_cols = ['disease_type', 'risk_type', 'spreader_type', 'TestResultsCode']
    label_ret = ['disease_type', 'risk_type', 'spreader_type']
    
    tr, va, te = apply_remove_outliers(train, validate, test)
    tr = apply_filter(train)
    va = apply_filter(validate)
    te = apply_filter(test)
    
    train_x = tr[[x for x in tr.columns if x not in label_cols]]
    validate_x = va[[x for x in va.columns if x not in label_cols]]
    test_x = te[[x for x in te.columns if x not in label_cols]]
    
    #fit scaler and imputer on training data only
    train_x = pd.DataFrame(imputer.fit_transform(train_x), columns = train_x.columns)
    train_x = pd.DataFrame(scaler.fit_transform(train_x), columns = train_x.columns)
    
    #transform test data using training's fitting
    validate_x = pd.DataFrame(imputer.transform(validate_x), columns = validate_x.columns)
    validate_x = pd.DataFrame(scaler.transform(validate_x), columns = validate_x.columns)
    test_x = pd.DataFrame(imputer.transform(test_x), columns = test_x.columns)
    test_x = pd.DataFrame(scaler.transform(test_x), columns = test_x.columns)
    
    #apply wrappers
    train_x = apply_selection(train_x)
    validate_x = apply_selection(validate_x)
    test_x = apply_selection(test_x)
    
    return train_x, train[label_ret], validate_x, va[label_ret], test_x, te[label_ret];

def save_to_file(df, filepath):
    df.to_excel(filepath)
    
def read_split_transform_save_data(filepath, save_tr, save_va, save_te):
    #read and split file
    df = pd.read_csv(filepath, header=0)
    train, test = train_test_split(df, test_size=0.25, shuffle=False)
    test, validation = train_test_split(test, test_size=15/25, shuffle=False)
    
    #save splits unmodified for future referencing
    save_to_file(train, 'VIRGIN_' + save_tr)
    save_to_file(validation, 'VIRGIN_' + save_va)
    save_to_file(test, 'VIRGIN_' + save_te)
    
    #save original labels
    tr_y = train['TestResultsCode']
    va_y = validation['TestResultsCode']
    te_y = test['TestResultsCode']
    
    #transform datasets
    x_train, y_train, x_validate, y_validate, x_test, y_test = transform_train_validate_test_data(train, validation, test)
    
    #save to excel files
    save_to_file(pd.concat([x_train, tr_y], axis=1), save_tr)
    save_to_file(pd.concat([x_validate, va_y], axis=1), save_va)
    save_to_file(pd.concat([x_test, te_y], axis=1), save_te)
    
    
    return x_train, y_train, x_validate, y_validate, x_test, y_test


###### BONUS CODE #######
import random

def custom_backwards_eliminiation(data_x, data_y, amount_to_keep, classifier):
    current_amount = len(data_x.columns)
    
    #features that survive each round
    surviving_features = data_x.columns

    #loop until wanted amount is hit
    while(current_amount > amount_to_keep):
        scores = {}
        for feature in surviving_features:
            features_without = surviving_features.copy().drop(feature)
            
            unique = True
            tester_idx = 0
            #make sure that the random selection is not unique otherwise the classifier will fail
            while(unique == True):              
                tester_idx = random.randint(0, data_x.shape[0])
                for i in range(0, data_x.shape[0]):
                    if i == tester_idx:
                        continue
                    if data_y[i] == data_y[tester_idx]:
                        unique = False
                        break
                        
            score = find_closest(data_x[features_without][~(data_x.index.isin([tester_idx]))],
                                     data_y[~(data_y.index.isin([tester_idx]))],
                                     data_x[(data_x.index.isin([tester_idx]))],
                                     data_y[(data_y.index.isin([tester_idx]))],
                                        data_y.unique())
            scores[feature] = score
            
        #worst performer is one with highest score, highest distance
        worst_performer = max(scores.keys(), key=(lambda k: scores[k]))
        surviving_features = surviving_features.drop(worst_performer)
        current_amount -= 1
    
    return surviving_features

def find_closest(training_x, training_y, to_predict_x, to_predict_y, label_options):
    #training_x: contains features data. size NxM
    #training_y: contains labels that correspond to training_x. size Nx1
    #to_predict_x: is an entry from the featuers space that we want to classify. size 1xM
    #to_predict_y: the correct prediction for to_predict. 1x1
    #label_options: all the possible values for labels (numeric range)
    
    closest_foreach_label = dict()
    
    for label in label_options:        
        closest_data_entry_distance = 9999
        
        #predict by closest in training_x
        for data_idx in training_x.index:
            #interested only in data entries that have the same label
            if(data_idx not in training_y.index or training_y.loc[data_idx] != label):
                continue
                
            #calculate total distance from data entry (square of numeric distance per feature)
            total_distance = 0
            for feature in training_x.columns:
                total_distance += pow(to_predict_x[feature] - training_x.at[data_idx, feature], 2)
            
            #initial assignation
            if closest_data_entry_distance == -1:
                closest_data_entry_distance = float(total_distance)
            
            #update if closer
            if float(total_distance) < closest_data_entry_distance:
                closest_data_entry_distance = float(total_distance)
        
        #update closest for label
        closest_foreach_label[label] = closest_data_entry_distance
    
    closest_label_score = min(closest_foreach_label.keys(), key=(lambda k: closest_foreach_label.get(k)))
    
    #correct prediction
    if closest_label_score == int(to_predict_y):
        return 0 #best score
    
    #return distance of scores
    return abs(closest_foreach_label.get(int(to_predict_y)) - closest_label_score)
    
    
    
#### EXTRA, WE USED THIS FOR PLOTTING GRAPH, THOUGHT ITS WORTH INCLUDING
def plot2(df, feature1, feature2, plot_type, lst=[]):
    if plot_type == 'disease':
        return sns.scatterplot(x=feature1, y=feature2, hue=df.disease_type.to_list(), data=df)
    if plot_type == 'risk':
        return sns.scatterplot(x=feature1, y=feature2, hue=df.risk_type.to_list(), data=df)
    if plot_type == 'spreader':
        return sns.scatterplot(x=feature1, y=feature2, hue=df.spreader_type.to_list(), data=df)
    if plot_type == 'specific diseases':
        return sns.scatterplot(x=feature1, y=feature2, hue=df.disease_type.apply(lambda x: x if x in lst else np.nan).to_list(),
                        data=df)

    if plot_type == 'singles1':
        types = ['not_detected', 'cold', 'flue']
        fig, axs = plt.subplots(1,3, figsize=(15,4))
        i = 0
        for ty in types:
            sns.scatterplot(x=feature1, y=feature2, hue=df.disease_type.apply(lambda x: x if x == ty else np.nan).to_list(),
                        data=df, ax=axs[i])
            i += 1
        return axs
    
    if plot_type == 'singles2':
        types = ['covid', 'cmv', 'measles']
        fig, axs = plt.subplots(1,3, figsize=(15,4))
        i = 0
        for ty in types:
            sns.scatterplot(x=feature1, y=feature2, hue=df.disease_type.apply(lambda x: x if x == ty else np.nan).to_list(),
                        data=df, ax=axs[i])
            i += 1
        return axs
        
    if plot_type == 'all':
        fig, axs = plt.subplots(1,3, figsize=(15,15))
        sns.scatterplot(x=feature1, y=feature2, hue=df.disease_type.to_list(), data=df, ax=axs[0])
        sns.scatterplot(x=feature1, y=feature2, hue=df.risk_type.to_list(), data=df, ax=axs[1])
        sns.scatterplot(x=feature1, y=feature2, hue=df.spreader_type.to_list(), data=df, ax=axs[2])
        return axs