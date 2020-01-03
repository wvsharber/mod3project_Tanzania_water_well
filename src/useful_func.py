import pandas as pd
import seaborn as sns
from datetime import datetime, date
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def drop_useless_cols(df, drop_values = []):
    """
    Drops columns from df that are specificied in drop_values. Won't drop values from continuous data columns, but will raise an error if you try. Returns DataFrame with columns dropped.
    """
    
    continuous_columns = ['amount_tsh', 'date_recorded', 'gps_height', 'population', 'construction_year']
    for cont in continuous_columns:
        if cont in drop_values:
            print(f'you cannot drop column: {cont}')
            return
        
    try:
        df_dropped = df.drop(drop_values, axis = 1)
        return df_dropped
    except:
        return df
    
def load_data(string1, string2):
    """
        Pass in two strings containg .csv file paths. This function will load the two dataframes and merge them along the column 'id'. Returns merged DataFrame.
    """
    df_1 = pd.read_csv(string1)
    df_2 = pd.read_csv(string2)
    #merging dataframes
    df = pd.merge(df_1, df_2, on = 'id', how = 'inner')
    return df

    
def fix_dates(df):
    """ 
    Takes the date of 01/01/2020 and subtracts it from the 'date_recorded' column. This information will be stored in column called 'days_since_recording' and drops the 'date_recorded' column from the DataFrame. Returns DataFrame.
    """
    basedate = datetime(2020, 1, 1)
    df['days_since_recording'] = df.loc[:,'date_recorded'].map(lambda x: (basedate - datetime.strptime(x, "%Y-%m-%d")).days)
    df.drop(['date_recorded'], axis = 1, inplace = True)
    return df

def clean_data(df, threshold = 100):
    """
    Replaces all NaN values in DataFrame with 'Not Known'. For categorical columns, replaces all values with a count less than 100 (threshold value) with 'other'. Returns edited DataFrame.
    """
    
    # replaces NaN with a string 'not known'
    df = df.fillna('Not Known')
    
    uvdict = {}

    for column in df.select_dtypes(exclude=['int','float']):
        values_list = df[column].unique()
        uvdict[column] = len(values_list)

    target_list = list(filter(lambda x: uvdict[x] > threshold, uvdict.keys()))
                       
                       
    for col in target_list:
        valued_dict = dict(df[col].value_counts())
        safe_values = list(key for key, value in valued_dict.items() if value >= 50)
    #     replace_values = list(filter(lambda x: x not in safe_values, all_values))
        df.loc[:, col] = df.loc[:, col].map(lambda y: 'other' if y not in safe_values else y)
    
    
    return df

def bin_me(df):
    """
    Creates bins for construction_year based on 5 year increments. In addition, values stored as year 0 will be transformed to 'not_available'. Returns edited DataFrame.
    """
    try:
        basedate = datetime(2020, 1, 1)
        a = list(range(1955,2016,5))
        cut_bins = [-1]
        cut_bins.extend(a)
        cut_labels = ['not available', '56-60','61-65','66-70','71-75','76-80','81-85','86-90','91-95','96-00','01-05','06-10','11-15']
        df.loc[:, 'construction_year_bin'] = pd.cut(df['construction_year'], bins = cut_bins, labels = cut_labels)
        df.drop(['construction_year'], axis = 1, inplace = True)
        return df
    except:
        if 'construction_year_bin' in df.columns:
            print('action already performed')
        else:
            print('you messed up')

def onehotmess(df):
    """
    Uses pd.getdummies() to one hot encode categorical variables in DataFrame. Returns edited DataFrame and target DataFrame.
    """
    df_objects = df.select_dtypes(exclude=['int','float']).drop(['status_group'], axis = 1)
    df_nums = df.select_dtypes(include=['int','float'])

    df_onehot = pd.get_dummies(df_objects)

    df_final = pd.concat([df_nums, df_onehot], axis = 1)
    
    return df_final, df.status_group

def normalize_func(df_values, df_target):
    """
    Takes DataFrame of training data and target values, performs a train-test split, and then scales the data using MinMaxScaler. Returns train and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(df_values, df_target, test_size = .05, random_state = 42)
    scaler = MinMaxScaler()
    
    X_train_ = scaler.fit_transform(X_train)
    X_test_ = scaler.transform(X_test)
    
    return X_train_, X_test_, y_train, y_test


def do_everything(string1, string2, drop_values, thresh = 200):
    """
    This function wraps previously defined data cleaning and preprocessing functions and returns processed train and test data sets.
    """
    loaded_data = load_data(string1, string2)
    df_dropped = drop_useless_cols(loaded_data, drop_values)
    fixed_date = fix_dates(df_dropped)
    cleaner_df = clean_data(fixed_date, thresh)
    df_binned = bin_me(cleaner_df)
    ohm_df, target_df = onehotmess(df_binned)
    X_train, X_test, y_train, y_test = normalize_func(ohm_df, target_df)
    
    return X_train, X_test, y_train, y_test, ohm_df, target_df


def create_voting_classifier(X_train, X_test, y_train, y_test, RFC_num = 200, LR_iter = 1000, GBC_num = 300):
    """
    This function will take in the X_train, X_test, y_train, and y_test.
    The function also allows the user to adjust the number of estimators for Random Forest and Gradient Boosted Forest using RFC_num and GBC_num respectively.
    The number of iterations for the logistic regression can be adjusted using LR_iter.
    This function will return a fitted voting_classifier object using VotingClassifier with soft max for voting, the accuracy score of that classifier, and a plot of the confusion matrix.
    """
    RF1 = RandomForestClassifier(n_estimators = RFC_num)
    LR1 = LogisticRegression(max_iter = LR_iter)
    GBR1 = GradientBoostingClassifier(n_estimators = GBC_num)
    KNN1 = KNeighborsClassifier()
    
    #Hard-coded estimates of accuracy from previously fit models
    lr_weight, rf_weight, gbr_weight, knn_weight = .77, .80, .78, .78
    
    eclf_soft = VotingClassifier(estimators = [('lr', LR1),
                                               ('rf', RF1),
                                               ('gbr', GBR1),
                                               ('knn', KNN1)],
                                 weights = [lr_weight, rf_weight, gbr_weight, knn_weight],
                                 voting = 'soft')
    eclf_soft.fit(X_train, y_train)
    score = eclf_soft.score(X_test, y_test)
    
    # plot and save confusion matrix
    plot_confusion_matrix(eclf_soft, X_test, y_test, xticks_rotation=45, display_labels = ['Functional', 'Needs Repair \n but Functional', 'Nonfunctional'],
                         normalize='pred', cmap='Blues')
    plt.title(f'Soft Voting Classifier Confusion Matrix \n Accuracy: {score.round(2)}')
    plt.tight_layout()
    plt.savefig('./reports/Voting_classifier_confusion_matrix.png', dpi = 300, transparent = True)
    return eclf_soft, score

def create_graph(df_target, df_values):
    """
    Creates a graph of the top and bottom ten most correlated values to 'functional' water wells.
    """

    df_target = pd.get_dummies(df_target)
    df = pd.concat([df_target, df_values], axis = 1)

    df_corr = df.corr()
    df_corr.drop('functional', inplace = True)
    df_corr.drop('non functional', inplace = True)
    df_corr.drop('functional needs repair', inplace = True)

    df_func_10_pos = df_corr.sort_values(by=['functional'], ascending = False)[['functional']].head(10)
    df_func_10_neg = df_corr.sort_values(by=['functional'], ascending = False)[['functional']].tail(11)
    df_func_10_neg.drop('extraction_type_class_other', inplace = True)

    df_func_10 = pd.concat([df_func_10_pos, df_func_10_neg])


    ### creates graph
    labels = list(df_func_10.index)
    x_pos = np.arange(len(labels))
    values = list(df_func_10.functional)
    color_b, color_r = ['#64b3ff']*(len(labels)//2), ['#ffb364']*(len(labels)//2)
    color_b.extend(color_r)

    fig_size = (12,9)

    plt.figure(figsize = fig_size)
    plt.bar(x_pos, values, align='center', color = color_b, alpha = 1)
    plt.grid(zorder=0, alpha= .5, linestyle = '--')
    plt.axvline(x=9.5, color = 'black', linestyle = '--')
    plt.axhline(y=0, color = 'black')
    plt.xticks(x_pos, labels, rotation=80, )
    plt.xlabel('Features', fontsize = (fig_size[0])*3//2)
    plt.ylabel('Percent Correlation', fontsize = (fig_size[0])*3//2)
    plt.ylim(-.4, .4)
    plt.title('Correlation with Well Functionality', fontsize=(fig_size[0])*2, y = 1.03)

    plt.tight_layout()
    plt.savefig('reports/corr_well_func.png', transparent = True)


