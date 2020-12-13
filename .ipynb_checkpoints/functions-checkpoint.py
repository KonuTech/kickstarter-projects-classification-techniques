# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# https://www.kaggle.com/fk0728/feature-engineering-with-sklearn-pipelines
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

# https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#impute-the-missing-data-and-score
# https://scikit-learn.org/stable/modules/impute.html
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
#https://scikit-learn.org/stable/modules/feature_selection.html#variance-threshold

# https://scikit-lego.readthedocs.io/en/latest/index.html

from sklearn.datasets import load_boston, load_iris, load_diabetes, load_digits
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, SelectFromModel
from sklearn.tree import ExtraTreeRegressor

import os
import pandas as pd
import numpy as np
from io import BytesIO
from io import TextIOWrapper
import zipfile
from zipfile import ZipFile
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, normaltest, kstest
# from scipy.stats import normaltest
import warnings
from IPython.display import Image
# from patsy import PatsyModel, PatsyTransformer
import itertools


warnings.filterwarnings('ignore')
sns.set()
pd.options.display.max_columns = None
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def get_dataset(dictionary):
    
    """
    https://scikit-learn.org/0.16/datasets/index.html
    https://scikit-learn.org/stable/datasets/index.html
    """

    for values in dictionary.values():
        #key = pd.DataFrame.from_dict(dictionary.values)
        if np.isscalar(values):
            pass
        else:
            #print(pd.DataFrame.from_dict(values))
            feature_names = dictionary["feature_names"]
            data = pd.DataFrame(dictionary["data"], columns=feature_names)
            target = pd.DataFrame(dictionary["target"], columns=["TARGET"])
            output = pd.concat([data,target],axis=1)
        
        return output


#for dataset in [load_boston(), load_iris(), load_diabetes()]:
#print(get_dataset(dataset)[:5])

def get_current_working_directory():

        """
        :return:
        """
        
        current_path = os.getcwd()

        return current_path


def change_current_working_directory(directory):
    """
    :param directory:
    :return:
    """
    if os.path.exists(directory):
        os.chdir(directory)
    else:
        print("\n" + "Directory Does Not Exists. Working Directory Have Not Been Changed." + "\n")

    current_path = str(os.getcwd())
    
    return current_path

def get_list_of_files_from_directory(directory):
    """
    :param directory:
    :return:
    """
    list_of_files = []

    for item in os.listdir(directory):
        list_of_files.append(item)

    return list_of_files

def get_list_of_zip_files(directory):
    """
    :param directory:
    :return:
    """
    os.chdir(directory)
    zip_files = []

    for root, dirs, files in os.walk("."):
        for filename in files:
            if filename.endswith(".zip"):
                zip_files.append(filename)

    return zip_files

def get_list_of_files_by_extension(directory, extension):
    """
    :param directory:
    :param extension:
    :return:
    """
    list_of_files = []

    for item in os.listdir(directory):
        if item.endswith("." + extension):
            list_of_files.append(item)

    return list_of_files

def unzip_files(directory, output_directory, zip_file_name):
    """
    :param input_directory:
    :param output_directory:
    :return:
    """

    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(output_directory)

    print("Unpacked " + str(zip_file_name) + " to: " + str(output_directory) + "\n")
    
def get_list_of_files_by_extension(directory, extension):
    """
    :param directory:
    :param extension:
    :return:
    """
    list_of_files = []

    for item in os.listdir(directory):
        if item.endswith("." + extension):
            list_of_files.append(item)

    return list_of_files

def count_unique_values(dataframe, variables):
    """
    """
    for column in variables:
        count_unique = dataframe[str(column)].value_counts()
        count_null = pd.Series(dataframe[str(column)].isnull().sum(),index=["nan"])
        count_unique = count_unique.append(count_null, ignore_index=False)
        
        print(column + " count distinct:")
        print(count_unique)
        print()

def visualise_floats(dataframe, variables, target):
    """
    """
    for column in variables:
        ax = sns.distplot(dataframe[column].dropna(), fit=norm)
        ax.set_title("Histogram of " + str(column) + " before imputation")
        ax.set_xlabel(str(column))
        ax.set_ylabel("Frequency Rate")
        fig = plt.figure()
        
        res = stats.probplot(dataframe[column], plot=plt)
        fig = plt.figure()
        
#         target_column = pd.DataFrame(dataframe.iloc[:,-1])
#         test_output = pd.merge(target_column, dataframe[variables], left_index=True, right_index=True)
#         ax = sns.jointplot(x=column, y=target, data=test_output, kind='reg', marker="+", color="b")
#         ax.fig.suptitle("Scatter plot of " + str(column) + "vs. " + target + " before imputation")
#         plt.figure()

def choose_imputer_and_visualise_floats(dataframe, variables, target, imputer=None, strategy=None, weights=None):
    """ 
    :SimpleImputer:
    :IterativeImputer:
    :KNNImputer:
    
    :SimpleImputer strategy:
    "mean"
    "median"
    "most_frequent"
    "constant"
    
    :KNNImputer weights:
    "uniform"
    "distance"
    "callable" 
    """
    
    #print("$ Counts before Imputation:")
    #for column in variables:
    #    print(count_unique_values(dataframe, column))
    #    print()
    
    if imputer == None:
        output = pd.DataFrame(dataframe.fillna(0), columns=variables)
        
    elif imputer == SimpleImputer and strategy != None:
        SI = SimpleImputer(missing_values=np.nan, strategy=str(strategy))
        SI.fit(dataframe[variables])
        output = pd.DataFrame(SI.transform(dataframe[variables]), columns=variables)
        
    elif imputer ==  IterativeImputer:
        II = IterativeImputer(max_iter=10, random_state=0)
        II.fit(dataframe[variables])
        output = pd.DataFrame(II.transform(dataframe[variables]), columns=variables)
        
    elif imputer == KNNImputer and weights != None:
        KNNI = KNNImputer(missing_values=np.nan, weights=str(weights), add_indicator=False)
        output = pd.DataFrame(KNNI.fit_transform(dataframe[variables]), columns=variables)
        
    else:
        output = "error"
    
    #print("$ Counts after Imputation:")
    #for column in output.columns:
    #    count_unique = output[column].value_counts()
    #    print(column)
    #    print(count_unique)
    #    print()
        
    
    for column in variables:
        ax = sns.distplot(output[column], fit=norm)
        ax.set_title("Histogram of " + str(column) + " after imputation")
        ax.set_xlabel(str(column))
        ax.set_ylabel("Frequency Rate")
        fig = plt.figure()
        
        res = stats.probplot(output[column], plot=plt)
        fig = plt.figure()
        
        if target != None:
        
            target_column = pd.DataFrame(dataframe.iloc[:,-1])
            test_output = pd.merge(target_column, output, left_index=True, right_index=True)
            ax = sns.jointplot(x=column, y=target, data=test_output, kind='reg', marker="+", color="b")
            ax.fig.suptitle("Scatter plot of " + str(column) + "vs. " + target + " after imputation")
            plt.figure()


    return output

def choose_imputer_and_visualise_categories(dataframe, variables, target, imputer=None, strategy=None, weights=None):
    """ 
    :SimpleImputer:
    :IterativeImputer:
    :KNNImputer:
    
    :SimpleImputer strategy:
    "mean"
    "median"
    "most_frequent"
    "constant"
    
    :KNNImputer weights:
    "uniform"
    "distance"
    "callable" 
    """
    
    #print("$ Counts before Imputation:")
    #for column in variables:
    #    print(count_unique_values(dataframe, column))
    #    print()
    
    if imputer == None:
        output = pd.DataFrame(dataframe.fillna(0), columns=variables)
        
    elif imputer == SimpleImputer and strategy != None:
        SI = SimpleImputer(missing_values=np.nan, strategy=str(strategy))
        SI.fit(dataframe[variables])
        output = pd.DataFrame(SI.transform(dataframe[variables]), columns=variables)
        
    elif imputer ==  IterativeImputer:
        II = IterativeImputer(max_iter=10, random_state=0)
        II.fit(dataframe[variables])
        output = pd.DataFrame(II.transform(dataframe[variables]), columns=variables)
        
    elif imputer == KNNImputer and weights != None:
        KNNI = KNNImputer(missing_values=np.nan, weights=str(weights), add_indicator=False)
        output = pd.DataFrame(KNNI.fit_transform(dataframe[variables]), columns=variables)
        
    else:
        output = "error"
        
    #print("$ Counts after Imputation:")
    #for column in range(len(output.columns)):
    #    count_unique = output[column].value_counts()
    #    print(count_unique)
    #    print()
        
    for column in variables:
        ax = sns.countplot(output[column], palette="Paired")
        ax.set_title("Bar plot of " + str(column) + " after imputation")
        ax.set_xlabel(str(column))
        fig = plt.figure()
            
    return output

def add_deviation_features(dataframe, variables_floats, variables_objects):
    
    """
    feature numeric
    category object
    """
    
    data = []

    #categories = pd.DataFrame(dataframe.select_dtypes(include=['object'])).columns
    categories = variables_objects
    #features = pd.DataFrame(dataframe.select_dtypes(include=['float64'])).columns
    features = variables_floats
    
    for category in categories:
        for feature in features:
            category_feature = str(category) + "_DEVIATION_" + str(feature)

            category_gb = dataframe.groupby(category)[feature]
            category_mean = category_gb.transform(lambda x: x.mean())
            category_std = category_gb.transform(lambda x: x.std())
            
            deviation_feature = ((dataframe[feature] - category_mean) / category_std).rename(category_feature)
            data.append(deviation_feature)
    
    output = pd.DataFrame(data).T
    dataframe = pd.concat([dataframe, output], axis=1)
    
    return dataframe