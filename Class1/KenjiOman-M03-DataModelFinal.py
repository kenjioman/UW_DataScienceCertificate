#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Kenji Oman
Created on: 03-15-2018
Purpose: To fulfill requirements for the UW Data Science Certificate,
    Data Science: Process and Tools class, Milestone 3

Note:  This structure of this file is largely unchanged from Assignment 9, but
I've added in another classifier with the corresponding metrics for it.  I've
also added commentary along the way, culminating in my thoughts on this
process/ future work (at the very end).  Thanks for the great course!

PS, My data came split, so I just kept it that way.  I showed how to split
data in the assignment for lesson 8, just to show that I can do it.
"""

import pandas as pd
import requests
import matplotlib.pyplot as plt  # For plotting
#import random as rand  # For replacing outlier values
import seaborn as sns  # For extra plots
from sklearn.preprocessing import MinMaxScaler  # to normalize numeric values
from sklearn.ensemble import RandomForestClassifier  # one of my classifiers
from sklearn.naive_bayes import GaussianNB  # my other classifier
import sklearn.metrics as skm

# Define the URL's for the training and test data
TRAIN_URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
             'adult/adult.data')

TEST_URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
            'adult/adult.test')

# And define the URL of the file which contains the header names
HEADER_URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
              'adult/adult.names')

class DataProject(object):
    """
    Just use a class to group all methods together, especially if we're going
    to be building on our work through each assignment.
    """

    def __init__(self, train, test, head):
        """Initialize the class when we instantiate it.

        Args:
            train (str): URL of the training set data
            test (str): URL of the test set data
            head (str): URL of the header location

        Attributes:
            self.train_url = train
            self.test_url = test
            self.head_url = head
            self.train_df = pd.DataFrame of trainig Data
            self.test_df = pd.DataFrame of the test Data
            self.train_X/Y = pd.DataFrame of the training features/ labels
                (unscaled numeric columns dropped, education column
                    (pd.category) converted to int)
            self.test_X/Y = pd.DataFrame of the test features/ labels (unscaled
                numeric columns dropped, education column (pd.category)
                converted to int)
        """

        # Save the URL's, in case we want it later
        self.train_url = train
        self.test_url = test
        self.head_url = head

        # Grab the data, and store the internal variables
        self._get_data()

        # Clean the numeric data
        self._clean_numeric_data()

        # Now, clean the categorical columns
        self._clean_categorical_data()

        # Normalize the numeric data
        self._normalize_numerics()

        # And curate the columns to drop redundant dummy columns
        self._curate_columns()

        # Build the ML arrays
        self._make_ml_data()


    def _get_data(self):
        """"Internal method to get the data and associate it with the class.
        Used in self.__init__
        """

        # Grab the data. Note, the separator is actually ', ', not just a
        # comma, so specify. Also, recognize the "?" as an NA value
        # (I think it is easier to have pandas catch the NA values instead
        #  of manually searching for and parsing these in the future).
        # Finally, set the engine to python, since having a separator greater
        # than one character automatically does this, and prints a warning
        # message.  By explicitly telling it to use python, we suppress the
        # warning.
        self.train_df = pd.read_csv(self.train_url, sep=', ', header=None,
                                    na_values='?', engine='python')

        # For the training data, have one comment row, so need to ignore
        self.test_df = pd.read_csv(self.test_url, sep=', ', header=None,
                                   skiprows=1, na_values='?', engine='python')

        # Get the header data
        response = requests.get(self.head_url)
        header = response.text.split('\n')

        # Now, filter to grab the header lines:
        # First, make sure there is at least one character for the line, and
        # ignore lines that start with the comment character for the file "|"
        header = [row for row in header if len(row) > 0 and row[0] != '|']

        # Ignore the first row, since it is just identifying the classifier
        # task and, get just the header values
        header = [head.split(':')[0] for head in header[1:]]

        # Finally, we need to add a header name for the last column (if <= or >
        # income of 50k)
        header.append('income')

        # Since it's hard to access df elements if they have a '-' (we have
        # to explicitly use df[colname] notation, instead of just df.colname)
        # let's convert all hyphens in column names into underscores
        header = ['_'.join(col.split('-')) for col in header]

        # Now, set the header for the data sets
        self.train_df.columns = header
        self.test_df.columns = header

    def _clean_numeric_data(self):
        """Consolidate the work done to clean the numeric columns. Acts on:
        self.train_df and self.test_df.
        """

        # We saw that capital_gain and capital_loss is only != 0 if the other
        # = 0. So, let's collapse these columns
        self.train_df.loc[:,'capital_diff'] = (
            self.train_df.capital_gain - self.train_df.capital_loss)
        self.test_df.loc[:,'capital_diff'] = (
            self.test_df.capital_gain - self.test_df.capital_loss)

        # And, drop the original capital gain/ capital loss columns
        self.train_df.drop('capital_gain capital_loss'.split(), axis=1,
                            inplace=True)
        self.test_df.drop('capital_gain capital_loss'.split(), axis=1,
                            inplace=True)

    def _clean_categorical_data(self):
        """Cleanup the categorical data, using all the methods we did in
        Lesson 6.
        """

        # Replace the missing data with "NaN"s, since we think there
        # are good reasons why people chose not to provide these
        # pieces of information
        workclass = self.train_df.workclass
        occupation = self.train_df.occupation
        native_country = self.train_df.native_country
        self.train_df.loc[workclass.isnull(), 'workclass'] = 'NaN'
        self.train_df.loc[occupation.isnull(), 'occupation'] = 'NaN'
        self.train_df.loc[native_country.isnull(), 'native_country'] = 'NaN'

        # And for the test data
        workclass = self.test_df.workclass
        occupation = self.test_df.occupation
        native_country = self.test_df.native_country
        self.test_df.loc[workclass.isnull(), 'workclass'] = 'NaN'
        self.test_df.loc[occupation.isnull(), 'occupation'] = 'NaN'
        self.test_df.loc[native_country.isnull(), 'native_country'] = 'NaN'

        # Consolodate the native country field to US, Mexico, NaN, and Other
        countries = 'United-States Mexico NaN'.split()
        self.train_df.loc[~self.train_df.native_country.isin(countries),
                          'native_country'] = 'Other'

        # And for the test data
        self.test_df.loc[~self.test_df.native_country.isin(countries),
                          'native_country'] = 'Other'

        # Consolodate the workinglass values
        self.train_df.loc[
            self.train_df.workclass.isin(
                'Local-gov State-gov Federal-gov'.split()), 'workclass'] = 'Gov'
        self.train_df.loc[
            self.train_df.workclass.isin(
                'Without-pay Never-worked'.split()), 'workclass'] = 'Unemployed'

        # And for the test data
        self.test_df.loc[
            self.test_df.workclass.isin(
                'Local-gov State-gov Federal-gov'.split()), 'workclass'] = 'Gov'
        self.test_df.loc[
            self.test_df.workclass.isin(
                'Without-pay Never-worked'.split()), 'workclass'] = 'Unemployed'

        # Consolodate the occupation column, based off of groupings people
        # frequently make.
        self.train_df.loc[
            self.train_df.occupation.isin(
                'Priv-house-serv Other-service'.split()), 'occupation'] = (
                'Service')
        self.train_df.loc[
            self.train_df.occupation.isin(
                ('Handlers-cleaners Farming-fishing Machine-op-inspct '
                'Transport-moving Craft-repair').split()), 'occupation'] = (
                    'Blue_collar')
        self.train_df.loc[
            self.train_df.occupation.isin(
                'Armed-Forces'.split()), 'occupation'] = 'Military'
        self.train_df.loc[
            self.train_df.occupation.isin(
                'Adm-clerical'.split()), 'occupation'] = 'Admin'
        self.train_df.loc[
            self.train_df.occupation.isin(
                'Tech-support Protective-serv'.split()), 'occupation'] = (
                    'Other_ocupation')
        self.train_df.loc[
            self.train_df.occupation.isin(
                'Prof-specialty Exec-managerial'.split()), 'occupation'] = (
                    'White_collar')

        # And for the test data
        self.test_df.loc[
            self.test_df.occupation.isin(
                'Priv-house-serv Other-service'.split()), 'occupation'] = (
                    'Service')
        self.test_df.loc[
            self.test_df.occupation.isin(
                ('Handlers-cleaners Farming-fishing Machine-op-inspct '
                'Transport-moving Craft-repair').split()), 'occupation'] = (
                    'Blue_collar')
        self.test_df.loc[
            self.test_df.occupation.isin(
                'Armed-Forces'.split()), 'occupation'] = 'Military'
        self.test_df.loc[
            self.test_df.occupation.isin(
                'Adm-clerical'.split()), 'occupation'] = 'Admin'
        self.test_df.loc[
            self.test_df.occupation.isin(
                'Tech-support Protective-serv'.split()), 'occupation'] = (
                    'Other_ocupation')
        self.test_df.loc[
            self.test_df.occupation.isin(
                'Prof-specialty Exec-managerial'.split()), 'occupation'] = (
                    'White_collar')

        # Drop the education_num column since the education column contains the
        # same information, and we want to condense the values somewhat
        self.train_df.drop('education_num', axis=1, inplace=True)
        self.test_df.drop('education_num', axis=1, inplace=True)

        # Now, group the different types of education people had based on HS
        # graduation or not, and a few other groupings
        self.train_df.loc[
            self.train_df.education.isin(
                'Preschool 1st-4th 5th-6th 7th-8th 9th 10th 11th 12th'.split()),
            'education'] = 'Dropout'
        self.train_df.loc[
            self.train_df.education.isin(
                'HS-grad Some-college'.split()), 'education'] = 'HS_grad'
        self.train_df.loc[
            self.train_df.education.isin(
                'Assoc-acdm Assoc-voc'.split()), 'education'] = 'Associates'

        # and for the test data
        self.test_df.loc[
            self.test_df.education.isin(
                'Preschool 1st-4th 5th-6th 7th-8th 9th 10th 11th 12th'.split()),
            'education'] = 'Dropout'
        self.test_df.loc[
            self.test_df.education.isin(
                'HS-grad Some-college'.split()), 'education'] = 'HS_grad'
        self.test_df.loc[
            self.test_df.education.isin(
                'Assoc-acdm Assoc-voc'.split()), 'education'] = 'Associates'

        # Rename the a few race values to make them easier to access later
        self.train_df.loc[
            self.train_df.race == 'Asian-Pac-Islander', 'race'] = 'Asian'
        self.train_df.loc[
            self.train_df.race == 'Amer-Indian-Eskimo', 'race'] = 'Amer_Indian'

        # And on the test data
        self.test_df.loc[
            self.test_df.race == 'Asian-Pac-Islander', 'race'] = 'Asian'
        self.test_df.loc[
            self.test_df.race == 'Amer-Indian-Eskimo', 'race'] = 'Amer_Indian'

        # Now, let's fix the marital_status column
        self.train_df.loc[
            self.train_df.marital_status.isin(
                'Separated Married-spouse-absent Divorced'.split()),
            'marital_status'] = 'Broken_marriage'
        self.train_df.loc[
            self.train_df.marital_status.isin(
                'Never-married'.split()), 'marital_status'] = 'Never_married'
        self.train_df.loc[
            self.train_df.marital_status.isin(
                'Married-AF-spouse Married-civ-spouse'.split()),
            'marital_status'] = 'Married'

        # And for the test data
        self.test_df.loc[
            self.test_df.marital_status.isin(
                'Separated Married-spouse-absent Divorced'.split()),
            'marital_status'] = 'Broken_marriage'
        self.test_df.loc[
            self.test_df.marital_status.isin(
                'Never-married'.split()), 'marital_status'] = 'Never_married'
        self.test_df.loc[
            self.test_df.marital_status.isin(
                'Married-AF-spouse Married-civ-spouse'.split()),
            'marital_status'] = 'Married'

        # Now, create dummy variables
        cols = self.train_df.select_dtypes(include=['object']).columns.tolist()

        # Then remove the education column from this list
        cols.remove('education')

        # Now, for the remaining columns, lets make dummy columns
        self.train_df = pd.merge(
            self.train_df,
            pd.get_dummies(
                self.train_df[cols], prefix=cols, columns=cols),
            right_index=True, left_index=True, how='outer')

        # And delete the columns we no longer need
        self.train_df.drop(cols, axis=1, inplace=True)

        # Repeate for the test data
        self.test_df = pd.merge(
            self.test_df,
            pd.get_dummies(
                self.test_df[cols], prefix=cols, columns=cols),
            right_index=True, left_index=True, how='outer')
        self.test_df.drop(cols, axis=1, inplace=True)

        # Now, create the category list/ order for the education column
        education_categories = ['Dropout', 'HS_grad', 'Associates', 'Bachelors',
                                'Masters', 'Prof-school', 'Doctorate']

        # And, set the category type for the education column with these values
        self.train_df.education = pd.Categorical(
            self.train_df.education, categories=education_categories,
            ordered=True
        )
        self.test_df.education = pd.Categorical(
            self.test_df.education, categories=education_categories,
            ordered=True
        )

        # Fix the test_df income columns (data had a period at the end of the
        # values)
        self.test_df.rename({'income_<=50K.': 'income_<=50K',
                             'income_>50K.': 'income_>50K'}, axis='columns',
                            inplace=True)

    def _normalize_numerics(self):
        """Normalize the numeric columns of the dataframes.
        """

        # 'fnlwgt' is still a feature in the dataframe -- it is a scaling
        # factor to relate the row to the population.  Since there isn't a
        # whole lot of documentation on what exactly this means, let's
        # actually drop this column from our dataset
        self.train_df.drop('fnlwgt', axis=1, inplace=True)
        self.test_df.drop('fnlwgt', axis=1, inplace=True)

        # We see that the non-dummy_variable columns are of type int64
        # (longlong) so use that to grab these column names
        cols = self.train_df.select_dtypes(include=['longlong']).columns\
            .tolist()

        # And apply min-max scaling to each of these columns. Call them
        # "scaled_***"
        for name in cols:
            minmax_scale = MinMaxScaler().fit(self.train_df[name].to_frame())
            self.train_df['scaled_' + name] = minmax_scale.transform(
                self.train_df[name].to_frame()
            )
            self.test_df['scaled_' + name] = minmax_scale.transform(
                self.test_df[name].to_frame()
            )

    def _curate_columns(self):
        """Drop the extra dummy columns that we don't need, since they contain
        redundant information.
        """

        # Create list to drop some:
        drop_col = [
            'workclass_Unemployed',
            'marital_status_Never_married',
            'occupation_NaN',
            'relationship_Not-in-family',
            'race_Other',
            'sex_Male',
            'native_country_NaN',
            'income_<=50K'
        ]

        # Now, drop these columns:
        self.train_df.drop(drop_col, axis='columns', inplace=True)
        self.test_df.drop(drop_col, axis='columns', inplace=True)

    def get_na_cols(self, df):
        """Get columns with null values from the specified dataframe that
        are associated with this class
        
        Args:
            df (str): Name of the dataframe associated with this class instance
                For example, if the instance of this class is mydata,
                then check if mydata.train_df has any columns with null values
                by calling mydata.get_na_cols('train_df')

        Returns:
            pd.Index: A pd.Index object with all the column names that have
                a null value.
        """

        # Make sure we are given a string
        assert type(df) == str, 'Need to give a string!'

        # Try to get this dataframe from this class
        try:
            df = getattr(self, df)

            # Assert that what we got is indeed a pd.DataFrame
            assert type(df) is pd.core.frame.DataFrame, "Didn't grab a df!"

        except AttributeError:
            print("\"{}\" isn't a part of the class!".format(df))
            raise

        # Now, return columns with a null
        return df.columns[df.isnull().any()]

    def _make_ml_data(self):
        """Construct self.train/test_X/Y based on the dataframes. Also, further
        curate the dataframe columns to remove unscaled numeric columns.
        """

        # Copy the train/test data
        self.train_X = self.train_df.copy(deep=True)
        self.test_X = self.test_df.copy(deep=True)

        # Now, drop the unscaled version of the numeric columns:
        drop_col = ['age', 'hours_per_week', 'capital_diff']
        self.train_X.drop(drop_col, axis='columns', inplace=True)
        self.test_X.drop(drop_col, axis='columns', inplace=True)

        # Also, sklearn needs numerical values, so convert the Education column
        # to the integer equivalent of the categories
        self.train_X.education = self.train_X.education.cat.codes
        self.test_X.education = self.test_X.education.cat.codes

        # Now, set the ml data
        self.train_Y = self.train_X['income_>50K']
        self.train_X = self.train_X.drop('income_>50K', axis='columns')
        
        self.test_Y = self.test_X['income_>50K']
        self.test_X = self.test_X.drop('income_>50K', axis='columns')

    def setup_RandomForest(self, **kwargs):
        """Set up a random forest classifier based on any input parameters
        given in **kwargs, and fit to the training data.

        Sets up Attribute
        self.rfc = sklearn.RandomForest classifier, with .fit() run on training
            data.
        """

        # Instantiate the classifier
        self.rfc = RandomForestClassifier(**kwargs)

        # And fit to the training data
        self.rfc.fit(self.train_X, self.train_Y)

    def setup_GaussianNB(self, **kwargs):
        """Set up a Gaussian Naive Bayes classifier based on input parameters
        given in **kwargs, and fit to the training data

        Sets up attribute
        self.nbc = sklearn.naive_bayes.GaussianNB classifier with .fit() run on
            training data.
        """

        # Instantiate the classifier
        self.nbc = GaussianNB(**kwargs)

        # Fit to the training data
        self.nbc.fit(self.train_X, self.train_Y)


def main():
    """Run if executing this script directly. Do this for things asked for
    from this assignment (otherwise, move this work up into __init__ in
    the class)
    """

    # Instantiate the class
    my_data = DataProject(TRAIN_URL, TEST_URL, HEADER_URL)

    # For my project, I chose the Random Forest classifier, and the Naive
    # Bayes classifier.  I chose random forest because, initially, I read
    # that it can be used as one way of pruning features.  However, I ended
    # up not having the time to explore this method.  As for the Naive
    # Bayes, it was one that I've heard about before, so I just wanted to
    # try it out to see how it would perform.
    #
    # Both are classifiers, and so are appropriate for this dataset, where
    # we have a classification problem.

    # Set up and fit the Random Forest classifier
    my_data.setup_RandomForest(n_estimators=20, random_state=0)

    # And the Naive Bayes classifier (we don't have any information about
    # priors, so leave blank)
    my_data.setup_GaussianNB()

    # Now, predict, the test set, and create a dataframe that shows
    # predictions vs actual
    results = pd.DataFrame({
        'RF_predict': my_data.rfc.predict(my_data.test_X),
        'NB_predict': my_data.nbc.predict(my_data.test_X),
        'actual': my_data.test_Y
    })

    # Show these predictions
    print(results.head())

    # And find the fraction of predictions that matched
    print((results.actual == results.RF_predict).sum() /
           float(results.shape[0]))
    print((results.actual == results.NB_predict).sum() /
           float(results.shape[0]))

    # So, we see that we correctly predicted 84% of the test cases for the
    # Random Forest Classifier, and the naive bayes classifier only got
    # 73% of the cases.

    # Let's print the confusion matrix for each. First, for the
    # RandomForest:
    print(skm.confusion_matrix(results.actual, results.RF_predict))
    print(skm.confusion_matrix(results.actual, results.NB_predict))

    # Also, let's make a plot of both matricies
    RF_con_matrix = pd.DataFrame(
        skm.confusion_matrix(results.actual, results.RF_predict),
        index=['Actual_<=50K', 'Actual_>50K'],
        columns=['RF_Predict_<=50K', 'RF_Predict_>50K'])
    NB_con_matrix = pd.DataFrame(
        skm.confusion_matrix(results.actual, results.NB_predict),
        index=['Actual_<=50K', 'Actual_>50K'],
        columns=['NB_Predict_<=50K', 'NB_Predict_>50K'])

    # Normalize by the total, so we get a fraction of all cases per cell
    f = plt.figure()
    ax = f.add_subplot(121)
    sns.heatmap(RF_con_matrix/RF_con_matrix.sum().sum(), annot=True, ax=ax)
    ax = f.add_subplot(122)
    sns.heatmap(NB_con_matrix/NB_con_matrix.sum().sum(), annot=True, ax=ax)
    plt.show()

    # For the random forest, it looks like we did a decent job of predicting,
    # but we have a higher number of false positives in the Naive Bayes
    # classifier.  Once again, here are the accuracy measures:
    RF_AR = skm.accuracy_score(results.actual, results.RF_predict)
    NB_AR = skm.accuracy_score(results.actual, results.NB_predict)
    print(RF_AR, NB_AR)

    # We get more details if we look at Sensitivity and Specificity
    # (TPR and 1-FPR)
    rf_tn, rf_fp, rf_fn, rf_tp = RF_con_matrix.values.ravel()
    nb_tn, nb_fp, nb_fn, nb_tp = NB_con_matrix.values.ravel()

    rf_sensitivity = rf_tp/(rf_tp+rf_fn)
    rf_specificity = rf_tn/(rf_tn+rf_fp)
    nb_sensitivity = nb_tp/(nb_tp+nb_fn)
    nb_specificity = nb_tn/(nb_tn+nb_fp)
    print(rf_sensitivity, rf_specificity, nb_sensitivity, nb_specificity)

    # We were pretty good at identifying people that didn't make over 50k, but
    # only got 60% of people correct who had indeed made over 50k for the
    # Random Forest  classifier.  Interestingly, the Naive Bayes classifier
    # was better at predicting people who made over 50k than correctly
    # distinguishing people that did not (so, it was more likely to classify
    # someone as making over 50k than not, even to the extent of
    # over-classifying high earners)

    # To get a better picture of these algorithms as a whole, let's look at
    # their ROC curves
    rf_predict_prob = my_data.rfc.predict_proba(my_data.test_X)[:, 1]
    nb_predict_prob = my_data.nbc.predict_proba(my_data.test_X)[:, 1]
    rf_fpr, rf_tpr, rf_thresholds = skm.roc_curve(results.actual,
                                                  rf_predict_prob)
    nb_fpr, nb_tpr, nb_thresholds = skm.roc_curve(results.actual,
                                                  nb_predict_prob)

    rf_auc_score = skm.roc_auc_score(results.actual, rf_predict_prob)
    nb_auc_score = skm.roc_auc_score(results.actual, nb_predict_prob)

    plt.plot(rf_fpr, rf_tpr, label='RFC, AUC: {}'.format(rf_auc_score))
    plt.plot(nb_fpr, nb_tpr, label='NBC, AUC: {}'.format(nb_auc_score))
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title('ROC curve for Adult Census Classifiers')
    plt.xlabel('FPR (1 - Specificity)')
    plt.ylabel('TPR (Sensitivity)')
    plt.show()

    # So, we see an AUC of 0.88 for the Random Forest classifier,
    # which isn't bad.  However, the Naive Bayes classifier only had an
    # AUC of 0.867, so it doesn't seem to be performing as well on our data.

    # Let's maximize the Sensitivity and Specificity, so first, let's put
    # them all in a df
    rf_metrics = pd.DataFrame({
        'rf_sensitivity': rf_tpr,
        'rf_specificity': (1-rf_fpr),
        'rf_threshold': rf_thresholds})
    nb_metrics = pd.DataFrame({
        'nb_sensitivity': nb_tpr,
        'nb_specificity': (1-nb_fpr),
        'nb_threshold': nb_thresholds})
    
    # Now, let's find the max sensitivity + specificity
    print("RFC_Max:",
        (rf_metrics.rf_sensitivity + rf_metrics.rf_specificity).max(),
        "NBC_Max:",
        (nb_metrics.nb_sensitivity + nb_metrics.nb_specificity).max()
    )

    # And, the probability threshold at this value (along with the sensitivity/
    # specificity)
    print("RFC_Threshold:\n",
          rf_metrics.loc[(rf_metrics.rf_sensitivity +
          rf_metrics.rf_specificity).idxmax()],
          "NBC_Threshold:\n",
          nb_metrics.loc[(nb_metrics.nb_sensitivity +
          nb_metrics.nb_specificity).idxmax()])

    # So, it looks like the "best" we can do is ~85% sensitivity and ~75%
    # specificity, with a likelihood threshold of ~15.8% for Random Forests,
    # while for Naive Bayes, we have ~80% sensitivity, ~78% specificity,
    # and a likelihood threshold of ~0.98%.  This makes sense -- since
    # Naive Bayes seemed to over-call high earners, increasing the threshold
    # of what you will accept as a call for an high earner should improve
    # our ability to distinguish those that really are.

    ### Using the same application as we had for Lesson 9 ###
    # If we were to assume that we are trying to predict people's income in
    # order to determine who we can sell something to (say, something fairly
    # expensive, so we want to target high earners), it makes more sense for
    # us to want to capture those who can actually afford the product
    # (maximize sensitivity), even if it means we end up sending advertisements
    # to some individuals who we don't think can afford it (who knows, maybe
    # they'll take out loans to buy our product).  In that sense, let's say
    # sending advertisements to half the people that aren't actually in the
    # high earning braket is ok (but let's set a minimum sensitivity threhshold
    # of 85%. Also, in actuality, we would probably want to send ads to lower
    # income people based on what their conversion rate is -- how likely
    # are the lower earners to actually buy our product, all of this balanced
    # by what advertising budget we have).
    #
    # With all of that, here is our range of sensitivity and specificities
    # we'd find acceptable
    print(rf_metrics[(rf_metrics.rf_sensitivity >= 0.85) &
                     (rf_metrics.rf_specificity >= 0.5)],
          nb_metrics[(nb_metrics.nb_sensitivity >= 0.85) &
                     (nb_metrics.nb_specificity >= 0.5)])

    # But that has 183 values for the random forest classifier, and 957
    # rows for the naive bayes, so lets just look at the ranges
    rf_mask = ((rf_metrics.rf_sensitivity >= 0.85) &
               (rf_metrics.rf_specificity >= 0.5))
    print("RF Ranges:")
    print(rf_metrics.loc[rf_metrics[rf_mask].rf_sensitivity.idxmax()])
    print(rf_metrics.loc[rf_metrics[rf_mask].rf_specificity.idxmax()])

    # So, we see that we can get Sensitivity of 95.5% and specificity of
    # 55% with a threshold of 0.48%, as well sensitivity/ specificity
    # thresholds out to sensitivity = 85%, specificty = 75.5%, and the
    # prediction threshold of 16%, for the random forest. And, for the
    # Naive Bayes classifier

    nb_mask = ((nb_metrics.nb_sensitivity >= 0.85) &
               (nb_metrics.nb_specificity >= 0.5))
    print("\nNB Ranges:")
    print(nb_metrics.loc[nb_metrics[nb_mask].nb_sensitivity.idxmax()])
    print(nb_metrics.loc[nb_metrics[nb_mask].nb_specificity.idxmax()])

    # We see we can get a sensitivity of 94.3% and specificity of 50%
    # with a threshold of 2*10**-5.  At the other end, we can get
    # sensitivity = 85%, specificity = 72.4%, and a threshold of 88.9%
    #
    # Looking at that, our original value of 85.3% sensitivity, 75.5%
    # specificity, with a threshold of 15.8% for the Random Forest classifier
    # is not bad!

    # For our final conclusions, we saw this in the ROC curve as well,
    # but for the vast majority of possible threshold values, the Random
    # Forest classifier seems to outperform the Gaussian Naive Bayes
    # classifier, so it probably makes sense to just use it for future work.
    #
    # However, there may be other confounding factors that I am not aware of
    # for **why** the naive bayes classifier didn't perform as well (or,
    # perhaps this is in general true, since the random forest classifier is
    # an ensemble method).
    #
    # For future work, I would like to look into how we can use the Random
    # Forest classifier as a means of pruning my features, but I did not have
    # the time to explore this as of yet.  Perhaps we would get different
    # performance on these classifiers if I indeed had constrained the features
    # some more.

if __name__ == '__main__':
    main()
