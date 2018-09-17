#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Kenji Oman
Created on: 02-22-2018
Purpose: To fulfill requirements for the UW Data Science Certificate,
    Data Science: Process and Tools class, Assignment 7

Note: I split the train_df into a "training" and a "test" dataset at the end,
    just to fulfill the requirements of the course.  However, my data camee
    already split, so for the purposes of the model, I"ve just use the
    pre-split data.
"""

import pandas as pd
import requests
import matplotlib.pyplot as plt  # For plotting
#import random as rand  # For replacing outlier values
import seaborn as sns  # For extra plots
from sklearn.preprocessing import MinMaxScaler  # to normalize numeric values
from sklearn.ensemble import RandomForestClassifier # My chosen classifier
from sklearn.model_selection import train_test_split   # To split the data

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


def main():
    """Run if executing this script directly. Do this for things asked for
    from this assignment (otherwise, move this work up into __init__ in
    the class)
    """

    # Instantiate the class
    my_data = DataProject(TRAIN_URL, TEST_URL, HEADER_URL)

    # From last time, we had every combination of my categorical columns.
    # Since this causes some columns to contain redundant information
    # (for a given categorical column with N categories, only need N-1 dummy
    # columns to fully capture all the information), let's look at these
    # columns and drop some.
    my_data.train_df.columns

    ## Note, looking at the test data's columns
    my_data.test_df.columns

    # we see that the income columns have a period appended to them, so
    # we need to rename them (I traced this actually to the original data
    # downlaod -- for some reason, the test data had periods at the end of
    # their income column values)
    my_data.test_df.rename({'income_<=50K.': 'income_<=50K',
                            'income_>50K.': 'income_>50K'}, axis='columns',
                           inplace=True)

    # Now, create list to drop some:
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

    # Note, I had also created scaled versions of the numeric columns. I
    # originally left the "raw" columns in in case we want to see what
    # these values actually are, but for the purposes of the classifier, let's
    # drop them too.
    drop_col += ['age', 'hours_per_week', 'capital_diff']

    # Now, drop these columns:
    my_data.train_df.drop(drop_col, axis='columns', inplace=True)
    my_data.test_df.drop(drop_col, axis='columns', inplace=True)

    # Also, sklearn needs numerical values, so convert the Education column
    # to the integer equivalent of the categories
    my_data.train_df.education = my_data.train_df.education.cat.codes
    my_data.test_df.education = my_data.test_df.education.cat.codes

    # Now, prepare for the model by grabbing the features/ outcome
    train_X = my_data.train_df.drop('income_>50K', axis='columns').values
    train_Y = my_data.train_df['income_>50K'].values
    test_X = my_data.test_df.drop('income_>50K', axis='columns').values
    test_Y = my_data.test_df['income_>50K'].values

    # Now, setup the classifier variables (I'll be using Random Forests, since
    # I read it is useful for curating the features to a smaller list for other
    # classifiers too)
    estimators = 20  # the number of trees to run
    mss = 2  # Minimum samples per split parameter
    
    # Now, define the classifier
    rfc = RandomForestClassifier(n_estimators=estimators,
                                 min_samples_split=mss)

    # Fit the data
    rfc.fit(train_X, train_Y)

    # Now, predict, the test set, and create a dataframe that shows
    # predictions vs actual
    results = pd.DataFrame({
        'predict':rfc.predict(test_X), 'actual': test_Y
    })

    # Show these predicitons
    print(results.head())

    # And find the fraction of predictions that matched
    print((results.actual == results.predict).sum() / float(results.shape[0]))

    # So, we see that we correctly predicted 84% of the test cases.

    # Since my data already came separated in a training and test set, I didn't
    # bother splitting it, but for the purposes of the assignment, I demonstrate
    # splitting the training set further (if I were doing this for real, I
    # would probably want to combine the two data sets first). Also, in real
    # life, we would set the random seed to be random (or allow it to pick
    # its own seed), but for the purposes of the assingment, I set it here
    # to an integer value
    demo_train_X, demo_train_Y, demo_test_X, demo_test_Y = train_test_split(
        my_data.train_df.drop('income_>50K', axis='columns').values,
        my_data.train_df['income_>50K'].values, random_state=0)
    )


if __name__ == '__main__':
    main()
