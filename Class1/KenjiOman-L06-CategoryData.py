#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Kenji Oman
Created on: 02-20-2018
Purpose: To fulfill requirements for the UW Data Science Certificate,
    Data Science: Process and Tools class, Assignment 6

Note: Below (in main()), I've gone through all the categorical columns to
consolidate their values (grouping similar things together). For the
few instances where there were missing values, I kept them as NaN, since
I thought there could be reasons why people decided not to respond to the
questions in these categories, so I thought it was significant to keep them
blank.  Finally, I made dummy variables for all the values from these
categorical columns (except for education, since it is ordinal, not nominal),
and I deleted the original columns.

Vijay, what is the most appropriate way to handle ordinal data?
"""

import pandas as pd
import requests
import matplotlib.pyplot as plt  # For plotting
import random as rand  # For replacing outlier values
import seaborn as sns  # For extra plots

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

    """Some exploratory work with categorical columns, since I have the time
    """

    # First, check which columns have NaN
    my_data.get_na_cols('train_df')

    # We see that 'workclass', 'occupation', and 'native_country' all have
    # NaN values, with none of the other columns having an NaN. Let's make a
    # plot of these categories
    workclass = my_data.train_df.workclass
    workclass.value_counts(dropna=False).plot.bar()
    plt.show()

    # We see that a lot of values are NaN, even more so than the Never-worked
    # and without-pay individuals. Let's check for occupation:
    occupation = my_data.train_df.occupation
    occupation.value_counts(dropna=False).plot.bar()
    plt.show()

    # We see that the NaNs is again roughly in the middle of the list. And
    # finally, for native_country
    native_country = my_data.train_df.native_country
    native_country.value_counts(dropna=False).plot.bar()
    plt.show()

    # Here, we see that the vast majority of individuals were from the US,
    # with NaNs showing up as #3 on the list.  These NaN individuals could be
    # people born in the US (and thought it was trivial to answer the
    # country of origin question), or could indeed be from other countries,
    # so we need to consider what we want to do with the NaN values in each of
    # these columns.  Let's consider these columns together
    sns.countplot(x='workclass', hue='occupation',
                  data=my_data.train_df.fillna('na'))
    plt.show()

    # From this, it at least looks like workclass and occupation only have
    # NaNs when the other also has a NaN. Let's check if this is true
    print(my_data.train_df.workclass.isna().sum())
    print(my_data.train_df.occupation.isna().sum())
    print((my_data.train_df.workclass.isna() &
           my_data.train_df.occupation.isna()).sum())

    # We see that there are only 7 instances where occupation has a NaN and
    # workclass does not, and all NaNs in workclass are shared by a NaN in
    # occupation.  This is compared to:
    print(my_data.train_df.native_country.isna().sum())
    print((my_data.train_df.native_country.isna() &
           my_data.train_df.occupation.isna()).sum())
    print((my_data.train_df.native_country.isna() &
           my_data.train_df.occupation.isna() &
           my_data.train_df.workclass.isna()).sum())

    # Where we see that Native_country has 583 NaNs, with only 27 of them
    # shared for when occupation and workclass also has a NaN. So, there is
    # likely a different reason for why people don't give their native country,
    # while people don't share their workingclass and occupation for largely
    # the same reason (different from the reason they don't share their native
    # country).  Although there are a lot of things we could do to try to
    # impute these missing values, for the sake of time, let's set them
    # to "NaN" (the string), and consolidate some columns.
    my_data.train_df.loc[workclass.isnull(), 'workclass'] = 'NaN'
    my_data.train_df.loc[occupation.isnull(), 'occupation'] = 'NaN'
    my_data.train_df.loc[native_country.isnull(), 'native_country'] = (
        'NaN')

    # And for the test data
    workclass = my_data.test_df.workclass
    occupation = my_data.test_df.occupation
    native_country = my_data.test_df.native_country
    my_data.test_df.loc[workclass.isnull(), 'workclass'] = 'NaN'
    my_data.test_df.loc[occupation.isnull(), 'occupation'] = 'NaN'
    my_data.test_df.loc[native_country.isnull(), 'native_country'] = (
        'NaN')

    # Now, let's look at the native country:
    my_data.train_df.native_country.value_counts().plot.bar()
    plt.show()

    # Just curious, let's see how this relates to income, normalized per group
    df = my_data.train_df.groupby('native_country')['income']\
        .value_counts(normalize=True).rename('fraction').reset_index()\
        .sort_values('fraction')
    sns.barplot(x='native_country', y='fraction', hue='income', data=df)
    plt.xticks(rotation=90)
    plt.show()

    # Considering how there are vastly more people in the US, Mexico, and NaN,
    # let's consolidate to 4 categories: United-States, Mexico, NaN, and
    # Other. So, need to grab all the Other countries and set their value
    countries = 'United-States Mexico NaN'.split()
    my_data.train_df.loc[
        ~my_data.train_df.native_country.isin(countries),
        'native_country'] = 'Other_countries'

    # Now, for workingclass:
    print(my_data.train_df.workclass.value_counts())

    # Once again, let's look at how these compare to income, normalized
    # per group
    df = my_data.train_df.groupby('workclass')['income']\
        .value_counts(normalize=True).rename('fraction').reset_index()\
        .sort_values('fraction')
    sns.barplot(x='workclass', y='fraction', hue='income', data=df)
    plt.show()

    # We see local and state government individuals to have similar
    # income distributions, but federal isn't too far off, so let's put
    # them in the same category.  Also, the never-worked and without-pay
    # individuals are both likely unemployed, so let's combine them.
    my_data.train_df.loc[
        my_data.train_df.workclass.isin(
            'Local-gov State-gov Federal-gov'.split()), 'workclass'] = 'Gov'
    my_data.train_df.loc[
        my_data.train_df.workclass.isin(
            'Without-pay Never-worked'.split()), 'workclass'] = 'Unemployed'

    # And for the test data
    my_data.test_df.loc[
        my_data.test_df.workclass.isin(
            'Local-gov State-gov Federal-gov'.split()), 'workclass'] = 'Gov'
    my_data.test_df.loc[
        my_data.test_df.workclass.isin(
            'Without-pay Never-worked'.split()), 'workclass'] = 'Unemployed'

    # Now, look at Occupation
    print(my_data.train_df.occupation.value_counts())

    # And by income fractions
    df = my_data.train_df.groupby('occupation')['income']\
        .value_counts(normalize=True).rename('fraction').reset_index()\
        .sort_values('fraction')
    sns.barplot(hue='income', y='fraction', x='occupation', data=df)
    plt.show()

    # Let's try consolodating these a bit, based on what people might
    # traditionally call these types of occupations:
    my_data.train_df.loc[
        my_data.train_df.occupation.isin(
            'Priv-house-serv Other-service'.split()), 'occupation'] = 'Service'
    my_data.train_df.loc[
        my_data.train_df.occupation.isin(
            ('Handlers-cleaners Farming-fishing Machine-op-inspct '
             'Transport-moving Craft-repair').split()), 'occupation'] = (
                'Blue_collar')
    my_data.train_df.loc[
        my_data.train_df.occupation.isin(
            'Armed-Forces'.split()), 'occupation'] = 'Military'
    my_data.train_df.loc[
        my_data.train_df.occupation.isin(
            'Adm-clerical'.split()), 'occupation'] = 'Admin'
    my_data.train_df.loc[
        my_data.train_df.occupation.isin(
            'Tech-support Protective-serv'.split()), 'occupation'] = (
                'Other_ocupation')
    my_data.train_df.loc[
        my_data.train_df.occupation.isin(
            'Prof-specialty Exec-managerial'.split()), 'occupation'] = (
                'White_collar')

    # And for the test data
    my_data.test_df.loc[
        my_data.test_df.occupation.isin(
            'Priv-house-serv Other-service'.split()), 'occupation'] = 'Service'
    my_data.test_df.loc[
        my_data.test_df.occupation.isin(
            ('Handlers-cleaners Farming-fishing Machine-op-inspct '
             'Transport-moving Craft-repair').split()), 'occupation'] = (
                'Blue_collar')
    my_data.test_df.loc[
        my_data.test_df.occupation.isin(
            'Armed-Forces'.split()), 'occupation'] = 'Military'
    my_data.test_df.loc[
        my_data.test_df.occupation.isin(
            'Adm-clerical'.split()), 'occupation'] = 'Admin'
    my_data.test_df.loc[
        my_data.test_df.occupation.isin(
            'Tech-support Protective-serv'.split()), 'occupation'] = (
                'Other_ocupation')
    my_data.test_df.loc[
        my_data.test_df.occupation.isin(
            'Prof-specialty Exec-managerial'.split()), 'occupation'] = (
                'White_collar')

    # Now, taking a look at the other object columns:
    print(my_data.train_df.dtypes)

    # We see that there is an education column and an education-num column.
    # Checking what we have for these:
    print(my_data.train_df['education education_num'.split()]\
          .drop_duplicates().sort_values('education_num'))

    # We see that the education_num column is just an encoding of the
    # education column, so let's drop the education_num column
    my_data.train_df.drop('education_num', axis=1, inplace=True)
    my_data.test_df.drop('education_num', axis=1, inplace=True)

    # I dropped education_num above, but as I've thought about it,
    # the education column is actually ordinal, not nominal, so the ordering
    # given in education_num matters. As for how to deal with ordinal data is
    # another question ...  But, we should be able to consolidate some, and
    # still have an ordering intact. First let's check the relationship of
    # level of education with income
    df = my_data.train_df.groupby('education')['income']\
        .value_counts(normalize=True).rename('fraction').reset_index()\
        .sort_values('fraction')
    sns.barplot(hue='income', y='fraction', x='education', data=df)
    plt.show()

    # So, look like we can group people based on if they dropped out before
    # high school graduation or not, if people were HS graduates, and a few
    # more groupings
    my_data.train_df.loc[
        my_data.train_df.education.isin(
            'Preschool 1st-4th 5th-6th 7th-8th 9th 10th 11th 12th'.split()),
        'education'] = 'Dropout'
    my_data.train_df.loc[
        my_data.train_df.education.isin(
            'HS-grad Some-college'.split()), 'education'] = 'HS_grad'
    my_data.train_df.loc[
        my_data.train_df.education.isin(
            'Assoc-acdm Assoc-voc'.split()), 'education'] = 'Associates'

    # and for the test data
    my_data.test_df.loc[
        my_data.test_df.education.isin(
            'Preschool 1st-4th 5th-6th 7th-8th 9th 10th 11th 12th'.split()),
        'education'] = 'Dropout'
    my_data.test_df.loc[
        my_data.test_df.education.isin(
            'HS-grad Some-college'.split()), 'education'] = 'HS_grad'
    my_data.test_df.loc[
        my_data.test_df.education.isin(
            'Assoc-acdm Assoc-voc'.split()), 'education'] = 'Associates'


    # Now, need to check marital_status, relationship, race, columns to see
    # if there is anything else we can consolidate from them.
    nominal_counts = pd.DataFrame()
    for col in 'marital_status relationship race'.split():
        nominal_counts = pd.merge(
            nominal_counts,
            my_data.train_df[col].value_counts(dropna=False).reset_index()\
                .rename(columns={'index': col, col: (col + '_cnt')}),
            how='outer', left_index=True, right_index=True)

    print(nominal_counts)

    # Let's look at the relationship data in greater detail -- I think
    # there are probably some patterns related to gender, so let's look
    # at the fraction of individuals of each gender for this column
    df = my_data.train_df.groupby('relationship')['sex']\
        .value_counts(normalize=True).rename('fraction').reset_index()\
        .sort_values('fraction')
    sns.barplot(hue='sex', y='fraction', x='relationship', data=df)
    plt.show()

    # I would have thought that Own-child would mean a single-parent, and
    # so would have a higher number of women, but this doesn't seem to be
    # the case. Also, I would have thought Unmarried would mean people that
    # are living together, but aren't married (as opposed to Not-in-family)
    # but we see many more women in this category, so that can't be the case
    # either.  Since it's hard to decipher what the attributes mean, let's
    # leave this column alone. Let's rename the values in Race to condense
    # them some.
    my_data.train_df.loc[
        my_data.train_df.race == 'Asian-Pac-Islander', 'race'] = 'Asian'
    my_data.train_df.loc[
        my_data.train_df.race == 'Amer-Indian-Eskimo', 'race'] = 'Amer_Indian'
    my_data.train_df.loc[
        my_data.train_df.race == 'Ohter', 'race'] = 'Other_race'

    # And on the test data
    my_data.test_df.loc[
        my_data.test_df.race == 'Asian-Pac-Islander', 'race'] = 'Asian'
    my_data.test_df.loc[
        my_data.test_df.race == 'Amer-Indian-Eskimo', 'race'] = 'Amer_Indian'
    my_data.test_df.loc[
        my_data.test_df.race == 'Ohter', 'race'] = 'Other_race'

    # Now, for the last nominal column, let's compare marital_status with
    # income
    df = my_data.train_df.groupby('marital_status')['income']\
        .value_counts(normalize=True).rename('fraction').reset_index()\
        .sort_values('fraction')
    sns.barplot(hue='income', y='fraction', x='marital_status', data=df)
    plt.show()

    # Let's rename Never-married, and consolidate Separated,
    # Married-spouse-absent, and Divorced (since they're all related to being
    # separated, and their income distributions are similar), and the two
    # married groupings
    my_data.train_df.loc[
        my_data.train_df.marital_status.isin(
            'Separated Married-spouse-absent Divorced'.split()),
        'marital_status'] = 'Broken_marriage'
    my_data.train_df.loc[
        my_data.train_df.marital_status.isin(
            'Never-married'.split()), 'marital_status'] = 'Never_married'
    my_data.train_df.loc[
        my_data.train_df.marital_status.isin(
            'Married-AF-spouse Married-civ-spouse'.split()),
        'marital_status'] = 'Married'

    # And for the test data
    my_data.test_df.loc[
        my_data.test_df.marital_status.isin(
            'Separated Married-spouse-absent Divorced'.split()),
        'marital_status'] = 'Broken_marriage'
    my_data.test_df.loc[
        my_data.test_df.marital_status.isin(
            'Never-married'.split()), 'marital_status'] = 'Never_married'
    my_data.test_df.loc[
        my_data.test_df.marital_status.isin(
            'Married-AF-spouse Married-civ-spouse'.split()),
        'marital_status'] = 'Married'

    # Now that we've done some consolodating, let's take a look again at
    # all ordinal/ nominal columns and their value-counts
    factor_counts = pd.DataFrame()
    for col in my_data.train_df.select_dtypes(include=['object']).columns:
        factor_counts = pd.merge(
            factor_counts,
            my_data.train_df[col].value_counts(dropna=False).reset_index()\
                .rename(columns={'index': col, col: (col + '_cnt')}),
            how='outer', left_index=True, right_index=True)

    print(factor_counts)

    # Now, create dummy variables of all but education columns (since this
    # is the only ordinal column). First, grab all columns with object type
    cols = my_data.train_df.select_dtypes(include=['object']).columns.tolist()

    # Then remove the education column from this list
    cols.remove('education')

    # Now, for the remaining columns, lets make dummy columns
    my_data.train_df = pd.merge(
        my_data.train_df,
        pd.get_dummies(
            my_data.train_df[cols], prefix=cols, columns=cols),
        right_index=True, left_index=True, how='outer')

    # And delete the columns we no longer need
    my_data.train_df.drop(cols, axis=1, inplace=True)

    # Repeate for the test data
    my_data.test_df = pd.merge(
        my_data.test_df,
        pd.get_dummies(
            my_data.test_df[cols], prefix=cols, columns=cols),
        right_index=True, left_index=True, how='outer')
    my_data.test_df.drop(cols, axis=1, inplace=True)

    """For a Summary of what I've done, see the note in the top comment block,
    at the top of this file.
    """


if __name__ == '__main__':
    main()
