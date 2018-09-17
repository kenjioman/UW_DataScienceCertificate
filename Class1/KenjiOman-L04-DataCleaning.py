#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Kenji Oman
Created on: 02-01-2018
Purpose: To fulfill requirements for the UW Data Science Certificate,
    Data Science: Process and Tools class, Assignment 4

NOTE: I checked this dataset (at least, the training data), and it seems
    **there are no missing data** for any of the numerical columns.  The only
    missing data appeared in the categorical columns.  I tried to look at
    outlier columns, but I'm now feeling the "outlier" I identified below is
    actually not an outlier (but is instead, probably representative of
    house sales, but since houses are worth more than 100,000 in general, these
    were truncated down to 99,9999 -- the max value they allowed for this
    column).  I will try to explore the relationship between capital-gain and
    some of the other variable before the next assignment.

    PS, Since it sounds like we will be building on this code for the rest of
    the class, I've also taken the liberty of modifying it from my submission
    for assignment 3 to define a python class to compartmentalize the data
    to some extent.
"""

import pandas as pd
import requests
import matplotlib.pyplot as plt  # For plotting
import random as rand  # For replacing outlier values

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

        # Now, set the header for the data sets
        self.train_df.columns = header
        self.test_df.columns = header

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

    # Check which columns contain missing data
    print(my_data.get_na_cols('train_df'))

    # We see 'workclass', 'occupation', 'native-country' all have missing
    # values.  However, these are all categorical columns.  Let's try
    # seeing what we can from the numerical columns
    print(my_data.train_df.describe())

    """see:
    age goes from 17-90, but doesn't look too surprising.
    fnlwgt is a weight that is given to each line to show how representative
      of the full population it is (at least, how representative of the
      state it was drawn from??).
    education-num is actually a classification of different categories of
      education, and maps one-to-one with that column. I will drop this next
    capital-gain and capital-loss -- these seem reasonable, but are highly
      skewed to the right (up to the 75% is 0). So let's take a look at that
      distribution to see what it looks like
    hours-per-week shows nothing too extrodinary, except that the max is
      99 hours a week!  If this person is doing this weekly, although it is
      do-able, this is incredible!
    """
    
    # as we noted in our observation, education-num is just a numbering
    # of the education factors.  So, remove, since it doesn't give any extra
    # information
    my_data.train_df.drop('education-num', axis=1, inplace=True)
    my_data.test_df.drop('education-num', axis=1, inplace=True)

    # Now, let's see if we can spot anything if we visualize the data.
    # Note, the scatter_matrix method grabs all numerical columns and shows
    # a nice grid of plots to make it easy to see the distribution of the
    # values in these columns, and any correlations that might exist between
    # columns.
    #
    # (PS, I realize if we are just executing the script directly, the plots
    #  will just flash by, but I didn't want to clutter the HD with saved
    #  images, so I assume you'll be following along this part by just running
    #  the things in main() one by one)
    pd.plotting.scatter_matrix(my_data.train_df)
    plt.show()

    """see:
    A good amount of binning of age and hours-per-week into discrete values
      (probably integer values for each). Also, perhaps an overabundance of
      people at age 90 and that work 99 hours a week, but these are probably
      individuals that go even higher, but were truncated at these values.
    Overall, nothing really out of the ordinary, except for in capital-gain,
      it looks like there is a big jump from most of the values up to 99999
      (nothing between ~50k to 99,999), with the vast majority of the
      values at 0 (makes sense, since if we look at the capital-gain vs
      capital-loss plot, it looks like we can only have non-zero values for
      one of those categories.)  Let's try looking at captial-gain without
      the capital-loss individuals
    """

    # Plot capital-gains without people that have a capital-loss (who are given
    # capital-gain of 0)
    my_data.train_df['capital-gain']\
        [my_data.train_df['capital-gain'] != 0].hist()
    plt.show()

    """see:
    True enough, it looks like there is an over-abundance of values for
    individuals with a capital-gain of 99,999. Let's have pandas calculate
    the quartiles to confirm.
    """

    # Get the quartiles for the capital-gain column, excluding the capital-loss
    # individuals
    my_data.train_df['capital-gain']\
        [my_data.train_df['capital-gain'] != 0].describe()

    """Indeed, it looks like the 99,999 values are outliers, as the 75% of
    the non-0 values is at 14,084.
    """

    # I was originally going to try to fit a gaussian distribution to the
    # histogram to re-assign the 99,9999 individuals based on it, but if
    # we take a look at:
    my_data.train_df['capital-gain'][
        (my_data.train_df['capital-gain'] != 0) &
        (my_data.train_df['capital-gain'] < 75000)].hist(bins=100)
    plt.show()

    # We see this is a tri-modal distribution (it'll be interesting to
    # see what makes up the groups at the 7-8 thousand range, and the group
    # at ~15 tousand).So, instead, replace with a value randomly picked from
    # the exact non-zero, non-99,9999 distribution. First, need to grab
    # these values
    good_val = my_data.train_df['capital-gain'][
        (my_data.train_df['capital-gain'] != 0) &
        (my_data.train_df['capital-gain'] < 99999)].values

    # Now, define a function that will return a random instance of
    # the good values
    def get_good_val(vals=good_val):
        """quick helper function to allow easy re-assigning of the outlier
        values in the capital-gain column, preserving the non-zero
        distribution.

        Args:
            vals (np.ndarray): The non-zero, non-99,999 values from the
                capital-gain column

        Returns:
            int: A random value from the distribution
        """

        # Return a value from the distribution randomly
        return vals[rand.randint(0, (vals.shape[0] - 1))]

    # Now, use the helper function to reset outlier values based on the
    # non-zero, non-outlier distribution
    # First, define a mask to save the conditions we want
    mask = (my_data.train_df['capital-gain'] == 99999)

    # Now, for these outlier values, reset them based on the good values
    # distribution
    my_data.train_df.loc[mask, 'capital-gain'] = (
        my_data.train_df[mask]['capital-gain'].apply(lambda x: get_good_val())
        )

    # Check the distribution that it looks ok
    my_data.train_df[my_data.train_df['capital-gain'] != 0]['capital-gain']\
        .hist(bins=40)
    plt.show()

    # Looking at the test dataset:
    pd.plotting.scatter_matrix(my_data.test_df)
    plt.show()

    # It looks like we have a similar problem, so repeat the procedure
    # For the test set, using the test set's distribution
    good_val = my_data.test_df['capital-gain'][
        (my_data.test_df['capital-gain'] != 0) &
        (my_data.test_df['capital-gain'] < 99999)].values

    # Define the mask
    mask = (my_data.test_df['capital-gain'] == 99999)

    # Now, for these outlier values, reset them based on the good values
    # distribution
    my_data.test_df.loc[mask, 'capital-gain'] = (
        my_data.test_df[mask]['capital-gain'].apply(
            lambda x: get_good_val(good_val))
        )

    # Check the distribution that it looks ok
    my_data.test_df[my_data.test_df['capital-gain'] != 0]['capital-gain']\
        .hist(bins=40)
    plt.show()

if __name__ == '__main__':
    main()
