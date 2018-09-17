#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Kenji Oman
Created on: 02-11-2018
Purpose: To fulfill requirements for the UW Data Science Certificate,
    Data Science: Process and Tools class, Assignment 5

NOTE: At the conclusion of Assignment 4, I felt what I had done with removing
    the "outlier" in the capital-gain column was actually incorrect, as I
    believe these are probably due to house sales (whose values are much
    greater, but were capped at, 99,999).  Instead, this assignment fulfills
    the requirements of the assignment (make a histogram, do median-imputation,
    and outlier replacement), but in terms of the dataset, explores the
    correlation of the capital-gain "outlier" with the other data columns
    in greater detail.
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

    # Last time, I noticed captial-gain and capital-loss columns seem to
    # only have non-zero values when the other one is zero. Here, let's
    # explicitly check (if this is true, we can condense these columns into
    # one column)
    print(my_data.train_df[(my_data.train_df.capital_gain != 0) &
                     (my_data.train_df.capital_loss != 0)].shape[0])
    print(my_data.test_df[(my_data.test_df.capital_gain != 0) &
                     (my_data.test_df.capital_loss != 0)].shape[0])

    # So, we see that each column is only != 0 if the other = 0. So, let's
    # Collapse these columns
    my_data.train_df.loc[:,'capital_diff'] = (my_data.train_df.capital_gain -
                                              my_data.train_df.capital_loss)
    my_data.test_df.loc[:,'capital_diff'] = (my_data.test_df.capital_gain -
                                             my_data.test_df.capital_loss)

    # Now, we don't need the capital_gain or capital_diff columns, so let's
    # drop them
    my_data.train_df.drop('capital_gain capital_loss'.split(), axis=1,
                          inplace=True)
    my_data.test_df.drop('capital_gain capital_loss'.split(), axis=1,
                         inplace=True)

    # Let's plot the 99,999 valued capital_gain individuals compared to
    # everyone else (Note, I'm on a hi-dpi screen, so I've made the points
    # fairly large to make them visible, but they may need to be set to
    # something smaller for other screens)
    pd.plotting.scatter_matrix(
        my_data.train_df,
        c=my_data.train_df.capital_diff.apply(lambda x: 'red' if x == 99999
                                              else 'blue'),
        s=my_data.train_df.capital_diff.apply(lambda x: 50 if x == 99999
                                              else 10))
    plt.show()

    # Although it is harder to see the 99,999 individuals in anywhere but those
    # plots eplicitly comparing capital_diff to the other numerical categories,
    # it does seem like there would be interesting distributions in the other
    # columns, so let's explore this a bit more.

    # To make it easier to do these comparisons, just grab the individuals in
    # questions
    high_gain = my_data.train_df[my_data.train_df.capital_diff == 99999]

    # Now, let's make a scatter_matrix of these individuals
    pd.plotting.scatter_matrix(high_gain)
    plt.show()

    # If we look at the age distribution, it looks like the number of people
    # over 50 with capital_gain 99999 drastically drops. So, let's look at
    # this histogram in more detail
    high_gain.age.hist(bins=50)
    plt.show()

    '''This shows that, indeed, the number of people with capital gain = 99,999
    drops off fairly quickly after age 50, with a small number of individuals
    below the age of 30.  This is consistent with the hypothesis that
    capital gain = 99,999 individuals are those with a house sale, as young
    people are unlikely to have a house to sell, and those above a certain
    age are unlikely to want to sell (and instead, stay in their homes until
    they retire -- the data is consistent with this as well, as there is
    another small bump in indiviuals at age 65, and 70, typical ages of
    retirement).
    '''

    # Let's look at how the education_num compares to education
    print(my_data.train_df[['education', 'education_num']].drop_duplicates()
          .sort_values('education_num'))

    # With that, we have an idea of what the education numbers mean in the
    # histogram for high_gain individuals. First, store the mapping of
    # education to education_num
    edu_map = my_data.train_df[['education', 'education_num']]\
        .drop_duplicates().sort_values('education_num')

    # Now, let's plot the high gain individuals on the same plot as everyone
    # else, but normalize the histograms so we can compare the distributions
    # directly
    plt.hist(high_gain.education_num, bins=20, alpha=0.4, label='high_gain',
             normed=True)
    plt.hist(my_data.train_df.loc[my_data.train_df.capital_diff != 99999,
                                  'education_num'],
             bins=20, alpha=0.4, label='non_high', normed=True)
    plt.xticks(edu_map.education_num, edu_map.education)
    plt.legend(loc='upper left')
    plt.show()

    # Although this looks pretty good, the bins aren't matching up, so let's
    # instead do this as a bar chart. First, grab the value counts for each
    # of the education levels
    high_gain_counts = high_gain.education_num.value_counts().sort_index()
    non_high_counts = my_data.train_df[my_data.train_df.capital_diff != 99999]\
        ['education_num'].value_counts().sort_index()

    # Now, make the bar-charts
    plt.bar(non_high_counts.index,
            non_high_counts.values/ non_high_counts.sum(),
            label='non_high', alpha=0.4)
    plt.bar(high_gain_counts.index,
            high_gain_counts.values / high_gain_counts.sum(),
            label='high_gain', alpha=0.4)
    plt.xticks(edu_map.education_num, edu_map.education)
    plt.legend(loc='upper left')
    plt.show()

    '''We clearly see that the capital gain = 99,999 individuals tend to have a
    higher education, although it is surprising how large a fraction of
    individuals are also included from the high school education only group.
    This is compared to the non high capital gain group (capital gain !=
    99,999), which have a markedly larger proportion of individuals with only
    a high school education.

    This observation is also consistent with the hypothesis that high
    capital gain individuals are likely those that have sold a house, as it is
    more likely for individuals to have a house that they can sell if they
    had the education (income) to be able to buy one in the first place.

    To further support how this group correlates to income, let's compare with
    the income group next.
    '''

    # Let's grab the value counts for the high_gain and non_high_gain
    # individuals to make our bar charts
    high_gain_counts = high_gain.income.value_counts().sort_index()
    non_high_counts = my_data.train_df[my_data.train_df.capital_diff != 99999]\
        ['income'].value_counts().sort_index()

    # Now, let's make the plot
    plt.bar(non_high_counts.index,
            non_high_counts.values/ non_high_counts.sum(),
            label='non_high', alpha=0.4)
    plt.bar(high_gain_counts.index,
            high_gain_counts.values / high_gain_counts.sum(),
            label='high_gain', alpha=0.4)
    plt.legend(loc='upper left')
    plt.show()

    '''We see that indeed, all capital-gain = 99,999 individuals are high
    earners (>50k), while ~80% of non high capital gain individuals make less
    than 50k.

    Caveat: If the captial gain values were used in determining the income,
    it would make sense that anyone with captial-gain = 99,999 would be in
    the high-income group, since 99,999 is greater than 50k.

    --------------------------------------------------------------
    With the analysis above, I feel confident that the capital-gain = 99,999
    values are real -- these are likely those people who sold a house within
    the year the data was gathered (1994 census), but the actual capital-gains
    they claimed were all truncated down to 99,999, in the same way that age
    and hours-per-week also seem to have been truncated.

    With this, I believe there are no outliers in the numerical columns. So,
    I will do median imputation of a numeric value, just to fulfill the
    requirements of the assignment (but, ultimately, I don't think this is
    necessary for the real dataset)
    '''

    # For the purposes of the assignment, let's replace the capital-gain =
    # 99,999 individuals with the median of the remaining capital gain
    # individuals. First, identify who has capital-gain = 99,999
    replace_mask = (my_data.train_df.capital_diff == 99999)

    # Now, replace these values with the median of what's left -- but,
    # need to also ensure capital-diff is > 0 when calculating the median,
    # or else will pull this value down (since I've included capital-loss
    # values in this column now).  Although, also note -- I would technically
    # have to go to a point earlier in my code to where I hadn't
    # dropped the capital-gain column, since there are some individuals
    # with both capital-gain and capital-loss = 0, which my constraint here
    # will ignore in calculating the median.
    my_data.train_df.loc[replace_mask, 'capital_diff'] = (
        my_data.train_df[~replace_mask & (my_data.train_df.capital_diff > 0)]
        .capital_diff.median()
    )

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


if __name__ == '__main__':
    main()
