#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Kenji Oman
Created on: 01-31-2018
Purpose: To fulfill requirements for the UW Data Science Certificate,
    Data Science: Process and Tools class, Assignment 3
"""

import pandas as pd
import requests

# Define the URL's for the training and test data
TRAIN_URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
             'adult/adult.data')

TEST_URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
            'adult/adult.test')

# And define the URL of the file which contains the header names
HEADER_URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
              'adult/adult.names')


def main():
    """Run if executing this script directly
    """

    # Grab the data
    train_df = pd.read_csv(TRAIN_URL, header=None)
    # For the training data, have one comment row, so need to ignore
    test_df = pd.read_csv(TEST_URL, header=None, skiprows=1)

    # Get the header data
    response = requests.get(HEADER_URL)
    header = response.text.split('\n')

    # Now, filter to grab the header lines:
    # First, make sure there is at least one character for the line, and
    # ignore lines that start with the comment character for the file "|"
    header = [row for row in header if len(row) > 0 and row[0] != '|']

    # Ignore the first row, since it is just identifying the classifier task
    # and, get just the header values
    header = [head.split(':')[0] for head in header[1:]]

    # Finally, we need to add a header name for the last column (if <= or >
    # income of 50k)
    header.append('income')

    # Now, set the header for the data sets
    train_df.columns = header
    test_df.columns = header

    # Now, print the first five rows of the training set
    print(train_df.head())


if __name__ == '__main__':
    main()
