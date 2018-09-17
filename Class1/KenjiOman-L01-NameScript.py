#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Kenji Oman
Created on: 01-18-2018
Purpose: To fulfill requirements for the UW Data Science Certificate,
    Data Science: Process and Tools class, Assignment 1
"""

import time


def im_kenji():
    """Function that returns my name
    """
    return 'Kenji Oman'


im_kenji()

# Formatting of time code derived from
# https://docs.python.org/3/library/time.html#time.struct_time
print(time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))

