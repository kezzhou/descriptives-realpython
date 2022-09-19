#### Imports ####

import math

import statistics

import numpy as np

import scipy.stats

import pandas as pd




#### Descriptive statistics ####

## we can start by creating arbitrary data

x = [8.0, 1, 2.5, 4, 28.0] ## dataset we define as x 

x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0] ## dataset that matches x but has a NaN value

x

x_with_nan ## let's see how they look

