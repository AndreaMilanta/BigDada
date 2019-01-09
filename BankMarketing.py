#                                           #
#------------ IMPORTS ----------------------#
#                                           #
import numpy as np
import pandas as pd
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
# import seaborn as sns
# from matplotlib import pyplot

#                                           #
#------------ CONSTANTS --------------------#
#
BankMarketingURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
BankMarketingName = "bank-full.csv"

#                                           #
#------------ PREPROCESSING ----------------#
#                                           #

# import dataset from URL and open it as dataframe
bmzip = urlopen(BankMarketingURL)
bmcsv = ZipFile(BytesIO(bmzip.read())).extract(BankMarketingName)
df = pd.read_csv(bmcsv, sep=';')
print(df.info);