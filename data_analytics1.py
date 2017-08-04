__author__ = 'Umesh'

import os, sys
import pandas as pd
import numpy as np
from sklearn import model_selection

class data_analytics(object):

    models = []

    def __init__(self, inputFile, sFactor=1.0):
        """
            initializes the data
        :return:
        """
        self.inputFile = inputFile
        self.read_csv()
        self.cleanData()
        self.inputData.head()
        self.nRows = self.inputData.shape[0]
        self.nCols = self.inputData.shape[1]
        self.splitFactor = sFactor
        self.split_data()

    def cleanData(self):

        """
            Function used for cleaning the data
        :return:
        """
        #print(self.inputData)
        self.getNullInd()
        self.inputData = self.inputData.drop(self.indices)
        print(self.inputData.head())
        # for row in self.inputData:
        #     print(row)
        return self.inputData

    def getNullInd(self):
        """

        :return:
        """
        self.indices = pd.isnull(self.inputData).any(1).nonzero()[0]
        print(self.indices)
        return self.indices

    def read_csv(self):
        """

        :return:
        """
        self.inputData = pd.read_csv(self.inputFile, na_values = [])

    def regression_model(self, dependentVariable):
        """
            method used for creating regression model with considering the input data
            :param dependentVariable : it is variable which we have to predict
        :return:
        """

    def split_data(self):
        """
            Method use to split data as per requirement in training & testing dataset
        :return:
        """
        #self.inputData_training = self.inputData.take(np.random.permutation(len(self.inputData))[:self.splitFactor])
        # Split-out validation dataset
        self.inputDataArray = self.inputData.values
        self.independent = self.inputDataArray[:, 0:(self.nCols-1)]
        print("Independent dataset : \n")
        print(self.independent[0,:])
        print("dependent dataset : \n")
        self.dependent = self.inputDataArray[:, (self.nCols-1)]
        print(self.dependent[0:5])
        validation_size = 0.20
        seed = 7
        self.inputDataTrain_X, self.inputDataTest_X, self.inputDataTrain_Y, self.inputDataTest_Y = model_selection.train_test_split(self.independent, self.dependent,
                                                                                                                                    test_size=(1 - self.splitFactor), random_state=seed)
        return self.inputDataTrain_X, self.inputDataTest_X, self.inputDataTrain_Y, self.inputDataTest_Y

if __name__ == '__main__':

    if len(sys.argv) == 3:
        inputCSV = sys.argv[1]
        splitFact = float(sys.argv[2])

    else:
        print("Please give input path of the csv file & splitFactor for training dataset (default is 1.0) :")
        exit()

    DA = data_analytics(inputCSV, sFactor=splitFact)
    print("InputData after cleaning : \n",DA.inputData.shape)
    print("InputData after splitting : \n",DA.inputDataTrain_X.shape)
    print("InputData after splitting : \n",DA.inputDataTrain_Y.shape)
    print("InputData after splitting : \n",DA.inputDataTest_X.shape)
    print("InputData after splitting : \n",DA.inputDataTest_Y.shape)
