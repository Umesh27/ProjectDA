__author__ = 'Umesh'

import os, sys
import pandas as pd

class data_analytics(object):

    models = []

    def __init__(self, inputFile):
        """
            initializes the data
        :return:
        """
        self.inputFile = inputFile
        self.read_csv()
        #self.inputData = inputD
        self.cleanData()

    def cleanData(self):

        """
            Function used for cleaning the data
        :return:
        """
        print(self.inputData)
        self.getNullInd()
        self.inputData = self.inputData.drop(self.indices)
        print(self.inputData)
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



if __name__ == '__main__':

    if len(sys.argv) == 2:
        inputCSV = sys.argv[1]

    else:
        print("Please give input path of the csv file :")
        exit()

    DA = data_analytics(inputCSV)
    print("InputData after cleaning : \n",DA.inputData[0])
