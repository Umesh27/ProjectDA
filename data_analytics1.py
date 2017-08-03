__author__ = 'Umesh'

import os, sys

class data_analytics(object):

    inputData = []
    models = []

    def __init__(self, inputD):
        """
            initializes the data
        :return:
        """

        self.inputData = inputD

    def cleanData(self):

        """
            Function used for cleaning the data
        :return:
        """

        for row in self.inputData:
            print(row)



if __name__ == '__main__':

    if len(sys.argv) == 2:
        inputCSV = sys.argv[1]

    else:
        print("Please give input path of the csv file :")
        exit()
