__author__ = 'Umesh'

import os, sys
import pandas as pd
import numpy as np
from sklearn import model_selection, linear_model
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

class data_analytics(object):

    models = []

    def __init__(self, inputFile, sFactor=0.0):
        """
            initializes the data
        :return:
        """
        self.inputFile = inputFile
        self.parentPath = os.path.split(self.inputFile)[0]
        self.figuresPath = os.path.join(self.parentPath, "Figures")
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

    def plot_model(self, X, Y, label_, color_="black"):
        """

        :return:
        """

        self.ax.scatter(X, Y, label=label_, color=color_)
        #ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
        self.ax.set_xlabel('Measured')
        self.ax.set_ylabel('Predicted')

    def lr_model1(self):
        """

        :return:
        """

        lm = linear_model.LinearRegression()
        model = lm.fit(self.inputDataTrain_X, self.inputDataTrain_Y)
        # print("intercept for train model : ", lm.intercept_)
        # print("coefficients for train model : ")
        # for i in range(lm.coef_.shape[0]):
        #     for j in range(lm.coef_.shape[1]):
        #         print(float(lm.coef_[i][j]))

        fig, self.ax = plt.subplots()

        predictionsTrain_model1 = lm.predict(self.inputDataTrain_X)
        print(predictionsTrain_model1.shape, self.inputDataTrain_Y.shape)

        print("Coefficient of determination train: ", lm.score(self.inputDataTrain_X, self.inputDataTrain_Y))
        self.plot_model(self.inputDataTrain_Y, predictionsTrain_model1, "trainData", color_="blue")

        if self.splitFactor < 1.0:
            print("Coefficient of determination test: ", lm.score(self.inputDataTest_X, self.inputDataTest_Y))
            predictionsTest_model1 = lm.predict(self.inputDataTest_X)
            print(predictionsTest_model1.shape, self.inputDataTest_Y.shape)
            self.plot_model(self.inputDataTest_Y, predictionsTest_model1, "testData", color_="red")

        lr_model1_path = os.path.join(self.figuresPath, "LR_Train_Test_model1.pdf")
        plt.legend(loc="upper left")
        plt.savefig(lr_model1_path)
        plt.show()
        plt.close()

    def lr_model2(self):
        """

        :return:
        """

        #predictions_cv = model_selection.cross_val_predict(lm, X, Y, cv = 10)
        lm = linear_model.LinearRegression()
        predictionsTrain_model2 = model_selection.cross_val_predict(lm, self.inputDataTrain_X, self.inputDataTrain_Y, cv = 10)

        #print("Coefficient of determination train: ", lm.score(self.inputDataTrain_X, self.inputDataTrain_Y))
        #print("Coefficient of determination test: ", lm.score(self.inputDataTest_X, self.inputDataTest_Y))
        fig, self.ax = plt.subplots()

        #predictionsTrain_model2 = lm.predict(self.inputDataTrain_X)
        self.plot_model(self.inputDataTrain_Y, predictionsTrain_model2, "trainData")

        # predictionsTest_model2 = lm.predict(self.inputDataTest_X)
        # self.plot_model(self.inputDataTest_Y, predictionsTest_model2)
        lr_model2_path = os.path.join(self.figuresPath, "LR_Train_Test_model2.pdf")
        plt.legend(loc="upper left")
        plt.savefig(lr_model2_path)
        plt.show()
        plt.close()

    def lr_model3(self):
        """

        :return:
        """

        from sklearn.isotonic import IsotonicRegression

        #predictions_cv = model_selection.cross_val_predict(lm, X, Y, cv = 10)
        ir = IsotonicRegression()
        predictionsTrain_model3 = ir.fit_transform(self.inputDataTrain_X, self.inputDataTrain_Y)

        fig, self.ax = plt.subplots()

        #predictionsTrain_model2 = lm.predict(self.inputDataTrain_X)
        self.plot_model(self.inputDataTrain_Y, predictionsTrain_model3, "trainData")

        # predictionsTest_model2 = lm.predict(self.inputDataTest_X)
        # self.plot_model(self.inputDataTest_Y, predictionsTest_model2)
        lr_model3_path = os.path.join(self.figuresPath, "LR_Train_Test_model3.pdf")
        plt.legend(loc="upper left")
        plt.savefig(lr_model3_path)
        plt.show()
        plt.close()

    def decision_tree_model1(self, maxDepth=None):
        """

        :return:
        """

        from sklearn.tree import DecisionTreeRegressor

        #predictions_cv = model_selection.cross_val_predict(lm, X, Y, cv = 10)
        decisionTreeRegr = DecisionTreeRegressor(max_depth=maxDepth)
        model4 = decisionTreeRegr.fit(self.inputDataTrain_X, self.inputDataTrain_Y)
        fig, self.ax = plt.subplots()

        print("Coefficient of determination train: ", decisionTreeRegr.score(self.inputDataTrain_X, self.inputDataTrain_Y))
        predictionsTrain_model4 = decisionTreeRegr.predict(self.inputDataTrain_X)
        self.plot_model(self.inputDataTrain_Y, predictionsTrain_model4, "trainData", color_="blue")

        if self.splitFactor < 1.0:
            print("Coefficient of determination test: ", decisionTreeRegr.score(self.inputDataTest_X, self.inputDataTest_Y))
            predictionsTest_model4 = decisionTreeRegr.predict(self.inputDataTest_X)
            self.plot_model(self.inputDataTest_Y, predictionsTest_model4, "trainData", color_="red")

        lr_model3_path = os.path.join(self.figuresPath, "DecisionTree_Train_Test_%s.pdf"%str(maxDepth))
        if os.path.exists(lr_model3_path):
            lr_model3_path = os.path.join(self.figuresPath, "DecisionTree_Train_Test_depth_%s.pdf"%str(maxDepth))

        plt.legend(loc="upper left")
        plt.savefig(lr_model3_path)
        plt.show()
        plt.close()

    def bagging_regressor(self):
        """

        :return:
        """

        from sklearn.ensemble import BaggingRegressor

        #predictions_cv = model_selection.cross_val_predict(lm, X, Y, cv = 10)
        baggingRegr = BaggingRegressor()
        model4 = baggingRegr.fit(self.inputDataTrain_X, self.inputDataTrain_Y)
        fig, self.ax = plt.subplots()

        print("Coefficient of determination train: ", baggingRegr.score(self.inputDataTrain_X, self.inputDataTrain_Y))
        predictionsTrain_model4 = baggingRegr.predict(self.inputDataTrain_X)
        print("coefficients : ")
        #print(baggingRegr.get_param())
        print(type(predictionsTrain_model4))

        x_all = self.inputData.values[:,1:self.nCols]
        predictionsTrain_model4_all = baggingRegr.predict(x_all)
        predictOutputCSV_all = os.path.join(self.parentPath, "prediction_output_all.csv")
        np.savetxt(predictOutputCSV_all, predictionsTrain_model4_all, delimiter=",")

        predictOutputCSV = os.path.join(self.parentPath, "prediction_output.csv")
        np.savetxt(predictOutputCSV, predictionsTrain_model4, delimiter=",")

        targetOutputCSV = os.path.join(self.parentPath, "target_output.csv")
        np.savetxt(targetOutputCSV, self.inputDataTrain_Y, delimiter=",")

        self.plot_model(self.inputDataTrain_Y, predictionsTrain_model4, "trainData", color_="blue")

        if self.splitFactor < 1.0:
            print("Coefficient of determination test: ", baggingRegr.score(self.inputDataTest_X, self.inputDataTest_Y))
            predictionsTest_model4 = baggingRegr.predict(self.inputDataTest_X)
            self.plot_model(self.inputDataTest_Y, predictionsTest_model4, "trainData", color_="red")

        lr_model3_path = os.path.join(self.figuresPath, "BaggingRegressor_Train_Test.pdf")

        plt.legend(loc="upper left")
        plt.savefig(lr_model3_path)
        plt.show()
        plt.close()

    def linear_regression_model(self):
        """
            method used for creating regression model with considering the input data
        :return:
        """

        # Linear Regression model
        #self.lr_model1()

        # Cross Validation Model
        #self.lr_model2()

        # Isotonic Regression Model
        #self.lr_model3()

        # decision tree model
        #self.decision_tree_model1(maxDepth=4)

        #self.decision_tree_model1()

        # Bagging Regressor model
        self.bagging_regressor()

    def split_data(self):
        """
            Method use to split data as per requirement in training & testing dataset
        :return:
        """
        #self.inputData_training = self.inputData.take(np.random.permutation(len(self.inputData))[:self.splitFactor])
        # Split-out validation dataset
        # self.inputDataArray = self.inputData.values
        # self.independent = self.inputDataArray[:, 0:(self.nCols-1)]
        self.colNames = self.inputData.columns.tolist()
        self.feature_names = self.colNames[1:self.nCols-1]
        self.target_name = [self.colNames[self.nCols-1]]
        self.independent = pd.DataFrame(self.inputData, columns=self.feature_names)
        print("Independent dataset : \n")
        print(self.independent.head())
        print("dependent dataset : \n")
        #self.dependent = self.inputDataArray[:, (self.nCols-1)]
        self.dependent = pd.DataFrame(self.inputData, columns=self.target_name)
        print(self.dependent.head())
        validation_size = 0.20
        seed = 7
        self.inputDataTrain_X, self.inputDataTest_X, self.inputDataTrain_Y, self.inputDataTest_Y = model_selection.train_test_split(self.independent, self.dependent,
                                                                                                                                    test_size=(1 - self.splitFactor), random_state=seed)
        return self.inputDataTrain_X, self.inputDataTest_X, self.inputDataTrain_Y, self.inputDataTest_Y

    def plot_data(self):
        """

        :return:
        """
        for colnames, col in self.inputData.iteritems():
            print(colnames)#, self.inputData[colnames])
            self.inputData.plot(x=colnames, y="TopLoad", kind="scatter")
        plt.show()

if __name__ == '__main__':

    # if len(sys.argv) == 3:
    #     inputCSV = sys.argv[1]
    #     splitFact = float(sys.argv[2])
    # else:
    #     print("Please give input path of the csv file & splitFactor for training dataset (default is 1.0) :")
    #     exit()

    inputCSV = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Regression_Training_DataSet_Secondary_FEA.csv"
    splitFact = 1.0
    DA = data_analytics(inputCSV, sFactor=splitFact)
    # print("InputData after cleaning : \n",DA.inputData.shape)
    # print("InputData after splitting : \n",DA.inputDataTrain_X.shape)
    # print("InputData after splitting : \n",DA.inputDataTrain_Y.shape)
    # print("InputData after splitting : \n",DA.inputDataTest_X.shape)
    # print("InputData after splitting : \n",DA.inputDataTest_Y.shape)
    # DA.plot_data()
    DA.linear_regression_model()