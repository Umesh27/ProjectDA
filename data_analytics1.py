__author__ = 'Umesh'

import os, sys
import pandas as pd
import numpy as np
from sklearn import model_selection, linear_model
from sklearn.tree import DecisionTreeRegressor
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
        print(self.inputFile)
        self.parentPath = os.path.split(os.path.split(self.inputFile)[0])[0]
        self.figuresPath = os.path.join(self.parentPath, "Figures")
        if not os.path.exists(self.figuresPath):
            os.mkdir(self.figuresPath)

        self.outputPath = os.path.join(self.parentPath, "Output")
        if not os.path.exists(self.outputPath):
            os.mkdir(self.outputPath)

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
        print(self.inputData.shape)
        #print(self.inputData.head())
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
                Here small variation in the parameters will have huge impact on the prediction values
        :return:
        """
        self.modelName = "sklearn_LR"
        lm = linear_model.LinearRegression()
        model = lm.fit(self.inputDataTrain_X, self.inputDataTrain_Y)
        print(self.inputDataTrain_X.shape, self.inputDataTrain_Y.shape)
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
        #plt.show()
        plt.close()

        return lm

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
        #plt.show()
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
        #plt.show()
        plt.close()
        return ir

    def lr_model4(self):
        """

        :return:
        """
        self.modelName = "sklearn_RidgeRegr"
        lm = linear_model.Ridge(alpha = 0.5)
        model = lm.fit(self.inputDataTrain_X, self.inputDataTrain_Y)
        print(self.inputDataTrain_X.shape, self.inputDataTrain_Y.shape)
        # print("intercept for train model : ", lm.intercept_)
        # print("coefficients for train model : ")
        # for i in range(lm.coef_.shape[0]):
        #     for j in range(lm.coef_.shape[1]):
        #         print(float(lm.coef_[i][j]))

        fig, self.ax = plt.subplots()

        predictionsTrain_model1 = lm.predict(self.inputDataTrain_X)
        print(predictionsTrain_model1.shape, self.inputDataTrain_Y.shape)

        trainScore = lm.score(self.inputDataTrain_X, self.inputDataTrain_Y)
        print("Coefficient of determination train: ", lm.score(self.inputDataTrain_X, self.inputDataTrain_Y))
        self.plot_model(self.inputDataTrain_Y, predictionsTrain_model1, "trainData", color_="blue")

        Y_prediction2 = lm.predict(self.independent)
        predictOutputCSV_allData = os.path.join(self.outputPath, "trainData_%s.csv"%self.modelName)
        np.savetxt(predictOutputCSV_allData, Y_prediction2, delimiter=",")

        if self.splitFactor < 1.0:
            print("Coefficient of determination test: ", lm.score(self.inputDataTest_X, self.inputDataTest_Y))
            predictionsTest_model1 = lm.predict(self.inputDataTest_X)
            print(predictionsTest_model1.shape, self.inputDataTest_Y.shape)
            self.plot_model(self.inputDataTest_Y, predictionsTest_model1, "testData", color_="red")

        lr_model1_path = os.path.join(self.figuresPath, "score_Vs_actual_%s.pdf"%self.modelName)
        plt.legend(loc="upper left")
        plt.savefig(lr_model1_path)
        #plt.show()
        plt.close()

        return lm, trainScore

    def decision_tree_model1(self, maxDepth=None):
        """

        :return:
        """

        from sklearn.tree import DecisionTreeRegressor
        self.modelName = "DTRegressor"
        #predictions_cv = model_selection.cross_val_predict(lm, X, Y, cv = 10)
        decisionTreeRegr = DecisionTreeRegressor(max_depth=maxDepth)
        model4 = decisionTreeRegr.fit(self.inputDataTrain_X, self.inputDataTrain_Y)
        fig, self.ax = plt.subplots()
        trainScore = decisionTreeRegr.score(self.inputDataTrain_X, self.inputDataTrain_Y)
        print("Coefficient of determination train: ", decisionTreeRegr.score(self.inputDataTrain_X, self.inputDataTrain_Y))
        predictionsTrain_model4 = decisionTreeRegr.predict(self.inputDataTrain_X)
        self.plot_model(self.inputDataTrain_Y, predictionsTrain_model4, "trainData", color_="blue")

        Y_prediction2 = decisionTreeRegr.predict(self.independent)
        predictOutputCSV_allData = os.path.join(self.outputPath, "trainData_%s.csv"%self.modelName)
        np.savetxt(predictOutputCSV_allData, Y_prediction2, delimiter=",")

        if self.splitFactor < 1.0:
            print("Coefficient of determination test: ", decisionTreeRegr.score(self.inputDataTest_X, self.inputDataTest_Y))
            predictionsTest_model4 = decisionTreeRegr.predict(self.inputDataTest_X)
            self.plot_model(self.inputDataTest_Y, predictionsTest_model4, "trainData", color_="red")

        lr_model3_path = os.path.join(self.figuresPath, "DecisionTree_Train_Test_%s.pdf"%str(maxDepth))
        if os.path.exists(lr_model3_path):
            lr_model3_path = os.path.join(self.figuresPath, "DecisionTree_Train_Test_depth_%s.pdf"%str(maxDepth))

        plt.legend(loc="upper left")
        plt.savefig(lr_model3_path)
        #plt.show()
        plt.close()

        return decisionTreeRegr, trainScore

    def adaBoosting_DTregressor(self, maxDepth=None):
        """

        :return:
        """

        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import AdaBoostRegressor

        self.modelName = "adaBoostDT"
        #predictions_cv = model_selection.cross_val_predict(lm, X, Y, cv = 10)
        decisionTreeRegrAdaBoost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=maxDepth), n_estimators=300)
        model4 = decisionTreeRegrAdaBoost.fit(self.inputDataTrain_X, self.inputDataTrain_Y)
        fig, self.ax = plt.subplots()
        trainScore = decisionTreeRegrAdaBoost.score(self.inputDataTrain_X, self.inputDataTrain_Y)
        print("Coefficient of determination train: ", decisionTreeRegrAdaBoost.score(self.inputDataTrain_X, self.inputDataTrain_Y))
        predictionsTrain_model4 = decisionTreeRegrAdaBoost.predict(self.inputDataTrain_X)
        self.plot_model(self.inputDataTrain_Y, predictionsTrain_model4, "trainData", color_="blue")

        Y_prediction2 = decisionTreeRegrAdaBoost.predict(self.independent)
        predictOutputCSV_allData = os.path.join(self.outputPath, "trainData_%s_300_%s.csv"%(self.modelName, maxDepth))
        np.savetxt(predictOutputCSV_allData, Y_prediction2, delimiter=",")

        if self.splitFactor < 1.0:
            print("Coefficient of determination test: ", decisionTreeRegrAdaBoost.score(self.inputDataTest_X, self.inputDataTest_Y))
            predictionsTest_model4 = decisionTreeRegrAdaBoost.predict(self.inputDataTest_X)
            self.plot_model(self.inputDataTest_Y, predictionsTest_model4, "trainData", color_="red")

        lr_model3_path = os.path.join(self.figuresPath, "%s_Train_Test_%s_300.pdf"%(self.modelName, str(maxDepth)))
        if os.path.exists(lr_model3_path):
            lr_model3_path = os.path.join(self.figuresPath, "%s_Train_Test_depth_%s.pdf"%(self.modelName, str(maxDepth)))

        plt.legend(loc="upper left")
        plt.savefig(lr_model3_path)
        #plt.show()
        plt.close()
        return decisionTreeRegrAdaBoost, trainScore

    def adaBoosting_RidgeRegressor(self):
        """

        :return:
        """

        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import AdaBoostRegressor

        self.modelName = "adaBoostRidgeRegr"
        #predictions_cv = model_selection.cross_val_predict(lm, X, Y, cv = 10)
        ridgeRegrAdaBoost = AdaBoostRegressor(linear_model.Ridge(alpha=0.5), n_estimators=300)
        model4 = ridgeRegrAdaBoost.fit(self.inputDataTrain_X, self.inputDataTrain_Y)
        fig, self.ax = plt.subplots()
        trainScore = ridgeRegrAdaBoost.score(self.inputDataTrain_X, self.inputDataTrain_Y)
        print("Coefficient of determination train: ", ridgeRegrAdaBoost.score(self.inputDataTrain_X, self.inputDataTrain_Y))
        predictionsTrain_model4 = ridgeRegrAdaBoost.predict(self.inputDataTrain_X)
        self.plot_model(self.inputDataTrain_Y, predictionsTrain_model4, "trainData", color_="blue")

        Y_prediction2 = ridgeRegrAdaBoost.predict(self.independent)
        predictOutputCSV_allData = os.path.join(self.outputPath, "trainData_%s_300.csv"%(self.modelName))
        np.savetxt(predictOutputCSV_allData, Y_prediction2, delimiter=",")

        if self.splitFactor < 1.0:
            print("Coefficient of determination test: ", ridgeRegrAdaBoost.score(self.inputDataTest_X, self.inputDataTest_Y))
            predictionsTest_model4 = ridgeRegrAdaBoost.predict(self.inputDataTest_X)
            self.plot_model(self.inputDataTest_Y, predictionsTest_model4, "trainData", color_="red")

        lr_model3_path = os.path.join(self.figuresPath, "%s_Train_Test_300.pdf"%(self.modelName))
        if os.path.exists(lr_model3_path):
            lr_model3_path = os.path.join(self.figuresPath, "%s_Train_Test_depth.pdf"%(self.modelName))

        plt.legend(loc="upper left")
        plt.savefig(lr_model3_path)
        #plt.show()
        plt.close()
        return ridgeRegrAdaBoost, trainScore

    def bagging_regressor(self):
        """

        :return:
        """

        from sklearn.ensemble import BaggingRegressor

        self.modelName = "baggingRegr"
        #predictions_cv = model_selection.cross_val_predict(lm, X, Y, cv = 10)
        baggingRegr = BaggingRegressor()
        model4 = baggingRegr.fit(self.inputDataTrain_X, self.inputDataTrain_Y)
        fig, self.ax = plt.subplots()
        trainScore = baggingRegr.score(self.inputDataTrain_X, self.inputDataTrain_Y)
        print("Coefficient of determination train: ", baggingRegr.score(self.inputDataTrain_X, self.inputDataTrain_Y))
        predictionsTrain_model4 = baggingRegr.predict(self.inputDataTrain_X)
        print("coefficients : ")
        #print(baggingRegr.get_param())
        print(type(predictionsTrain_model4))
        self.plot_model(self.inputDataTrain_Y, predictionsTrain_model4, "trainData", color_="blue")

        if self.splitFactor < 1.0:
            print("Coefficient of determination test: ", baggingRegr.score(self.inputDataTest_X, self.inputDataTest_Y))
            predictionsTest_model4 = baggingRegr.predict(self.inputDataTest_X)
            self.plot_model(self.inputDataTest_Y, predictionsTest_model4, "trainData", color_="red")

        lr_model3_path = os.path.join(self.figuresPath, "FEA_All_Features_%s.pdf"%self.modelName)
        plt.legend(loc="upper left")
        plt.savefig(lr_model3_path)
        #plt.show()
        plt.close()

        predictionsTrain_model = baggingRegr.predict(self.independent)
        predictOutputCSV_allData = os.path.join(self.outputPath, "trainDataPrediction_%s.csv"%self.modelName)
        np.savetxt(predictOutputCSV_allData, predictionsTrain_model, delimiter=",")

        return baggingRegr, trainScore

    def bagging_DTRegressor(self):
        """

        :return:
        """

        from sklearn.ensemble import BaggingRegressor

        self.modelName = "baggingDTRegr"

        #predictions_cv = model_selection.cross_val_predict(lm, X, Y, cv = 10)
        baggingDTRegr = BaggingRegressor(base_estimator=DecisionTreeRegressor())#, n_estimators=50, bootstrap=True, oob_score=True)
        model4 = baggingDTRegr.fit(self.inputDataTrain_X, self.inputDataTrain_Y)
        fig, self.ax = plt.subplots()
        trainScore = baggingDTRegr.score(self.inputDataTrain_X, self.inputDataTrain_Y)
        print("Coefficient of determination train: ", baggingDTRegr.score(self.inputDataTrain_X, self.inputDataTrain_Y))
        predictionsTrain_model4 = baggingDTRegr.predict(self.inputDataTrain_X)
        print("coefficients : ")
        #print(baggingRegr.get_param())
        print(type(predictionsTrain_model4))

        x_all = self.inputData.values[:,1:self.nCols]
        self.plot_model(self.inputDataTrain_Y, predictionsTrain_model4, "trainData", color_="blue")

        if self.splitFactor < 1.0:
            print("Coefficient of determination test: ", baggingDTRegr.score(self.inputDataTest_X, self.inputDataTest_Y))
            predictionsTest_model4 = baggingDTRegr.predict(self.inputDataTest_X)
            self.plot_model(self.inputDataTest_Y, predictionsTest_model4, "trainData", color_="red")

        lr_model3_path = os.path.join(self.figuresPath, "FEA_All_Features_%s.pdf"%self.modelName)
        predictionsTrain_model = baggingDTRegr.predict(self.independent)
        predictOutputCSV_allData = os.path.join(self.outputPath, "trainDataPrediction_%s.csv"%self.modelName)
        np.savetxt(predictOutputCSV_allData, predictionsTrain_model, delimiter=",")
        plt.legend(loc="upper left")
        plt.savefig(lr_model3_path)
        #plt.show()
        plt.close()
        return baggingDTRegr, trainScore

    def bagging_RidgeRegressor(self):
        """

        :return:
        """

        from sklearn.ensemble import BaggingRegressor

        self.modelName = "baggingRidgeRegr3000"
        #predictions_cv = model_selection.cross_val_predict(lm, X, Y, cv = 10)
        baggingRidgeRegr = BaggingRegressor(base_estimator=linear_model.Ridge(alpha = 0.5), n_estimators=3000)#, n_estimators=50, bootstrap=True, oob_score=True)
        model4 = baggingRidgeRegr.fit(self.inputDataTrain_X, self.inputDataTrain_Y)
        fig, self.ax = plt.subplots()
        trainScore = baggingRidgeRegr.score(self.inputDataTrain_X, self.inputDataTrain_Y)
        print("Coefficient of determination train: ", baggingRidgeRegr.score(self.inputDataTrain_X, self.inputDataTrain_Y))
        predictionsTrain_model4 = baggingRidgeRegr.predict(self.inputDataTrain_X)
        #print("coefficients : ")
        #print(baggingRegr.get_param())
        #print(type(predictionsTrain_model4))

        x_all = self.inputData.values[:,1:self.nCols]
        self.plot_model(self.inputDataTrain_Y, predictionsTrain_model4, "trainData", color_="blue")

        if self.splitFactor < 1.0:
            print("Coefficient of determination test: ", baggingRidgeRegr.score(self.inputDataTest_X, self.inputDataTest_Y))
            predictionsTest_model4 = baggingRidgeRegr.predict(self.inputDataTest_X)
            self.plot_model(self.inputDataTest_Y, predictionsTest_model4, "trainData", color_="red")

        lr_model3_path = os.path.join(self.figuresPath, "Scores_Vs_targets_%s.pdf"%self.modelName)
        predictionsTrain_model = baggingRidgeRegr.predict(self.independent)
        predictOutputCSV_allData = os.path.join(self.outputPath, "trainDataPrediction_%s.csv"%self.modelName)
        np.savetxt(predictOutputCSV_allData, predictionsTrain_model, delimiter=",")

        plt.legend(loc="upper left")
        plt.savefig(lr_model3_path)
        #plt.show()
        plt.close()
        return baggingRidgeRegr, trainScore

    def svm_model(self):
        """

        :return:
        """
        from sklearn import svm

        self.modelName = "svc_linear"
        svc = svm.SVC(C=1, kernel='linear')
        print(self.inputDataTrain_X.head())
        print(self.inputDataTrain_Y.head())
        model = svc.fit(self.inputDataTrain_X, self.inputDataTrain_Y)
        print(model.score(self.inputDataTrain_X, self.inputDataTrain_Y))
        predicted_TrainY = model.predict(self.inputDataTrain_X)
        plt.scatter(predicted_TrainY, self.inputDataTrain_Y)
        plt.xlabel("predictedY")
        plt.ylabel("targetY")
        #plt.show()

        return predicted_TrainY


    def statsModel_lr(self):
        """
            Using linear regression from statsmodel library
        :return:
        """
        from statsmodels.formula.api import ols

        self.modelName = "stats_ols"

        self.inputDataTrain = pd.concat([self.inputDataTrain_X, self.inputDataTrain_Y], axis=1)
        self.inputDataTest = pd.concat([self.inputDataTest_X, self.inputDataTest_Y], axis=1)

        lm_new = ols(formula="TopLoad ~ Volume + Prim_X_Copy + Prim_Y_Copy + Prim_Length + Prim_Breadth + Prim_Height "
                             "+ Sec_Length + Sec_Breadth + Sec_Height + Prim_Flap + Prim_Cal + Prim_E1 + Prim_E2 + Headspace + Sec_Flap + Sec_Caliper + Sec_E1 + Sec_E2", data = self.inputDataTrain).fit()
        print(lm_new.summary())
        predicted_val = lm_new.predict(self.independent)
        predictOutputCSV_allData = os.path.join(self.outputPath, "prediction_trainAll.csv")
        np.savetxt(predictOutputCSV_allData, predicted_val, delimiter=",")

        # OutputCSV_allData = os.path.join(self.outputPath, "target_trainAll.csv")
        # np.savetxt(OutputCSV_allData, self.inputDataTrain_Y, delimiter=",")
        plt.scatter(predicted_val, self.dependent)
        plt.figure()

        trainPredicted = lm_new.predict(self.inputDataTrain_X)
        plt.scatter(trainPredicted, self.inputDataTrain_Y)

        testPredicted = lm_new.predict(self.inputDataTest_X)
        plt.scatter(testPredicted, self.inputDataTest_Y)

        #plt.show()
        return lm_new

    def statsmodels_sm(self):
        """

        :return:
        """
        import statsmodels.api as sm

        self.modelName = "sm_OLS"

        #X = sm.add_constant(self.inputDataTrain_X)
        X = self.inputDataTrain_X
        print(X.shape)#, X.columns)
        OLS_Model = sm.OLS(self.inputDataTrain_Y, X).fit()

        print(OLS_Model.summary())

        Y_prediction = OLS_Model.predict(X)
        Y_prediction2 = OLS_Model.predict(self.independent)
        float_formatter = lambda x: "%.2f" % x
        np.set_printoptions(formatter={'float_kind':float_formatter})
        #print(self.independent.values[0], Y_prediction2[0])
        predictOutputCSV_allData = os.path.join(self.outputPath, "trainData_%s.csv"%self.modelName)
        print("Output path : ",predictOutputCSV_allData)
        np.savetxt(predictOutputCSV_allData, Y_prediction2, delimiter=",")


        plt.scatter(Y_prediction, self.inputDataTrain_Y)
        #plt.show()

        return OLS_Model, OLS_Model.rsquared

    def linear_regression_model(self, model = "LinearRegression"):
        """
            method used for creating regression model with considering the input data
        :return:
        """

        # self.regression_models = {"LinearRegression", "RidgeRegression", "DecisionTree", "DecisionTreeAdaBoosting",
        #                           "DecisionTreeBagging", "RidgeRegressionAdaBoosting", "RidgeRegressionBagging"}

        self.allModels = [self.statsmodels_sm, self.lr_model4, self.decision_tree_model1, self.adaBoosting_DTregressor, self.adaBoosting_RidgeRegressor, self.bagging_RidgeRegressor,
                          self.bagging_DTRegressor]

        self.regression_models = {"LinearRegression":self.statsmodels_sm, "RidgeRegression":self.lr_model4, "DecisionTree":self.decision_tree_model1,
                                  "DecisionTreeAdaBoosting":self.adaBoosting_DTregressor, "DecisionTreeBagging":self.bagging_DTRegressor,
                                  "RidgeRegressionBagging":self.bagging_RidgeRegressor, "RidgeRegressionAdaBoosting":self.adaBoosting_RidgeRegressor, "all":self.allModels}

        try:
            if model in self.regression_models:
                print(model)
                if type(model) is list:
                    for i in range(len(model)):
                        self.regressorModel, self.score = self.regression_models[model][i]()
                else:
                    self.regressorModel, self.score = self.regression_models[model]()
                return self.regressorModel, self.score

        except Exception as ex:
            print(ex)

        """
        # 1 Statsmodels linear regression
        #self.regressorModel = self.statsmodels_sm()

        # 2 Ridge Regression model
        #self.regressorModel = self.lr_model4()

        # 3 decision tree model
        self.regressorModel = self.decision_tree_model1(maxDepth=16)

        # 4 Bagging Regressor model with Ridge Regressor
        #self.regressorModel = self.bagging_RidgeRegressor()

        # 5 Bagging Regressor model with Decision Tree
        #self.regressorModel = self.bagging_DTRegressor()

        # 6 Bagging Regressor model
        #self.regressorModel = self.bagging_regressor()

        # 7 decision tree regressor with adaptive boosting algorithm
        #self.regressorModel = self.adaBoosting_DTregressor(16)

        # Linear Regression model
        #self.regressorModel = self.lr_model1()

        # Cross Validation Model
        #self.lr_model2()

        # Isotonic Regression Model
        #self.lr_model3()

        #self.decision_tree_model1()


        #self.regressorModel = self.decision_tree_adaBoosting_model2()

        # Statsmodels linear regression
        #self.regressorModel = self.statsModel_lr()

        # sklearn SVM

        #self.regressorModel = self.svm_model()
        """

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
        self.feature_names = self.colNames[0:self.nCols-1]
        self.target_name = [self.colNames[self.nCols-1]]
        self.independent = pd.DataFrame(self.inputData, columns=self.feature_names)
        #print("Independent dataset : \n")
        #print(self.independent.head())
        #print("dependent dataset : \n")
        #self.dependent = self.inputDataArray[:, (self.nCols-1)]
        self.dependent = pd.DataFrame(self.inputData, columns=self.target_name)
        #print(self.dependent.head())
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
        ##plt.show()

    def testModel(self, testCSV):
        """

        :return:
        """
        allData = pd.read_csv(testCSV)
        nCols = allData.shape[1]
        colNames = allData.columns.tolist()
        feature_names = colNames[0:nCols-1]
        target_name = [colNames[nCols-1]]
        testDataX = pd.DataFrame(allData, columns=feature_names)
        #print(testDataX)
        print(testDataX.shape)#, testDataX.columns)
        testDataY = pd.DataFrame(allData, columns=target_name)
        print(testDataY.shape)
        predictTestDataY = self.regressorModel.predict(testDataX)
        #print(type(predictTestDataY))
        float_formatter = lambda x: "%.2f" % x
        np.set_printoptions(formatter={'float_kind':float_formatter})
        #print(testDataY, predictTestDataY)
        try:
            print("Coefficient of determination train: ", self.regressorModel.score(testDataX, testDataY))
            self.testScore = self.regressorModel.score(testDataX, testDataY)
        except Exception as ex:
            self.testScore = "NA"
            print(ex)
        plt.scatter(testDataY, predictTestDataY)
        ##plt.show()
        #predictOutputCSV_allData = os.path.join(DA.parentPath, "Output", "prediction_output_allData_adaBoost.csv")

        predictOutputCSV_allData = os.path.join(self.outputPath, "physicalDataPrediction_%s.csv"%self.modelName)

        if os.path.exists(predictOutputCSV_allData):
            tempPathList = predictOutputCSV_allData.split(".")
            print(tempPathList)
            predictOutputCSV_allData = "".join([tempPathList[0], "_n.", tempPathList[-1]])
        np.savetxt(predictOutputCSV_allData, predictTestDataY, delimiter=",")

        return self.testScore

if __name__ == '__main__':

    # if len(sys.argv) == 3:
    #     inputCSV = sys.argv[1]
    #     splitFact = float(sys.argv[2])
    # else:
    #     print("Please give input path of the csv file & splitFactor for training dataset (default is 1.0) :")
    #     exit()

    inputCSV = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Input\Regression_Training_DataSet_Secondary_FEA.csv"
    inputCSV = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Input\Secondary_FEA_52Cases_test1.csv"

    """
     README

     Dataset should consist of first column as sequence & last column as Topload( Target Variable) as this two columns will be removed from the dataset for training the model
        the remaining columns will be considered as features.

        Ex.
            dataset = [('Sequence', 'Volume','Prim_X_Copy','Prim_Y_Copy','Prim_Length','Prim_Breadth','Prim_Height','Sec_Length','Sec_Breadth','Sec_Height','Prim_Flap',
                        'Prim_Cal','Prim_E1','Prim_E2','Headspace','Sec_Flap','Sec_Caliper','Sec_E1','Sec_E2', 'TopLoad'),
                        (1,117.96875,2,3,6.25,2,9.4375,12.5,6,9.4375,1.6,18,5029.933,1050.000,0.75,3,3.6,465,316,479.5)]

            In above dataset:
                (First column is Sequence & last one is "TopLoad" will be removed & remaining will be features as follows (only feature names))
                (Last column will be considered as Target Variable)

            features_names = ['Volume','Prim_X_Copy','Prim_Y_Copy','Prim_Length','Prim_Breadth','Prim_Height','Sec_Length','Sec_Breadth','Sec_Height','Prim_Flap',
                        'Prim_Cal','Prim_E1','Prim_E2','Headspace','Sec_Flap','Sec_Caliper','Sec_E1','Sec_E2']
            Target Variable = ["TopLoad"]
     """
    inputCSV = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Input\test1.csv"
    inputCSV = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Input\FEA_2x5_Features.csv"
    inputCSV = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Input\FEA_All_Features.csv"
    inputCSV = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Input\FEA_All_Features_new.csv"
    inputCSV = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Input\cement_strength.csv"
    inputCSV = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Input\PeakLoad_FEA_RegressionInput_TopBottom.csv"
    splitFact = 0.75#1.0 #0.20#
    DA = data_analytics(inputCSV, sFactor=splitFact)
    # self.regression_models = {"LinearRegression", "RidgeRegression", "DecisionTree", "DecisionTreeAdaBoosting",
    #                           "DecisionTreeBagging", "RidgeRegressionAdaBoosting", "RidgeRegressionBagging"}
    DA.linear_regression_model("LinearRegression")#"DecisionTreeAdaBoosting")#"RidgeRegressionBagging")#"RidgeRegressionAdaBoosting")#
    #"""
    testCSVPath = os.path.join(DA.parentPath, "Input", "test1_physical.csv")
    testCSVPath = os.path.join(DA.parentPath, "Input", "physical_test_input.csv")
    testCSVPath = os.path.join(DA.parentPath, "Input", "physical_test_input_new.csv")
    testCSVPath = os.path.join(DA.parentPath, "Input", "cement_strength_test.csv")
    testCSVPath = os.path.join(DA.parentPath, "Input", "physical_test_input_2.csv")
    DA.testModel(testCSVPath)
    #"""