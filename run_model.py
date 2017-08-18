__author__ = 'Umesh'

from tkinter import *
from tkinter import filedialog
import data_analytics1

class App:
    def __init__(self, master):

        self.score = {}
        self.testScore = {}
        self.master = master
        self.frame = Frame(self.master)
        self.frame.pack()

        self.button1 = Button(self.frame, text="Get training File", command=self.open_file1)
        self.button1.grid(row=0, column=0)
        self.label1 = Label(self.frame)
        self.label1.grid(row=0, column=2)
        # self.entry1 = Entry(self.frame, width=100)
        # self.entry1.grid(row=0, column=2)

        self.button2 = Button(self.frame, text="Get testing File", command=self.open_file2)
        self.button2.grid(row=1, column=0)
        self.label2 = Label(self.frame)
        self.label2.grid(row=1, column=2)
        # self.entry2 = Entry(self.frame, width=100)
        # self.entry2.grid(row=1, column=2)

        self.button2 = Button(self.frame, text="QUIT", command=self.frame.quit)
        self.button2.grid(row=5, column=1)

        # Create a Tkinter variable
        self.tkvar = StringVar(self.frame)

        self.regression_models = {"LinearRegression", "RidgeRegression", "DecisionTree", "DecisionTreeAdaBoosting", "DecisionTreeBagging", "RidgeRegressionAdaBoosting", "RidgeRegressionBagging",
                                  "all"}
        self.all_regression_models = ["LinearRegression", "RidgeRegression", "DecisionTree", "DecisionTreeAdaBoosting", "DecisionTreeBagging", "RidgeRegressionAdaBoosting", "RidgeRegressionBagging"]

        self.tkvar.set("LinearRegression") # set the default option
        self.modelName = "LinearRegression"


        popupMenu = OptionMenu(self.frame, self.tkvar, *self.regression_models, command=self.getValue)
        Label(self.frame, text="Choose a model").grid(row = 2, column = 1)
        popupMenu.grid(row = 2, column =2)
        # link function to change dropdown
        self.tkvar.trace('w', self.change_dropdown)

        self.button3 = Button(self.frame, text="BuildModel", command=self.buildModel)
        self.button3.grid(row=3, column=1)

        self.button4 = Button(self.frame, text="testModel", command=self.testModel)
        self.button4.grid(row=3, column=2)

        self.button5 = Button(self.frame, text="Score", command=self.printScore)
        self.button5.grid(row=4, column=2)

        # textbox=Text(self.frame)
        # textbox.pack()
        # button1=Button(self.frame, text='output', command=lambda : print('printing to GUI'))
        # button1.pack()

    def printScore(self):
        """

        :return:
        """

        # create child window
        win = Toplevel()
        # display message
        #message = "This is the child window"
        #Label(win, text=message).pack()
        # quit child window and return to root window
        # the button is optional here, simply use the corner x of the child window

        #frame2.pack_forget()
        print("In score button")
        self.entry_list = []
        self.entry_list2 = []
        self.label_list = []
        count = 0

        self.label_list.append(Label(win, text="Model_Name", width=50))
        self.label_list[0].grid(row=count, column=0)
        self.label_list.append(Label(win, text="Coefficient_Of_Determination_train", width=50))
        self.label_list[1].grid(row=0, column=1)
        self.label_list.append(Label(win, text="Coefficient_Of_Determination_test", width=50))
        self.label_list[2].grid(row=0, column=2)
        print("In print function :", self.score)
        try:
            for key, value in self.score.items():
                print("count",count)
                self.label_list.append(Label(win, text=key, width=50))
                self.label_list[count+3].grid(row=count+1, column=0)
                self.entry_list.append(Entry(win, width=50))
                self.entry_list[count].grid(row=count+1, column=1)
                self.entry_list[count].insert(0,"%s"%(self.score[key]))
                #if len(self.testScore) > 0:
                self.entry_list2.append(Entry(win, width=50))
                self.entry_list2[count].grid(row=count+1, column=2)
                self.entry_list2[count].insert(0,"%s"%(self.testScore.get(key, "NA")))
                #else:
                    #continue
                count = count + 1
                print(key, "\t", value)
        except Exception as ex:
            print(ex)

        button = Button(win, text='OK', command=win.destroy)
        button.grid(row=count+1, column=1)
        #frame2.destroy()

    def getValue(self, value):
        self.modelName = value
        print("method name :", self.modelName)

    def change_dropdown(self, *args):
        print(self.tkvar.get())

    def open_file1(self):
        """
        :return:
        """
        fName = filedialog.askopenfilename()
        #self.entry1.insert(0, fName)
        self.label1.config(text=fName)

    def open_file2(self):
        """
        :return:
        """
        fName = filedialog.askopenfilename()
        self.label2.config(text=fName)
        #self.entry2.insert(0, fName)

    def buildModel(self):
        """
        """
        #inputCSV = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Input\FEA_All_Features.csv"
        splitFact = 1.0
        self.DA = data_analytics1.data_analytics(self.label1.cget('text'), sFactor=splitFact)
        if self.modelName == "all":
            for i in range(len(self.all_regression_models)):
                self.DA.regressor, trainScore = self.DA.linear_regression_model(self.all_regression_models[i])
                print(self.all_regression_models[i], trainScore)
                self.score.update({self.all_regression_models[i]:trainScore})
                try:
                    testScore = self.DA.testModel(self.label2.cget('text'))
                    self.testScore.update({self.all_regression_models[i]:testScore})
                    print(self.testScore)
                except Exception as ex:
                    print(ex)
        else:
            self.DA.regressor, trainScore = self.DA.linear_regression_model(self.modelName)
            self.score.update({self.modelName:trainScore})
        print(self.score)
        print(self.testScore)

    def testModel(self):
        """
        :return:
        """
        #self.DA.testModel(self.entry2.get())
        testScore = self.DA.testModel(self.label2.cget('text'))
        self.testScore.update({self.modelName:testScore})

    def write_slogan(self):
        #filedialog.askopenfilename()
        print("Tkinter is easy to use!")

    def openFile(self):#, entry):
        #fName = filedialog.askdirectory()
        fName = filedialog.askopenfilename()
        print(fName)
        #entry.insert(0,fName)
        print("Tkinter is easy to use!")

if __name__ == '__main__':

    root = Tk()
    app = App(root)
    root.mainloop()
    # print(app.entry_1.get())
    # for i in range(len(app.entry_list)):
    #     print(app.entry_list[i].get())

