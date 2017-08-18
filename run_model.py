__author__ = 'Umesh'

from tkinter import *
from tkinter import filedialog
import data_analytics1

class App:
    def __init__(self, master):
        frame = Frame(master)
        frame.pack()
        #"""
        #self.button_list = []
        self.entry_list = []
        self.label_list = []
        self.label_list1 = ['Volume','Prim_X_Copy','Prim_Y_Copy','Prim_Length','Prim_Breadth','Prim_Height','Sec_Length',
                            'Sec_Breadth','Sec_Height','Prim_Flap','Prim_Cal','Prim_E1','Prim_E2','Headspace','Sec_Flap','Sec_Caliper','Sec_E1','Sec_E2']

        self.label_list2 = [117.96875,2,3,6.25,2,9.4375,12.5,6,9.4375,1.6,18,5029.933,1050.000,0.75,3,3.6,465,316]

        for i in range(len(self.label_list1)):
            #print()
            self.entry_list.append(Entry(frame, width=50))
            self.entry_list[i].grid(row=i+1, column=2)
            self.entry_list[i].insert(0,self.label_list2[i])
            self.label_list.append(Label(frame, text=self.label_list1[i], width=50))
            self.label_list[i].grid(row=i+1, column=0)
            #self.button_list.append(Button(frame, text="file%s"%(i+1), command=self.openFile))
            #self.button_list[i].grid(row=i, column=0)

        """
        self.entry_1 = Entry(frame, text="Hello", width=50)
        self.entry_1.grid(row=0, column=0)
        self.trainData = Button(frame, text="trainCSV", command=self.openFile)#(self.entry_1))
        #self.trainData.pack()
        #self.trainData.place(height=50, width=50)
        self.trainData.grid(row=1, column=0)


        self.entry_2 = Entry(frame, text="Hello", width=50)
        self.entry_2.grid(row=0, column=1)
        self.testData = Button(frame, text="testCSV", command=self.openFile)#(self.entry_2))
        #self.testData.pack()
        #self.testData.place(height=150, width=50)
        self.testData.grid(row=1, column=1)
        """


        self.button = Button(frame, text="QUIT", fg="red", command=frame.quit)
        self.button.grid(row=len(self.label_list1)+3, column=1)

        self.button2 = Button(frame, text="BuildModel", fg="blue", command=self.runModel)
        self.button2.grid(row=len(self.label_list1)+1, column=0)

        self.button2 = Button(frame, text="PeakLoad", fg="blue", command=self.get_peakload)
        self.button2.grid(row=len(self.label_list1)+2, column=0)

        self.entry_2 = Entry(frame, text="", width=50)
        self.entry_2.grid(row=len(self.label_list1)+2, column=2)
        #self.button.pack(side=LEFT)
        #self.slogan.pack(side=LEFT)

    def runModel(self):
        """
        """
        inputCSV = r"D:\Umesh\LSPP\1Aug\DataAnalysis\ProjectDA\Input\FEA_All_Features.csv"
        splitFact = 1.0
        self.DA = data_analytics1.data_analytics(inputCSV, sFactor=splitFact)
        self.DA.linear_regression_model()

    def get_peakload(self):
        """
        """
        self.entry_2.delete(0,END)
        import pandas as pd
        self.data = []
        for i in range(len(self.entry_list)):
            print(self.label_list1[i], self.entry_list[i].get())
            self.data.append(self.entry_list[i].get())

        input_data = pd.DataFrame.from_records([self.data], columns=self.label_list1)
        print(input_data)
        predictTestDataY = self.DA.regressorModel.predict(input_data)
        print(predictTestDataY)
        self.entry_2.insert(0, predictTestDataY[0])

    def write_slogan(self):
        #filedialog.askopenfilename()
        print("Tkinter is easy to use!")


    """
    def openFile(self):
        fName = filedialog.askopenfilename()
        self.entry_list.insert(0,fName)
        print("Tkinter is easy to use!")
        """

    def openFile(self):#, entry):
        #fName = filedialog.askdirectory()
        fName = filedialog.askopenfilename()
        #entry.insert(0,fName)
        print("Tkinter is easy to use!")

if __name__ == '__main__':

    root = Tk()
    app = App(root)
    root.mainloop()
    # print(app.entry_1.get())
    # for i in range(len(app.entry_list)):
    #     print(app.entry_list[i].get())

