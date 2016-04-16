import Rbfc
from termcolor import colored
import general_functions as gf
from PyQt4 import QtGui
from rbfc_gui import Ui_MainWindow
import sys
import easygui
import plotly
import numpy as np
from PyQt4 import QtCore

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

class main(QtGui.QMainWindow,Ui_MainWindow):
#main class that creates the GUI and its functions

    def __init__(self):
        self.train_ratio = 0.5 #the default ratio is 0,5
        super(self.__class__, self).__init__()
        self.setupUi(self) #create the GUI

        #the code below creates the functions in the GUI
        self.browse_button.clicked.connect(self.browse_button_clicked)
        self.classify_sample_button.clicked.connect(self.classify_sample_button_clicked)
        self.show_membership_functions_button.clicked.connect(self.show_membership_functions_button_clicked)
        self.show_train_set_button.clicked.connect(self.show_train_set_button_clicked)
        self.show_rules_button.clicked.connect(self.show_rules_button_clicked)
        self.train_to_test_ratio_slider.valueChanged.connect(self.train_to_test_ratio_slider_value_changed) #here
        self.train_to_test_ratio_spinbox.valueChanged.connect(self.train_to_test_ratio_spinbox_value_changed) #here
        self.train_button.clicked.connect(self.train_button_clicked)
        self.classify_test_set_button.clicked.connect(self.classify_test_set_button_clicked)

    def browse_button_clicked(self):
        #open a filechooser and pick the dataset used for the training and testing
        path=easygui.fileopenbox()

        if path:
            self.dataset_path_input.setText(path)

    def classify_sample_button_clicked(self):
        #take the parameters from the inputs calculate the category of the sample those belong to
        # try:
            '''
            self.param1=self.first_input.text()
            self.param2=self.second_input.text()
            self.param3=self.third_input.text()
            self.param4=self.fourth_input.text()
            if self.param1 is not '' and self.param2 is not '' and self.param3 is not '' and self.param4 is not '': #check to see if the inputs are empty
                if float(self.param1) > 0 and float(self.param2) > 0 and float(self.param3) > 0 and float(self.param4) > 0:
                    self.ins=[]
                    self.ins.append(self.param1)
                    self.ins.append(self.param2)
                    self.ins.append(self.param3)
                    self.ins.append(self.param4)
                    print(self.ins)
            '''
            self.ins = []
            for characteristic_input in self.characteristics_inputs:
                self.ins.append(characteristic_input.text())

            self.rbfc.classify_samples(np.array([self.ins]).astype(float))
            '''
            else:
                print("One or more of the values is a non-positive number.")
            else:
                print("One or more of the parameters is not given.")
        except ValueError:
            print("Please give numerical values as parameters.")
        '''

    def show_membership_functions_button_clicked(self):
        #show the membership functions created during the training
        self.rbfc.show_membership_functions_plot()

    def show_train_set_button_clicked(self):
        #show the training set
        self.rbfc.show_train_set()

    def show_rules_button_clicked(self):
        #print the rules generated during the training into the console
        self.rbfc.print_rules()

    def train_to_test_ratio_slider_value_changed(self):
        #change the ratio of train/test samples both in variable and in spinner
        self.train_to_test_ratio_spinbox.setValue(self.train_to_test_ratio_slider.value())
        self.train_ratio = 1-self.train_to_test_ratio_slider.value()/100

    def train_to_test_ratio_spinbox_value_changed(self):
        #change the ratio of train/test samples both in variable and in slider
        self.train_to_test_ratio_slider.setValue(self.train_to_test_ratio_spinbox.value())
        self.train_ratio = 1-self.train_to_test_ratio_spinbox.value()/100

    def train_button_clicked(self):
        #begin the training
        try:
            (data_set, targets) = gf.csv_to_data_set(self.dataset_path_input.text())
            self.accs = []
            (self.train_set, self.test_set, self.train_targets, self.test_targets) = gf.split_data_set(data_set, targets, test_size=self.train_ratio)
            print("Training was successful.")

            if 'characteristics_inputs' in self.__dict__:
                for characteristic_input in self.characteristics_inputs:
                    characteristic_input.setParent(None)
                self.characteristics_inputs = []

            # Dynamically add inputs
            self.characteristics_inputs = []
            for i in range(data_set.shape[1]):
                self.characteristics_inputs.append(QtGui.QLineEdit(self.gridLayoutWidget))
                self.characteristics_inputs[i].setStyleSheet(_fromUtf8("background-color: rgb(255, 255, 255);"))
                # self.characteristics_inputs[i].setObjectName(_fromUtf8("first_input"))
                self.horizontalLayout_2.addWidget(self.characteristics_inputs[i])

            self.rbfc = Rbfc(self.train_set, self.train_targets)
            self.show_membership_functions_button.setEnabled(True)
            self.show_rules_button.setEnabled(True)
            self.show_train_set_button.setEnabled(True)
            self.classify_test_set_button.setEnabled(True)
            self.classify_sample_button.setEnabled(True)
            '''
            self.first_input.setEnabled(True)
            self.second_input.setEnabled(True)
            self.third_input.setEnabled(True)
            self.fourth_input.setEnabled(True)
            '''
        except FileNotFoundError:
            print("A file was not specified.")
        except:
           print("Something went wrong.")

    def classify_test_set_button_clicked(self):
        #begin the testing of the samples found in the test set
        self.classifications = self.rbfc.classify_samples(self.test_set)
        self.accs.append(gf.accuracy(self.classifications, self.test_targets))
        for i in range(len(self.classifications)):
            if self.test_targets[i] == self.classifications[i]:
                print(colored("True category: {}, Classification: {}".format(self.test_targets[i], self.classifications[i]), 'green'))
            else:
                print(colored("True category: {}, Classification: {}".format(self.test_targets[i], self.classifications[i]), 'red'))
        print(colored("\nAccuracy\n-----------",'cyan'))
        print(colored(gf.accuracy(self.classifications, self.test_targets),'cyan'))


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    rbfc_window = main()
    rbfc_window.show()
    sys.exit(app.exec_())