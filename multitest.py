from Rbfc import Rbfc
from termcolor import colored
import general_functions as gf
import plotly
import numpy as np


times_to_test = 100

if __name__ == '__main__':
    (data_set, targets) = gf.csv_to_data_set("iris.data")
    accs = []
    for y in range(times_to_test):
        
        print("-"*20, "Test No: ", y+1, "-"*20)
        # Train - test set split
        (train_set, test_set, train_targets, test_targets) = gf.split_data_set(data_set, targets, test_size=.6)
        # Train model
        rbfc = Rbfc(train_set, train_targets)
        print(colored("Training was successfull.", "yellow"))
        rbfc.show_data_set()
        # Classify test set
        predictions = rbfc.classify_samples(test_set)
        # Print confusion matrix
        labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        gf.show_confusion_matrix(test_targets, predictions, labels)
        """
        for i in range(len(classifications)):
            if test_targets[i] == classifications[i]:
                print(colored("True category: {}, Classification: {}".format(test_targets[i], classifications[i]), 'green'))
            else:
                print(colored("True category: {}, Classification: {}".format(test_targets[i], classifications[i]), 'red'))
        """
        print(colored("-"*10 + " Accuracy " + "-"*10,'cyan'))
        # Append accuracy to accs array
        accs.append(gf.accuracy(predictions, test_targets))
        print(colored(accs[y],'cyan'))
        
print(colored("Average accuracy: " + str(sum(accs)/times_to_test), "green"))
# rbfc.show_data_set_plot()