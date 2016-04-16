import matplotlib.pyplot as plot
import numpy as np
import copy
import time
from termcolor import colored
from membership_function import MembershipFunction
from rule import Rule
import general_functions as gf
from plotly.offline import plot
import plotly.graph_objs as go

class Rbfc:

    def __init__(self, data_set, targets):
        self.data_set = data_set
        self.targets = targets
        self._sort_data_set()
        self._generate_membership_functions()
        self._generate_rules()

    def _sort_data_set(self):
        """
        Sorts dataset's individual columns and inserts them in self._sorted_data_set.
        """
        self._sorted_data_set = copy.copy(self.data_set)
        for i in range(self._sorted_data_set.shape[1]):
            self._sorted_data_set[self._sorted_data_set[:, i].sort()]

    def show_data_set(self):
        """
        Prints each sample in data set along with its target
        """
        for i in range(self.data_set.shape[0]):
            print("Sample: ", self.data_set[i], " Target: ", self.targets[i])

    def classify_samples(self, samples):
        """
        Classifies the samples using the existing rules.
        :param samples: Samples array.
        :return: Numpy array with categories of the classified samples.
        """
        # Used to hold classifications of all samples for return
        classifications = []
        # Used to hold category name as key and number of times category
        # was found during classification of sample (ex. categories{"setosa": 2, "virginia": 2, ...}
        categories = {}

        for sample in samples:
            # Initialize categories values to 0
            for cat in self.target_categories:
                categories[cat] = 0
            # Generate rule for sample
            sample_rule = self._generate_rule(sample)
            # Classify sample
            for rule in self.rules:
                if (sample_rule.rule==rule.rule).all():
                    categories[rule.target] += 1

            # Sort categories dict base on its values
            classifications.append(sorted(categories, key=categories.get, reverse=True)[0])
        # print("classify samples",classifications)
        return np.array(classifications)

    def show_membership_functions_plot(self):
        smoothing_value = 1
        for key, mem_funcs in enumerate(self._membership_functions):
            left_plot_smoothed = go.Scatter(x=list(range(len(mem_funcs.membership_function_left))),
                                            y=gf.smooth_data(mem_funcs.membership_function_left, smoothing_value), mode="lines",
                                            marker=go.Marker(color="#df80ff"), name="Left Membership function")
            right_plot_smoothed = go.Scatter(x=list(range(len(mem_funcs.membership_function_left))),
                                             y=gf.smooth_data(mem_funcs.membership_function_right, smoothing_value), mode="lines",
                                             marker=go.Marker(color="#8600b3"), name="Right Membership function")
            left_plot = go.Scatter(x=list(range(len(mem_funcs.membership_function_left))),
                                   y=mem_funcs.membership_function_left, mode="markers",
                                   marker=go.Marker(color="#df80ff"), opacity=0.3, name="Left Membership function")
            right_plot = go.Scatter(x=list(range(len(mem_funcs.membership_function_left))),
                                    y=mem_funcs.membership_function_right, mode="markers",
                                    marker=go.Marker(color="#8600b3"), opacity=0.3, name="Right Membership function")

            plot([left_plot_smoothed, right_plot_smoothed, left_plot, right_plot], filename="Characteristic {} membership function.html".format(key+1))
            # Add delay between plots showing to avoid crashing of browser
            time.sleep(1.5)

            """ # Create plots using matplotlib
            fig = plot.figure(facecolor="#eeeeee")
            fig.canvas.set_window_title("Membership Function: " + str(i+1))
            for j in range(len(self._membership_functions[i].membership_function_left)):
                plot.plot(x=j, y=self._membership_functions[i].membership_function_left[j], marker=">", color="#ff99ff")
                plot.scatter(x=j, y=self._membership_functions[i].membership_function_right[j], marker="<", color="#660066")

        plot.show()
        """

    def show_data_set_plot(self):

        data_plots = []
        for i in range(self.data_set.shape[1]):
            data_plots.append(go.Scatter(x=list(range(self.data_set.shape[0])), y=self.data_set[:, i], mode="markers", name="Characteristic "+str(i+1)))

        plot(data_plots, filename="Dataset.html")

        # Add delay between plots showing to avoid crashing of browser
        time.sleep(1.5)

    def _generate_membership_functions(self):

        self._membership_functions = []
        for i in range(self._sorted_data_set.shape[1]):  # Loop through columns
            self._membership_functions.append(MembershipFunction(self._sorted_data_set[:, i]))

    def _generate_rules(self):

        self.rules = []
        for key, sample in enumerate(self.data_set):
            self.rules.append(self._generate_rule(sample, self.targets[key]))

        # Remove duplicate rules
        # self.rules = gf.remove_doubles(self.rules)
        # Keep categories names in self.target_categories (ex. ["Iris-setosa", "Iris-virginica", ...])
        self.target_categories = gf.remove_doubles(self.targets)

    def print_rules(self):
        for rule in self.rules:
            print(rule)

    def _generate_rule(self, sample, target=None):
        mem_activation = []
        for key, mem_fun in np.ndenumerate(self._membership_functions):
            mem_activation.append(mem_fun.membership_functions_activation(sample[key[0]]))

        activations_left, activations_right = [], []
        for (activ_l, activ_r) in mem_activation:
            activations_left.append(activ_l > 0)
            activations_right.append(activ_r > 0)

        return Rule(rule=np.concatenate((activations_left, activations_right)), target=target)


if __name__=="__main__":
    """
    iris = data_sets.load_iris()

    # Create membership functions
    sorted_all = (sorted(iris.data[:,1]))
    sepal_width_sorted = (sorted(iris.data[:,1]))
    petal_length_sorted = (sorted(iris.data[:,2]))
    petal_width_sorted = sorted(iris.data[:,3])


    # Show increasing function graph
    for key in range(len(petal_length_sorted)):
        plot.scatter(x=key, y=sorted_all[key], marker=".", color="#ffffff")
        plot.scatter(x=key, y=sorted_all[key], marker=".",color="#ffff00")
        plot.scatter(x=key, y=sorted_all[key], marker="o",color="#00ffff")
        plot.scatter(x=key, y=sorted_all[key], marker=".",color="#ff00ff")

    #plot.show()

    # Generate membership functions
    petal_length_membership = MembershipFunction(petal_length_sorted, overlap_factor=1)

    i=0
    for key, value in np.ndenumerate(iris.data[:,2]):

        (mem_left, mem_right) = petal_length_membership.membership_functions_activation(value)

        if key[0] < 50:
            color1 = "#ffebcc"
            color2 = "#995c00"
        elif key[0] < 100:
            color1 = "#ecb3ff"
            color2 = "#600080"
        else:
            color1 = "#b3f0ff"
            color2 = "#003d4d"
        plot.scatter(x=i, y=mem_left, color=color1)
        plot.scatter(x=i, y=mem_right, marker="h", color=color2)

        i+=1
    print("max: {}, min: {}".format(petal_length_membership.increasing_function_max, petal_length_membership.increasing_function_min))
    #plot.show()
    for key in range(len(petal_length_membership.membership_function_left)):
        plot.scatter(x=key, y=petal_length_membership.membership_function_left[key])
        plot.scatter(x=key, y=petal_length_membership.membership_function_right[key], marker=".", s=2, color="#ff00ff")

    plot.show()
    """
