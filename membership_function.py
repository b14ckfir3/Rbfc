import numpy as np
import general_functions as gf
import math


class MembershipFunction:

    def __init__(self, increasing_function, horizontal_cut=0.5, overlap_factor=0.5):
        """
        Generate membership function using the increasing_function and store increasing_function's max and min values
        :param increasing_function: Function used to generate the membership function
        :param horizontal_cut: Where to cut the functions horizontally
        :param overlap_factor: How much should the membership functions overlapping. 1 is for full overlap, 0 is for no overlap
        :return:
        """
        self.increasing_function = increasing_function
        self.increasing_function_min = np.min(increasing_function)
        self.increasing_function_max = np.max(increasing_function)
        (self.membership_function_left,
         self.membership_function_right) = self._increasing_function_to_membership_functions(increasing_function,
                                                                                             horizontal_cut,
                                                                                             overlap_factor)

    def _increasing_function_to_membership_functions(self, increasing_function, horizontal_cut=0.5, overlap_factor=0.5):
        """
        Converts one increasing function (array with increasing values) and converts it to two (left-right) membership functions
        :param increasing_function: Array with increasing values representing the increasing function
        :param horizontal_cut: Where to cut the functions horizontally
        :param overlap_factor: How much should the membership functions overlapping. 1 is for full overlap, 0 is for no overlap
        :returns: increasing_function if horizontal_cut <= 0.1
        """

        if horizontal_cut <= 0.1:
            return increasing_function

        # Bring function down to horizontal_cut
        increasing_function = increasing_function - (np.max(increasing_function) - np.min(increasing_function)) \
                                                    * gf.clamp(horizontal_cut) - np.min(increasing_function)

        membership_function_left = increasing_function[increasing_function < 0] * -1
        membership_function_right = increasing_function[increasing_function >= 0]

        # Bring functions down to 0
        membership_function_left -= np.min(membership_function_left)
        membership_function_right -= np.min(membership_function_right)

        # Convert to 0 - 1 vertical space
        membership_function_left /= np.max(membership_function_left)
        membership_function_right /= np.max(membership_function_right)

        # Calculate how many indices should the membership_function_right be moved
        displacement = math.floor(len(membership_function_left) * (1 - gf.clamp(overlap_factor)))

        membership_function_right = np.concatenate(([0] * displacement, membership_function_right))
        membership_function_left = np.concatenate((membership_function_left,
                                                   [0] * (
                                                       len(membership_function_right) - len(membership_function_left))))

        return (membership_function_left, membership_function_right)

    def membership_functions_activation(self, check_value):
        """
        Returns a tuple indicating how much left and right membership functions are activated by the check_value
        :param check_value: Value used to check which membership functions are activated and how much
        :return: Tuple with 2 float numbers between 0 & 1. (ex. (0.2, 0.7) = left membership function is activated by .2 and right is activated by .7)
        """
        return (self._membership_function_activation(self.membership_function_left, check_value),
                self._membership_function_activation(self.membership_function_right, check_value))

    def _membership_function_activation(self, membership_function, check_value):
        """
        Checks how much a membership function is being activated by the check value
        :param membership_functions_tuple: Membership function
        :param check_value: Value used to check how much is the membership functions being activated
        :returns: Float number between 0 & 1
        """

        # Convert from self.increasing_function_min - self.increasings_function_max to 0 - len(membership_function)
        check_value_transformed = gf.lerp(0, len(self.membership_function_left),
                                          alpha=(check_value-self.increasing_function_min)/(self.increasing_function_max-self.increasing_function_min))
        """
        ((check_value - self.increasing_function_min) * (len(membership_function))
                                   / (self.increasing_function_max - self.increasing_function_min)) / check_value
        """
        # Find the two closest indices of membership_functions for check_value_transformed
        (left_index, right_index) = self._between_indices(check_value_transformed)
        
        # Return activation of memeberhip functions using linear interpolation
        return gf.lerp(membership_function[left_index], membership_function[right_index],
                       alpha=(check_value_transformed - left_index) / (right_index - left_index))

    def _between_indices(self, check_value):
        prev_index = math.floor(check_value)
        next_index = math.floor(check_value) + 1
        if 0>=next_index:
            return 0,1
        if next_index < len(self.membership_function_right)-1:
            return prev_index, next_index
        else:
            return len(self.membership_function_right)-1, len(self.membership_function_right)-1
