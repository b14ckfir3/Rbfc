class Rule:

    def __init__(self, rule, target=None):
        """
        :param rule: Tuple with 0s & 1s, representing the rule. (ex. (0,0,1,0,1,1,0,1))
        :param target: Target of rule. (ex. "Iris-Setosa")
        :return:
        """
        self.rule = rule
        self.target = target

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        val = bytearray(self.target.encode() if self.target is not None else "".encode())+bytearray(self.rule)
        return int.from_bytes(val, byteorder='big')

    def __str__(self):
        ret_val = 'If '
        for value in self.rule:
            ret_val += str(value)
            ret_val += " and "

        return ret_val[:-4] + "then target is " + str(self.target)
