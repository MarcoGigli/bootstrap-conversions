class Policy(object):

    def __init__(self, T: int, m: int):
        """
        Constructor.
        """
        self.T = T
        self.m = m


    def selectArm(self, arms):
        """
        This functions selects L arms among the K ones depending on statistics
        over the past observations.
        """
        raise NotImplementedError("Method `selectArms` is not implemented.")

    def updateState(self, disclosure):
        """
        This function updates the statistics given the new observations.
        """
        raise NotImplementedError("Method `updateState` is not implemented.")

    @staticmethod
    def id():
        raise NotImplementedError("Static method `id` is not implemented.")

    @staticmethod
    def recquiresInit():
        raise NotImplementedError("Static method `recquiresInit` is not \
                                   implemented.")