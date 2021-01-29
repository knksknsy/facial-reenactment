import os
import sys
import logging

from configs import TestOptions

class Test():
    def __init__(self, options: TestOptions):
        self.options = options

        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        self._test()


    def _test(self):
        pass