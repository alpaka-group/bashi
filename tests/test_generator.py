import unittest

from bashi.generator import add


class TestSum(unittest.TestCase):
    def test_sum(self):
        self.assertTrue(add(1, 2) == 3)
