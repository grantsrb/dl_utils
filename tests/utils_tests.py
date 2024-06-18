import unittest
import sys
sys.path.append("../")

import utils

class TestUtilsMethods(unittest.TestCase):

    def test_get_one_hots_1d_list(self):
        # 1d array
        arr = [i for i in range(3)]
        one_hots = utils.get_one_hots(arr, 3)
        for i in range(3):
            self.assertEqual(one_hots[i][i], 1)

        arr = [i if i!=1 else -1 for i in range(3)]
        one_hots = utils.get_one_hots(arr, 3)
        self.assertEqual(one_hots[0][0], 1)
        self.assertEqual(one_hots[1][0], 0)
        self.assertEqual(one_hots[1][1], 0)
        self.assertEqual(one_hots[1][2], 0)
        self.assertEqual(one_hots[2][2], 1)

    def test_get_one_hots_2d_list(self):
        # 2d array
        arr = [[i for i in range(3)] for j in range(2)]
        one_hots = utils.get_one_hots(arr, 3)
        for j in range(2):
            for i in range(3):
                self.assertEqual(one_hots[j][i][i], 1)

        arr = [[i if i!=1 else -1 for i in range(3)] for j in range(2)]
        one_hots = utils.get_one_hots(arr, 3)
        print(one_hots)
        self.assertEqual(one_hots[0][0][0], 1)
        self.assertEqual(one_hots[0][1][0], 0)
        self.assertEqual(one_hots[0][1][1], 0)
        self.assertEqual(one_hots[0][1][2], 0)
        self.assertEqual(one_hots[0][2][2], 1)




if __name__ == '__main__':
    unittest.main()
