import unittest
import pandas as pd
import pandas.tests as pd_testing
from main import scale_bounding_boxes

input_array = pd.DataFrame([[281.2590449, 187.0350708, 327.7279305, 223.225547],
                            [15.16353111, 187.0350708, 120.3299566, 236.4301802],
                            [239.1924747, 176.7648005, 361.9681621, 236.4301802]])


output_array = pd.DataFrame([[93.198263, 110.252252, 108.596237, 131.585586],
                             [5.024602, 110.252252, 39.872648, 139.369369],
                             [79.259045, 104.198198, 119.942113, 139.369369]])


class TestCalculus(unittest.TestCase):

    def test_scaling(self):
        input_array_shape = (380, 676)
        output_array_shape = (224, 224)
        pd.testing.assert_frame_equal(scale_bounding_boxes(input_array, input_array_shape, output_array_shape),
                                      output_array)

    def test_scaling_max(self):
        output_array_shape = (224, 224)
        self.assertLessEqual(output_array.iloc[:, 0].max(), output_array_shape[1])
        self.assertLessEqual(output_array.iloc[:, 2].max(), output_array_shape[1])
        self.assertLessEqual(output_array.iloc[:, 1].max(), output_array_shape[0])
        self.assertLessEqual(output_array.iloc[:, 3].max(), output_array_shape[0])

    def test_scaling_min(self):
        self.assertGreaterEqual(output_array.min().min(), 0)
