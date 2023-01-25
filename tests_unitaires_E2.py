# python -m unittest tests_unitaires_E2.py -v
import unittest
import pandas as pd
from app import inputs_to_df, preprocess_inputs, predict


class MyTestCase(unittest.TestCase):

    def test_inputs_to_df_type(self):
        self.assertEqual(
            type(inputs_to_df("-124.13", "40.80", 1259, "2.2478", "NEAR OCEAN")), pd.core.frame.DataFrame)

    def test_inputs_to_df_output_nbcol(self):
        self.assertEqual(
            len(inputs_to_df("-124.13", "40.80", 1259, "2.2478", "NEAR OCEAN").columns), 5)

    def test_preprocess_inputs_output_nbcol(self):
        self.assertEqual(len(preprocess_inputs(pd.DataFrame({'longitude': [float("-124.13")], 'latitude': [float(
            "40.80")], 'population': [1259], 'median_income': [float("2.2478")], 'ocean_proximity': ["NEAR OCEAN"]})).columns), 5)

    def test_preprocess_inputs_type(self):
        self.assertEqual(
            type(preprocess_inputs(pd.DataFrame({'longitude': [float("-124.13")], 'latitude': [float("40.80")], 'population': [1259], 'median_income': [float("2.2478")], 'ocean_proximity': ["NEAR OCEAN"]}))), pd.core.frame.DataFrame)


if __name__ == '__main__':
    unittest.main()
