import unittest
import fbsm
import numpy as np

class ControlTest(unittest.TestCase):
    def test_set_values(self):
        def _f():
            return None
        t_min_max = [-1, 7]
        ini = [1, -3, 7]
        t_values = [-1, 2, 3, 7]
        control = fbsm.Control(_f, t_min_max, ini, t_values)
        self.assertEqual(control.eqs, _f)
        self.assertEqual(control.t_span, t_min_max)
        self.assertEqual(control.x0, ini)
        self.assertEqual(control.t_eval, t_values)


class StrictIncrDecrTest(unittest.TestCase):
    def test_increasing_values(self):
        v = [-12, 0, 3, 4, 7, 9, 15]
        self.assertTrue(fbsm.strict_incr(v))
        self.assertFalse(fbsm.strict_decr(v))

    def test_decreasing_values(self):
        v = [37, 32, 10, 8, -2, -4]
        self.assertFalse(fbsm.strict_incr(v))
        self.assertTrue(fbsm.strict_decr(v))

    def test_non_strict(self):
        v = [2, 2]
        self.assertFalse(fbsm.strict_incr(v))
        self.assertFalse(fbsm.strict_decr(v))

    def test_empty(self):
        v = []
        self.assertTrue(fbsm.strict_incr(v))
        self.assertTrue(fbsm.strict_decr(v))


class FlipAllTest(unittest.TestCase):
    def test_flip(self):
        arrs = [
            np.array([1, 3, 4]),
            np.array([7, 6, 5]),
            np.array([0, -1, 2])
        ]
        expected = [
            np.array([4, 3, 1]),
            np.array([5, 6, 7]),
            np.array([2, -1, 0])
        ]
        actual = fbsm.flip_all(arrs)
        self.assertEqual(len(expected), len(actual))
        for i in range(len(expected)):
            self.assertTrue((expected[i] == actual[i]).all())

    def test_flip_empty(self):
        self.assertEqual([], fbsm.flip_all([]))


class WithArrEvTest(unittest.TestCase):
    def do_test(self, x):
        # Function that would not work with an array variable
        def f(i):
            if i < 10:
                return i * 2
            return i * 3
        try:
            _ = f(x)
            #ValueError if x is numpy.ndarray or TypeError if it is a Sequence
            self.fail("Should have thrown ValueError or TypeError. Didn't")
        except ValueError as e:
            pass
        except TypeError as e:
            pass
        f2 = fbsm.with_array_evaluation(f)
        return f2(x)

    def test_with_numpy(self):
        actual = self.do_test(np.array([2, 7, 34]))
        expected = np.array([4, 14, 102])
        self.assertTrue((expected == actual).all())

    def test_with_seq(self):
        actual = self.do_test([1, 5, 12])
        expected = [2, 10, 36]
        self.assertEqual(expected, actual)



if __name__ == '__main__':
    unittest.main()
