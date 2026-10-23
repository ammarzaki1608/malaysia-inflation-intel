"""Unit tests for user_profile.py. Run: python -m unittest test_user_profile"""

import unittest

from user_profile import DIVISIONS, decode_profile, default_profile, encode_profile

STATES = ["Johor", "Pulau Pinang", "Selangor"]


class RoundTrip(unittest.TestCase):
    def test_encode_then_decode_gives_the_same_profile(self):
        p = default_profile("Johor")
        p.update(state="Pulau Pinang", income=6200, pay_rise=3.5, horizon=6)
        p["spending"]["01"] = 1100
        back, from_url = decode_profile(encode_profile(p), STATES, "Johor")
        self.assertTrue(from_url)
        self.assertEqual(back, p)

    def test_zero_spending_is_left_out_of_the_url(self):
        sp = encode_profile(default_profile("Johor"))["sp"]
        self.assertNotIn("02:", sp)
        self.assertIn("01:900", sp)

    def test_no_pay_rise_means_no_pr_key(self):
        self.assertNotIn("pr", encode_profile(default_profile("Johor")))


class BadLinks(unittest.TestCase):
    def test_empty_params_give_defaults(self):
        p, from_url = decode_profile({}, STATES, "Johor")
        self.assertFalse(from_url)
        self.assertEqual(p, default_profile("Johor"))

    def test_garbage_values_fall_back_safely(self):
        params = {"s": "Atlantis", "inc": "lots", "pr": "x", "h": "7",
                  "sp": "01:abc,99:500,04:250,07"}
        p, from_url = decode_profile(params, STATES, "Johor")
        self.assertTrue(from_url)
        self.assertEqual(p["state"], "Johor")
        self.assertEqual(p["income"], 4500)
        self.assertIsNone(p["pay_rise"])
        self.assertEqual(p["horizon"], 12)
        self.assertEqual(p["spending"]["04"], 250)
        self.assertEqual(p["spending"]["01"], 0)
        self.assertEqual(sorted(p["spending"]), DIVISIONS)

    def test_values_are_clamped(self):
        p, _ = decode_profile({"inc": "999999", "pr": "500", "sp": "01:-5,04:999999"},
                              STATES, "Johor")
        self.assertEqual(p["income"], 50_000)
        self.assertEqual(p["pay_rise"], 100.0)
        self.assertEqual(p["spending"]["01"], 0)
        self.assertEqual(p["spending"]["04"], 10_000)


if __name__ == "__main__":
    unittest.main()
