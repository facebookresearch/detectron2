# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os
import tempfile
import unittest

from detectron2.utils.events import CommonMetricPrinter, EventStorage, JSONWriter


class TestEventWriter(unittest.TestCase):
    def testScalar(self):
        with tempfile.TemporaryDirectory(
            prefix="detectron2_tests"
        ) as dir, EventStorage() as storage:
            json_file = os.path.join(dir, "test.json")
            writer = JSONWriter(json_file)
            for k in range(60):
                storage.put_scalar("key", k, smoothing_hint=False)
                if (k + 1) % 20 == 0:
                    writer.write()
                storage.step()
            writer.close()
            with open(json_file) as f:
                data = [json.loads(l) for l in f]
                self.assertTrue([int(k["key"]) for k in data] == [19, 39, 59])

    def testScalarMismatchedPeriod(self):
        with tempfile.TemporaryDirectory(
            prefix="detectron2_tests"
        ) as dir, EventStorage() as storage:
            json_file = os.path.join(dir, "test.json")

            writer = JSONWriter(json_file)
            for k in range(60):
                if k % 17 == 0:  # write in a differnt period
                    storage.put_scalar("key2", k, smoothing_hint=False)
                storage.put_scalar("key", k, smoothing_hint=False)
                if (k + 1) % 20 == 0:
                    writer.write()
                storage.step()
            writer.close()
            with open(json_file) as f:
                data = [json.loads(l) for l in f]
                self.assertTrue([int(k.get("key2", 0)) for k in data] == [17, 0, 34, 0, 51, 0])
                self.assertTrue([int(k.get("key", 0)) for k in data] == [0, 19, 0, 39, 0, 59])
                self.assertTrue([int(k["iteration"]) for k in data] == [17, 19, 34, 39, 51, 59])

    def testPrintETA(self):
        with EventStorage() as s:
            p1 = CommonMetricPrinter(10)
            p2 = CommonMetricPrinter()

            s.put_scalar("time", 1.0)
            s.step()
            s.put_scalar("time", 1.0)
            s.step()

            with self.assertLogs("detectron2.utils.events") as logs:
                p1.write()
            self.assertIn("eta", logs.output[0])

            with self.assertLogs("detectron2.utils.events") as logs:
                p2.write()
            self.assertNotIn("eta", logs.output[0])

    def testSmoothingWithWindowSize(self):
        with tempfile.TemporaryDirectory(
            prefix="detectron2_tests"
        ) as dir, EventStorage() as storage:
            json_file = os.path.join(dir, "test.json")
            writer = JSONWriter(json_file, window_size=10)
            for k in range(20):
                storage.put_scalar("key1", k, smoothing_hint=True)
                if (k + 1) % 2 == 0:
                    storage.put_scalar("key2", k, smoothing_hint=True)
                if (k + 1) % 5 == 0:
                    storage.put_scalar("key3", k, smoothing_hint=True)
                if (k + 1) % 10 == 0:
                    writer.write()
                storage.step()

            num_samples = {k: storage.count_samples(k, 10) for k in ["key1", "key2", "key3"]}
            self.assertEqual(num_samples, {"key1": 10, "key2": 5, "key3": 2})
            writer.close()
            with open(json_file) as f:
                data = [json.loads(l) for l in f]
                self.assertEqual([k["key1"] for k in data], [4.5, 14.5])
                self.assertEqual([k["key2"] for k in data], [5, 15])
                self.assertEqual([k["key3"] for k in data], [6.5, 16.5])
