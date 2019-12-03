import os
from unittest import TestCase

from cvat.api import CVATAPI, Label, Attribute
import sys


class TestCVATAPI(TestCase):
    def setUp(self) -> None:
        self.api = CVATAPI(os.environ["CVAT_HOST"])

    def test_login(self):
        result = self.api.login(os.environ["CVAT_USER"], os.environ["CVAT_PASSWORD"]).json()
        print(f"Log in result: {result}")

    def test_create_task(self):
        self.test_login()
        result = self.api.create_task("test", 5000, [
            Label("glass beads", [Attribute("cropped", "checkbox", ["yes", "no"], "no")])
        ])

        print(f"create task result: {result.json()}")

    def test_get_shared_files(self):
        self.test_login()
        result = self.api.get_server_files()
        print(result.json())

    def test_export(self):
        self.test_login()
        result = self.api.export_data(11)
        print(result.json)