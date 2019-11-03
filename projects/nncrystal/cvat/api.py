import time
from typing import List, Union

import requests
import rx
from rx import scheduler


class Attribute:
    def __init__(self, name: str, input_type: str, values: Union[List[str], str], default_value: str):
        self.name = name

        assert input_type in ["checkbox", "radio", "number", "text", "select"]
        self.input_type = input_type

        self.default_value = default_value

        self.mutable = True  # should only affect continuous case
        self.values = values

    def dict(self):
        return self.__dict__


class Label:
    def __init__(self, name, attrs: List[Attribute]):
        self.name = name
        self.attributes = [attr.__dict__ for attr in attrs]

    def dict(self):
        return self.__dict__


class CVATAPI:
    def __init__(self, host: str):
        self.host = host
        self.session = requests.session()
        self.csrf_token = None

    def update_csrf_token(self):
        self.session.get(self.host)
        if 'csrftoken' in self.session.cookies:
            # Django 1.6 and up
            self.csrf_token = self.session.cookies['csrftoken']
        else:
            # older versions
            self.csrf_token = self.session.cookies['csrf']

    def login(self, username: str, password: str):
        url = f"{self.host}/api/v1/auth/login"

        self.update_csrf_token()
        resp = self.session.post(url, json={
            "username": username,
            "password": password
        }, headers={"Referer": url, "X-CSRFToken": self.csrf_token})
        return resp

    def create_task(self, name: str, segment_size: int, labels: List[Label]):
        image_quality = 50
        url = f"{self.host}/api/v1/tasks"
        self.update_csrf_token()
        resp = self.session.post(url, json={
            "name": name,
            "segment_size": segment_size,
            "image_quality": image_quality,
            "labels": [label.dict() for label in labels],
        }, headers={"Referer": url, "X-CSRFToken": self.csrf_token})
        return resp

    def list_task(self, id_=None):
        url = f"{self.host}/api/v1/tasks" + (f"/{id_}" if id_ is not None else "")
        resp = self.session.get(url)
        return resp

    def add_task_data(self, server_files):
        pass

    def get_server_files(self):
        url = f"{self.host}/api/v1/server/share"
        resp = self.session.get(url)
        return resp

    def get_frame(self, task_id, frame_id):
        url = f"{self.host}/api/v1/tasks/{task_id}/frames/{frame_id}"
        resp = self.session.get(url)
        return resp

    def export_data(self, task_id, filename=None, format="COCO JSON 1.0"):
        if filename is None:
            tasks = self.list_task().json()
            results = tasks["results"]
            result = list(filter(lambda r: r["id"] == task_id, results))
            assert len(result)
            result = result[0]
            filename = result["name"]
        url = f"{self.host}/api/v1/tasks/{task_id}/annotations/{filename}?format={format}&action=download"
        while True:
            result = self.session.get(url)
            if result.status_code == 200:
                return result
            else:
                time.sleep(0.5)

    def get_job(self, job_id):
        url = f"{self.host}/api/v1/jobs/{job_id}"
        resp = self.session.get(url)
        return resp

    def upload_annotations(self, job_id, buffer, format="CVAT XML 1.1"):
        url = f"{self.host}/api/v1/jobs/{job_id}/annotations?format={format}"
        self.update_csrf_token()
        self.session.put(url, files={
            "annotation_file": buffer
        }, headers={"Referer": url, "X-CSRFToken": self.csrf_token})

        result = self.session.put(url, headers={"Referer": url, "X-CSRFToken": self.csrf_token})
        return result
