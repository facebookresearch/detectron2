import argparse
import os
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from cvat.api import CVATAPI
from cvat.argument import cvat_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--task_id", action="append", type=int)
    parser.add_argument("--job_id", action="append", type=int)
    parser.add_argument("--output_dir", required=True)
    cvat_args(parser)
    args = parser.parse_args()

    api = CVATAPI(args.cvat_host)
    api.login(args.cvat_username, args.cvat_password)
    if args.all:
        tasks = api.list_task()
        task_ids = [x["id"] for x in tasks]

    else:
        task_ids = []
        if args.task_id:
            task_ids.extend(args.task_id)
        if args.job_id:
            for job_id in args.job_id:
                job = api.get_job(job_id).json()
                task_id = job["task_id"]
                task_ids.append(task_id)

    os.makedirs(args.output_dir, exist_ok=True)
    for task in task_ids:
        content = api.export_data(task, format="CVAT XML 1.1 for images").content
        root: Element = ElementTree.fromstring(content)
        name = root.find('meta/task/name').text
        with open(os.path.join(args.output_dir, f"{name}.xml"), "wb") as f:
            f.write(content)


