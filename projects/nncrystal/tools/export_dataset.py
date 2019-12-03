import argparse
import json
import os
from xml.etree import ElementTree

from cvat.api import CVATAPI
from cvat.argument import cvat_args
from cvat.cvat_xml import cvat_xml_to_coco

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    cvat_args(parser)
    parser.add_argument("--job_id", action="append", type=int, required=True)
    parser.add_argument("--output_dir", default="./datasets")
    args = parser.parse_args()

    api = CVATAPI(args.cvat_host)
    api.login(args.cvat_username, args.cvat_password)

    for job_id in args.job_id:
        job = api.get_job(job_id).json()
        task_id = job["task_id"]
        data = api.export_data(task_id, format="CVAT XML 1.1 for images").content
        root = ElementTree.fromstring(data)
        result = cvat_xml_to_coco(root, ignore_crowded=True,
                                    occluded_as_crowded=True,
                                    ignore_attributes=True
                                    )
        with open(os.path.join(args.output_dir, f"export_job_{job_id}.coco.json"), "w") as f:
            json.dump(result, f)
