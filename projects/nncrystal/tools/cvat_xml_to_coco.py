import argparse
import json
from xml.etree import ElementTree

from cvat.api import CVATAPI
from cvat.argument import cvat_args
from cvat.cvat_xml import cvat_xml_to_coco

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ignore_attributes", action="store_true")
    parser.add_argument("--ignore_crowded", action="store_true")
    parser.add_argument("--occluded_as_crowded", action="store_true")
    parser.add_argument("--file")
    parser.add_argument("--job_id")
    cvat_args(parser)
    args = parser.parse_args()

    if args.file is not None:
        root = ElementTree.parse(args.file)

    elif args.job_id is not None:
        api = CVATAPI(args.cvat_host)
        api.login(args.cvat_username, args.cvat_password)
        job = api.get_job(args.job_id).json()
        task_id = job["task_id"]
        xml_content = api.export_data(task_id, format="CVAT XML 1.1 for images").content
        root = ElementTree.fromstring(xml_content)
    else:
        raise ValueError("Task id or xml file should be specified")
    print(json.dumps(cvat_xml_to_coco(root, ignore_crowded=args.ignore_crowded,
                                      occluded_as_crowded=args.occluded_as_crowded,
                                      ignore_attributes=args.ignore_attributes,
                                      cvat_base_dir=args.cvat_base)))
