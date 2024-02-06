import os
from typing import Dict, Any, Tuple
import yaml


def load_od_config() -> Dict[str, Any]:
    root = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(root, 'project_config.yaml')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    config['anchor_matcher']['thresholds'] = [float(i) for i in config['anchor_matcher']['thresholds']]
    config['proposal_matcher']['thresholds'] = [float(i) for i in config['proposal_matcher']['thresholds']]

    config['id_to_label_name'] = {v: k for k, v in config['label_name_to_id'].items()}
    return config


CONFIG = load_od_config()
