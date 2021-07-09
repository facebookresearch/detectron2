# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import LazyConfig

# equivalent to relative import
dir1a_str, dir1a_dict = LazyConfig.load_rel("dir1_a.py", ("dir1a_str", "dir1a_dict"))

dir1b_str = dir1a_str + "_from_b"
dir1b_dict = dir1a_dict

# Every import is a reload: not modified by other config files
assert dir1a_dict.a == 1
