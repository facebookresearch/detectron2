---
name: "Unexpected behaviors / Bugs"
about: Report unexpected behaviors or bugs in detectron2
title: Please read & provide the following

---

If you do not know the root cause of the problem / bug, and wish someone to help you, please
post according to this template:

## Instructions To Reproduce the Issue:

1. what changes you made (`git diff`) or what code you wrote
```
<put diff or code here>
```
2. what exact command you run:
3. what you observed (including the full logs):
```
<put logs here>
```
4. please also simplify the steps as much as possible so they do not require additional resources to
	 run, such as a private dataset.

## Expected behavior:

If there are no obvious error in "what you observed" provided above,
please tell us the expected behavior.

If you expect the model to converge / work better, note that we do not give suggestions
on how to train a new model.
Only in one of the two conditions we will help with it:
(1) You're unable to reproduce the results in detectron2 model zoo.
(2) It indicates a detectron2 bug.

## Environment:

Run `python -m detectron2.utils.collect_env` in the environment where you observerd the issue, and paste the output.
If detectron2 hasn't been successfully installed, use `python detectron2/utils/collect_env.py` (after getting this file from github).

If your issue looks like an installation issue / environment issue,
please first try to solve it yourself with the instructions in
https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md#common-installation-issues
