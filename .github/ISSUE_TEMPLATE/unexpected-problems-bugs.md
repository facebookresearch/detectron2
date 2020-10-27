---
name: "Unexpected behaviors"
about: Run into unexpected behaviors when using detectron2
title: Please read & provide the following

---

If you do not know the root cause of the problem, and wish someone to help you, please
post according to this template:

## Instructions To Reproduce the Issue:

Check https://stackoverflow.com/help/minimal-reproducible-example for how to ask good questions.
Simplify the steps to reproduce the issue using suggestions from the above link, and provide them below:

1. Full runnable code or full changes you made:
```
If making changes to the project itself, please use output of the following command:
git rev-parse HEAD; git diff

<put code or diff here>
```
2. What exact command you run:
3. __Full logs__ you observed:
```
<put logs here>
```

## Expected behavior:

If there are no obvious error in "what you observed" provided above,
please tell us the expected behavior.

If you expect the model to converge / work better, note that we do not give suggestions
on how to train a new model.
Only in one of the two conditions we will help with it:
(1) You're unable to reproduce the results in detectron2 model zoo.
(2) It indicates a detectron2 bug.

## Environment:

Provide your environment information using the following command:
```
wget -nc -q https://github.com/facebookresearch/detectron2/raw/master/detectron2/utils/collect_env.py && python collect_env.py
```

If your issue looks like an installation issue / environment issue,
please first try to solve it with the instructions in
https://detectron2.readthedocs.io/tutorials/install.html#common-installation-issues
