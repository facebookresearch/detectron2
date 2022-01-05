---
name: "😩 Unexpected behaviors"
about: Report unexpected behaviors when using detectron2
title: Please read & provide the following

---

If you do not know the root cause of the problem, please post according to this template:

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
3. __Full logs__ or other relevant observations:
```
<put logs here>
```

## Expected behavior:

If there are no obvious crash in "full logs" provided above,
please tell us the expected behavior.

If you expect a model to converge / work better, we do not help with such issues, unless
a model fails to reproduce the results in detectron2 model zoo, or proves existence of bugs.

## Environment:

Paste the output of the following command:
```
wget -nc -nv https://github.com/facebookresearch/detectron2/raw/main/detectron2/utils/collect_env.py && python collect_env.py
```

If your issue looks like an installation issue / environment issue,
please first check common issues in https://detectron2.readthedocs.io/tutorials/install.html#common-installation-issues
