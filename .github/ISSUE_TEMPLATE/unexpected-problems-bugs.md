---
name: "Unexpected Problems / Bugs"
about: Report unexpected problems or bugs in detectron2

---

If you do not know the root cause of the problem / bug, and wish someone to help you, please
include:

## How To Reproduce the Issue

1. what changes you made (`git diff`) or what code you wrote
2. what exact command you run
3. what you observed (including the full logs):
```
<put logs in a formatted block>
```

## Expected behavior

If there are no obvious error in "what you observed" provided above,
please tell us the expected behavior.

If you expect the model to work better, note that we do not help you train your model.
Only in one of the two conditions we will help with it:
(1) You're unable to reproduce the results in detectron2 model zoo.
(2) It indicates a detectron2 bug.

## Environment

Please paste the output of `python -m detectron2.utils.collect_env`.
If detectron2 hasn't been successfully installed,
use `python detectron2/utils/collect_env.py`.
