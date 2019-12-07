# Contributing to detectron2
We want to make contributing to this project as easy and transparent as
possible.

## Issues
We use GitHub issues to track public bugs and questions.
Please make sure to follow one of the
[issue templates](https://github.com/facebookresearch/detectron2/issues/new/choose)
when reporting any issues.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Pull Requests
We actively welcome your pull requests.

However, if you're adding any significant features, please
make sure to have a corresponding issue to discuss your motivation and proposals,
before sending a PR. We do not always accept new features, and we take the following
factors into consideration:

1. Whether the same feature can be achieved without modifying detectron2.
Detectron2 is designed so that you can implement many extensions from the outside, e.g.
those in [projects](https://github.com/facebookresearch/detectron2/tree/master/projects).
If some part is not as extensible, you can also bring up the issue to make it more extensible.
2. Whether the feature is potentially useful to a large audience, or only to a small portion of users.
3. Whether the proposed solution has a good design / interface.
4. Whether the proposed solution adds extra mental/practical overhead to users who don't
   need such feature.
5. Whether the proposed solution breaks existing APIs.

When sending a PR, please do:

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints with `./dev/linter.sh`.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## License
By contributing to detectron2, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
