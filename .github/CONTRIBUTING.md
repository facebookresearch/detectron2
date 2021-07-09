# Contributing to detectron2

## Issues
We use GitHub issues to track public bugs and questions.
Please make sure to follow one of the
[issue templates](https://github.com/facebookresearch/detectron2/issues/new/choose)
when reporting any issues.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Pull Requests
We actively welcome pull requests.

However, if you're adding any significant features (e.g. > 50 lines), please
make sure to discuss with maintainers about your motivation and proposals in an issue
before sending a PR. This is to save your time so you don't spend time on a PR that we'll not accept.

We do not always accept new features, and we take the following
factors into consideration:

1. Whether the same feature can be achieved without modifying detectron2.
   Detectron2 is designed so that you can implement many extensions from the outside, e.g.
   those in [projects](https://github.com/facebookresearch/detectron2/tree/master/projects).
   * If some part of detectron2 is not extensible enough, you can also bring up a more general issue to
     improve it. Such feature request may be useful to more users.
2. Whether the feature is potentially useful to a large audience (e.g. an impactful detection paper, a popular dataset,
   a significant speedup, a widely useful utility),
   or only to a small portion of users (e.g., a less-known paper, an improvement not in the object
   detection field, a trick that's not very popular in the community, code to handle a non-standard type of data)
   * Adoption of additional models, datasets, new task are by default not added to detectron2 before they
     receive significant popularity in the community.
     We sometimes accept such features in `projects/`, or as a link in `projects/README.md`.
3. Whether the proposed solution has a good design / interface. This can be discussed in the issue prior to PRs, or
   in the form of a draft PR.
4. Whether the proposed solution adds extra mental/practical overhead to users who don't
   need such feature.
5. Whether the proposed solution breaks existing APIs.

To add a feature to an existing function/class `Func`, there are always two approaches:
(1) add new arguments to `Func`; (2) write a new `Func_with_new_feature`.
To meet the above criteria, we often prefer approach (2), because:

1. It does not involve modifying or potentially breaking existing code.
2. It does not add overhead to users who do not need the new feature.
3. Adding new arguments to a function/class is not scalable w.r.t. all the possible new research ideas in the future.

When sending a PR, please do:

1. If a PR contains multiple orthogonal changes, split it to several PRs.
2. If you've added code that should be tested, add tests.
3. For PRs that need experiments (e.g. adding a new model or new methods),
   you don't need to update model zoo, but do provide experiment results in the description of the PR.
4. If APIs are changed, update the documentation.
5. We use the [Google style docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) in python.
6. Make sure your code lints with `./dev/linter.sh`.


## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## License
By contributing to detectron2, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
