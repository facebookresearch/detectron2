# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

from .torchscript_patch import patch_instances


def export_torchscript_with_instances(model, fields):
    """
    Run :func:`torch.jit.script` on a model that uses the :class:`Instances` class. Since
    attributes of :class:`Instances` are "dynamically" added in eager mode，it is difficult
    for torchscript to support it out of the box. This function is made to support scripting
    a model that uses :class:`Instances`. It does the following:

    1. Create a scriptable ``new_Instances`` class which behaves similarly to ``Instances``,
       but with all attributes been "static".
       The attributes need to be statically declared in the ``fields`` argument.
    2. Register ``new_Instances`` to torchscript, and force torchscript to
       use it when trying to compile ``Instances``.

    After this function, the process will be reverted. User should be able to script another model
    using different fields.

    Example:
        Assume that ``Instances`` in the model consist of two attributes named
        ``proposal_boxes`` and ``objectness_logits`` with type :class:`Boxes` and
        :class:`Tensor` respectively during inference. You can call this function like:

        ::
            fields = {"proposal_boxes": "Boxes", "objectness_logits": "Tensor"}
            torchscipt_model =  export_torchscript_with_instances(model, fields)

    Note:
        Currently we only support models in evaluation mode.

    Args:
        model (nn.Module): The input model to be exported to torchscript.
        fields (Dict[str, str]): Attribute names and corresponding type annotations that
            ``Instances`` will use in the model. Note that all attributes used in ``Instances``
            need to be added, regarldess of whether they are inputs/outputs of the model.
            Custom data type is not supported for now.

    Returns:
        torch.jit.ScriptModule: the input model in torchscript format
    """

    assert (
        not model.training
    ), "Currently we only support exporting models in evaluation mode to torchscript"

    with patch_instances(fields):
        scripted_model = torch.jit.script(model)
        return scripted_model
