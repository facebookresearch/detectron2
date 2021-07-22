#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# -*- coding: utf-8 -*-

import argparse

template = """<details><summary> install </summary><pre><code>\
python -m pip install detectron2{d2_version} -f \\
  https://dl.fbaipublicfiles.com/detectron2/wheels/{cuda}/torch{torch}/index.html
</code></pre> </details>"""
CUDA_SUFFIX = {
    "11.1": "cu111",
    "11.0": "cu110",
    "10.2": "cu102",
    "10.1": "cu101",
    "10.0": "cu100",
    "9.2": "cu92",
    "cpu": "cpu",
}


def gen_header(torch_versions):
    return '<table class="docutils"><tbody><th width="80"> CUDA </th>' + "".join(
        [
            '<th valign="bottom" align="left" width="100">torch {}</th>'.format(t)
            for t in torch_versions
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2-version", help="detectron2 version number, default to empty")
    args = parser.parse_args()
    d2_version = f"=={args.d2_version}" if args.d2_version else ""

    all_versions = (
        [("1.7", k) for k in ["11.0", "10.2", "10.1", "9.2", "cpu"]]
        + [("1.8", k) for k in ["11.1", "10.2", "10.1", "cpu"]]
        + [("1.9", k) for k in ["11.1", "10.2", "cpu"]]
    )

    torch_versions = sorted({k[0] for k in all_versions}, key=float, reverse=True)
    cuda_versions = sorted(
        {k[1] for k in all_versions}, key=lambda x: float(x) if x != "cpu" else 0, reverse=True
    )

    table = gen_header(torch_versions)
    for cu in cuda_versions:
        table += f""" <tr><td align="left">{cu}</td>"""
        cu_suffix = CUDA_SUFFIX[cu]
        for torch in torch_versions:
            if (torch, cu) in all_versions:
                cell = template.format(d2_version=d2_version, cuda=cu_suffix, torch=torch)
            else:
                cell = ""
            table += f"""<td align="left">{cell} </td> """
        table += "</tr>"
    table += "</tbody></table>"
    print(table)
