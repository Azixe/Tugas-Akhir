"""Strip the ZipMap node from a CatBoost ONNX export.

CatBoost exports ``probabilities`` as ``seq(map(int64, float))``. That type is
awkward for the browser extension (ort-web) and for flat-tensor consumers, so
this replaces it with the underlying ``[N, 2]`` float tensor that the
``TreeEnsembleClassifier`` node already produces.

Usage:
    python Catboost/fix_onnx_output.py sampling_results/cb_full.onnx [more.onnx ...]

The file is rewritten in place (a ``.zipmap.bak`` copy is kept on first run).
Prints a verification line per file (outputs after conversion).
"""

from __future__ import annotations

import shutil
import sys

import onnx
from onnx import TensorProto, helper


def strip_zipmap(path: str) -> str:
    model = onnx.load(path)
    graph = model.graph

    zipmaps = [n for n in graph.node if n.op_type == 'ZipMap']
    if not zipmaps:
        return f"{path}: no ZipMap node (already tensor output?)"

    for node in zipmaps:
        tensor_name = node.input[0]
        graph.node.remove(node)
        # replace the sequence output with the underlying tensor, keeping order
        for i, out in enumerate(graph.output):
            if out.name == node.output[0]:
                graph.output.remove(out)
                graph.output.insert(
                    i, helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, ['N', 2]))
                break

    onnx.checker.check_model(model)
    backup = path + '.zipmap.bak'
    try:
        open(backup, 'x').close()
        shutil.copy2(path, backup)
    except FileExistsError:
        pass
    onnx.save(model, path)

    outs = [f"{o.name}:{o.type.WhichOneof('value')}" for o in graph.output]
    return f"{path}: converted -> outputs: {outs}"


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        print(strip_zipmap(p))
