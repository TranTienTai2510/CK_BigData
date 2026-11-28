"""Focused tests for the isolated purchase-intent architecture zoo."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from ml import features as FT
from ml_test.benchmark_models import _export_onnx
from ml_test.models_zoo import MODEL_SPECS, _torch_flat_features, build_model, window_to_flat_features


class MemoryDataset:
    def __init__(self, x_cat, x_num, length, shop):
        self.x_cat, self.x_num, self.length, self.shop = x_cat, x_num, length, shop
        self.y = np.asarray([0, 1], dtype=np.float32)
        self.k = length
        self.shop_id = np.asarray(["synthetic", "synthetic"], dtype=object)

    def __len__(self):
        return len(self.y)

    def get_batch(self, indices):
        return (self.x_cat[indices], self.x_num[indices], self.length[indices],
                self.shop[indices], self.y[indices])


def main():
    torch.manual_seed(17)
    spec = FT.make_spec(
        ["synthetic"],
        [f"synthetic|cat{i}" for i in range(4)],
        ["synthetic|brand"],
    )
    batch, width = 2, spec["max_len"]
    x_cat = np.zeros((batch, width, FT.N_CAT), dtype=np.int64)
    x_num = np.zeros((batch, width, FT.N_NUM), dtype=np.float32)
    lengths = np.asarray([1, 4], dtype=np.int64)
    for row, length in enumerate(lengths):
        begin = width - int(length)
        x_cat[row, begin:, 0] = np.arange(3, 3 + length)
        x_cat[row, begin:, 1] = np.arange(3, 3 + length)
        x_cat[row, begin:, 2] = 3
        x_num[row, begin:, 0] = np.arange(1, length + 1, dtype=np.float32)
        x_num[row, begin:, 2] = 0.1
        x_num[row, begin:, 3] = np.arange(length, dtype=np.float32) * 0.2
        x_num[row, begin:, 5] = np.log1p(np.arange(1, length + 1))
        x_num[row, begin:, 6] = np.log1p(np.arange(length, dtype=np.float32))
        x_num[row, begin:, 7] = 0.0
        x_num[row, begin:, 8] = np.arange(length, dtype=np.float32)
    flat = window_to_flat_features(x_cat, x_num, lengths)
    tx_cat = torch.from_numpy(x_cat)
    tx_num = torch.from_numpy(x_num)
    tx_len = torch.from_numpy(lengths)
    valid = (torch.arange(width)[None, :] >= (width - tx_len)[:, None]).to(torch.float32).unsqueeze(-1)
    flat_torch = _torch_flat_features(tx_cat, tx_num, valid, tx_len).detach().numpy()
    checks = {
        "flat_shape_31": (flat.shape, (batch, 31)),
        "numpy_torch_flat_parity": (float(np.max(np.abs(flat - flat_torch))) < 1e-5, True),
    }

    shop = torch.full((batch,), spec["shops"]["synthetic"], dtype=torch.int64)
    test_ds = MemoryDataset(x_cat, x_num, lengths, shop.numpy())
    for name in MODEL_SPECS:
        model = build_model(name, spec, d_model=16, dropout=0.0)
        prob = model(tx_cat, tx_num, tx_len, shop)
        prob.sum().backward()
        checks[name + "_output_and_grad"] = (
            tuple(prob.shape) == (batch,) and bool(torch.isfinite(prob).all())
            and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
            True,
        )
        with tempfile.TemporaryDirectory(prefix="models_zoo_") as tmp_dir:
            parity = _export_onnx(model, os.path.join(tmp_dir, name + ".onnx"), spec, test_ds,
                                  verify_samples=2)
        checks[name + "_onnx_parity"] = (
            parity["onnx_parity_check"]["passed"]
            and parity["onnx_parity_check"]["max_abs_diff"] < 1e-4,
            True,
        )

    failed = []
    for name, (got, expected) in checks.items():
        passed = got == expected
        print(f"{'OK' if passed else 'BAD'} {name}: got={got} expected={expected}")
        if not passed:
            failed.append(name)
    print("TEST", "PASS" if not failed else f"FAIL {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
