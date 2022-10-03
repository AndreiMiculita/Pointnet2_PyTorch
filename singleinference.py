import os

import hydra
import hydra.experimental
import numpy as np
import torch

pytest_plugins = ["helpers_namespace"]

hydra.experimental.initialize(
    os.path.join(os.path.dirname(__file__), "pointnet2/config")
)

ckpt_path = os.path.join(os.path.dirname(__file__), "outputs/cls-ssg/epoch=80-val_loss=1.26-val_acc=0.557.ckpt")


def get_model(overrides=[]):
    cfg = hydra.experimental.compose("config.yaml", overrides)
    return hydra.utils.instantiate(cfg.task_model, cfg)


def _test_loop(model, inputs, labels):
    # Load weights from ckpt
    model.load_from_checkpoint(ckpt_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    prev_loss = 1e10
    for _ in range(5):
        optimizer.zero_grad()
        res = model.training_step((inputs, labels), None)
        loss = res["loss"]
        loss.backward()
        optimizer.step()

        print(loss.item())

        assert loss.item() < prev_loss + 1.0, "Loss spiked upwards"

        prev_loss = loss.item()


# @pytest.mark.parametrize("use_xyz", ["True", "False"])
# @pytest.mark.parametrize("model", ["ssg", "msg"])
def test_cls(use_xyz, model):
    model = get_model(
        ["task=cls", f"model={model}", f"model.use_xyz={use_xyz}"]
    )

    B, N = 4, 2048  # INFO: B is batch size, N is number of points

    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B)).cuda()

    model.cuda()
    _test_loop(model, inputs, labels)  # TODO: change this to a single test


if __name__ == "__main__":
    test_cls("True", "ssg")
