# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html

import os

import hydra.experimental
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

pytest_plugins = ["helpers_namespace"]

hydra.experimental.initialize(
    os.path.join(os.path.dirname(__file__), "pointnet2/config")
)

ckpt_path = os.path.join(os.path.dirname(__file__), "outputs/entr-ssg/epoch=151-val_loss=0.02.ckpt")


overrides = ["task=entr", f"model=ssg", f"model.use_xyz=True"]
cfg = hydra.experimental.compose("config.yaml", overrides)
model = hydra.utils.instantiate(cfg.task_model, cfg)

B, N = 4, 2048  # INFO: B is batch size, N is number of points

# inputs = torch.randn(B, N, 6).cuda()
# labels = torch.from_numpy(np.random.randint(0, 3, size=B)).cuda()
model.cuda()

# Load weights from ckpt
model.load_from_checkpoint(ckpt_path)

model.prepare_data()
data_loader = model.val_dataloader()

# Run inference
model.eval()

print("Running inference...")
# Only do first 10 batches
for i, batch in enumerate(data_loader):
    if i < 40:
        continue
    if i > 50:
        break

    # model.validation_step(batch, i)
    # break

    print(batch[0].shape)
    print(batch[1].shape)
    print(batch[1])

    # Show all items in batch as 3d scatter plots, fit plots to point cloud
    for i in range(batch[0].shape[0]):
        ax = plt.axes(projection='3d')
        ax.scatter3D(batch[0][i, :, 0], batch[0][i, :, 1], batch[0][i, :, 2])
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        # plt.show()

    # Compare labels to predictions
    inputs, labels = batch
    inputs = inputs.cuda()

    for i in range(inputs.shape[0]):
        # Run inference, display every layer's output
        start = time.time()
        logits = model(inputs[i:i+1, :, :])
        end = time.time()
        print(f'time{end - start}')
        # print("logits" + str(logits))
        # print("labels" + str(labels[i:i+1, :]))
        # print("")
        # print("xyz" + str(xyz))
        # print("features" + str(features))

    # labels = torch.argmax(labels, dim=1)
    # preds = torch.argmax(preds, dim=1)

    # Set display to show full width in terminal, avoid scientific notation
    np.set_printoptions(linewidth=500, suppress=True)
    torch.set_printoptions(linewidth=500, sci_mode=False)

    # import code; code.interact(local=locals())

    # Print labels and predictions with names
    print("Inputs:", inputs)
    print("Labels:", labels)
    print("Predictions: ", logits)
