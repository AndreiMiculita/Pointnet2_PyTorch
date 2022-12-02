# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html

import os
import time

import hydra.experimental
import matplotlib.pyplot as plt
import numpy as np
import torch

from pointnet2.models import PointNet2EntropySSG

# Set display to show full width in terminal, avoid scientific notation
np.set_printoptions(linewidth=500, suppress=True)
torch.set_printoptions(linewidth=500, sci_mode=False)

pytest_plugins = ["helpers_namespace"]

hydra.experimental.initialize(
    os.path.join(os.path.dirname(__file__), "pointnet2/config")
)

ckpt_path = os.path.join(os.path.dirname(__file__), "outputs/entr-ssg/epoch=151-val_loss=0.02.ckpt")

overrides = ["task=entr", f"model=ssg", f"model.use_xyz=True"]
cfg = hydra.experimental.compose("config.yaml", overrides)


def single_inference(
        model: PointNet2EntropySSG,
        sample: torch.Tensor,
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
) -> torch.Tensor:
    """
    Run inference on a single sample. This can be used in the application ("production").
    """
    # Make sure both are on the device
    model.to(device)
    sample.to(device)

    # Disable gradients
    with torch.no_grad():
        # nn.Module.forward() will expect a batch, so create a pseudo batch of size 1
        pseudo_batch = sample.unsqueeze(0)
        # Run inference
        output = model(pseudo_batch)
        # Remove batch dimension
        output = output.squeeze(0)
        # Return output
        return output


def main():
    model = PointNet2EntropySSG.load_from_checkpoint(ckpt_path)
    # Print type of model
    print(f'model type: {type(model)}')
    model.eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            print(f'{name}:\n{param}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.prepare_data()
    data_loader = model.val_dataloader()

    # Run inference
    model.eval()

    print(f'Running inference...')

    # Print devices for each parameter in model
    for name, param in model.named_parameters():
        print(f'{name} {param.shape}')

    # Print the model parameters
    print(f'model.parameters() {model.parameters()}')

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            # Only do first 10 batches
            if idx >= 10:
                break

            point_clouds, labels = batch

            print(f'idx {idx}')
            print(f'point_clouds.shape {point_clouds.shape}')
            print(f'point_clouds {point_clouds}')
            print(f'labels.shape {labels.shape}')
            print(f'labels {labels}')

            # Show all items in batch as 3d scatter plots, fit plots to point cloud, show as 4x8 grid
            fig = plt.figure(figsize=(8, 4))
            for i in range(32):
                ax = fig.add_subplot(4, 8, i + 1, projection='3d')
                ax.scatter(point_clouds[i, :, 0], point_clouds[i, :, 1], point_clouds[i, :, 2])
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)

            plt.show()

            # Compare labels to predictions
            point_clouds = point_clouds.to(device)
            labels = labels.to(device)

            for point_cloud, label in zip(point_clouds, labels):
                # Run inference
                start = time.time()
                logits = single_inference(model, point_cloud)
                end = time.time()
                print(f'time taken {end - start}')
                print(f"logits {logits}")
                print(f"max logits {torch.argmax(logits)}")
                print(f"label  {label}")
                print(f"max label  {torch.argmax(label)}")

            print(f'\n')

            # Outputs are always the same, to debug we print the parameters of the model
            for name, param in model.named_parameters():
                print(f'param {name} on device {param.device}')
                print(f'\tparam {name} {param}')
                # With the following line we can see that the parameters are always the same
                print(f'\tparam {name} sums to {torch.sum(param)}')


if __name__ == "__main__":
    # We give a little demo here, but the single_inference() function is what you're probably looking for
    main()
