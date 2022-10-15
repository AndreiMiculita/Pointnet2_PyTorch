import os
import os.path as osp

import lmdb
import msgpack_numpy
import numpy as np
import torch
import torch.utils.data as data
import tqdm
import pandas as pd

import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class ModelNetEntropyLoader(data.Dataset):
    def __init__(self, num_points, transforms=None, train=True, download=True):
        super().__init__()

        self.transforms = transforms

        self.set_num_points(num_points)
        self._cache = os.path.join(BASE_DIR, "modelnet10_andrei_partial_entropy_cache")

        if not osp.exists(self._cache):
            self.folder = "modelnet10_andrei_partial"
            self.data_dir = os.path.join(BASE_DIR, self.folder)
            self.url = (
                "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"
            )

            if not os.path.exists(self.data_dir):
                print("Need data dir!")
                exit(1)

            self.train = train
            self.set_num_points(num_points)

            self.catfile = os.path.join(self.data_dir, "modelnet10_shape_names.txt")
            self.cat = [line.rstrip() for line in open(self.catfile)]
            self.classes = dict(zip(self.cat, range(len(self.cat))))

            entropy_table = pd.read_csv(os.path.join(self.data_dir, "entropy_dataset_mnet10_10samplesfull.csv"))

            os.makedirs(self._cache)

            print("Converted to LMDB for faster dataloading while training")
            for split in ["train", "test"]:
                if split == "train":
                    shape_ids = [
                        line.rstrip()
                        for line in open(
                            os.path.join(self.data_dir, "modelnet10_train.txt")
                        )
                    ]
                else:
                    shape_ids = [
                        line.rstrip()
                        for line in open(
                            os.path.join(self.data_dir, "modelnet10_test.txt")
                        )
                    ]

                print("Shape IDs:", shape_ids)

                shape_names = ["_".join(x.split("_")[0:-7]) for x in shape_ids]
                print("Shape names:", shape_names)

                shape_indexes = ["_".join(x.split("_")[-7:-6]) for x in shape_ids]

                # Convert to int
                shape_indexes = [int(x) for x in shape_indexes]

                print(set(shape_indexes))

                # exit()

                # list of (shape_name, shape_txt_file_path) tuple
                self.datapath = [
                    (
                        shape_names[i],
                        os.path.join(self.data_dir, shape_names[i], shape_ids[i])
                        + ".pcd",
                    )
                    for i in range(len(shape_ids))
                ]

                with lmdb.open(
                    osp.join(self._cache, split), map_size=1 << 36
                ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                    print(len(self.datapath))

                    idx = 0
                    pbar = tqdm.trange(len(self.datapath))
                    for i in pbar:
                        fn = self.datapath[i]

                        point_set = np.asarray(o3d.io.read_point_cloud(fn[1]).points, dtype=np.float32)

                        # Only take point clouds with more than 1e3 points
                        if point_set.shape[0] > 1e3:
                            # Pad point_set with 0s up to 6 columns
                            # point_set = np.pad(point_set, ((0, 0), (0, 6 - point_set.shape[1])), "constant")

                            cls = self.classes[self.datapath[i][0]]
                            cls = int(cls)

                            # Read entropies where shape name and obj_ind match
                            entropies_for_obj = np.asarray(entropy_table[
                                (entropy_table["label"] == fn[0])
                                & (entropy_table["obj_ind"] == shape_indexes[i])
                            ]["entropy"])

                            # Add tqdm message with entropies
                            pbar.set_description(f"entr for label {fn[0]} and obj_ind {shape_indexes[i]} {entropies_for_obj}")

                            # print(f"entr for label {fn[0]} and obj_ind {shape_indexes[i]} {entropies_for_obj}")
                            if entropies_for_obj.shape[0] == 0:
                                print("No entropies for this object")
                                exit()

                            txn.put(
                                str(idx).encode(),
                                msgpack_numpy.packb(
                                    dict(pc=point_set, lbl=cls, entr=entropies_for_obj), use_bin_type=True
                                ),
                            )
                            idx += 1

            # shutil.rmtree(self.data_dir)

        self._lmdb_file = osp.join(self._cache, "train" if train else "test")
        with lmdb.open(self._lmdb_file, map_size=1 << 36) as lmdb_env:
            self._len = lmdb_env.stat()["entries"]

        self._lmdb_env = None

    def __getitem__(self, idx):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._lmdb_file, map_size=1 << 36, readonly=True, lock=False
            )

        with self._lmdb_env.begin(buffers=True) as txn:
            ele = msgpack_numpy.unpackb(txn.get(str(idx).encode()), raw=False)

        point_set = ele["pc"]

        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        point_set = point_set[pt_idxs, :]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # Pad point_set with 0s up to 6 columns
        point_set = np.pad(point_set, ((0, 0), (0, 6 - point_set.shape[1])), "constant")

        if self.transforms is not None:
            point_set = self.transforms(point_set)

        # get entropies as float, divide by the max in the dataset to get a value between 0 and 1
        entropies = ele["entr"].astype(np.float32) / 5.46
        entropies = entropies.astype(np.float32)

        return point_set, entropies

    def __len__(self):
        return self._len

    def set_num_points(self, pts):
        self.num_points = min(int(1e3), pts)


if __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            d_utils.PointcloudScale(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
        ]
    )
    dset = ModelNetEntropyLoader(16, train=True, transforms=transforms)
    print(dset[0][0])
    print(dset[0][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)