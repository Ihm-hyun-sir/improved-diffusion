from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

from torch.utils.data import Sampler
import random 


class MajorMinorSampler(Sampler):
    def __init__(self, classes, batch_size, major_classes, minor_classes):
        """
        Sampler for creating batches containing only Major or Minor classes.

        :param classes: List of class labels for the dataset.
        :param batch_size: Number of samples per batch.
        :param major_classes: List of Major Class indices.
        :param minor_classes: List of Minor Class indices.
        """
        self.classes = classes
        self.batch_size = batch_size
        self.major_classes = major_classes
        self.minor_classes = minor_classes

        # 인덱스를 Major와 Minor로 분리
        self.major_indices = [i for i, cls in enumerate(classes) if cls in major_classes]
        self.minor_indices = [i for i, cls in enumerate(classes) if cls in minor_classes]

        # 각각 배치 생성
        self.major_batches = self._create_batches(self.major_indices)
        self.minor_batches = self._create_batches(self.minor_indices)

        # Major와 Minor 배치를 합치고 순서를 랜덤하게 셔플
        self.all_batches = self.major_batches + self.minor_batches
        random.shuffle(self.all_batches)

    def _create_batches(self, indices):
        """
        Create batches from a list of indices.
        """
        random.shuffle(indices)
        return [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)
                if len(indices[i:i + self.batch_size]) == self.batch_size]

    def __iter__(self):
        """
        Yield batches.
        """
        for batch in self.all_batches:
            yield batch

    def __len__(self):
        return len(self.all_batches)

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=True  # No shuffle
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        major_classes = []
        minor_classes = []
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        class_counts = {}
        for name in class_names:
            class_counts[name] = class_counts.get(name, 0) + 1
        sorted_classes = {
        x: i for i, (x, _) in enumerate(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))
        }
        classes = [sorted_classes[x] for x in class_names]

        index_of_classes_sorted = list(sorted_classes.values())
        major_classes = [x for x in index_of_classes_sorted[:len(index_of_classes_sorted)//2]]
        minor_classes = [x for x in index_of_classes_sorted[len(index_of_classes_sorted)//2 :len(index_of_classes_sorted) ]]
        
        print("Major classes : ",major_classes)
        print("minor classes  : ",minor_classes)


    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )

    sampler = MajorMinorSampler(
        dataset.local_classes, batch_size, major_classes, minor_classes
    )

    loader = DataLoader(
            dataset,batch_sampler=sampler, num_workers=1
        )
    # else:
    #     loader = DataLoader(
    #         dataset, batch_sampler=sampler, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    #     )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict
