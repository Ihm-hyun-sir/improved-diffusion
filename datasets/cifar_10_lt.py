import numpy as np
import torchvision.datasets as datasets
import os
import tempfile
from tqdm.auto import tqdm
import pickle
import shutil

class ImbalanceCIFAR10(datasets.CIFAR100):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        if train:
            self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.num_per_cls_dict = dict()
            self.gen_imbalanced_data(self.img_num_list)

        self.root = root

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    
    def get_cls_name_list(self):
        meta_loc = self.root +"/cifar-10-batches-py/batches.meta"
        with open(meta_loc, "rb") as f:
            meta_data = pickle.load(f)
        fine_label_names = meta_data['label_names']
        return fine_label_names
        


class ImbalanceCIFAR100(datasets.CIFAR100):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(ImbalanceCIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.num_per_cls_dict = dict()
        self.gen_imbalanced_data(self.img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    


if __name__ == "__main__":

    #dataset = ImbalanceCIFAR10( root="/home2/ihmhyunsir/WorkingSpace/lab_proj/diffusion_imbalance/improved-diffusion/dataset", train= True, download=True)

    for split in ["train", "test"]:
        out_dir = f"cifar_lt_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("downloading...")
        current_dir = os.getcwd()
        #with os.getcwd() as tmp_dir:
        dataset = ImbalanceCIFAR10( root=current_dir, train= split == "train", download=True)
        CLASSES = dataset.get_cls_name_list()
        print("dumping images...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
            image.save(filename)
        
        #print(rf"downloading {split} set done! {"\n"} {len(dataset)} samples {"\n"} Classes : {CLASSES} {"\n"} Num of Classes : {dataset.img_num_list}")


    temp_dir = os.path.join(os.getcwd(), "cifar-10-batches-py")
    if os.path.exists(temp_dir):
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)  # 폴더와 내부 파일 삭제
    else:
        print(f"Temporary directory not found: {temp_dir}")

    
    temp_dir = os.path.join(os.getcwd(), "cifar-10-python.tar.gz")
    if os.path.exists(temp_dir):
        print(f"Cleaning up temporary directory: {temp_dir}")
        os.remove(temp_dir)  # 폴더와 내부 파일 삭제
    else:
        print(f"Temporary directory not found: {temp_dir}")
    

    #train [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
