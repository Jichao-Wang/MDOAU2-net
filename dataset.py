from torch.utils.data import Dataset
import PIL.Image as Image
import os
import stable_seed

stable_seed.setup_seed()
picture_shape = 128


def make_dataset(root, dataset_usage='train', k=0, total_folds=0):
    data_set = []
    imgs, labels, n_image, n_label = [], [], 0, 0
    if dataset_usage == 'train':
        # 组织用于k-fold交叉验证的训练集
        for j in range(total_folds):
            if j == k:
                continue
            mid_image_dir = os.listdir(root + '/train/image/' + str(j))
            mid_label_dir = os.listdir(root + '/train/label/' + str(j))
            mid_image_dir.sort()
            mid_label_dir.sort()
            n_image = len(mid_image_dir)
            n_label = len(mid_label_dir)
            if n_image != n_label:
                print("Error: The number of images and labels are not equal. Please check the dataset!")
            for i in range(n_image):
                imgs.append(
                    os.path.join(root + '/train/image/' + str(j), mid_image_dir[i]))
            for i in range(n_label):
                labels.append(
                    os.path.join(root + '/train/label/' + str(j), mid_label_dir[i]))
        for i in range(len(imgs)):
            data_set.append((imgs[i], labels[i]))
    elif dataset_usage == 'val':
        # 组织验证集  有label有image但是未用于训练的为验证样本
        # 用于非交叉验证的训练集组织方式，也适用于IRM算法中的单个env建立
        for j in range(total_folds):
            if j != k:
                continue
            mid_image_dir = os.listdir(root + '/train/image/' + str(j))
            mid_label_dir = os.listdir(root + '/train/label/' + str(j))
            mid_image_dir.sort()
            mid_label_dir.sort()
            n_image = len(mid_image_dir)
            n_label = len(mid_label_dir)
            if n_image != n_label:
                print("Error: The number of images and labels are not equal. Please check the dataset!")
            for i in range(n_image):
                imgs.append(
                    os.path.join(root + '/train/image/' + str(j), mid_image_dir[i]))
            for i in range(n_label):
                labels.append(
                    os.path.join(root + '/train/label/' + str(j), mid_label_dir[i]))
        for i in range(len(imgs)):
            data_set.append((imgs[i], labels[i]))
    elif dataset_usage == 'test':
        # 组织测试集  没有label的为测试样本
        mid_image_dir = os.listdir(root + '/test/image')
        mid_image_dir.sort()
        for i in range(len(mid_image_dir)):
            data_set.append(os.path.join(root + '/test/image', mid_image_dir[i]))
    return data_set


class Single_Train_Dataset(Dataset):
    def __init__(self, root, args, transform=None, target_transform=None, k_single_set=0):
        imgs = make_dataset(root, dataset_usage='val', k=k_single_set, total_folds=args.total_folds)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB').resize(size=(picture_shape, picture_shape))
        img_y = Image.open(y_path).convert('L').resize(size=(picture_shape, picture_shape))
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)


class Train_Dataset(Dataset):
    def __init__(self, root, args, transform=None, target_transform=None):
        imgs = make_dataset(root, dataset_usage='train', k=args.k_fold, total_folds=args.total_folds)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB').resize(size=(picture_shape, picture_shape))
        img_y = Image.open(y_path).convert('L').resize(size=(picture_shape, picture_shape))
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)


class Validation_Dataset(Dataset):
    def __init__(self, root, args, transform=None, target_transform=None):
        imgs = make_dataset(root, dataset_usage='val', k=args.k_fold, total_folds=args.total_folds)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB').resize(size=(picture_shape, picture_shape))
        img_y = Image.open(y_path).convert('L').resize(size=(picture_shape, picture_shape))
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y, x_path

    def __len__(self):
        return len(self.imgs)


class Test_Dataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root, dataset_usage='test')
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB').resize(size=(picture_shape, picture_shape))
        if self.transform is not None:
            img_x = self.transform(img_x)
        # print(x_path) # extract test picture name/number
        pic_name = x_path.split(sep='/')[-1].split(sep='\\')[-1]
        print(type(pic_name), pic_name)
        return img_x, pic_name

    def __len__(self):
        return len(self.imgs)


# def make_dataset(root, dataset_usage='train'):
#     data_set = []
#     if dataset_usage == 'train':
#         imgs, labels = [], []
#         train_image_dir = os.listdir(root + '/train/image')
#         train_label_dir = os.listdir(root + '/train/label')
#         train_image_dir.sort()
#         train_label_dir.sort()
#         n_image = len(train_image_dir)
#         n_label = len(train_label_dir)
#         for i in range(n_image):
#             imgs.append(os.path.join(root + '/train/image', train_image_dir[i]))
#         for i in range(n_label):
#             labels.append(os.path.join(root + '/train/label', train_label_dir[i]))
#         for i in range(len(imgs)):
#             data_set.append((imgs[i], labels[i]))
#     elif dataset_usage == 'val':
#         imgs, labels = [], []
#         val_image_dir = os.listdir(root + '/val/image')
#         val_label_dir = os.listdir(root + '/val/label')
#         val_image_dir.sort()
#         val_label_dir.sort()
#         n_image = len(val_image_dir)
#         n_label = len(val_label_dir)
#         for i in range(n_image):
#             imgs.append(os.path.join(root + '/val/image', val_image_dir[i]))
#         for i in range(n_label):
#             labels.append(os.path.join(root + '/val/label', val_label_dir[i]))
#         for i in range(len(imgs)):
#             data_set.append((imgs[i], labels[i]))
#     elif dataset_usage == 'test':
#         for i in range(len(os.listdir(root + '/test'))):
#             data_set.append(os.path.join(root + '/test', os.listdir(root + '/test')[i]))
#
#     return data_set
