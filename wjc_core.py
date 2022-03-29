import os
import torch
import argparse
import numpy as np
import PIL.Image as Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim, autograd
from torchvision.transforms import transforms
from dataset import Train_Dataset, Validation_Dataset, Test_Dataset, Single_Train_Dataset
import skimage.io as io
import shutil
import stable_seed

stable_seed.setup_seed()

threshold = 0.5  # 二分类阈值
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

l2_regularizer_weight = 0.001
lr = 0.001
penalty_anneal_iters = 300
penalty_weight = 0.01


def makedir(new_path):
    folder = os.path.exists(new_path)
    if not folder:
        os.makedirs(new_path)
    else:
        shutil.rmtree(new_path)
        os.makedirs(new_path)


def init_work_space(args):
    makedir('./' + args.project_name + '/results')
    makedir(args.ckpt)
    makedir('./' + args.project_name + '/runs')


def train_model(args, writer, model, criterion, optimizer, dataload, regular=''):
    save_epoch, best_val_acc, best_val_mIoU = 0, -0.1, -0.1
    for epoch in range(args.epoch):
        print('Epoch {}/{}'.format(epoch, args.epoch - 1))
        print('-' * 10)
        epoch_loss = 0
        epoch_correct_pixels, epoch_total_pixels = [], []
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs).to(device)
            del inputs
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate accuracy
            predicted = outputs.detach().cpu().numpy()
            predicted[predicted >= threshold] = 1
            predicted[predicted < threshold] = 0
            correct = (predicted == labels.detach().cpu().numpy()).sum()
            del predicted
            pixel_num = 1.0
            for i in range(len(labels.size())):
                pixel_num *= labels.size()[i]

            epoch_correct_pixels.append(correct)
            epoch_total_pixels.append(pixel_num)
            epoch_loss += float(loss.item())
            del labels
            del loss
        val_accuracy, val_mIoU = validation(args, model, method='train')
        epoch_loss = epoch_loss / step
        epoch_train_accuracy = np.mean(epoch_correct_pixels) / np.mean(epoch_total_pixels)
        print(
            "epoch %d loss:%0.3f train accuracy:%0.3f val accuracy:%0.3f val mIoU:%0.3f best_val_acc:%0.4f best_mIoU:%0.4f" % (
                epoch, epoch_loss, epoch_train_accuracy, val_accuracy, val_mIoU, best_val_acc, best_val_mIoU))
        writer.add_scalar('loss', epoch_loss / step, global_step=epoch)
        writer.add_scalar('train accuracy', epoch_train_accuracy, global_step=epoch)
        writer.add_scalar('validated accuracy', val_accuracy, global_step=epoch)
        writer.add_scalars('accuracy/group',
                           {'train_accuracy': epoch_train_accuracy, 'validated accuracy': val_accuracy},
                           global_step=epoch)
        if best_val_acc < val_accuracy:
            # save_epoch = epoch
            # torch.save(model, args.ckpt + '/' + args.model + '.pth')
            best_val_acc = val_accuracy
        if best_val_mIoU < val_mIoU:
            save_epoch = epoch
            torch.save(model, args.ckpt + '/' + args.model + '.pth')
            best_val_mIoU = val_mIoU
    print("Model:", args.model)
    print("Dataset:", args.data_file)
    print("Best epoch is" + str(save_epoch))
    print("Best val acc is " + str(best_val_acc))
    print("Best val mIoU is " + str(best_val_mIoU))
    torch.cuda.empty_cache()
    return model


def mean_accuracy(outputs, labels):
    predicted = outputs.detach().cpu().numpy()
    predicted[predicted >= threshold] = 1
    predicted[predicted < threshold] = 0
    correct = (predicted == labels.detach().cpu().numpy()).sum()

    pixel_num = 1.0
    for i in range(len(labels.size())):
        pixel_num *= labels.size()[i]
    epoch_train_accuracy = np.mean(correct) / np.mean(pixel_num)
    return epoch_train_accuracy
    # debug code
    # a = torch.rand([10, 1, 20, 20])
    # b = torch.randint(low=0, high=2, size=(10, 1, 20, 20))
    # print(mean_accuracy(a, b)) # =0.5


def meanIoU(imgPredict, imgLabel, numClass=2):
    # ref:https://blog.csdn.net/sinat_29047129/article/details/103642140
    imgLabel = imgLabel.cpu()
    imgPredict[imgPredict >= threshold] = 1
    imgPredict[imgPredict < threshold] = 0

    def genConfusionMatrix(numClass, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < numClass)
        label = numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=numClass ** 2)  # 核心代码
        confusionMatrix = count.reshape(numClass, numClass)
        return confusionMatrix

    confusionMatrix = genConfusionMatrix(numClass, imgPredict, imgLabel)
    # Intersection = TP Union = TP + FP + FN
    # IoU = TP / (TP + FP + FN)
    intersection = np.diag(confusionMatrix)  # 取对角元素的值，返回列表
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(
        confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
    IoU = intersection / union  # 返回列表，其值为各个类别的IoU
    mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
    return mIoU


def penalty(logits, y, criterion=nn.BCELoss()):
    # torch.nan_to_num(mid, nan=1e-8, posinf=1.0 - 1e-8)
    scale = torch.tensor(1.).requires_grad_()
    loss = criterion(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[
        0]  # reference https://blog.csdn.net/qq_36556893/article/details/91982925
    return torch.sum(grad ** 2)


def train_IRM_model(args, writer, model, criterion, optimizer, env_dataloaders, regular=''):
    # def penalty(logits, y):
    #     scale = torch.tensor(1.).to(device).requires_grad_()
    #     loss = criterion(logits * scale, y)
    #     grad = autograd.grad(loss, [scale], create_graph=True)[
    #         0]  # reference https://blog.csdn.net/qq_36556893/article/details/91982925
    #     return torch.sum(grad ** 2)
    global l2_regularizer_weight
    global lr
    global penalty_anneal_iters
    global penalty_weight
    save_epoch, best_val_acc = 0, -0.1
    for epoch in range(args.epoch):
        print('Epoch {}/{}'.format(epoch, args.epoch - 1))
        print('-' * 10)
        optimizer.zero_grad()
        envs = []
        for env_dataloader in env_dataloaders:
            env = {'nll': [], 'acc': [], 'penalty': []}
            for x, y in env_dataloader:
                inputs = x  # .half()
                labels = y  # .half()
                outputs = model(inputs.to(device)).cpu()
                torch.nan_to_num(outputs, nan=1e-8, posinf=1.0 - 1e-8)
                torch.nan_to_num(labels, nan=1e-8, posinf=1.0 - 1e-8)
                env['nll'].append(criterion(outputs, labels))
                env['acc'].append(mean_accuracy(outputs, labels))
                env['penalty'].append(penalty(outputs, labels))
                del inputs, labels, outputs
                torch.cuda.empty_cache()
            envs.append(env)
        mid_train_nll, mid_train_acc, mid_train_penalty = [], [], []
        for i_env in range(len(env_dataloaders) - 1):
            mid_train_nll.extend(envs[i_env]['nll'])
            mid_train_acc.extend(envs[i_env]['acc'])
            mid_train_penalty.extend(envs[i_env]['penalty'])
        print('mid_train_penalty', mid_train_penalty)
        train_nll = torch.stack(mid_train_nll).mean()
        train_acc = float(np.mean(mid_train_acc))
        train_penalty = torch.stack(mid_train_penalty).mean()

        weight_norm = torch.tensor(0.).to(device)
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone().to(device)
        loss += l2_regularizer_weight * weight_norm
        # penalty_weight = (penalty_weight if epoch >= penalty_anneal_iters else 1.0)
        # loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        loss.backward()
        optimizer.step()

        val_accuracy, val_mIoU = validation(args, model, method='train')
        print("epoch %d loss:%0.3f train accuracy:%0.3f val accuracy:%0.3f train_penalty:%0.4f  best_val_acc:%0.4f" % (
            epoch, float(loss.item()), train_acc, val_accuracy, penalty_weight * train_penalty, best_val_acc))
        writer.add_scalar('loss', train_nll, global_step=epoch)
        writer.add_scalar('train accuracy', train_acc, global_step=epoch)
        writer.add_scalar('validated accuracy', val_accuracy, global_step=epoch)
        writer.add_scalars('accuracy/group',
                           {'train_accuracy': train_acc, 'validated accuracy': val_accuracy},
                           global_step=epoch)
        if best_val_acc < val_accuracy:
            save_epoch = epoch
            torch.save(model, args.ckpt + '/' + args.model + '.pth')
            best_val_acc = val_accuracy
    print("Model:", args.model)
    print("Dataset:", args.data_file)
    print("Best epoch is" + str(save_epoch))
    print("Best val acc is " + str(best_val_acc))
    torch.cuda.empty_cache()
    return model


# 训练模型
def train(args, writer, model, regular=''):
    # args == IRM
    # generate 4 single_train datasets as envs
    # define new loss function
    # model.half()
    if args.loss == "IRM":
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()  # nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), )
        env_dataloaders = []
        for i in range(args.total_folds):
            if i == args.k_fold:
                continue
            mid_env_dataset = Single_Train_Dataset(args.data_file, args, transform=x_transforms,
                                                   target_transform=y_transforms, k_single_set=i)
            mid_env_dataloader = DataLoader(mid_env_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
            print('mid_env_dataloader', len(mid_env_dataloader))
            env_dataloaders.append(mid_env_dataloader)
        train_IRM_model(args, writer, model, criterion, optimizer, env_dataloaders, regular)
    else:
        model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), )
        liver_dataset = Train_Dataset(args.data_file, args, transform=x_transforms, target_transform=y_transforms)
        dataloaders = DataLoader(liver_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        train_model(args, writer, model, criterion, optimizer, dataloaders, regular)


# 用于测试模型在有image有label的数据中的表现
def validation(args, model, print_each=False, method='train'):
    liver_dataset = Validation_Dataset(args.data_file, args, transform=x_transforms, target_transform=y_transforms)  #
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    if method == 'train':
        dataloaders = DataLoader(liver_dataset, batch_size=8)
    model.eval()
    epoch_correct_pixels, epoch_total_pixels, epoch_acc, epoch_mIoU = [], [], [], []
    with torch.no_grad():
        for x, y, x_path in dataloaders:
            inputs = x.to(device)
            labels = y.to(device)
            predicted = model(inputs).detach().cpu().numpy()
            predicted[predicted >= threshold] = 1
            predicted[predicted < threshold] = 0
            correct = (predicted == labels.detach().cpu().numpy()).sum()
            pixel_num = 1.0
            for i in range(len(labels.size())):
                pixel_num *= labels.size()[i]
            epoch_correct_pixels.append(correct)
            epoch_total_pixels.append(pixel_num)
            epoch_mIoU.append(meanIoU(predicted, labels))
            if print_each:
                print(x_path, 'acc', correct / pixel_num)
                mid_x_path = x_path
                epoch_acc.append(correct / pixel_num)
        if print_each:
            print('\nepoch_acc', epoch_acc, '\nepoch_mIoU', epoch_mIoU)
    return np.mean(epoch_correct_pixels) / np.mean(epoch_total_pixels), np.mean(epoch_mIoU)


# 用于测试只有image但没有label的数据
def test(args, save_gray=False, manual=False, weight_path=''):
    model = None
    if not manual:
        model = torch.load(args.ckpt + '/' + args.model + '.pth', map_location='cpu')
    if manual:
        model = torch.load(weight_path, map_location='cpu')  # use certain model weight.

    liver_dataset = Test_Dataset(args.data_file, transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)

    model.eval()
    with torch.no_grad():
        for x, pic_name_i in dataloaders:
            pic_name_i = pic_name_i[0]
            mid_x = torch.squeeze(x).numpy()
            if len(mid_x.shape) == 2:
                io.imsave(args.project_name + "/results/" + pic_name_i.split('.')[0] + "_x.png", mid_x)
            elif len(mid_x.shape) == 3:
                mid_x_image = np.array(mid_x[0])
                # io.imsave(args.project_name + "/results/" + pic_name_i.split('.')[0] + "_x.png", mid_x)
            predict = model(x)
            predict = torch.squeeze(predict).detach().numpy()
            if save_gray:
                io.imsave(args.project_name + "/results/" + pic_name_i.split('.')[0] + "_gray_pre.png", predict)

            predict[predict >= threshold] = 1
            predict[predict < threshold] = 0
            io.imsave(args.project_name + "/results/" + pic_name_i.split('.')[0] + "_label_pre.png", predict)


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def model_forward_visualization(image_path, weight_path, model_name=''):
    """输入一张测试图像和训练好的模型权重，可视化每一步卷积的结果"""
    model = torch.load(weight_path, map_location='cpu')  # load trained model

    save_output = SaveOutput()  # register hooks for each layer
    hook_handles, k1, k2 = [], 0, 0
    for layer in model.modules():
        k1 += 1
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            k2 += 1
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)

    x = x_transforms(Image.open(image_path).convert('L').resize(size=(512, 512))).unsqueeze(0)
    print(x, x.dtype)
    y = model(x)

    def module_output_to_numpy(tensor):
        return tensor.detach().to('cpu').numpy()

    for layer_idx in range(len(save_output.outputs)):
        images = module_output_to_numpy(save_output.outputs[layer_idx])
        # 这里的0代表读取output里第一个卷积层的输出

        print(type(images))
        print(images.shape)
        mid_1 = images.shape[1]
        mid_idx = 0
        while mid_idx < mid_1:
            # mid_idx is the index of feature
            with plt.style.context("seaborn-white"):
                plt.figure(frameon=False)
            for idx in range(64):
                # idx is the index of subplot
                if mid_idx == mid_1:
                    break
                plt.subplot(8, 8, idx + 1)
                plt.imshow(images[0, mid_idx])
                mid_idx += 1
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            plt.savefig(
                './model_visualization/' + model_name + '/layer_' + str(layer_idx) + '_mid_' + str(mid_idx) + '.png')
            plt.cla()
            plt.close('all')


def model_print(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
