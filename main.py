import time
import torch
import wjc_core
import argparse
from tensorboardX import SummaryWriter
from attention_unet import AttU_Net
from segnet import SegNet
from unet import Unet
from Unet_plus_plus import NestedUNet
from Res_net import ResNet50, ResNet101
from MDOAU_net import MDOAU_net
from new_model import New_model1, New_model2, Offset_NestedUNet
from Deeplab_v3_plus import DeepLabv3_plus
from MDOAU2_net import MDOAU2_net_1, MDOAU2_net_2, MDOAU2_net_3, MDOAU2_net_4
import stable_seed

stable_seed.setup_seed()

if __name__ == '__main__':
    model, name = None, 'None'
    parse = argparse.ArgumentParser()
    parse.add_argument("--project_name", type=str, default=name)
    parse.add_argument("--data_file", type=str, default="d2")  # data
    parse.add_argument("--total_folds", type=int, default=3)
    parse.add_argument("--k_fold", type=int, default=0)
    parse.add_argument("--model", type=str, default=name)  # model
    parse.add_argument("--input_ch", type=int, help="number of input channels", default=1)
    parse.add_argument("--output_ch", type=int, help="number of output channels", default=1)
    parse.add_argument("--batch_size", type=int, default=4)  # train params
    parse.add_argument("--epoch", type=int, default=20)
    parse.add_argument("--loss", type=str, default="")  # IRM train strategy. "" or "IRM"
    parse.add_argument("--dynamic_learn_rate", type=str, default="")  # ReduceLROnPlateau
    parse.add_argument("--re", type=str, default="0")
    parse.add_argument("--ckpt", type=str, help="the path of model weight file",
                       default="./" + name + "/weights")  # save path
    args = parse.parse_args()

    project_name = args.data_file + "_" + args.model + "_k" + str(args.k_fold) + "_" + str(
        args.total_folds) + "_in" + str(args.input_ch) + "_batch" + str(args.batch_size) + "_epoch" + str(
        args.epoch)
    if len(args.dynamic_learn_rate) > 0:
        project_name = project_name + "_rate" + args.dynamic_learn_rate
    project_name = project_name + "_re" + args.re
    parse.set_defaults(project_name=project_name)
    parse.set_defaults(ckpt="./" + project_name + "/weights")
    args = parse.parse_args()
    print('*' * 10, "project name:", args.project_name)
    if "AttU_Net" == args.model:
        print("AttU_Net")
        model = AttU_Net(args.input_ch, args.output_ch)
    elif "NestedUNet" == args.model:
        print("NestedUNet")
        model = NestedUNet(args.input_ch, args.output_ch)
    elif "SegNet" == args.model:
        print("SegNet")
        model = SegNet(args.input_ch, args.output_ch)
    elif "ResNet50" == args.model:
        print("ResNet50")
        model = ResNet50(args.input_ch, args.output_ch)
    elif "ResNet100" == args.model:
        print("ResNet100")
        model = ResNet50(args.input_ch, args.output_ch)
    elif "Unet" == args.model:
        print("Unet")
        model = Unet(args.input_ch, args.output_ch)
    elif "New_model1" == args.model:
        print("New_model1")
        model = New_model1(args.input_ch, args.output_ch)
    elif "New_model2" == args.model:
        print("New_model2")
        model = New_model2(args.input_ch, args.output_ch)
    elif "NestedUNet_IRM" == args.model:
        print("NestedUNet_IRM")
        model = New_model1(args.input_ch, args.output_ch)
    elif "MDOAU_net" == args.model:
        print("MDOAU_net")
        model = MDOAU_net(args.input_ch, args.output_ch)
    elif "Offset_NestedUNet" == args.model:
        print("Offset_NestedUNet")
        model = Offset_NestedUNet(args.input_ch, args.output_ch)
    elif "DeepLabv3_plus" == args.model:
        print("DeepLabv3_plus")
        model = DeepLabv3_plus(args.input_ch, args.output_ch)
    elif "MDOAU2_net_1" == args.model:
        print("MDOAU2_net_1")
        model = MDOAU2_net_1(3, 1)
    elif "MDOAU2_net_2" == args.model:
        print("MDOAU2_net_2")
        model = MDOAU2_net_2(3, 1)
    elif "MDOAU2_net_3" == args.model:
        print("MDOAU2_net_3")
        model = MDOAU2_net_3(3, 1)
    elif "MDOAU2_net_4" == args.model:
        print("MDOAU2_net_4")
        model = MDOAU2_net_4(3, 1)
    elif "MDOAU_net_superpixel" == args.model:
        print("MDOAU_net")
        model = MDOAU_net(3, 1)

    # Prepare a space for saving trained model and predicted results.
    wjc_core.init_work_space(args)

    # Train a model.
    start_time = time.time()
    writer = SummaryWriter('./' + args.project_name + '/runs')
    print("Start training at", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    wjc_core.train(args, writer, model)
    writer.close()
    end_time = time.time()
    print("Training cost %.3f" % ((end_time - start_time) / 3600), " hours")
    print("Finish training at", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # # Test a model.
    # start_time = time.time()
    # # test the model trained
    wjc_core.test(args)
    # # or test a certain model
    # wjc_core.test(args, save_gray=True, manual=True, weight_path='./weights/MDOAU_net.pth')
    # end_time = time.time()
    # print("Testing cost %.3f" % ((end_time - start_time) / 60), " minutes")

    # Print the validation accuracy of the MODAU-net model. *You can change the pth file.
    # print(wjc_core.validation(args, torch.load('./weights/NestedUNet.pth'), print_each=True, method=''))
    # Visualize feature maps with an input image and a certain trained model.
    # wjc_core.model_forward_visualization(image_path="./data/train/image/8.png",
    #                                      weight_path="./weights/Attention_Unet.pth")
    # Print parameter number of each models.
    print(wjc_core.model_print(model))

