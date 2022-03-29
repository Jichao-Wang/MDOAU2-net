import cv2
import os
import stable_seed

stable_seed.setup_seed()


# transform black-red label images to black-white label mode
def transform_label_format():
    png_path = "./d2/train/label/4/"
    png_names = os.listdir(png_path)
    print(len(png_names), "pictures")
    for i in range(len(png_names)):
        image = cv2.imread(png_path + png_names[i])
        for j in range(image.shape[0]):
            for k in range(image.shape[1]):
                if image[j, k, 2] >= 127:
                    image[j, k] = [0, 0, 0]
                else:
                    image[j, k] = [255, 255, 255]
        cv2.imwrite(png_path + png_names[i], image)


# transform_label_format()
