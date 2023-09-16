from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

# open colour image
def Horizontal(image):
    # plt.gcf().clear()
    image_raw = Image.open(image)
    image_gray = image_raw.convert('L')
    # image_gray.show()
    # plt.figure('sunflower')  # name
    plt.imshow(image_gray, cmap='gray')  # Color mapping
    plt.axis('off')
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.axis('off')  # Turn off grid lines
    # f = plt.gcf()
    return plt.gcf()
    # image_black_white.save('./images/black_white_sunflower.jpg')
    # plt.show()
    # return 0


def cv2_BGR2Gray(image):
    # ======================== Reading images =======================
    # Image path, relative path
    image_path = image
    # Read image in BGR format
    image = cv2.imread(image_path)

    # ============== Convert to grayscale image and display grayscale image ================
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


if __name__ == '__main__':
    # from_root = r"processed/validate/Freeform"
    # from_root = r"E:/AVEC2014/Depression/Depression/processed/CCANet_test/Freeform"
    # from_root = r"E:/AVEC2014/Depression/Depression/processed/CCANet_test/Northwind"
    # from_root = r"E:/AVEC2014/Depression/Depression/processed/CCANet_train/Freeform"
    # from_root = r"E:/AVEC2014/Depression/Depression/processed/CCANet_train/Northwind"
    # from_root = r"E:/AVEC2014/Depression/Depression/processed/validate/Freeform"
    from_root = r"E:/AVEC2014/Depression/Depression/processed/validate/Northwind"

    # save_root = r"enhance_processed/validate/Freeform"
    # save_root = r"E:/AVEC2014/Depression/Depression/Grayscale_processed/CCANet_test/Freeform"
    # save_root = r"E:/AVEC2014/Depression/Depression/Grayscale_processed/CCANet_test/Northwind"
    # save_root = r"E:/AVEC2014/Depression/Depression/Grayscale_processed/CCANet_train/Freeform"
    # save_root = r"E:/AVEC2014/Depression/Depression/Grayscale_processed/CCANet_train/Northwind"
    # save_root = r"E:/AVEC2014/Depression/Depression/Grayscale_processed/validate/Freeform"
    save_root = r"E:/AVEC2014/Depression/Depression/Grayscale_processed/validate/Northwind"


    for root, dirs, files in os.walk(from_root):# Path, folder name, file name
        print(len(files))
        for file_i in files:
            file_i_path = os.path.join(root, file_i)

            split = os.path.split(file_i_path)# Split [0] is the path, split [1] is the file name
            dir_loc = os.path.split(split[0])[1]# Name of the cut file
            save_path = os.path.join(save_root, dir_loc)# Splice the file name with the enhanced data folder and save the path

            # print(file_i_path)
            # print(save_path)

            if os.path.isdir(save_path) == False:
                os.makedirs(save_path)

            img_i = cv2.imdecode(np.fromfile(file_i_path, dtype=np.uint8), -1)   # Reading images

            cv2.imencode('.jpg', img_i)[1].tofile(os.path.join(save_path, file_i[:-4] + "_original.jpg"))

            # img_horizontal = Horizontal(file_i_path)
            img_horizontal = cv2_BGR2Gray(file_i_path)

            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_Grayscale.jpg"), img_horizontal)
    print('save successfully')