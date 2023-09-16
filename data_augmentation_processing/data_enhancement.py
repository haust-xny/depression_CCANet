'''
    Using flipping for data augmentation
'''
import os
import cv2
import numpy as np

# flip horizontal
def Horizontal(image):
    return cv2.flip(image, 1, dst=None) # Horizontal Mirror

# flip vertical
def Vertical(image):
    return cv2.flip(image, 0, dst=None) # mirror vertically

if __name__ == '__main__':
    # from_root = r"processed/validate/Freeform"
    from_root = r"processed/validate/Northwind"
    # save_root = r"enhance_processed/validate/Freeform"
    save_root = r"enhance_processed/validate/Northwind"

    # threshold = 200

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

            cv2.imencode('.jpg', img_i)[1].tofile(os.path.join(save_path, file_i[:-4] + "_original.jpg"))    # Save Picture
            # print('save successfully')

            # if len(files) < threshold:

            img_horizontal = Horizontal(img_i)
            cv2.imencode('.jpg', img_horizontal)[1].tofile(os.path.join(save_path, file_i[:-4] + "_horizontal.jpg"))    # Save Picture
            # print('save successfully')

            img_vertical = Vertical(img_i)
            cv2.imencode('.jpg', img_vertical)[1].tofile(os.path.join(save_path, file_i[:-4] + "_vertical.jpg")) # Save Picture
    print('save successfully')

            # else:
            #     pass
