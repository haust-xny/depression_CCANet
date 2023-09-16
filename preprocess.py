import os
import pandas as pd
import cv2
from tqdm import tqdm
from mtcnn import MTCNN


def get_files(path):
    file_info = os.walk(path)
    file_list = []
    for r, d, f in file_info:
        file_list += f
    return file_list


def get_dirs(path):
    file_info = os.walk(path)
    dirs = []
    for d, r, f in file_info:
        dirs.append(d)
    return dirs[1:]


def generate_label_file():
    print('get label....')
    base_url = 'AVEC2014/label/DepressionLabels/'
    file_list = get_files(base_url)
    labels = []
    loader = tqdm(file_list)
    for file in loader:
        label = pd.read_csv(base_url + file, header=None)
        labels.append([file[:file.find('_Depression.csv')], label[0][0]])
        loader.set_description('file:{}'.format(file))
    pd.DataFrame(labels, columns=['file', 'label']).to_csv('../processed/label.csv', index=False)
    return labels


def generate_img(path, v_type, img_path):
    videos = get_files(path)
    loader = tqdm(videos)
    for video in loader:
        name = video[:5]
        save_path = img_path + v_type + '/' + name
        os.makedirs(save_path, exist_ok=True)
        cap = cv2.VideoCapture(path + video)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        gap = int(n_frames / 100)
        for i in range(n_frames):
            success, frame = cap.read()
            if success and i % gap == 0:
                cv2.imwrite(save_path + '/{}.jpg'.format(int(i / gap)), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                loader.set_description("data:{} type:{} video:{} frame:{}".format(path.split('/')[2], v_type, name, i))
        cap.release()

# 获取视频帧图片并保存
def get_img():
    print('get video frames....')
    train_f = 'AVEC2014/CCANet_train/Freeform/'
    train_n = 'AVEC2014/CCANet_train/Northwind/'
    test_f = 'AVEC2014/CCANet_test/Freeform/'
    test_n = 'AVEC2014/CCANet_test/Northwind/'
    validate_f = 'AVEC2014/dev/Freeform/'
    validate_n = 'AVEC2014/dev/Northwind/'
    dirs = [train_f, train_n, test_f, test_n, validate_f, validate_n]
    types = ['Freeform', 'Northwind', 'Freeform', 'Northwind', 'Freeform', 'Northwind']
    img_path = ['dataset_img/CCANet_train/', 'dataset_img/CCANet_train/', 'dataset_img/CCANet_test/', 'dataset_img/CCANet_test/', 'dataset_img/validate/', 'dataset_img/validate/']
    os.makedirs('dataset_img/train', exist_ok=True)
    os.makedirs('dataset_img/test', exist_ok=True)
    os.makedirs('dataset_img/validate', exist_ok=True)
    for i in range(6):
        generate_img(dirs[i], types[i], img_path[i])

# 使用MTCNN裁剪图片面部图像
def get_face():
    print('get frame faces....')
    detector = MTCNN()
    save_path = ['processed/CCANet_train/Freeform/', 'processed/CCANet_train/Northwind/', 'processed/CCANet_test/Freeform/',
                 'processed/CCANet_test/Northwind/', 'processed/validate/Freeform/', 'processed/validate/Northwind/']
    paths = ['dataset_img/CCANet_train/Freeform/', 'dataset_img/CCANet_train/Northwind/', 'dataset_img/CCANet_test/Freeform/', 'dataset_img/CCANet_test/Northwind/',
             'dataset_img/validate/Freeform/', 'dataset_img/validate/Northwind/']
    for index, path in enumerate(paths):
        dirs = get_dirs(path)
        loader = tqdm(dirs)
        for d in loader:
            os.makedirs(save_path[index] + d.split('/')[-1], exist_ok=True)
            files = get_files(d)
            for file in files:
                img_path = d + '/' + file
                s_path = save_path[index] + d.split('/')[-1] + '/' + file
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                print(img.shape)
                info = detector.detect_faces(img)
                if (len(info) > 0):
                    x, y, width, height = info[0]['box']
                    confidence = info[0]['confidence']
                    b, g, r = cv2.split(img)
                    img = cv2.merge([r, g, b])
                    print(img.shape)
                    img = img[y:y + height, x:x + width, :]
                    print(img.shape)
                    cv2.imwrite(s_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    loader.set_description('confidence:{:4f} dataset_img:{}'.format(confidence, img_path))


if __name__ == '__main__':
    os.makedirs('dataset_img', exist_ok=True)
    os.makedirs('../processed', exist_ok=True)
    os.makedirs('../processed/train', exist_ok=True)
    os.makedirs('../processed/test', exist_ok=True)
    os.makedirs('../processed/validate', exist_ok=True)
    label = generate_label_file()
    get_img()
    get_face()
