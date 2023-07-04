import os.path
import cv2
import dlib
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL.Image import Resampling
from tqdm import tqdm
import pandas as pd
import face_recognition
from PIL import Image
from dataset.dataloader import show_pic
face_shape = (96, 96)
face_detector_model_path = 'mmod_human_face_detector.dat'
detector = dlib.cnn_face_detection_model_v1(face_detector_model_path)
# detector = dlib.get_frontal_face_detector()


def get_face_dlib(file_path, target_path, margin=0):
    filename = os.path.split(file_path)[1]
    img = dlib.load_rgb_image(file_path)
    # img = cv2.imread(file_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(img, 1)
    if not dets:
        return file_path
    rect = dets[0]    # 0: rect, 1:confidence
    left = rect.left()
    top = rect.top()
    right = rect.right()
    bottom = rect.bottom()
    cropped = img[top - margin // 2:bottom + margin // 2,
              left - margin // 2:right + margin // 2, :]
    aligned = cv2.resize(cropped, face_shape)
    # cv2.imshow('face', aligned)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(target_path, filename.replace('.jpg', '_face.jpg')), aligned)
    return None


def get_face(file_path, target_path):
    filename = os.path.split(file_path)[1]
    img = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=1, model='cnn')
    if not face_locations:
        return file_path
    for face_location in face_locations:
        top, right, bottom, left = face_location
        # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
        #                                                                                             right))
        face_image = img[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        resize_image = pil_image.resize(face_shape, Resampling.LANCZOS)
        resize_image.save(os.path.join(target_path, filename.replace('.jpg', '_face.jpg')))
    return None


if __name__ == "__main__":
    # print(dlib.DLIB_USE_CUDA)
    import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    ''' # 初步提取
    Error = []
    for i in range(5):
        frame = 'frame_{}/'.format(i)
        folder = 'D:/Project/Pycharm/cav-mae-master/src/preprocess/sample_frames/trainval/' + frame
        loop =  tqdm(os.listdir(folder))
        for file in loop:
            error = get_face(os.path.join(folder, file), 'D:/github/ACMMM2023/dataset/face_2/' + frame)
            if error:
                Error.append(error)
                # print(error)
                # img = torchvision.io.image.read_image(error)
                # img = torch.transpose(img, 0, 2)
                # img = torch.transpose(img, 0, 1)
                # show_pic(img.numpy(), os.path.split(error)[1])
            loop.set_postfix(Error=len(Error))
    df = pd.DataFrame(Error)
    df.to_csv('./facedetection2_error.csv', index=True)
    '''
    '''
    # 对提取失败的重新提取
    Error2 = []
    df = pd.read_csv('./facedetection2_error.csv')
    loop = tqdm(df['0'])
    for i in range(5):
        frame = 'frame_{}/'.format(i)
        for file in loop:
            error2 = get_face_dlib(file, 'D:/github/ACMMM2023/dataset/face_2_1/' + frame)
            if error2:
                Error2.append(error2)
            loop.set_postfix(Error2=len(Error2))
    df2 = pd.DataFrame(Error2)
    df.to_csv('./facedetection2_1_error.csv', index=True)
    '''