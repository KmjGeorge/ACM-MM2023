import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from deepface import DeepFace
from tqdm import tqdm
import os
import pandas as pd

backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
]


def show_pic(picture, title):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title)
    axs.imshow(picture, aspect="auto")
    plt.axis('off')
    plt.show(block=False)


def get_face(file_path, target_path):
    filename = os.path.split(file_path)[1]
    try:
        result = DeepFace.extract_faces(img_path=file_path, target_size=(96, 96),
                                        detector_backend=backends[4], align=True)
        face = result[0]['face']
        pil_image = Image.fromarray(np.uint8(face * 255))
        pil_image.save(os.path.join(target_path, filename.replace('.jpg', '_face.jpg')))
    except:
        return file_path
    return None


def no_face_mask(csv_path, target_path, num_frames=15):
    df = pd.read_csv(csv_path)
    for i in range(num_frames):
        loop = tqdm(df['0'])
        frame = 'frame_{}/'.format(i)
        for file in loop:
            filename = os.path.split(file)[1]
            array = np.zeros(shape=(96, 96, 3), dtype=np.uint8)
            pil_image = Image.fromarray(array)
            pil_image.save(os.path.join(target_path+frame, filename.replace('.jpg', '_face.jpg')))
            # orig_image = Image.open(file)
            # orig_image.save(os.path.join(target_path+frame, filename))


if __name__ == '__main__':
    Error = []
    for i in range(15):
        frame = 'frame_{}/'.format(i)
        folder = 'D:/Project/Pycharm/cav-mae-master/src/preprocess/sample_frames/test15/' + frame
        loop = tqdm(os.listdir(folder))
        for file in loop:
            error = get_face(os.path.join(folder, file), 'D:/Datasets/NextSpeaker/face_deepface15/test/' + frame)
            if error:
                Error.append(error)
            loop.set_postfix(Error=len(Error))
    df = pd.DataFrame(Error)
    df.to_csv('D:/github/ACMMM2023/dataset/facedetection_test15_deepface_error.csv', index=True)

    # no_face_mask('D:/github/ACMMM2023/dataset/facedetection_deepface_error.csv', 'D:/github/ACMMM2023/dataset/no_face/')
