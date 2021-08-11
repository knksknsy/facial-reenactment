import os
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def select_random_frames(frames, num_frames):
    idx = np.random.choice(len(frames)-1, num_frames, replace=False)
    return [frames[sample] for sample in idx]


def sanitize_csv(csv_path: str, sort_col: str):
    df = pd.read_csv(csv_path, sep=',', header=0)
    df = df.drop_duplicates(keep=False)
    df = df.sort_values(by=[sort_col], ascending=True)
    df.to_csv(csv_path, sep=',', index=False)


def prune_videos(source, max_num_videos=None, max_bytes_videos=None):
    # Split large videos into chunks (files length)
    if max_num_videos is not None:
        for root, dirs, files in os.walk(source):
            if len(files) >= max_num_videos and len(dirs) <= 0:
                split_large_video(root, files, chunk_size=50)

    # Split large videos into chunks (files size)
    if max_bytes_videos is not None:
        for root, dirs, files in os.walk(source):
            if len(files) > 0 and len(dirs) <= 0:
                total_size = 0
                for f in files:
                    fp = os.path.join(root, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
                if total_size >= max_bytes_videos:
                    split_large_video(root, files, chunk_size=len(files)//3)


def split_large_video(source, files, chunk_size):
    files = sorted(files)
    chunks = divide_chunks(files, chunk_size)

    for i, chunk in enumerate(chunks):
        destination = source + f'_{i+1}'

        if not os.path.isdir(destination):
            os.makedirs(destination)
        
        for f in chunk:
            shutil.move(os.path.join(source, f), destination)
    #shutil.rmtree(source)


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i: i+n]


def contains_only_videos(files, extension='.mp4'):
        return len([x for x in files if os.path.splitext(x)[1] != extension]) == 0


def select_random_frames(frames, num_frames):
    idx = np.random.choice(len(frames)-1, num_frames, replace=False)
    return [frames[sample] for sample in idx]


def crop_face(frame, frames_total, fa, padding, padding_color, output_res, method='pyplot'):
    frame = cv2.copyMakeBorder(frame, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=padding_color)
    landmarks = fa.get_landmarks_from_image(frame)
    face_detected = landmarks is not None
    if not face_detected:
        id_x = np.random.randint(len(frames_total))
        frame = frames_total[id_x]
        return crop_face(frame, frames_total, fa, padding, padding_color, output_res, method)
    else:
        landmarks = landmarks[0]
        frame = crop_frame(frame, landmarks, output_res, padding, method=method)
        if frame is None:
            id_x = np.random.randint(len(frames_total))
            frame = frames_total[id_x]
            return crop_face(frame, frames_total, fa, padding, padding_color, output_res, method)
        else:
            return frame


def detect_face(frame, frames_total, fa, padding, padding_color, output_res, method='pyplot'):
    frame = cv2.copyMakeBorder(frame, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=padding_color)
    landmarks = fa.get_landmarks_from_image(frame)
    face_detected = landmarks is not None
    if not face_detected:
        id_x = np.random.randint(len(frames_total))
        frame = frames_total[id_x]
        return detect_face(frame, frames_total, fa, padding, padding_color, output_res, method)
    else:
        landmarks = landmarks[0]
        frame = crop_frame(frame, landmarks, output_res, padding, method=method)
        if frame is None:
            id_x = np.random.randint(len(frames_total))
            frame = frames_total[id_x]
            return detect_face(frame, frames_total, fa, padding, padding_color, output_res, method)
        else:
            landmarks = fa.get_landmarks_from_image(frame)
            face_detected = landmarks is not None
            if not face_detected:
                id_x = np.random.randint(len(frames_total))
                frame = frames_total[id_x]
                return detect_face(frame, frames_total, fa, padding, padding_color, output_res, method)
            else:
                return frame, landmarks[0]


def detect_crop_face(shape, frame, padding, face_alignment, crop=True):
    if crop:
        frame = cv2.copyMakeBorder(frame, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    landmarks = face_alignment.get_landmarks_from_image(frame)
    face_detected = landmarks is not None
    if not face_detected:
        return None, None
    else:
        landmarks = landmarks[0]
        frame, rmin, rmax, cmin, cmax = get_bounding_box(frame, landmarks, (shape,shape), padding, method='cv2', crop=crop)

        if frame is None:
            return None, None
                
        return frame, rmin, rmax, cmin, cmax


def extract_frames(video, n_frames=None):
    cap = cv2.VideoCapture(video)

    if n_frames is None:    
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = np.empty((n_frames, h, w, 3), np.dtype('uint8'))

    fn, ret = 0, True
    while fn < n_frames and ret:
        ret, img = cap.read()
        frames[fn] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fn += 1

    cap.release()
    return frames


def crop_frame(frame, landmarks, dimension, padding, method='pyplot'):
    heatmap = plot_landmarks(landmarks=landmarks, landmark_type='boundary', channels=3, output_res=(frame.shape[0], frame.shape[1]), input_res=(frame.shape[0], frame.shape[1]), method=method)

    rows = np.any(heatmap, axis=1)
    cols = np.any(heatmap, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    frame = frame[rmin-padding:rmax+padding, cmin-padding:cmax+padding]
    
    try:
        frame = cv2.resize(frame, dimension)
    except Exception as e:
        return None

    return frame


def get_bounding_box(frame, landmarks, dimension, padding, method='cv2', crop=True):
    heatmap = plot_landmarks(landmarks=landmarks, landmark_type='boundary', channels=3, output_res=(frame.shape[0], frame.shape[1]), input_res=(frame.shape[0], frame.shape[1]), method=method)

    rows = np.any(heatmap, axis=1)
    cols = np.any(heatmap, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    if crop:
        frame = frame[rmin-padding:rmax+padding, cmin-padding:cmax+padding]
    
    try:
        frame = cv2.resize(frame, dimension)
    except Exception as e:
        return None

    return frame, rmin, rmax, cmin, cmax


def plot_landmarks(landmarks, landmark_type, channels, output_res, input_res, method='pyplot'):
    if method == 'pyplot':
        return pyplot_landmarks(landmarks, landmark_type, channels, output_res, input_res)
    elif method == 'cv2':
        return cv2_landmarks(landmarks, landmark_type, channels, output_res, input_res)


def cv2_landmarks(landmarks, landmark_type, channels, output_res, input_res):
    if isinstance(output_res, int) and isinstance(input_res, int):
        ratio = input_res / output_res
        landmarks = landmarks / ratio
    elif isinstance(output_res, tuple) and isinstance(input_res, tuple):
        ratio_y = input_res[0] / output_res[0]
        ratio_x = input_res[1] / output_res[1]
        ratio = ratio_x / ratio_y
        landmarks_y = landmarks[:,0] / ratio_y
        landmarks_x = landmarks[:,1] / ratio_x
        landmarks = np.stack((landmarks_y, landmarks_x), axis=1)

    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    white = (255,255,255)

    groups = [
        # Head
        [np.arange(0,17,1), white if channels == 1 else red],
        # Eyebrows
        [np.arange(17,22,1), white if channels == 1 else green],
        [np.arange(22,27,1), white if channels == 1 else green],
        # Nose
        [np.arange(27,31,1), white if channels == 1 else blue],
        [np.arange(31,36,1), white if channels == 1 else blue],
        # Eyes
        [list(np.arange(36,42,1))+[36], white if channels == 1 else green],
        [list(np.arange(42,48,1))+[42], white if channels == 1 else green],
        # Mouth
        [list(np.arange(48,60,1))+[48], white if channels == 1 else blue],
        # Inner-Mouth
        [list(np.arange(60,68,1))+[60], white if channels == 1 else blue]
    ]

    # black background
    if isinstance(output_res, int):
        image = np.zeros((output_res, output_res, channels), dtype=np.float32)
    elif isinstance(output_res, tuple):
        image = np.zeros((output_res[0], output_res[1], channels), dtype=np.float32)
    
    # # white background
    # if isinstance(output_res, int):
    #     image = np.ones((output_res, output_res, channels), dtype=np.float32)
    #     image = image * 255.0
    # elif isinstance(output_res, tuple):
    #     image = np.ones((output_res[0], output_res[1], channels), dtype=np.float32)
    #     image = image * 255.0

    for g in groups:
        for i in range(len(g[0]) - 1):
            if landmark_type == 'boundary':
                s = int(landmarks[g[0][i]][0]), int(landmarks[g[0][i]][1])
                e = int(landmarks[g[0][i+1]][0]), int(landmarks[g[0][i+1]][1])
                cv2.line(image, s, e, g[1], 1)
            elif landmark_type == 'keypoint':
                c = int(landmarks[g[0][i]][0]), int(landmarks[g[0][i]][1])
                cv2.circle(image, c, 1, g[1], -1)
    return image


def pyplot_landmarks(landmarks, landmark_type, channels, output_res, input_res):
    dpi = 100

    if isinstance(output_res, int) and isinstance(input_res, int):
        ratio = input_res / output_res
        landmarks = landmarks / ratio
        fig = plt.figure(figsize=(output_res / dpi, output_res / dpi), dpi=dpi)
    elif isinstance(output_res, tuple) and isinstance(input_res, tuple):
        ratio_y = input_res[0] / output_res[0]
        ratio_x = input_res[1] / output_res[1]
        ratio = ratio_x / ratio_y
        landmarks_y = landmarks[:,0] / ratio_y
        landmarks_x = landmarks[:,1] / ratio_x
        landmarks = np.stack((landmarks_y, landmarks_x), axis=1)
        fig = plt.figure(figsize=(output_res[0] / dpi, output_res[1] / dpi), dpi=dpi)

    ax = fig.add_subplot(111)
    ax.axis('off')

    if isinstance(output_res, int):
        if channels >= 3:    
            plt.imshow(np.zeros((output_res, output_res, channels)))
        else:
            plt.imshow(np.zeros((output_res, output_res)))
    elif isinstance(output_res, tuple):
        if channels >= 3:
            plt.imshow(np.zeros((output_res[0], output_res[1], channels)))
        else:
            plt.imshow(np.zeros((output_res[0], output_res[1])))
        
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    red = (1,0,0)
    green = (0,1,0)
    blue = (0,0,1)
    white = (1,1,1)

    if landmark_type == 'boundary':
        marker_size = 3*72./dpi/ratio
        # Head
        ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', lw=marker_size, color=white if channels == 1 else red)
        # Eyebrows
        ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', lw=marker_size, color=white if channels == 1 else green)
        ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', lw=marker_size, color=white if channels == 1 else green)
        # Nose
        ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', lw=marker_size, color=white if channels == 1 else blue)
        ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', lw=marker_size, color=white if channels == 1 else blue)
        # Eyes
        ax.fill(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', lw=marker_size, color=white if channels == 1 else green, fill=False)
        ax.fill(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', lw=marker_size, color=white if channels == 1 else green, fill=False)
        # Mouth
        ax.fill(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', lw=marker_size, color=white if channels == 1 else blue, fill=False)
        # Inner-Mouth
        ax.fill(landmarks[60:68, 0], landmarks[60:68, 1], linestyle='-', lw=marker_size, color=white if channels == 1 else blue, fill=False)
    elif landmark_type == 'keypoint':
        marker_size = (3*72./dpi/ratio)**2
        # Head
        ax.scatter(landmarks[0:17, 0], landmarks[0:17, 1], marker='o', s=marker_size, color=white if channels == 1 else red)
        # Eyebrows
        ax.scatter(landmarks[17:22, 0], landmarks[17:22, 1], marker='o', s=marker_size, color=white if channels == 1 else green)
        ax.scatter(landmarks[22:27, 0], landmarks[22:27, 1], marker='o', s=marker_size, color=white if channels == 1 else green)
        # Nose
        ax.scatter(landmarks[27:31, 0], landmarks[27:31, 1], marker='o', s=marker_size, color=white if channels == 1 else blue)
        ax.scatter(landmarks[31:36, 0], landmarks[31:36, 1], marker='o', s=marker_size, color=white if channels == 1 else blue)
        # Eyes
        ax.scatter(landmarks[36:42, 0], landmarks[36:42, 1], marker='o', s=marker_size, color=white if channels == 1 else green)
        ax.scatter(landmarks[42:48, 0], landmarks[42:48, 1], marker='o', s=marker_size, color=white if channels == 1 else green)
        # Mouth
        ax.scatter(landmarks[48:60, 0], landmarks[48:60, 1], marker='o', s=marker_size, color=white if channels == 1 else blue)
        # Inner-Mouth
        ax.scatter(landmarks[60:68, 0], landmarks[60:68, 1], marker='o', s=marker_size, color=white if channels == 1 else blue)

    fig.canvas.draw()
    buffer = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8)
    canvas_shape = fig.canvas.get_width_height()
    data = np.reshape(buffer, (canvas_shape[0], canvas_shape[1], 3))
    plt.close(fig)

    if channels == 1:
        data = data[:,:,1]

    return data


def plot_mask(landmarks, output_res, input_res):
    if isinstance(output_res, int) and isinstance(input_res, int):
        ratio = input_res / output_res
        landmarks = landmarks / ratio
    elif isinstance(output_res, tuple) and isinstance(input_res, tuple):
        ratio_y = input_res[0] / output_res[0]
        ratio_x = input_res[1] / output_res[1]
        ratio = ratio_x / ratio_y
        landmarks_y = landmarks[:,0] / ratio_y
        landmarks_x = landmarks[:,1] / ratio_x
        landmarks = np.stack((landmarks_y, landmarks_x), axis=1)

    white = (255,255,255)
    head_pts = list(np.arange(0,17,1))+[0]

    if isinstance(output_res, int):
        image = np.zeros((output_res, output_res, 1), dtype=np.float32)
    elif isinstance(output_res, tuple):
        image = np.zeros((output_res[0], output_res[1], 1), dtype=np.float32)

    shape_pts = []
    for i in range(len(head_pts) - 1):
        pt = [int(landmarks[head_pts[i]][0]), int(landmarks[head_pts[i]][1])]
        shape_pts.append(pt)

    shape_pts = np.asarray(shape_pts, dtype=np.int32)[None,:,:]
    image = cv2.fillPoly(image, shape_pts, white)
    image = cv2.GaussianBlur(image, (5,5), 0)

    return image
