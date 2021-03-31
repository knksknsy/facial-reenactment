import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize(data, mean, std):
    data = (data - mean) / std
    return data


def denormalize(data, mean, std):
    data = (data * std) + mean
    return data


def extract_frames(video):
    cap = cv2.VideoCapture(video)

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

    if isinstance(output_res, int):
        image = np.zeros((output_res, output_res, channels), dtype=np.float32)
    elif isinstance(output_res, tuple):
        image = np.zeros((output_res[0], output_res[1], channels), dtype=np.float32)

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
        plt.imshow(np.zeros((output_res, output_res, channels)))
    elif isinstance(output_res, tuple):
        plt.imshow(np.zeros((output_res[0], output_res[1], channels)))
        
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
