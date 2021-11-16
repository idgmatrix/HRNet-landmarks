import cv2
from utils_inference import get_lmks_by_img, get_model_by_name, get_preds, decode_preds, crop
from utils_landmarks import show_landmarks, get_five_landmarks_from_net, alignment_orig


def draw_landmarks(img, lmks, color, line_width):

    for i in range(0, 16):
        x1, y1 = int(lmks[i][0]), int(lmks[i][1])
        x2, y2 = int(lmks[i + 1][0]), int(lmks[i + 1][1])
        cv2.line(img, (x1, y1), (x2, y2), color, line_width)

    # right eyebrow
    for i in range(17, 21):
        x1, y1 = int(lmks[i][0]), int(lmks[i][1])
        x2, y2 = int(lmks[i + 1][0]), int(lmks[i + 1][1])
        cv2.line(img, (x1, y1), (x2, y2), color, line_width)
    # left eyebrow
    for i in range(22, 26):
        x1, y1 = int(lmks[i][0]), int(lmks[i][1])
        x2, y2 = int(lmks[i + 1][0]), int(lmks[i + 1][1])
        cv2.line(img, (x1, y1), (x2, y2), color, line_width)

    # nose
    for i in range(27, 35):
        x1, y1 = int(lmks[i][0]), int(lmks[i][1])
        x2, y2 = int(lmks[i + 1][0]), int(lmks[i + 1][1])
        cv2.line(img, (x1, y1), (x2, y2), color, line_width)
    x0, y0 = int(lmks[30][0]), int(lmks[30][1])
    cv2.line(img, (x0, y0), (x2, y2), color, line_width)

    # right eye
    x0, y0 = int(lmks[36][0]), int(lmks[36][1])
    for i in range(36, 41):
        x1, y1 = int(lmks[i][0]), int(lmks[i][1])
        x2, y2 = int(lmks[i + 1][0]), int(lmks[i + 1][1])
        cv2.line(img, (x1, y1), (x2, y2), color, line_width)
    cv2.line(img, (x0, y0), (x2, y2), color, line_width)
    # left eye
    x0, y0 = int(lmks[42][0]), int(lmks[42][1])
    for i in range(42, 47):
        x1, y1 = int(lmks[i][0]), int(lmks[i][1])
        x2, y2 = int(lmks[i + 1][0]), int(lmks[i + 1][1])
        cv2.line(img, (x1, y1), (x2, y2), color, line_width)
    cv2.line(img, (x0, y0), (x2, y2), color, line_width)

    # lips
    x0, y0 = int(lmks[48][0]), int(lmks[48][1])
    for i in range(48, 59):
        x1, y1 = int(lmks[i][0]), int(lmks[i][1])
        x2, y2 = int(lmks[i + 1][0]), int(lmks[i + 1][1])
        cv2.line(img, (x1, y1), (x2, y2), color, line_width)
    cv2.line(img, (x0, y0), (x2, y2), color, line_width)
    # mouth
    x0, y0 = int(lmks[60][0]), int(lmks[60][1])
    for i in range(60, 67):
        x1, y1 = int(lmks[i][0]), int(lmks[i][1])
        x2, y2 = int(lmks[i + 1][0]), int(lmks[i + 1][1])
        cv2.line(img, (x1, y1), (x2, y2), color, line_width)
    cv2.line(img, (x0, y0), (x2, y2), color, line_width)


model = get_model_by_name('300W', device='cuda')
#img = cv2.imread('./images/chester.jpg')
#lmks = get_lmks_by_img(model, img)
#show_landmarks(img, lmks)

cam = cv2.VideoCapture(0)
while True:
    # webcam
    success, img = cam.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    img = cv2.flip(img, 1)

    lmks = get_lmks_by_img(model, img)
    #print(lmks)
    for i in range(68):
        cv2.circle(img, (int(lmks[i][0]), int(lmks[i][1])), 2, (0, 255, 0))
    draw_landmarks(img, lmks, (0, 255, 0), 2)

    cv2.imshow('Facial Landmark Detection', img)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cam.release()

