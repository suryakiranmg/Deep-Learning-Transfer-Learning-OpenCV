import cv2

def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('Suki')
    while True:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

