import cv2
from src.app import drawsi

cap = cv2.VideoCapture(0)

def main():
    obj = drawsi()
    try:
        Fmodel = obj.face_cal
        obj.main_fun(cap, Fmodel)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()