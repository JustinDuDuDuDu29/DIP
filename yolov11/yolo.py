import os
import glob

from ultralytics import YOLO


def main(folder_path : str):

        image_folder = glob.glob(os.path.join(folder_path + "*.png"))
        print(len(image_folder))

        for i in range(len(image_folder)):
                model = YOLO("./model/yolo11x.pt")
                result = model.predict(
                        source=image_folder[i],
                        mode="predict",
                        save=True,
                        device="cpu"
                )

        print("done")

if __name__ == "__main__":
        folder_path = "../enhance_nk/"
        main(folder_path)