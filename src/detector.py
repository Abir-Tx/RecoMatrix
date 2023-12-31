import cv2, time, os, tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(20)


class Detector:
    def __init__(self):
        pass

    def readClasses(self, classesFilePath):
        with open(classesFilePath, "r") as f:
            self.classesList = f.read().splitlines()
        # colors list
        self.colorList = np.random.uniform(
            low=0, high=255, size=(len(self.classesList), 3)
        )
        # print(len(self.classesList), len(self.colorsList))

    def downloadModel(self, modelURL):
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[: fileName.index(".")]

        # print(fileName)
        # print(self.modelName)

        self.cacheDir = "../pretrained_models"  # where Detector.py file located

        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(
            fname=fileName,
            origin=modelURL,
            cache_dir=self.cacheDir,
            cache_subdir="checkpoints",
            extract=True,
        )

    def loadModel(self):
        print("Loading Model " + self.modelName)

        # tf.keras.backend_clear_session()
        tf.compat.v1.reset_default_graph()
        self.model = tf.saved_model.load(
            os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model")
        )

        print("Model " + self.modelName + "loaded successfully...")

    def createBoundingBox(self, image, threshold=0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]

        detections = self.model(inputTensor)
        bboxs = detections["detection_boxes"][0].numpy()
        classIndexes = detections["detection_classes"][0].numpy().astype(np.int32)
        classScores = detections["detection_scores"][0].numpy()

        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(
            bboxs,
            classScores,
            max_output_size=50,
            iou_threshold=threshold,
            score_threshold=threshold,
        )

        bboxIdx = bboxIdx.numpy()  # Convert to a numpy array

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100 * classScores[i])
                classIndex = classIndexes[i]

                if classIndex < len(self.classesList):  # Check if classIndex is valid
                    classLabelText = self.classesList[classIndex].upper()
                    classColor = self.colorList[classIndex]

                    displayText = "{}: {}%".format(classLabelText, classConfidence)

                    ymin, xmin, ymax, xmax = bbox
                    xmin, xmax, ymin, ymax = (
                        xmin * imW,
                        xmax * imW,
                        ymin * imH,
                        ymax * imH,
                    )
                    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                    cv2.rectangle(
                        image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1
                    )
                    cv2.putText(
                        image,
                        displayText,
                        (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,  # Use correct font style
                        1,
                        classColor,
                        2,
                    )
        return image

    def predictImage(self, imagePath, threshold=0.5):
        image = cv2.imread(imagePath)

        bboxImage = self.createBoundingBox(image, threshold)

        # Saving the detected image
        file_name = imagePath.split("/")[-1]
        imageName = file_name.split(".")[0]
        cv2.imwrite("../output/" + imageName + "_detected.jpg", bboxImage)

        # Show the detected image
        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predictImagesInFolder(self, folderPath, threshold=0.5, show=False):
        # Create the output folder if it doesn't exist
        output_folder = "../output/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Loop through all files in the specified folder
        for filename in os.listdir(folderPath):
            if filename.endswith(".jpg") or filename.endswith(
                ".png"
            ):  # You can add more image extensions if needed
                imagePath = os.path.join(folderPath, filename)

                # Perform the image prediction for each image in the folder
                image = cv2.imread(imagePath)
                bboxImage = self.createBoundingBox(image, threshold)

                # Saving the detected image
                imageName = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, imageName + "_detected.jpg")
                cv2.imwrite(output_path, bboxImage)

                # Show the detected image if specified
                if show:
                    cv2.imshow("Result", bboxImage)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print("Info: All the detected images has been saved\n")
                else:
                    print(
                        "\033[91m Info: \033[0m Detected image named \033[92m"
                        + imageName
                        + "\033[0m  has been saved in \033[92m"
                        + output_folder
                        + "\033[0m"
                    )

    def predictVideo(self, videoPath, threshold=0.5):
        cap = cv2.VideoCapture(videoPath)

        if cap.isOpened == False:
            print("Error opening file...")
            return
        (success, image) = cap.read()

        startTime = 0

        while success:
            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime

            bboxImage = self.createBoundingBox(image, threshold)
            cv2.putText(
                bboxImage,
                "FPS: " + str(int(fps)),
                (20, 70),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Result", bboxImage)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success, image) = cap.read()
        cv2.destroyAllWindows()
