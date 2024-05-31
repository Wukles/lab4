import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import numpy as np


class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")

        self.image_frame = tk.Frame(root)
        self.image_frame.pack()

        self.original_label = Label(self.image_frame)
        self.original_label.pack(side="left", padx=10, pady=10)

        self.processed_label = Label(self.image_frame)
        self.processed_label.pack(side="left", padx=10, pady=10)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side="bottom", pady=10)

        self.load_button = Button(self.button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side="left", padx=5)

        self.process_button = Button(self.button_frame, text="Process Image", command=self.process_image)
        self.process_button.pack(side="left", padx=5)

        self.contour_button = Button(self.button_frame, text="Find Contours", command=self.find_contours)
        self.contour_button.pack(side="left", padx=5)

        self.primitive_button = Button(self.button_frame, text="Find Primitives", command=self.find_primitives)
        self.primitive_button.pack(side="left", padx=5)

        self.original_image = None
        self.processed_image = None
        self.binarized_image = None
        self.contours = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image, self.original_label)

    def process_image(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            self.processed_image = blurred_image
            self.display_image(self.processed_image, self.processed_label)

    def find_contours(self):
        if self.processed_image is not None:
            _, self.binarized_image = cv2.threshold(self.processed_image, 80, 255, cv2.THRESH_BINARY)
            self.contours, _ = cv2.findContours(self.binarized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            contour_image = np.zeros((self.binarized_image.shape[0], self.binarized_image.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(contour_image, self.contours, -1, (0, 255, 0), 1)

            self.display_image(contour_image, self.processed_label)

    def find_primitives(self):
        if self.contours is not None:
            threshold_value = simpledialog.askinteger("Input", "threshold", minvalue=0,
                                                      maxvalue=255)
            min_area = simpledialog.askfloat("Input", "minArea")
            min_distance = simpledialog.askinteger("Input", "minDistance",
                                                   minvalue=1)
            ac_threshold = simpledialog.askinteger("Input", "acTreshold",
                                                   minvalue=1)

            _, binarized_image = cv2.threshold(self.processed_image, threshold_value, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binarized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            contour_image = np.zeros((binarized_image.shape[0], binarized_image.shape[1], 3), dtype=np.uint8)

            for contour in contours:
                approx_contour = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
                if cv2.contourArea(approx_contour) > min_area:
                    if len(approx_contour) == 3:
                        cv2.drawContours(contour_image, [approx_contour], -1, (0, 255, 255), 2)
                    elif len(approx_contour) == 4 and self.is_rectangle(approx_contour):
                        rect = cv2.minAreaRect(approx_contour)
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)  # Use np.intp instead of np.int0
                        cv2.drawContours(contour_image, [box], 0, (0, 255, 0), 2)

            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
            circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, 1, min_distance,
                                       param1=100, param2=ac_threshold, minRadius=0, maxRadius=0)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(contour_image, (i[0], i[1]), i[2], (255, 0, 0), 2)

            self.display_image(contour_image, self.processed_label)

    def is_rectangle(self, contour):
        contour = contour.reshape(4, 2)
        angles = []
        for i in range(4):
            p1 = contour[i]
            p2 = contour[(i + 1) % 4]
            p3 = contour[(i + 2) % 4]
            angle = self.angle(p1, p2, p3)
            angles.append(angle)
        return all(80 <= angle <= 100 for angle in angles)

    def angle(self, pt1, pt2, pt3):
        vec1 = pt1 - pt2
        vec2 = pt3 - pt2
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return np.degrees(np.arccos(cos_angle))

    def display_image(self, image, label):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        label.configure(image=image)
        label.image = image


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()
