from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
import pytesseract
import cv2
import numpy as np

class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = MDBoxLayout(orientation='vertical')
        self.image = Image()
        layout.add_widget(self.image)
        self.save_image_button = MDRaisedButton(
            text='Click to pic',
            pos_hint={'center_x': .5, 'center_y': .5},
            size_hint=(None, None)
        )
        self.save_image_button.bind(on_press=self.take_picture)
        layout.add_widget(self.save_image_button)
        self.add_widget(layout)
        
        # Initialize the camera
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0 / 30.0)

        # Load the pre-trained EAST text detector
        self.net = cv2.dnn.readNet(r"assets/frozen_east_text_detection.pb")

    def load_video(self, *args):
        ret, frame = self.capture.read()
        if not ret:
            return

        self.image_frame = frame.copy()  # Copy for later use without bounding boxes
        
        # Resize the frame to improve speed
        (H, W) = frame.shape[:2]
        newW, newH = (320, 320)
        rW = W / float(newW)
        rH = H / float(newH)
        resized_frame = cv2.resize(frame, (newW, newH))

        # Prepare the frame for the EAST model
        blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        (scores, geometry) = self.net.forward(output_layers)

        # Decode the detections and extract bounding boxes
        boxes = self.decode_predictions(scores, geometry, min_confidence=0.5)

        # Merge boxes to create one large bounding box
        if boxes:
            boxes_array = np.array(boxes)
            x_min = np.min(boxes_array[:, 0])
            y_min = np.min(boxes_array[:, 1])
            x_max = np.max(boxes_array[:, 2])
            y_max = np.max(boxes_array[:, 3])
            startX = int(x_min * rW)
            startY = int(y_min * rH)
            endX = int(x_max * rW)
            endY = int(y_max * rH)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

        # Update the Kivy Image widget with the frame containing the bounding box
        buffer = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def decode_predictions(self, scores, geometry, min_confidence):
        detections = []
        confidences = []

        for i in range(0, scores.shape[2]):
            for j in range(0, scores.shape[3]):
                if scores[0, 0, i, j] < min_confidence:
                    continue

                (offsetX, offsetY) = (j * 4.0, i * 4.0)
                angle = geometry[0, 4, i, j]
                cosA = np.cos(angle)
                sinA = np.sin(angle)
                h = geometry[0, 0, i, j] + geometry[0, 2, i, j]
                w = geometry[0, 1, i, j] + geometry[0, 3, i, j]
                endX = int(offsetX + (cosA * geometry[0, 1, i, j]) + (sinA * geometry[0, 2, i, j]))
                endY = int(offsetY - (sinA * geometry[0, 1, i, j]) + (cosA * geometry[0, 2, i, j]))
                startX = int(endX - w)
                startY = int(endY - h)

                detections.append((startX, startY, endX, endY))
                confidences.append(float(scores[0, 0, i, j]))

        return detections

    def take_picture(self, *args):
        myconfig = r"--psm 4 --oem 3"
        
        # Convert to grayscale and apply adaptive thresholding
        captured_gray = cv2.cvtColor(self.image_frame, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(captured_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 71, 18)
        
        # Perform OCR on the entire image
        text_data = pytesseract.image_to_string(adaptive_thresh, lang='eng', config=myconfig)

        # Switch to the text display screen and pass the text data
        self.manager.get_screen('text_screen').display_text(text_data)
        self.manager.current = 'text_screen'

class TextDisplayScreen(Screen):
    def display_text(self, text):
        self.clear_widgets()
        label = Label(text=text, halign='left', valign='top')
        self.add_widget(label)

class MainApp(MDApp):
    
    def build(self):
        self.screen_manager = ScreenManager()
        
        # Create and add the camera screen
        camera_screen = CameraScreen(name='camera_screen')
        self.screen_manager.add_widget(camera_screen)
        
        # Create and add the text display screen
        text_screen = TextDisplayScreen(name='text_screen')
        self.screen_manager.add_widget(text_screen)
        
        return self.screen_manager

if __name__ == '__main__':
    MainApp().run()