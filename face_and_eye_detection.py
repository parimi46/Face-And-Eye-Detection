import tempfile

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2 as cv
import base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from streamlit_option_menu import option_menu

with open(r'D:\OpenCv\APP_PLATE\style.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Define the HTML content with Tailwind CSS and Streamlit navigation bar
html_content = """
<link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" />
<script src="https://unpkg.com/react@16/umd/react.production.min.js"></script>
<script src="https://unpkg.com/react-dom@16/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/@material-ui/core/umd/material-ui.production.min.js"></script>
<script src="app.min.js"></script>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Wix+Madefor+Text:ital,wght@0,400..800;1,400..800&display=swap" rel="stylesheet">

<div class="flex justify-center items-center h-screen">
    <div class="text-center">
        <h1 class="text-4xl font-bold mb-8" style="background: linear-gradient(to right, red, yellow); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">FACE AND EYE DETECTION</h1>
        <div id="accordion" class="w-full md:w-1/2 mx-auto">
            <div class="card">
                <!-- Add your content inside this card -->
            </div>
        </div>
    </div>
</div>
"""

# Display the combined HTML content
components.html(html_content)

# Load Haar cascades for face and eye detection
face_cascade = cv.CascadeClassifier(r'D:\OpenCv\Haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(r'D:\OpenCv\Haarcascades\haarcascade_eye_tree_eyeglasses.xml')


def detect_face_and_eye(img):
    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around the detected faces and eyes
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

    return img


def download_image(image_data, filename='output_image.jpg'):
    b64_img = base64.b64encode(image_data).decode('utf-8')
    href = f'<a href="data:image/jpeg;base64,{b64_img}" download="{filename}">Download Output Image</a>'
    st.markdown(href, unsafe_allow_html=True)


# Define the sidebar options
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Image', 'Live', 'Video'],
                           icons=['house', 'image', 'camera'], menu_icon="cast", default_index=1)

# Execute selected option
if selected == "Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

    if uploaded_file is not None:
        # Read the uploaded file bytes
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        # Decode the bytes into a cv2 image
        img = cv.imdecode(file_bytes, 1)

        # Display the original image with styled caption
        st.image(img, channels="BGR", caption='Original image', width=500)

        if st.button('Detect Face and Eye', key='detect_button', type="primary"):
            img_with_detection = detect_face_and_eye(img)
            st.image(img_with_detection, channels="BGR", caption='Image with Face and Eye Detection', width=500)
            download_image(cv.imencode('.jpg', img_with_detection)[1].tobytes())


elif selected == "Live":

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_with_detection = detect_face_and_eye(img)
            return img_with_detection


    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)


if selected == "Video":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
            # Write video bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())

            # Open temporary file as video
        cap = cv.VideoCapture(temp_file.name)

            # Get video properties
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv.CAP_PROP_FPS))

            # Define the codec and create VideoWriter object
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

            # Read video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

                # Detect faces and eyes
            frame_with_detections = detect_face_and_eye(frame)

                # Write the frame with detections to the output video
            out.write(frame_with_detections)

                # Display the frame with detections
            #st.image(frame_with_detections, channels="BGR")

            # Release the VideoCapture and VideoWriter objects
        cap.release()
        out.release()

            # Offer download button for the processed video
        with open('output.avi', 'rb') as f:
            video_bytes = f.read()
            st.download_button(label="Download Processed Video", data=video_bytes, file_name="processed_video.avi", mime="video/avi")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
