import os
import numpy as np
import cv2
from skimage import io, color, feature
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Function to load images from a folder
def load_images_from_folder(folder, size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img = io.imread(os.path.join(folder, filename))
            img = color.rgb2gray(img)  # Convert to grayscale
            img = resize(img, size)  # Resize to a consistent size
            images.append(img)
            labels.append(filename.split("_")[0])  # Assuming filenames are 'label_001.jpg'
    return np.array(images), np.array(labels)


# Extract HOG features from the images
def extract_hog_features(images):
    hog_features = []
    for img in images:
        features, _ = feature.hog(img, visualize=True, block_norm='L2-Hys')
        hog_features.append(features)
    return np.array(hog_features)


# Train an SVM classifier
def train_svm(images, labels):
    hog_features = extract_hog_features(images)
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(hog_features, labels)
    return clf


# Recognize faces in a single image using the trained classifier
def recognize_face(image, clf):
    img = color.rgb2gray(image)
    img = resize(img, (64, 64))
    hog_features = extract_hog_features([img])
    label = clf.predict(hog_features)
    return label[0]


# Load your dataset (train the model)
images, labels = load_images_from_folder('images')
hog_features = extract_hog_features(images)
predicted_label = "none"

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

# Train the SVM model
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# Open a video stream (use webcam by default)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from video stream
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame and convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces using OpenCV's built-in Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) from the frame
        face_roi = frame[y:y + h, x:x + w]

        # Resize the face region to 64x64 and convert to RGB for HOG extraction
        face_resized = cv2.resize(face_roi, (64, 64))

        # Predict the label using the trained classifier
        predicted_label = recognize_face(face_resized, clf)

        # Draw rectangle around face and label it
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame with detected faces and predictions
    cv2.imshow("Face Recognition", frame)
    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
print(f"Predicted label: {predicted_label}")
if predicted_label == "kyle" or predicted_label == "andrew":
    print("True")
else:
    print("False")
