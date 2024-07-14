# Parking Slot Detection using OpenCV, CNN, and Keras
This project aims to detect parking slots and identify empty parking spots using image processing techniques and machine learning models. The project utilizes OpenCV for image processing, Convolutional Neural Networks (CNN) for classification, and Keras as the deep learning framework.

## Installation
### Prerequisites
- Python 3.x
- OpenCV
- TensorFlow
- Keras
- NumPy
- Scikit-learn

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/soumyathatipamula/parking_slot_detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd parking_slot_detection
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Image Processing
The `ImageProcessing.py` script processes an image to detect contours and draw bounding boxes around potential cars.
```bash
python ImageProcessing.py
```
### Making Parking Spot Coordinates
The `parkingspotcoordinate.py` script allows users to manually mark parking spots on an image by drawing rectangles with the mouse.
```bash
python parkingspotcoordinate.py
```
### Train the Model
The `emptyparkingspotdetectionmodel.py` script trains a CNN model to classify parking spots as empty or occupied.
```bash
python emptyparkingspotdetectionmodel.py
```
### Detect Empty Parking Spots
The `emptyparkingspotdetection.py` script uses the trained model to detect empty parking spots in a given image.
```bash
python emptyparkingspotdetection.py
```

## Project Structure
`ImageProcessing.py`: Script for detecting and drawing bounding boxes around cars.
`parkingspotcoordinate.py`: Script for marking parking spot coordinates manually.
`emptyparkingspotdetectionmodel.py`: Script for training the CNN model.
`emptyparkingspotdetection.py`: Script for detecting empty parking spots using the trained model.

## Features
- Detects and draws bounding boxes around cars in an image.
- Allows manual marking of parking spots.
- Trains a CNN model to classify parking spots as empty or occupied.
- Detects empty parking spots in a given image.

