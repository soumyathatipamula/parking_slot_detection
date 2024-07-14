# Parking Slot Detection using OpenCV, CNN, and Keras

## Description
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
    git clone https://github.com/yourusername/parking-slot-detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd parking-slot-detection
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
