
---

# NeuroFruity Image Classification

## Overview

**NeuroFruity Image Classification** is a deep learning project designed to classify various types of fruits using TensorFlow and Keras. The model is capable of recognizing six different classes of fruits in real-time, either from webcam input or screen capture. It uses a custom neural network architecture optimized with mixed precision for improved performance and memory efficiency.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.4 or higher
- OpenCV 4.5 or higher
- Joblib 1.0 or higher
- Matplotlib 3.3 or higher

### Setting Up the Environment

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/NeuroFruity.git
    cd NeuroFruity
    ```

2. **Create a Virtual Environment and Activate It:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset should consist of various fruit images organized in subdirectories by fruit type. The structure is as follows:

```
data_sets/NeuroFruity_v_2/
│
├── Apple/
│   ├── apple_01.jpg
│   ├── apple_02.jpg
│   └── ...
│
├── Banana/
│   ├── banana_01.jpg
│   ├── banana_02.jpg
│   └── ...
│
└── Orange/
    ├── orange_01.jpg
    ├── orange_02.jpg
    └── ...
```

Make sure to adjust the dataset paths in the code as necessary.

## Preprocessing

The images are preprocessed using OpenCV for resizing and cropping to the target size. The `process_image()` function performs the following steps:

1. Reads the image.
2. Resizes the image while maintaining the aspect ratio.
3. Crops the image center to the specified dimensions.
4. Converts the image to RGB format.

This step can be parallelized using Joblib for faster processing.

## Model Architecture

The model uses a custom convolutional neural network with the following key components:

- **Input Layer:** Accepts images of size 640x640 with 3 channels.
- **Convolutional Layers:** Multiple blocks with Conv2D, BatchNormalization, and activation functions.
- **Detection Heads:** Three separate heads to classify the fruits at different scales of the input image.
- **Mixed Precision:** Utilizes TensorFlow’s mixed precision policy for improved performance.

## Training

The model is trained using the Adam optimizer with a learning rate of 0.0004. The training is performed for 100 epochs with a batch size of 2. The training data is shuffled and cached for efficiency.

To start training the model:

```bash
python train.py
```

This will start the training process and save the best model checkpoint based on the validation loss.

## Evaluation

The model is evaluated on a validation set which is 20% of the total dataset. The performance metrics include:

- Accuracy
- Loss

The evaluation results are stored and can be visualized using Matplotlib.

## Results

The final model achieves the following results on the validation set:

- **Training Accuracy:** 99.04%
- **Validation Accuracy:** 98.9%
- **Training Loss:** 0.0034
- **Validation Loss:** 0.0097

These results are visualized in the form of plots for both accuracy and loss.

## Visualization

The training and validation accuracy and loss can be visualized using the following plots:

```python
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training loss')
plt.plot(epochs_range, val_loss, label='Validation loss')
plt.title('Training and Validation loss')
plt.show()
```

## Usage

1. **Run Real-Time Object Detection:**

   To start real-time object detection using your webcam, run the following command:

   ```bash
   python main.py
   ```

2. **Use Screen Capture:**

   To use screen capture instead of a webcam, uncomment the corresponding lines in the `main.py` script:

   ```python
   # screen = ImageGrab.grab(area)  # Reads RGB
   ```

   Define the area of the screen you want to capture in the `area` variable.

3. **Test with a Static Image:**

   To test the model with a static image, uncomment the line that loads an image:

   ```python
   # screen = cv2.imread("path/to/your/image.jpg")  # Reads in BGR
   ```

   Replace `"path/to/your/image.jpg"` with the path to your test image.

4. **Make Predictions:**

   To use the trained model for prediction, follow these steps:

   1. **Load the Model:**
      ```python
      from tensorflow.keras.models import load_model
      model = load_model('CheckPoints/your_model.h5')
      ```

   2. **Preprocess the Image:**
      ```python
      img = process_image('path_to_image.jpg')
      img = np.expand_dims(img, axis=0)  # Add batch dimension
      ```

   3. **Make a Prediction:**
      ```python
      prediction = model.predict(img)
      ```

   4. **Interpret the Result:**
      ```python
      print("Predicted class:", np.argmax(prediction))
      ```

## Troubleshooting

- **Error: Unable to access the camera.**
  - Make sure your webcam is connected and accessible.
  - Check if the correct device index is being used in `cv2.VideoCapture()`.

- **Model not found or failed to load.**
  - Ensure that the trained model is saved in the `CheckPoints` directory and the path is correctly specified in the `main.py` script.

## Contributing

We welcome contributions to this project! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to all contributors and the open-source community for providing tools and libraries that made this project possible.

---
