This is how you should structure your data models.


```bash
dataset/
    train/
        unripe/
            unripe_image1.jpg
            unripe_image2.jpg
            ...
        semi-ripe/
            semi_ripe_image1.jpg
            semi_ripe_image2.jpg
            ...
        ripe/
            ripe_image1.jpg
            ripe_image2.jpg
            ...
        overripe/
            overripe_image1.jpg
            overripe_image2.jpg
            ...
    test/
        unripe/
            unripe_image1.jpg
            unripe_image2.jpg
            ...
        semi-ripe/
            semi_ripe_image1.jpg
            semi_ripe_image2.jpg
            ...
        ripe/
            ripe_image1.jpg
            ripe_image2.jpg
            ...
        overripe/
            overripe_image1.jpg
            overripe_image2.jpg
            ...
```
This works best with NVIDIA GPUs that supports CUDA.

## Installation
- Install the required packages
  ```bash
  pip install -r requirements.txt
  ```
- Install CUDA and cuDNN if you have an NVIDIA GPU
  - [CUDA](https://developer.nvidia.com/cuda-downloads)
  - [cuDNN](https://developer.nvidia.com/cudnn)

## Usage
- Inorder to get accurate results, run trainer.py and strucutre your dataset as shown above
- To run the model on your own dataset, change the path in trainer.py to your dataset path
- To use the trained model replace the model variable as below
  ```python
    from keras.models import load_model

    # Load the saved model from the .h5 file
    model = load_model('path/to/model.h5')

    # Use the loaded model to make predictions
    predictions = model.predict(data)
    ```
