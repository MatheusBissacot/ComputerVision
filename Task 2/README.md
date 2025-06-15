# Dependencies

The program was developed with Python 3.12. To run the project, you need to install the following dependencies:

```bash
pip install torch torchvision
pip install pillow
pip install numpy
pip install scikit-learn
pip install tqdm
```


# How to run:

 - **Download Models**
    Download the best models using this [link](https://drive.google.com/drive/folders/1aFbZSyAQH90F6GMMkB4ji3Br-8F3JtZB?usp=sharing). Place each model in the same folder as the Python file.

 - **Evaluate Results**
    To evaluate the results, execute the following command:

    ```bash
    python piece_detection.py <input.json> --model {best,yolo}
    ```

    - If the `--model` argument is not provided or is set to `best`, the script will use our best-performing model.

    - If the `--model` argument is set to any value other than `best`, the script will use our second-best performing model.

 - **Other models**
    Additional models used in this approach can be found using this [link](https://drive.google.com/drive/folders/1AoCPdrP-sQaQ5-RGTlmFc9fT5ysqES36?usp=sharing)