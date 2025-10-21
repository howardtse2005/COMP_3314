## Getting Started
1.  **Clone the repository:**
2.  **Install dependencies:**
    The code was tested on Python 3.10 in Conda Environment
    ```bash
    conda create -n 3314 python=3.10
    conda activate 3314
    conda install conda-forge::pytorch conda-forge::opencv conda-forge::tqdm anaconda::numpy conda-forge::einops
    ```
3.  **Train on your own images**
    Put all your image-ground truth pairs in the data directory.
    In the config.py, choose the preferred architecture (unet, hnet).
    Then, run
    ```bash
    python3 train.py
    ```
4. **Test on your own images**
    Put all your image-ground truth pairs in the data directory.
    In the config.py, choose the preferred architecture (unet, hnet). Then, put the path to the pretrained model (.pth file)
    Then, run
    ```bash
    python3 test.py
    ```
