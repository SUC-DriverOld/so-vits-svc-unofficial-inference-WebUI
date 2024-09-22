<div align="center">

# so-vits-svc-unofficial-inference-WebUI

This is an unofficial WebUI for [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc), it only supports inference.

</div>

## Usage

1. Clone this repository.
2. Install requirements. Make sure your pip version is lower than 24.1.

    ```bash
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
    ```

3. Download nessessary pretrained model files and put them into `pretrain` folder. You can see the [Official Repository's README](https://github.com/svc-develop-team/so-vits-svc/tree/4.1-Stable?tab=readme-ov-file#-pre-trained-model-files) for more information.
4. Put your trained model files into any folder like below. 

    ```
    YOUR_MODEL_FOLDER
    ├── model_folder_1
    │   ├── config.json             (must!)
    │   ├── model.pth               (must!)
    │   ├── diffusion.pt            (optional)
    │   ├── diffusion.yaml          (optional)
    │   ├── feature_and_index.pkl   (optional)
    │   └── kmeans_10000.pt         (optional)
    └── model_folder_2
        ├── config.json
        ├── model.pth
        └── ...
    ```

    - `config.json` and `model.pth` are must-have files.
    - `diffusion.pt`, `diffusion.yaml`, `feature_and_index.pkl`, `kmeans_10000.pt` are optional files.
    - You can change the model name to anything you like (do not change the postfix), but the kmeans_model must have `kmeans` in its name.

5. Run the WebUI.
    ```bash
    python webUI.py -m YOUR_MODEL_FOLDER_PATH
    ```

## CLI

```bash
usage: webUI.py [-h] [-m MODEL_FOLDER] [-n SERVER_NAME] [-p SERVER_PORT] [-s] [-c]

optional arguments:
  -h, --help                                    show this help message and exit
  -m MODEL_FOLDER, --model_folder MODEL_FOLDER  Path to the model folder
  -n SERVER_NAME, --server_name SERVER_NAME     Server IP name
  -p SERVER_PORT, --server_port SERVER_PORT     Server port
  -s, --share                                   Open share link
  -c, --clean_cache                             Clean cache before launch
```