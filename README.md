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

### webUI.py

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

### inference_main.py

```bash
usage: inference_main.py [-h] [-m MODEL_PATH] [-c CONFIG_PATH] [-cl CLIP] [-n CLEAN_NAMES [CLEAN_NAMES ...]] [-t TRANS [TRANS ...]] [-s SPK_LIST [SPK_LIST ...]] [-a]
                         [-cm CLUSTER_MODEL_PATH] [-cr CLUSTER_INFER_RATIO] [-lg LINEAR_GRADIENT] [-f0p F0_PREDICTOR] [-eh] [-shd] [-usm] [-lea LOUDNESS_ENVELOPE_ADJUSTMENT] [-fr]
                         [-dm DIFFUSION_MODEL_PATH] [-dc DIFFUSION_CONFIG_PATH] [-ks K_STEP] [-se] [-od] [-sd SLICE_DB] [-d DEVICE] [-ns NOICE_SCALE] [-p PAD_SECONDS] [-wf WAV_FORMAT]       
                         [-lgr LINEAR_GRADIENT_RETAIN] [-eak ENHANCER_ADAPTIVE_KEY] [-ft F0_FILTER_THRESHOLD]

optional arguments:
  -h, --help                                                show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH                    模型路径
  -c CONFIG_PATH, --config_path CONFIG_PATH                 配置文件路径
  -cl CLIP, --clip CLIP                                     音频强制切片，默认0为自动切片，单位为秒/s
  -n CLEAN_NAMES [CLEAN_NAMES ...], --clean_names CLEAN_NAMES [CLEAN_NAMES ...]
                                                            wav文件名列表，放在raw文件夹下
  -t TRANS [TRANS ...], --trans TRANS [TRANS ...]           音高调整，支持正负（半音）
  -s SPK_LIST [SPK_LIST ...], --spk_list SPK_LIST [SPK_LIST ...]
                                                            合成目标说话人名称
  -a, --auto_predict_f0                                     语音转换自动预测音高，转换歌声时不要打开这个会严重跑调
  -cm CLUSTER_MODEL_PATH, --cluster_model_path CLUSTER_MODEL_PATH
                                                            聚类模型或特征检索索引路径，留空则自动设为各方案模型的默认路径，如果没有训练聚类或特征检索则随便填
  -cr CLUSTER_INFER_RATIO, --cluster_infer_ratio CLUSTER_INFER_RATIO
                                                            聚类方案或特征检索占比，范围0-1，若没有训练聚类模型或特征检索则默认0即可
  -lg LINEAR_GRADIENT, --linear_gradient LINEAR_GRADIENT    两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值0，单位为秒
  -f0p F0_PREDICTOR, --f0_predictor F0_PREDICTOR            选择F0预测器,可选择crepe,pm,dio,harvest,rmvpe,fcpe默认为pm(注意：crepe为原F0使用均值滤波器)
  -eh, --enhance                                            是否使用NSF_HIFIGAN增强器,该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭
  -shd, --shallow_diffusion                                 是否使用浅层扩散，使用后可解决一部分电音问题，默认关闭，该选项打开时，NSF_HIFIGAN增强器将会被禁止
  -usm, --use_spk_mix                                       是否使用角色融合
  -lea LOUDNESS_ENVELOPE_ADJUSTMENT, --loudness_envelope_adjustment LOUDNESS_ENVELOPE_ADJUSTMENT
                                                            输入源响度包络替换输出响度包络融合比例，越靠近1越使用输出响度包络
  -fr, --feature_retrieval                                  是否使用特征检索，如果使用聚类模型将被禁用，且cm与cr参数将会变成特征检索的索引路径与混合比例
  -dm DIFFUSION_MODEL_PATH, --diffusion_model_path DIFFUSION_MODEL_PATH
                                                            扩散模型路径
  -dc DIFFUSION_CONFIG_PATH, --diffusion_config_path DIFFUSION_CONFIG_PATH
                                                            扩散模型配置文件路径
  -ks K_STEP, --k_step K_STEP                               扩散步数，越大越接近扩散模型的结果，默认100
  -se, --second_encoding                                    二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差
  -od, --only_diffusion                                     纯扩散模式，该模式不会加载sovits模型，以扩散模型推理
  -sd SLICE_DB, --slice_db SLICE_DB                         默认-40，嘈杂的音频可以-30，干声保留呼吸可以-50
  -d DEVICE, --device DEVICE                                推理设备，None则为自动选择cpu和gpu
  -ns NOICE_SCALE, --noice_scale NOICE_SCALE                噪音级别，会影响咬字和音质，较为玄学
  -p PAD_SECONDS, --pad_seconds PAD_SECONDS                 推理音频pad秒数，由于未知原因开头结尾会有异响，pad一小段静音段后就不会出现
  -wf WAV_FORMAT, --wav_format WAV_FORMAT                   音频输出格式
  -lgr LINEAR_GRADIENT_RETAIN, --linear_gradient_retain LINEAR_GRADIENT_RETAIN
                                                            自动音频切片后，需要舍弃每段切片的头尾。该参数设置交叉长度保留的比例，范围0-1,左开右闭
  -eak ENHANCER_ADAPTIVE_KEY, --enhancer_adaptive_key ENHANCER_ADAPTIVE_KEY
                                                            使增强器适应更高的音域(单位为半音数)|默认为0
  -ft F0_FILTER_THRESHOLD, --f0_filter_threshold F0_FILTER_THRESHOLD
                                                            F0过滤阈值，只有使用crepe时有效. 数值范围从0-1. 降低该值可减少跑调概率，但会增加哑音
```