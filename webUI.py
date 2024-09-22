import glob
import os
import sys
import time
import gradio as gr
import librosa
import numpy as np
import soundfile
import torch
import warnings
import argparse
import shutil
import tkinter as tk
from tkinter import filedialog
from inference.infer_tool import Svc

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60))
parser.add_argument("-m", "--model_folder", type=str, default="trained", help="Path to the model folder")
parser.add_argument("-n", "--server_name", type=str, default=None, help="Server IP name")
parser.add_argument("-p", "--server_port", type=int, default=None, help="Server port")
parser.add_argument("-s", "--share", action="store_true", default=False, help="Open share link")
parser.add_argument("-c", "--clean_cache", action="store_true", default=False, help="Clean cache before launch")
args = parser.parse_args()

model = None
spk = None
local_model_root = args.model_folder

if args.clean_cache and os.path.exists("cache"):
    shutil.rmtree("cache")
os.makedirs("cache", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("raw", exist_ok=True)
os.makedirs("pretrain", exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = os.path.abspath("cache")

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"

def select_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected_dir = filedialog.askdirectory()
    root.destroy()
    return selected_dir

def open_folder(folder):
    if folder == "":
        raise gr.Error("请先选择文件夹!")
    os.makedirs(folder, exist_ok=True)
    absolute_path = os.path.abspath(folder)
    if sys.platform == "win32":
        os.system(f"explorer {absolute_path}")
    elif sys.platform == "darwin":
        os.system(f"open {absolute_path}")
    elif sys.platform == "linux":
        os.system(f"xdg-open {absolute_path}")

def modelAnalysis(device, enhance, use_diffusion, use_feature, use_spk_mix, local_model_selection):
    global model
    try:
        device = cuda[device] if "CUDA" in device else device

        diff_model_path, diff_config_path, cluster_model_path, kmeans_model_path = None, None, None, None
        for file in os.listdir(os.path.join(local_model_root, local_model_selection)):
            if file.endswith(".pth"):
                model_path = os.path.join(local_model_root, local_model_selection, file)
            if file.endswith(".json"):
                config_path = os.path.join(local_model_root, local_model_selection, file)
            if file.endswith(".pt"):
                if "kmeans" in file:
                    kmeans_model_path = os.path.join(local_model_root, local_model_selection, file)
                else:
                    diff_model_path = os.path.join(local_model_root, local_model_selection, file)
            if file.endswith(".yaml"):
                diff_config_path = os.path.join(local_model_root, local_model_selection, file)
            if file.endswith(".pkl"):
                cluster_model_path = os.path.join(local_model_root, local_model_selection, file)
        print(model_path, config_path, diff_model_path, diff_config_path, cluster_model_path, kmeans_model_path)

        only_diffusion = False
        shallow_diffusion = False
        if use_diffusion == "不使用":
            diff_model_path = None
            diff_config_path = None
        if use_diffusion == "使用扩散模型" and diff_model_path and diff_config_path:
            shallow_diffusion = True
        if use_diffusion == "使用全扩散推理" and diff_model_path and diff_config_path:
            only_diffusion = True

        feature_retrieval = False
        feature_model_path = None
        if use_feature == "不使用":
            cluster_model_path = None
            kmeans_model_path = None
        if use_feature == "启用特征检索模型" and cluster_model_path:
            feature_model_path = cluster_model_path
            feature_retrieval = True
        if use_feature == "启用聚类模型" and kmeans_model_path:
            feature_model_path = kmeans_model_path
            feature_retrieval = False

        model = Svc(net_g_path = model_path,
                config_path = config_path,
                device = device if device != "Auto" else None,
                cluster_model_path = feature_model_path if feature_model_path else "",
                nsf_hifigan_enhance = enhance,
                diffusion_model_path = diff_model_path if diff_model_path else "",
                diffusion_config_path = diff_config_path if diff_config_path  else "",
                shallow_diffusion = shallow_diffusion,
                only_diffusion = only_diffusion,
                spk_mix_enable = use_spk_mix,
                feature_retrieval = feature_retrieval
                )
        spks = list(model.spk2id.keys())

        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
        msg = f"成功加载模型{os.path.basename(model_path)}到设备{device_name}上\n"
        if not diff_model_path:
            msg += "未加载扩散模型\n"
        else:
            msg += f"扩散模型{os.path.basename(diff_model_path)}加载成功\n"
        if not feature_model_path:
            msg += "未加载聚类模型或特征检索模型\n"
        elif cluster_model_path:
            msg += f"特征检索模型{os.path.basename(cluster_model_path)}加载成功\n"
        else:
            msg += f"聚类模型{os.path.basename(kmeans_model_path)}加载成功\n"
        msg += "\n当前模型的可用音色：\n"
        for i in spks:
            msg += i + " "
        sid = gr.Dropdown(label="选择音色（说话人）", choices=spks, value=spks[0], interactive=True)
        return sid, msg
    except Exception as e:
        raise gr.Error(e)

def modelUnload():
    global model
    if model is None:
        return gr.Dropdown(label="选择音色（说话人）"), "没有模型需要卸载!"
    else:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        return gr.Dropdown(label="选择音色（说话人）", choices=[], interactive=False), "模型卸载完毕!"

def vc_infer(output_format, sid, audio_path, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment, output_folder):
    global model
    _audio = model.slice_inference(
        raw_audio_path=audio_path,
        spk=sid,
        tran=vc_transform,
        slice_db=slice_db,
        cluster_infer_ratio=cluster_ratio,
        auto_predict_f0=auto_f0,
        noice_scale=noise_scale,
        pad_seconds=pad_seconds,
        clip_seconds=cl_num,
        lg_num=lg_num,
        lgr_num=lgr_num,
        f0_predictor=f0_predictor,
        enhancer_adaptive_key=enhancer_adaptive_key,
        cr_threshold=cr_threshold,
        k_step=k_step,
        use_spk_mix=use_spk_mix,
        second_encoding=second_encoding,
        loudness_envelope_adjustment=loudness_envelope_adjustment
    )
    model.clear_empty()

    os.makedirs("results", exist_ok=True)

    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    isdiffusion = "sovits"
    if model.shallow_diffusion:
        isdiffusion = "sovdiff"
    if model.only_diffusion:
        isdiffusion = "diff"

    output_file_name = 'result_' + truncated_basename + f'_{sid}_{key}{cluster}{isdiffusion}.{output_format}'
    output_file = os.path.join(output_folder, output_file_name)
    soundfile.write(output_file, _audio, model.target_sample, format=output_format)
    return output_file

def vc_fn(sid, input_audio, output_format, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment, output_folder):
    global model
    try:
        if input_audio is None:
            raise gr.Error("请上传音频")
        if model is None:
            raise gr.Error("请先加载模型")
        if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
            cluster_ratio = 0
        os.makedirs(output_folder, exist_ok=True)

        start_time = time.time()
        audio, _ = soundfile.read(input_audio)
        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        truncated_basename = os.path.basename(input_audio).split(".")[0]

        output_file = vc_infer(output_format, sid, input_audio, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment, output_folder)

        gr.Info(f"音频转换完成！耗时{time.time() - start_time:.2f}秒")
        return output_file
    except Exception as e:
        raise gr.Error(e)

def scan_local_models():
    res = []
    candidates = glob.glob(os.path.join(local_model_root, '**', '*.json'), recursive=True)
    candidates = set([os.path.dirname(c) for c in candidates])
    for candidate in candidates:
        jsons = glob.glob(os.path.join(candidate, '*.json'))
        pths = glob.glob(os.path.join(candidate, '*.pth'))
        if (len(jsons) >= 1 and len(pths) >= 1):
            res.append(os.path.basename(candidate))
    res.sort()
    return res

def local_model_refresh_fn():
    return gr.Dropdown(label='选择模型文件夹', choices=scan_local_models(), interactive=True, scale=3)

with gr.Blocks(
    theme=gr.Theme.load('gradio_theme.json')
) as app:
    gr.Markdown(value="""<div align="center"><font size=6><b>so-vits-svc4.1-Inference-WebUI</b></font></div>""")
    with gr.Accordion(label="基本模型设置", open=True):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    local_model_selection = gr.Dropdown(label='选择模型文件夹', choices=scan_local_models(), interactive=True, scale=3)
                    local_model_refresh_btn = gr.Button('刷新本地模型列表', scale=1)
                device = gr.Radio(label="选择推理设备", choices=["Auto",*cuda.keys(),"cpu"], value="Auto", interactive=True)
                use_diffusion = gr.Radio(label="扩散模型", choices=["不使用","使用扩散模型","使用全扩散推理"], value="使用扩散模型", interactive=True)
                use_feature = gr.Radio(label="特征检索模型/聚类模型", choices=["不使用","启用特征检索模型","启用聚类模型"], value="不使用", interactive=True)
                with gr.Row():
                    enhance = gr.Checkbox(label="使用NSF_HIFIGAN增强", value=False, interactive=True)
                    use_spk_mix = gr.Checkbox(label="动态声线融合", value=False, interactive=True)
            with gr.Column():
                model_load_button = gr.Button("加载模型", variant="primary")
                model_unload_button = gr.Button("卸载模型", variant="primary")
                sid_output = gr.Textbox(label="Output Message", lines=6)
                sid = gr.Dropdown(label="选择音色（说话人）", choices=[], interactive=False)
    with gr.Accordion(label="推理参数设置", open=True):
        with gr.Row():
            f0_predictor = gr.Radio(label="选择F0预测器", choices=["pm", "dio", "harvest", "crepe", "rmvpe", "fcpe"], value="rmvpe", interactive=True)
            output_format = gr.Radio(label="音频输出格式", choices=["wav", "flac", "mp3"], value = "wav", interactive=True)
        with gr.Row():
            vc_transform = gr.Number(label="变调（半音数量）", value=0, interactive=True)
            cl_num = gr.Number(label="音频自动切片，0为不切片，单位为秒", value=0, minimum=0, step=0.05, interactive=True)
            slice_db = gr.Number(label="切片阈值", value=-40, interactive=True)
            lg_num = gr.Number(label="两端音频切片的交叉淡入长度，单位为秒", value=0, minimum=0, step=0.05, interactive=True)
        with gr.Row():
            k_step = gr.Slider(label="浅扩散步数", value=30, minimum=10, maximum=1000, step=10, interactive=True)
            cluster_ratio = gr.Slider(label="聚类模型/特征检索混合比例", value=0.5, minimum=0, maximum=1, step=0.1, interactive=True)
            lgr_num = gr.Slider(label="每段切片交叉长度保留的比例", value=0.75, minimum=0.05, maximum=1, step=0.05, interactive=True)
            loudness_envelope_adjustment = gr.Slider(label="输入源响度包络替换输出响度包络融合比例", value=0, minimum=0, maximum=1, step=0.1, interactive=True)
        with gr.Row():
            noise_scale = gr.Number(label="噪音级别，玄学参数", value=0.4, step=0.05, interactive=True)
            pad_seconds = gr.Number(label="推理音频pad秒数", value=0.5, minimum=0, step=0.05, interactive=True)
            enhancer_adaptive_key = gr.Number(label="使增强器适应更高的音域（半音数量）", value=0, interactive=True)
            cr_threshold = gr.Number(label="F0过滤阈值，只有启动crepe时有效", value=0.05, minimum=0, maximum=1, step=0.05, interactive=True)
        with gr.Row():
            auto_f0 = gr.Checkbox(label="自动f0预测，配合聚类模型f0预测效果更好", value=False, interactive=True)
            second_encoding = gr.Checkbox(label="二次编码，浅扩散前会对原始音频进行二次编码", value=False, interactive=True)
    vc_input3 = gr.Audio(label="上传音频", type="filepath", interactive=True)
    with gr.Row():
        output_folder = gr.Textbox(label="音频输出文件夹", value="results", interactive=True, scale=3)
        select_output_folder = gr.Button("选择文件夹")
        open_output_folder = gr.Button("打开文件夹")
    vc_submit = gr.Button("音频转换", variant="primary")
    with gr.Row():
        vc_output = gr.Audio(label="Output Audio", type="filepath", interactive=False)

    local_model_refresh_btn.click(local_model_refresh_fn, outputs=local_model_selection)
    vc_submit.click(vc_fn, inputs=[sid, vc_input3, output_format, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds,cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment, output_folder], outputs=[vc_output])
    model_load_button.click(modelAnalysis,inputs=[device, enhance, use_diffusion, use_feature, use_spk_mix, local_model_selection], outputs=[sid, sid_output])
    model_unload_button.click(modelUnload, outputs=[sid, sid_output])
    select_output_folder.click(select_folder, outputs=[output_folder])
    open_output_folder.click(open_folder, inputs=[output_folder])

app.launch(inbrowser=True, server_name=args.server_name, server_port=args.server_port, share=args.share)