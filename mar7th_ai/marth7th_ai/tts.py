import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from itertools import chain
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch

from compress_model import removeOptimizer
from edgetts.tts_voices import SUPPORTED_LANGUAGES
from inference.infer_tool import Svc
from utils import mix_model

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
spk = None
debug = False

local_model_root = './trained'

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"

def modelAnalysis(device, enhance):
    global model

    try:
        device = cuda[device] if "CUDA" in device else device


        model_path = "C:/Users/诗乃琴音/Desktop/marth7th_ai/trained/mar7th_G_40000.pth"
        config_path = "C:/Users/诗乃琴音/Desktop/marth7th_ai/trained/mar7th.json"


        model = Svc(model_path,
                    config_path,
                    device = "cpu",
                    nsf_hifigan_enhance=enhance,


                    )

        spks = list(model.spk2id.keys())
        device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)

        msg = f"Successfully loaded the model to device {device_name}\n"


        msg += "Available speakers for the current model:\n"
        for spk in spks:
            msg += spk + " "

        return msg

    except Exception as e:
        if debug:
            traceback.print_exc()
        print(e)


def vc_infer(output_format, sid, audio_path, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    global model
    _audio = model.slice_inference(
        audio_path,
        sid,
        vc_transform,
        slice_db,
        cluster_ratio,
        auto_f0,
        noise_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment
    )
    model.clear_empty()

    if not os.path.exists("results"):
        os.makedirs("results")
    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    isdiffusion = "sovits"
    if model.shallow_diffusion:
        isdiffusion = "sovdiff"

    if model.only_diffusion:
        isdiffusion = "diff"

    output_file_name = 'result_'+truncated_basename+f'_{sid}_{key}{cluster}{isdiffusion}.{output_format}'
    output_file = os.path.join("results", output_file_name)
    soundfile.write(output_file, _audio, model.target_sample, format=output_format)
    return output_file

def vc_fn(sid, input_audio, output_format, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    global model

    try:
        if input_audio is None:
            return "You need to upload an audio"
        if model is None:
            return "You need to upload a model"
        if getattr(model, 'cluster_model', None) is None and not model.feature_retrieval:
            if cluster_ratio != 0:
                return "You need to upload a cluster model or feature retrieval model before assigning cluster ratio!"
        audio, sampling_rate = soundfile.read(input_audio)
        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        truncated_basename = Path(input_audio).stem[:-6]
        processed_audio = os.path.join("raw", f"{truncated_basename}.wav")
        soundfile.write(processed_audio, audio, sampling_rate, format="wav")
        output_file = vc_infer(output_format, sid, processed_audio, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)
        return "Success"
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise e

def text_clear(text):
    return re.sub(r"[\n\,\(\) ]", "", text)

def vc_fn2(_text, _lang, _gender, _rate, _volume, sid, output_format, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    global model

    try:
        if model is None:
            return "You need to upload a model"
        if getattr(model, 'cluster_model', None) is None and not model.feature_retrieval:
            if cluster_ratio != 0:
                return "You need to upload a cluster model or feature retrieval model before assigning cluster ratio!"
        _rate = f"+{int(_rate*100)}%" if _rate >= 0 else f"{int(_rate*100)}%"
        _volume = f"+{int(_volume*100)}%" if _volume >= 0 else f"{int(_volume*100)}%"
        if _lang == "Auto":
            _gender = "Male" if _gender == "男" else "Female"
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume, _gender])
        else:
            subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume])
        target_sr = 44100
        y, sr = librosa.load("tts.wav")
        resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        soundfile.write("tts.wav", resampled_y, target_sr, subtype="PCM_16")
        input_audio = "tts.wav"
        output_file_path = vc_infer(output_format, sid, input_audio, "tts", vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)
        os.remove("tts.wav")
        return "Success"
    except Exception as e:
        if debug:
            traceback.print_exc()
        raise e

def scan_local_models():
    res = []
    candidates = glob.glob(os.path.join(local_model_root, '**', '*.json'), recursive=True)
    candidates = set([os.path.dirname(c) for c in candidates])
    for candidate in candidates:
        jsons = glob.glob(os.path.join(candidate, '*.json'))
        pths = glob.glob(os.path.join(candidate, '*.pth'))
        if (len(jsons) == 1 and len(pths) == 1):
            res.append(candidate)
    return res

def local_model_refresh_fn():
    return scan_local_models()

if __name__ == "__main__":
    vc_transform = 0
    auto_f0 = True
    enhance = False
    cluster_ratio = 0
    slice_db = -40
    output_format = "wav"
    device = 'auto'
    noise_scale = 0.4
    k_step = 100
    pad_seconds = 0.5
    cl_num = 0
    lg_num = 0
    lgr_num = 0.75
    enhancer_adaptive_key = 0
    cr_threshold = 0.05
    loudness_envelope_adjustment = 0
    second_encoding = False
    use_spk_mix = False

    text2tts = "我是"
    tts_gender = "女"
    tts_lang = "Auto"
    tts_rate = 0.0
    tts_volume = 0.0

    sid = "mar7th"
    input_audio = "tts.wav"
    f0_predictor = "mp"


    # 刷新本地模型列表按钮点击事件，调用 local_model_refresh_fn 函数，并将结果更新到 local_model_selection
    local_model_refresh_fn()
    sid_output = print("成功加载模型到设备cpu上未加载聚类模型或特征检索模型未加载扩散模型当前模型的可用音色：mar7th ")
    modelAnalysis(device, enhance)


    vc_fn2(text2tts, tts_lang, tts_gender, tts_rate, tts_volume, sid, output_format, vc_transform, auto_f0,
                      cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, "pm",
                      enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding,
                      loudness_envelope_adjustment)


