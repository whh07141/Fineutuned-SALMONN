import json
import os

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
from transformers import WhisperFeatureExtractor

class EmoMultiTaskDataset(Dataset):
    """
    Multi-task dataset for SALMONN: Task1(Task1: Emotional Trajectory Detection),
    Task2(Emotional Reasoning), Task3(Empathy Assessment)
    """
    def __init__(self, ann_paths, whisper_path, max_text_len=512, device='cuda'):
        super().__init__()
        self.device = device
        self.max_text_len = max_text_len

        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]

        self.data = []
        for path in ann_paths:
            with open(path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                # 如果 JSON 是 {"annotation": [...]}
                if isinstance(file_data, dict) and 'annotation' in file_data:
                    file_data = file_data['annotation']
                self.data.extend(file_data)

        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        # 文本输入和目标
        input_text = item.get('input_text', '')
        target_text = item.get('target_text', '')
        emotion_seq = item.get('emotion_sequence', None)

        # task_type
        task_type = item.get('task', 'unknown')

        # 多轮音频拼接
        audio_list = item.get('audio_files', [])
        audio_concat = np.array([], dtype=np.float32)
        sr = 16000  # 假设采样率固定，可改为读取首个文件的采样率
        for audio_path in audio_list:
            audio, sr_tmp = sf.read(audio_path)
            # Ensure audio is 1D
            if len(audio.shape) == 2:
                audio = audio[:, 0]
            elif len(audio.shape) > 2:
                raise ValueError(f"Unexpected audio shape: {audio.shape}")
            # Ensure audio is 1D array
            audio = np.asarray(audio, dtype=np.float32).flatten()
            sr = sr_tmp
            audio_concat = np.concatenate([audio_concat, audio, np.zeros(1600, dtype=np.float32)])

        # pad/truncate
        if len(audio_concat) < sr:
            audio_concat = np.concatenate([audio_concat, np.zeros(sr - len(audio_concat), dtype=np.float32)])
        audio_concat = audio_concat[: sr * 30]

        spectrogram = self.wav_processor(audio_concat, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()

        # TTS 音频（Task3 可选）
        tts_audio = item.get('tts_audio', None)
        if tts_audio is not None and os.path.exists(tts_audio):
            tts_wav, _ = sf.read(tts_audio)
        else:
            tts_wav = None

        # 构造监督文本：
        # - 你的目标是生成自然语言的情绪轨迹说明（target_text），我们保持 target_text 为主体；
        # - 可选地在末尾追加一行标准化的情绪序列，作为对齐锚点，便于模型更稳定学习。
        #   通过环境变量 SALMONN_TASK1_ADD_SUFFIX 控制（默认开启）。
        supervised_text = target_text if target_text else input_text
        if task_type.startswith('task1') and emotion_seq:
            try:
                add_suffix = os.environ.get("SALMONN_TASK1_ADD_SUFFIX", "1") not in ["0", "false", "False"]
            except Exception:
                add_suffix = True
            if add_suffix:
                seq_str = " -> ".join([str(e).strip().lower() for e in emotion_seq])
                supervised_text = f"{supervised_text}\n情绪序列: {seq_str}"

        return {
            "spectrogram": spectrogram,
            "raw_wav": audio_concat,
            "text": supervised_text,
            "response_txt": target_text,
            "task": task_type,
            "id": item.get('dialogue_id', ''),
            "tts_audio": tts_wav,
        }

    def collater(self, samples):
        spectrograms = torch.stack([s['spectrogram'] for s in samples], dim=0)
        raw_wav = [torch.from_numpy(s['raw_wav']).float() for s in samples]
        raw_wav_length = torch.tensor([len(s['raw_wav']) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        padding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

        texts = [s['text'] for s in samples]
        responses = [s['response_txt'] for s in samples]
        tasks = [s['task'] for s in samples]
        ids = [s['id'] for s in samples]
        tts_audio = [s['tts_audio'] for s in samples]

        return {
            "spectrogram": spectrograms,
            "raw_wav": raw_wav,
            "padding_mask": padding_mask,
            "text": texts,
            "response_txt": responses,
            "task": tasks,
            "id": ids,
            "tts_audio": tts_audio,
        }

