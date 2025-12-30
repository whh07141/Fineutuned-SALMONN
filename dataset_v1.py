import json
import os
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperFeatureExtractor

class EmoMultiTaskDataset(Dataset):
    def __init__(self, ann_paths, whisper_path, max_text_len=1024):
        super().__init__()
        self.max_text_len = max_text_len
        self.data = []
        
        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]
            
        for path in ann_paths:
            with open(path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                if isinstance(file_data, dict) and 'annotation' in file_data:
                    file_data = file_data['annotation']
                self.data.extend(file_data)

        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        # 1. 音频处理：按照你的逻辑拼接，但略微增加静音间隙
        audio_list = item.get('audio_files', [])
        audio_concat = np.array([], dtype=np.float32)
        sr = 16000 
        
        # 增加到 0.5s 的静音间隙 (8000个采样点 @ 16kHz)
        gap_size = 8000 
        
        for audio_path in audio_list:
            if not os.path.exists(audio_path):
                continue
            audio, sr_tmp = sf.read(audio_path)
            if len(audio.shape) == 2:
                audio = audio[:, 0]
            sr = sr_tmp
            # 拼接音频 + 静音隙
            audio_concat = np.concatenate([audio_concat, audio, np.zeros(gap_size, dtype=np.float32)])

        # 2. 截断与 Pad (保持你的 30s 逻辑)
        if len(audio_concat) < sr:
            audio_concat = np.concatenate([audio_concat, np.zeros(sr - len(audio_concat), dtype=np.float32)])
        audio_concat = audio_concat[: sr * 30]

        # 3. 提取特征
        spectrogram = self.wav_processor(audio_concat, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()

        # 4. 文本处理：封装成 SALMONN 标准 Prompt
        # 将你构建好的 input_text 放入 USER 槽位
        raw_input_text = item.get('input_text', '')
        formatted_prompt = f"USER: <Speech><SpeechHere></Speech>\n{raw_input_text}\nASSISTANT: "
        target_text = item.get('target_text', '')

        return {
            "spectrogram": spectrogram,
            "raw_wav": audio_concat,
            "text": formatted_prompt, # 现在的 text 已经包含标签和上下文
            "response_txt": target_text,
            "task": item.get('task', 'unknown'),
            "id": item.get('dialogue_id', ''),
        }

    def collater(self, samples):
        # 保持你原始的 collater 逻辑不变，确保 raw_wav 和 padding_mask 正确
        spectrograms = torch.stack([s['spectrogram'] for s in samples], dim=0)
        raw_wav = [torch.from_numpy(s['raw_wav']).float() for s in samples]
        raw_wav_length = torch.tensor([len(s['raw_wav']) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        
        # 对应 raw_wav 的 mask
        padding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

        return {
            "spectrogram": spectrograms,
            "raw_wav": raw_wav,
            "padding_mask": padding_mask,
            "text": [s['text'] for s in samples],
            "response_txt": [s['response_txt'] for s in samples],
            "task": [s['task'] for s in samples],
            "id": [s['id'] for s in samples],
        }