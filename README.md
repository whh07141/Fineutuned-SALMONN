#Fintune SALMONN for multi-turn emotional trajectory

<h1 align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.herokuapp.com/?lines=Hello,+There!+👋;Welcome+to+double red finetuned SALMONN;&center=true&size=30">
  </a>
</h1>

🚀🚀 Welcome to the repo of **Fintuned SALMONN**!

SALMONN is a large language model (LLM) enabling **speech, audio events, and music inputs**, which is developed by the Department of Electronic Engineering at Tsinghua University and ByteDance. Instead of speech-only input or audio-event-only input, SALMONN can perceive and understand all kinds of audio inputs and therefore obtain emerging capabilities such as multilingual speech recognition and translation and audio-speech co-reasoning. This can be regarded as giving the LLM "ears" and cognitive hearing abilities, which makes SALMONN a step towards hearing-enabled artificial general intelligence.
We leavege the Human-like Spoken Dialogue Systems Challenge Track1 dataset to finetune SALMONN for accurately identify and concisely summarize a user’s emotional changes throughout a multi-turn conversation.

<div style='display:flex; gap: 0.25rem; '>
<a href='https://openreview.net/pdf?id=14rn7HpKVk'><img src='https://img.shields.io/badge/SALMONN_paper-PDF-green'></a>
<a href='https://huggingface.co/tsinghua-ee/SALMONN'><img src='https://img.shields.io/badge/SALMONN--13B-checkpoint-yellow'></a> 
<a href='https://huggingface.co/tsinghua-ee/SALMONN-7B'><img src='https://img.shields.io/badge/SALMONN--7B-checkpoint-yellow'></a>
</div>

## 🌟 Structure

The model architecture of SALMONN is shown below. A window-level Q-Former is used as the connection module to fuse the outputs from a Whisper speech encoder and a BEATs audio encoder as augmented audio tokens, which are aligned with the LLM input space. The LoRA adaptor aligns the augmented LLM input space with its output space. The text prompt is used to instruct SALMONN to answer open-ended questions about the general audio inputs and the answers are in the LLM text responses. 

<div align=center><img src="resource/structure.png" height="100%" width="75%"/></div>

## ⚡️ Demos

Compared with traditional speech and audio processing tasks such as speech recognition and audio caption, SALMONN leverages the general knowledge and cognitive abilities of the LLM to achieve a cognitively oriented audio perception, which dramatically improves the versatility of the model and the richness of the task. In addition, SALMONN is able to follow textual commands and even spoken commands with a relatively high degree of accuracy. Since SALMONN only uses training data based on textual commands, listening to spoken commands is also a cross-modal emergent ability.

Here are some examples of SALMONN.

| Audio                                                  | Response                                     |
| ------------------------------------------------------ | -------------------------------------------- |
| [gunshots.wav](./resource/audio_demo/gunshots.wav)     | ![sac](resource/response_demo/sac.png)       |
| [duck.wav](./resource/audio_demo/duck.wav)             | ![story](resource/response_demo/story.png)   |
| [music.wav](./resource/audio_demo/music.wav)           | ![mc](resource/response_demo/mc.png)         |

## ✨ Innovation
We preprocessed the training and testing JSON. During data loading in(dataset_v1.py), we concatenated multiple audio rounds and directly packaged the text into a standard SALMONN prompt.
The prompt for fine-tuning was: "task1_emotional_trajectory": [ "<SpeechHere>" ].

## 🌈 How to Finetune a model

For SALMONN-13B v1, you need to use the following dependencies:
1. Our environment: The python version is 3.9.17, and other required packages can be installed with the following command: ```pip install -r requirements.txt```.
2. Download [whisper large v2](https://huggingface.co/openai/whisper-large-v2/tree/main) to ```whisper_path```.
3. Download [Fine-tuned BEATs_iter3+ (AS2M) (cpt2)](https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea) to `beats_path`.
4. Download [vicuna 13B v1.1](https://huggingface.co/lmsys/vicuna-13b-v1.1/tree/main) to ```llama_path```.
5. prepare the training json.
   We use the convert.py to transform the Human-like Spoken Dialogue Systems Challenge training json for SALMONN finetuning.
7. Running with ```python3 train.py --cfg-path configs/config.yaml``` in A100-SXM-80GB.

## 🌈 How to inference in CLI

1. Same as **How to train a model: 1-4**.
2. Download [salmonn v1](https://huggingface.co/tsinghua-ee/SALMONN/blob/main/salmonn_v1.pth) to ```ckpt```.
3. Running with ```python3 cli_inference.py --cfg-path configs/decode_config.yaml``` in A100-SXM-80GB. Now you can input ```wav_path``` and ```prompt```. Enjoy yourself !


## ✨ Acknowledgements
Our work is based on the code provided by SALMONN（https://github.com/bytedance/SALMONN）.

