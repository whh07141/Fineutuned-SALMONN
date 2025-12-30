import json

def convert_raw_to_train_json_v2(input_path, output_path):
    transformed = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)
            except: continue

            turns = entry.get('turns', [])
            if not turns: continue

            # --- 核心改进：构建交织对话历史 ---
            dialogue_history = ""
            for i, turn in enumerate(turns):
                # 每一轮的文本信息
                dialogue_history += f"第{i+1}轮 用户说: \"{turn['text']}\"\n"
                
                # 如果不是最后一轮，把模型的回复也作为背景信息加入
                # 注意：最后一轮的 response_txt 是我们的训练目标 (Target)，不放进 Input
                if i < len(turns) - 1:
                    dialogue_history += f"第{i+1}轮 模型回: \"{turn['response_txt']}\"\n"
            
            # 构造最终的指令输入
            full_input = (
                f"以下是完整的对话历史记录：\n{dialogue_history}\n"
                f"请结合对应的多段语音特征（特别注意其中的笑声、叹息及语气起伏），"
                f"详细描述用户在这三轮对话中的情绪变化轨迹。"
            )
            
            # 训练目标：最后一轮的回答
            target = turns[-1].get('response_txt', '')
            
            transformed.append({
                "dialogue_id": entry.get('dialogue_id', f"idx_{line_num}"),
                "task": "task1_emotional_trajectory",
                "input_text": full_input,
                "target_text": target,
                "audio_files": [t['split_audio_file'] for t in turns]
            })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)
    print(f"转换完成！已包含模型回复历史。")

if __name__ == "__main__":
    convert_raw_to_train_json_v2('task1_3.jsonl', 'formatted_with_history.json')