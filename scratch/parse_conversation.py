# -*- coding: utf-8 -*-
import json
import sys

# JSONL 파일 경로 정의
jsonl_path = "/home/minum/.claude/projects/-home-minum-26CS-MoNaVLA/58e5cf05-0b84-4fb1-a86b-e7a1e7e92f5c.jsonl"

def extract_text(content_field):
    """
    다양한 형태의 content 필드(문자열, 리스트 등)에서 텍스트만 추출합니다.
    """
    if isinstance(content_field, str):
        return content_field
    elif isinstance(content_field, list):
        texts = []
        for item in content_field:
            if isinstance(item, dict):
                if "text" in item:
                    texts.append(item["text"])
                elif "content" in item:
                    texts.append(str(item["content"]))
            else:
                texts.append(str(item))
        return "\n".join(texts)
    elif isinstance(content_field, dict):
        if "text" in content_field:
            return content_field["text"]
        elif "content" in content_field:
            return str(content_field["content"])
    return str(content_field)

def parse_dialogue():
    conversation = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                msg_type = data.get("type")
                
                if msg_type == "user":
                    content = ""
                    if "message" in data and "content" in data["message"]:
                        content = extract_text(data["message"]["content"])
                    elif "content" in data:
                        content = extract_text(data["content"])
                    
                    # command 형태나 무의미한 caveat 등 필터링
                    if "<command-name>" in content:
                        continue
                    if "Caveat: The messages below were generated" in content:
                        continue
                        
                    conversation.append({
                        "index": idx,
                        "role": "USER",
                        "content": content
                    })
                    
                elif msg_type == "assistant":
                    content = ""
                    if "message" in data and "content" in data["message"]:
                        content = extract_text(data["message"]["content"])
                    elif "content" in data:
                        content = extract_text(data["content"])
                    
                    conversation.append({
                        "index": idx,
                        "role": "ASSISTANT",
                        "content": content
                    })
            except Exception as e:
                pass
                
    return conversation

if __name__ == "__main__":
    dialogue = parse_dialogue()
    print(f"총 추출된 대화 턴 수: {len(dialogue)}")
    
    # 마지막 20턴의 대화 내용을 요약 출력
    print("\n=== 최근 대화 턴 출력 (마지막 20턴) ===")
    for turn in dialogue[-20:]:
        role = turn["role"]
        idx = turn["index"]
        content_preview = turn["content"][:200].replace("\n", " ")
        print(f"[{idx}] {role}: {content_preview}...")
        
    # 마지막 5턴에 대해서는 본문을 전체 출력해 봅니다.
    print("\n=== 마지막 5턴 본문 출력 ===")
    for turn in dialogue[-5:]:
        role = turn["role"]
        idx = turn["index"]
        print(f"\n--- [{idx}] {role} ---")
        print(turn["content"])
