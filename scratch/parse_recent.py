# -*- coding: utf-8 -*-
import json
import os

filepath = "/home/minum/.claude/projects/-home-minum-26CS-MoNaVLA/58e5cf05-0b84-4fb1-a86b-e7a1e7e92f5c.jsonl"
file_size = os.path.getsize(filepath)

# 마지막 5MB 읽기 (더 넓은 문맥을 잡기 위해 5MB로 증설)
read_size = min(file_size, 5 * 1024 * 1024) 
with open(filepath, "rb") as f:
    f.seek(file_size - read_size)
    data = f.read()

lines = data.decode("utf-8", errors="ignore").split("\n")
if len(lines) > 1:
    lines = lines[1:]

dialogue = []
for idx, line in enumerate(lines):
    if not line.strip():
        continue
    try:
        d = json.loads(line)
        m_type = d.get("type")
        if m_type in ["user", "assistant"]:
            content = ""
            if "message" in d and "content" in d["message"]:
                c = d["message"]["content"]
                if isinstance(c, list):
                    content = "\n".join([item.get("text", "") for item in c if isinstance(item, dict)])
                else:
                    content = str(c)
            elif "content" in d:
                content = str(d["content"])
            
            if "<command-name>" in content or "Caveat: The messages below" in content:
                continue
            dialogue.append((m_type.upper(), content))
    except Exception as e:
        pass

# 최근 40턴 대화를 파일로 저장
out_path = "/home/minum/26CS/MoNaVLA/scratch/recent_dialogue.txt"
with open(out_path, "w", encoding="utf-8") as out:
    out.write(f"최근 추출된 대화 턴 수 (최대 60턴):\n")
    for r, c in dialogue[-60:]:
        out.write(f"\n==================== {r} ====================\n")
        out.write(c)
        out.write("\n")

print(f"recent_dialogue.txt 생성 완료. 경로: {out_path}")
