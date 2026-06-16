#!/bin/bash
DIR="docs/v5/grounding_comparison"
echo "========================================================================================================================"
echo "                   �� MoNaVLA Grounding Analysis Dashboard (Detailed Response Mode)"
echo "========================================================================================================================"
printf "%-15s | %-8s | %-15s | %-15s | %-8s | %-15s\n" "Model" "Progress" "L-Action (v,w)" "R-Action (v,w)" "Diff" "Status"
echo "----------------|----------|-----------------|-----------------|----------|---------------------------------------------"

for f in $DIR/log_*.txt; do
    [ -e "$f" ] || continue
    model=$(basename "$f" .txt | sed 's/log_//')
    
    # 진행도 파싱
    progress=$(grep -o "\[Sample [0-9]*/[0-9]*\]" "$f" | tail -1 | tr -d "[]" | sed 's/Sample //')
    [ -z "$progress" ] && progress=$(grep -o "[0-9]*%" "$f" | tail -1)
    [ -z "$progress" ] && progress="Starting"
    
    # 마지막 두 개의 Response Action 파싱 (최근 샘플의 Left/Right)
    actions=$(grep "Response Action:" "$f" | tail -2 | awk -F'[][]' '{print $2}' | awk '{printf "[%s,%s] ", $1, $3}')
    l_act=$(echo $actions | awk '{print $1}')
    r_act=$(echo $actions | awk '{print $2}')
    [ -z "$l_act" ] && l_act="N/A"
    [ -z "$r_act" ] && r_act="N/A"
    
    # 마지막 차이값(Diff) 파싱
    diff=$(grep -o "Difference = [0-9.]*" "$f" | tail -1 | awk '{print $3}')
    [ -z "$diff" ] && diff=$(grep -o "Diff: [0-9.]*" "$f" | tail -1 | awk '{print $2}')
    [ -z "$diff" ] && diff="N/A"
    
    # 상태 판별
    status="Running"
    last_line=$(tail -n 1 "$f")
    [[ "$last_line" == *"50/50"* || "$last_line" == *"Completed"* ]] && status="Done ✅"
    [[ "$last_line" == *"Error"* || "$last_line" == *"Exception"* ]] && status="Error ❌"
    
    printf "%-15s | %-8s | %-15s | %-15s | %-8s | %-15s\n" "$model" "$progress" "$l_act" "$r_act" "$diff" "$status"
done
echo "========================================================================================================================"
