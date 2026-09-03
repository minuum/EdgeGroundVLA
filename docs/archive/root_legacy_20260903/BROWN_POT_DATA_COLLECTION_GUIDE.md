# Brown Pot + Gray Basket Navigation - Data Collection Guide

**Date**: 2026-01-22  
**Objects**: Brown Pot (70% VLM) + Gray Basket (90% VLM)  
**Cost**: $0 (이미 보유!)  
**Expected Navigation**: 85-90% success

---

## 🎯 Quick Start

### Objects

**TARGET**: 갈색 화분 (Brown Pot)
- VLM Recognition: 70/100
- Prompt: `<grounding> Is there a brown pot?`
- 위치: 바닥에 배치

**OBSTACLE**: 회색 빨랫 바구니 (Gray Basket)
- VLM Recognition: 90/100 ⭐ (최고 점수!)
- Prompt: `<grounding> Is there a gray basket?`
- 위치: TARGET과 로봇 사이

---

## 📋 Data Collection Plan

### Phase 1: Brown Pot Only (200 episodes)

**Setup**:
```
Environment: 복도 또는 실내 공간
Target: Brown Pot (바닥에 배치)
Obstacle: None
Distance: 2-3m
```

**Episodes**:
- LEFT instruction: 100 episodes
- RIGHT instruction: 100 episodes

**Instructions**:
```python
LEFT_INSTRUCTIONS = [
    "Navigate around obstacles and reach the front of the brown pot on the left",
    "Navigate to the left of the brown pot",
    "Go to the brown pot on the left",
]

RIGHT_INSTRUCTIONS = [
    "Navigate around obstacles and reach the front of the brown pot on the right",
    "Navigate to the right of the brown pot", 
    "Go to the brown pot on the right",
]
```

---

### Phase 2: Brown Pot + Gray Basket (150 episodes)

**Setup**:
```
Environment: 복도
Target: Brown Pot
Obstacle: Gray Basket (로봇과 pot 사이)
Distance: 2-3m
```

**Episodes**:
- LEFT + obstacle: 75 episodes
- RIGHT + obstacle: 75 episodes

---

## 🤖 ROS2 Data Collection

### 1. Setup

```bash
# On robot server
cd /home/billy/25-1kp/vla

# Install dependencies (if needed)
pip install -r requirements.txt
```

---

### 2. Configure Objects

```bash
# Edit configs
vim Mobile_VLA/configs/brown_pot_left.json
vim Mobile_VLA/configs/brown_pot_right.json

# Update data_dir path
# "data_dir": "/home/billy/25-1kp/vla/ROS_action/brown_pot_dataset"
```

---

### 3. Data Collection Script

```bash
# Create collection directory
mkdir -p data/brown_pot_episodes

# Run teleoperation collection
roslaunch mobile_vla collect_episode.launch \
  instruction:="Navigate to the left of the brown pot" \
  output_dir:=data/brown_pot_episodes \
  episode_name:=episode_001_left
```

---

## 📊 Episode Structure

### H5 File Format

```python
episode_001_left.h5:
  /observations
    /image [T, H, W, 3]  # RGB images (224x224)
    /instruction [str]   # "Navigate to the left..."
  
  /actions
    /linear_x [T]  # Forward velocity
    /angular_z [T] # Turning velocity
  
  /metadata
    /target: "brown_pot"
    /obstacle: "gray_basket" or "none"
    /side: "LEFT" or "RIGHT"
    /success: true/false
```

---

## ✅ Quality Control

### Per Episode Check

```python
def validate_episode(episode_path):
    """
    Validate collected episode
    """
    checks = {
        'images': len(episode['observations/image']) > 30,  # Min 30 frames
        'smooth_actions': check_smooth_trajectory(episode['actions']),
        'reached_target': distance_to_target < 30cm,
        'correct_side': verify_left_right(episode),
    }
    return all(checks.values())
```

---

### Dataset Statistics

```bash
# After collection
python scripts/validate_dataset.py \
  --data_dir data/brown_pot_episodes \
  --output_report dataset_stats.json
```

---

## 🎯 VLM Prompts (Verified)

### Object Detection

```python
# TARGET (Brown Pot)
TARGET_PROMPT = "<grounding> Is there a brown pot?"

# Expected VLM response:
# "There is a brown metal pot on the floor"

# OBSTACLE (Gray Basket)  
OBSTACLE_PROMPT = "<grounding> Is there a gray basket?"

# Expected VLM response:
# "Yes, there is a gray plastic basket on the floor"
```

**Important**: 
- ✅ Use simple prompts (no JSON hints!)
- ✅ VLM does object detection only
- ✅ Action head learns instruction grounding

---

## 📈 Expected Performance

### VLM Recognition (Tested!)

```
Brown Pot:    70/100 ✅
Gray Basket:  90/100 ✅✅ (best!)
Average:      80/100
```

### Navigation Success (Predicted)

```
Current baseline (20% VLM): 60-70%
Brown Pot (80% VLM avg):    85-90%

Improvement: +15-25% ✅
```

---

## 🚀 Training

### After Data Collection

```bash
# Train LEFT model
python RoboVLMs_upstream/robovlms/train/train.py \
  --config Mobile_VLA/configs/brown_pot_left.json \
  --gpus 1

# Train RIGHT model
python RoboVLMs_upstream/robovlms/train/train.py \
  --config Mobile_VLA/configs/brown_pot_right.json \
  --gpus 1

# Training time: ~3 hours per model
```

---

## 📋 Timeline

### Week 1: Data Collection

```
Day 1-2: Brown Pot only (200 episodes)
  - 100 LEFT
  - 100 RIGHT

Day 3-4: Brown Pot + Gray Basket (150 episodes)
  - 75 LEFT with obstacle
  - 75 RIGHT with obstacle

Day 5: Quality check & validation
```

### Week 2: Training & Testing

```
Day 1: Training (6 hours total)
Day 2-3: Real robot testing
Day 4-5: Performance evaluation
```

---

## 🔧 Troubleshooting

### If VLM Recognition < 70%

**Check**:
1. Lighting (ensure good indoor lighting)
2. Object placement (center of view)
3. Camera height (30cm recommended)
4. Distance (2-3m optimal)

**Solutions**:
- Improve lighting
- Adjust camera angle
- Use larger pot if available

---

### If Navigation Success < 80%

**Options**:
1. Collect more data (500 episodes vs 350)
2. Tune hyperparameters (learning rate, hidden size)
3. Consider LoRA fine-tuning (+10-15% boost)

---

## 📊 Success Criteria

### Minimum Acceptable

```
VLM Recognition: >60%
Data Quality: >90% valid episodes
Navigation Success: >75%

Decision: Deploy if all met
```

### Target Performance

```
VLM Recognition: >75%
Data Quality: >95% valid episodes
Navigation Success: >85%

Decision: Production ready
```

---

## 💰 Cost Analysis

### Total Investment

```
Objects:
  Brown Pot:     $0 (보유 중)
  Gray Basket:   $0 (보유 중)
  
Compute:
  Training:      $0 (local GPU)
  
Time:
  Collection:    1 week
  Training:      6 hours
  Testing:       1 week
  
Total Cost:    $0 ✅✅✅
Total Time:    2 weeks
```

---

## 🎊 Key Advantages

1. **$0 Cost** - 이미 objects 보유
2. **80% VLM Recognition** - Red Ball과 동일
3. **No LoRA** - Frozen VLM sufficient
4. **Fast Timeline** - 2 weeks to deployment
5. **Gray Basket 90%** - Best obstacle recognition!

---

## 📚 References

- VLM Test Results: `docs/REAL_OBJECTS_VLM_TEST_RESULTS.md`
- Gray Basket Test: `docs/GRAY_BASKET_VLM_TEST_RESULTS.md`
- System Design: `docs/RED_BALL_SYSTEM_DESIGN.md` (같은 구조)
- Prompt Analysis: `docs/VLM_PROMPT_RESPONSE_DETAILED_ANALYSIS.md`

---

**Summary**: Brown Pot (70%) + Gray Basket (90%) = 80% average VLM recognition. Same as Red Ball but **FREE**! Ready for data collection on robot server. Expected navigation: 85-90% success. No LoRA needed! 🎯
