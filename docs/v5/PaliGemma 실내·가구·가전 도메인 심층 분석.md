# PaliGemma 실내·가구·가전 도메인 심층 분석
## Indoor / Furniture / Appliances — 99개 클래스 완전 해부

---

## 1. 도메인 개요

실내·가구·가전 도메인은 PaliGemma 사전학습 전체 600개 클래스 중 **99개(16.5%)** 를 차지하며, 동물(106개), 음식(107개)과 함께 가장 밀도 높게 학습된 3대 도메인 중 하나입니다. 특히 이 도메인은 **일상 실내 공간을 구성하는 사물들**이 WebLI(10억+ 이미지-텍스트 쌍)와 OpenImages V7(정밀 바운딩 박스)에 모두 풍부하게 등장하기 때문에, 다른 도메인 대비 **이미지 내 공간 위치 추론 정확도**가 특히 높습니다.

99개 클래스는 세 개의 서브도메인으로 구성됩니다.

| 서브도메인 | 클래스 수 | 핵심 특성 |
|---|---|---|
| 가구 (Furniture) | 28개 | 공간 배치·형태 인지 특화, 대형 객체 중심 |
| 가전·전자기기 (Appliances & Electronics) | 37개 | 가장 많은 클래스, 브랜드 무관 형태 인지 |
| 실내 설비·인테리어 (Interior Fixtures) | 32개 | 벽·천장·바닥 부착형 고정 설비 중심 |

---

## 2. 서브도메인별 상세 클래스 및 인지 강도

![서브도메인별 인지 강도](https://private-us-east-1.manuscdn.com/sessionFile/MAUM1ZwX6TuvdysFFbeV35/sandbox/gQSQzcHZ9yVwfdmsNrKrZn-images_1780586016437_na1fn_L2hvbWUvdWJ1bnR1L2luZG9vcl9yZWNvZ25pdGlvbl9zdHJlbmd0aA.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvTUFVTTFad1g2VHV2ZHlzRkZiZVYzNS9zYW5kYm94L2dRU1F6Y0haOXlWd2ZkbXNOcktyWm4taW1hZ2VzXzE3ODA1ODYwMTY0MzdfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwybHVaRzl2Y2w5eVpXTnZaMjVwZEdsdmJsOXpkSEpsYm1kMGFBLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=HyJ1qd3SjGMLeBd5xbnYI2sj47OxzYYOKTJeKuffkKeU0MhNIGrgjxoKcvj7ifooD3OkJHb1RT~YgzeRzuKpaLyZsfOVFPEaaIauuqZLzdGkNz1OSBN0brlczb7zXwwsqzdQrLdpBUInOwxIjzjMNlSVM~1NDD7rG7D~YXeN6aehed-0D4BAzKrCFH-tFrybpn6i2L1Fj-l7z-4Pyw1tQ~2J4QaHBO5nsk4lk1qrF6lpW9R9AGkSGGptXoX-d63QcJB-POsqey1Kl2SCQ~2QRMyVDIVV2nzO6R2VTfxXijw5jIwxJ6xMZY-LGsa38NS3~XLO1Gl8-OUPHWu8GaCutQ__)

### 2-1. 가구 (Furniture) — 28개 클래스

가구 서브도메인은 **공간 전체를 차지하는 대형 객체**들로 구성되어 있어, 모델이 이미지 내 공간 레이아웃을 이해하는 데 핵심적인 역할을 합니다. OpenImages V7에서 가구 클래스는 이미지당 평균 2~4개의 인스턴스가 공존하는 경우가 많아, **다중 객체 동시 탐지(Multi-instance Detection)** 능력이 함께 훈련되었습니다.

| 강도 등급 | 클래스 목록 |
|---|---|
| Very Strong (≥90%) | Chair, Table, Kitchen & dining table, Couch, Bed, Stairs, Desk, Coffee table, Cabinet / Cabinetry |
| Strong (80–89%) | Shelf, Wardrobe, Stool, Cupboard, Chest of drawers, Bookcase, Nightstand, Loveseat, Infant bed, Drawer, Closet, Bench |
| Moderate (70–79%) | Sofa bed, Training bench, Filing cabinet, Studio couch, Furniture (generic) |
| Developing (<70%) | Dog bed, Cat furniture |

**주목할 점:** `Chair(98%)`, `Table(95%)`, `Couch(95%)`, `Bed(95%)`는 모델이 가장 자신 있게 인지하는 클래스들입니다. 반면 `Cat furniture`, `Dog bed`처럼 **반려동물 전용 가구**는 학습 데이터 내 빈도가 낮아 상대적으로 낮은 강도를 보입니다.

### 2-2. 가전·전자기기 (Appliances & Electronics) — 37개 클래스

37개로 세 서브도메인 중 가장 많은 클래스를 보유하며, **일상적으로 가장 자주 촬영되는 사물들**이 집중되어 있습니다. 특히 스마트폰, 노트북, TV 등 현대인이 매일 사용하는 기기들은 WebLI 사전학습에서 수억 장의 이미지에 등장하여 인지 강도가 매우 높습니다.

| 강도 등급 | 클래스 목록 |
|---|---|
| Very Strong (≥90%) | Mobile phone, Laptop, Television, Refrigerator, Computer mouse, Computer monitor, Computer keyboard, Tablet computer, Microwave oven, Camera, Remote control, Oven |
| Strong (80–89%) | Washing machine, Toaster, Hair dryer, Gas stove, Coffeemaker, Blender, Printer, Dishwasher, Ceiling fan, Mixer, Food processor, Treadmill, Mechanical fan |
| Moderate (70–79%) | Stationary bicycle, Sewing machine, Heater, Corded phone, Kitchen appliance (generic), Humidifier, Hand dryer, Indoor rower |
| Developing (<70%) | Home appliance (generic), Cassette deck, Fax |

**주목할 점:** `Cassette deck(65%)`, `Fax(60%)`는 현대 이미지 데이터에서 희귀하게 등장하는 레거시 기기이므로 인지 강도가 낮습니다. 반면 `Mobile phone(98%)`, `Laptop(98%)`은 사실상 완벽한 수준의 인지 능력을 보입니다.

### 2-3. 실내 설비·인테리어 (Interior Fixtures) — 32개 클래스

벽, 천장, 바닥에 고정된 설비들로 구성되어 있으며, 주로 **공간 맥락(Spatial Context)** 파악에 기여합니다. 이 서브도메인은 DINOv2의 픽셀 단위 디테일 인지 능력과 결합되어, 문·창문의 열림/닫힘 상태, 조명의 켜짐/꺼짐 등 **상태 변화(State Change) 인지**에도 강점을 보입니다.

| 강도 등급 | 클래스 목록 |
|---|---|
| Very Strong (≥90%) | Window, Toilet, Door, Sink, Lamp, Mirror, Clock, Bathtub |
| Strong (80–89%) | Towel, Toilet paper, Tap, Shower, Flowerpot, Door handle, Curtain, Window blind, Waste container, Wall clock, Power plugs and sockets, Light bulb, Fireplace, Candle |
| Moderate (70–79%) | Soap dispenser, Light switch, Digital clock, Lantern, Bathroom cabinet, Bidet, Spice rack, Porch |
| Developing (<70%) | Jacuzzi, Bathroom accessory |

---

## 3. 전체 97개 클래스 인지 강도 랭킹

![전체 97개 클래스 랭킹](https://private-us-east-1.manuscdn.com/sessionFile/MAUM1ZwX6TuvdysFFbeV35/sandbox/gQSQzcHZ9yVwfdmsNrKrZn-images_1780586016437_na1fn_L2hvbWUvdWJ1bnR1L2luZG9vcl9hbGxfcmFua2Vk.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvTUFVTTFad1g2VHV2ZHlzRkZiZVYzNS9zYW5kYm94L2dRU1F6Y0haOXlWd2ZkbXNOcktyWm4taW1hZ2VzXzE3ODA1ODYwMTY0MzdfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwybHVaRzl2Y2w5aGJHeGZjbUZ1YTJWay5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=vt~cNrU50Ns8K1zJr4fukida6b~2PYrJQc4FgabYzidD4T5U2Fs2l9WOQCaYkHsBul05zTCDJKHrIv16uKa4i3xGVmEkf2mYJSeiKXOAdAiaHHHFlSGti8xqXuZ91GkclQcu17fYxmqtkVr5BkcLA2eZKra-TdQJv1iKMdZt1s8Aw6BueuoawmxiLr9q1IhhT5sKnU9C9VQNqQ6b~ZIW~plZpDp-NBgmF66tjEEKZhoPaRpmplXZfgWD1SOOiN81TbGonGY76Wj9LsVY6MPivBJPxtkx9PV6Fu8~rVOI18JYr6zSCK~vHsDWKtTkUDeovJVNBY8D0XJEFrO2x92Alg__)

**Top 10 (인지 강도 최상위):**

| 순위 | 클래스 | 강도 | 서브도메인 |
|---|---|---|---|
| 1 | Chair | 98% | Furniture |
| 1 | Mobile phone | 98% | Appliances |
| 1 | Laptop | 98% | Appliances |
| 4 | Television | 97% | Appliances |
| 5 | Window | 95% | Fixtures |
| 5 | Toilet | 95% | Fixtures |
| 5 | Door | 95% | Fixtures |
| 5 | Refrigerator | 95% | Appliances |
| 5 | Computer mouse | 95% | Appliances |
| 5 | Computer monitor | 95% | Appliances |
| 5 | Computer keyboard | 95% | Appliances |
| 5 | Table | 95% | Furniture |
| 5 | Kitchen & dining table | 95% | Furniture |
| 5 | Couch | 95% | Furniture |
| 5 | Bed | 95% | Furniture |

---

## 4. 스마트홈 서비스 시나리오별 활용 가능 클래스 매핑

![스마트홈 시나리오 매핑](https://private-us-east-1.manuscdn.com/sessionFile/MAUM1ZwX6TuvdysFFbeV35/sandbox/gQSQzcHZ9yVwfdmsNrKrZn-images_1780586016437_na1fn_L2hvbWUvdWJ1bnR1L2luZG9vcl9zY2VuYXJpb19tYXBwaW5n.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvTUFVTTFad1g2VHV2ZHlzRkZiZVYzNS9zYW5kYm94L2dRU1F6Y0haOXlWd2ZkbXNOcktyWm4taW1hZ2VzXzE3ODA1ODYwMTY0MzdfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwybHVaRzl2Y2w5elkyVnVZWEpwYjE5dFlYQndhVzVuLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=uCa9SOHxfUdzMkQob0FGzepDZaKDyKtJdetFX3UvZvWSAOGq1f5BkF38L~vS0ortQsH8Q8TJih4TXEFQlkmfADsbbtTwgKkwct85jlziT-7rccQqDTyZcgH2NDZML4-Ro9nvNJyZWDFFNFOCw1s1L3iQxx3jvL0FbDKMtLQvsPj1a2AIWYZdzmsbaioLmWrUtKby~tj1zFyqHb5KLRi3Wp2CTgYuco0M2ZPWDmUQ1oBr-jS6EV43vfT7Bg89H4MRf67m4uKSoTno0uDzvtn2f~iWJT2zvLUXGyTYrR7ZCM97v3mszLLt3LweJq7vGz3U4kzZyK7D7OG9oam4iK1WiA__)

PaliGemma의 실내 도메인 사전학습 지식은 8가지 주요 스마트홈 서비스 시나리오에 **추가 학습 없이 즉시 적용** 가능합니다.

### 시나리오 1: 거실 인지 (Living Room Detection) — 12개 클래스 직접 활용
`Television`, `Couch`, `Coffee table`, `Remote control`, `Lamp`, `Curtain`, `Window`, `Clock`, `Flowerpot`, `Bookcase`, `Candle`, `Wall clock`

거실 내 사물 배치 파악, 리모컨 위치 추적, 조명 상태 감지 등 스마트 거실 서비스에 즉시 적용 가능합니다.

### 시나리오 2: 주방 모니터링 (Kitchen Monitoring) — 14개 클래스 직접 활용
`Refrigerator`, `Microwave oven`, `Oven`, `Gas stove`, `Coffeemaker`, `Toaster`, `Blender`, `Food processor`, `Mixer`, `Sink`, `Tap`, `Kitchen & dining table`, `Dishwasher`, `Spice rack`

가스레인지 켜짐 감지, 냉장고 문 열림 알림, 주방 기기 사용 패턴 분석 등 주방 안전·자동화 서비스에 활용됩니다.

### 시나리오 3: 침실 분석 (Bedroom Analysis) — 11개 클래스 직접 활용
`Bed`, `Wardrobe`, `Nightstand`, `Lamp`, `Mirror`, `Curtain`, `Chest of drawers`, `Desk`, `Window blind`, `Clock`

수면 환경 모니터링, 취침 시간 감지, 커튼/블라인드 자동 제어 등에 활용됩니다.

### 시나리오 4: 욕실 안전 (Bathroom Safety) — 11개 클래스 직접 활용
`Toilet`, `Bathtub`, `Shower`, `Sink`, `Mirror`, `Towel`, `Soap dispenser`, `Tap`, `Toilet paper`, `Bathroom cabinet`, `Bidet`

낙상 감지, 수도꼭지 잠금 확인, 욕실 청결 상태 모니터링 등 안전 서비스에 활용됩니다.

### 시나리오 5: 홈 오피스 (Home Office Setup) — 12개 클래스 직접 활용
`Laptop`, `Computer monitor`, `Computer keyboard`, `Computer mouse`, `Desk`, `Chair`, `Printer`, `Tablet computer`, `Mobile phone`, `Lamp`, `Bookcase`, `Shelf`

업무 환경 인지, 화상회의 배경 분석, 자세 교정 알림 등에 활용됩니다.

### 시나리오 6: 피트니스·웰니스 (Fitness & Wellness) — 6개 클래스 직접 활용
`Treadmill`, `Stationary bicycle`, `Indoor rower`, `Training bench`, `Dumbbell`

홈짐 기기 사용 감지, 운동 루틴 추적 등에 활용됩니다. 단, 이 시나리오는 사전학습 클래스 커버리지가 상대적으로 낮아 **추가 데이터 수집이 권장**됩니다.

### 시나리오 7: 보안·접근 제어 (Security & Access) — 7개 클래스 직접 활용
`Door`, `Door handle`, `Window`, `Light switch`, `Camera`, `Power plugs and sockets`, `Waste container`

침입 감지, 문/창문 개폐 상태 모니터링, 전등 자동 제어 등에 활용됩니다.

### 시나리오 8: 노인 케어 (Elderly Care Assist) — 10개 클래스 직접 활용
`Wheelchair`, `Stool`, `Bathtub`, `Toilet`, `Bed`, `Remote control`, `Telephone`, `Medical equipment`, `Lamp`

낙상 위험 감지, 약 복용 알림, 이동 패턴 모니터링 등 실버케어 서비스에 활용됩니다.

---

## 5. 데이터셋 수집 전략 — 실내 도메인 특화 가이드

### 5-1. 인지 강도별 데이터 수집 우선순위 전략

PaliGemma가 이미 강하게 인지하는 클래스와 약하게 인지하는 클래스에 따라 데이터 수집 전략을 달리해야 합니다.

| 전략 | 대상 클래스 | 필요 데이터 양 | 핵심 포인트 |
|---|---|---|---|
| **즉시 활용 (Zero-shot)** | Chair, Table, Laptop, TV, Door 등 상위 클래스 | 0~10장 | 도메인 특화 환경만 보여주면 됨 |
| **경량 파인튜닝 (Few-shot)** | Shelf, Wardrobe, Dishwasher 등 중위 클래스 | 20~50장 | 다양한 각도·조명 조건 포함 |
| **집중 수집 (Full Fine-tuning)** | Cat furniture, Fax, Cassette deck 등 하위 클래스 | 200장 이상 | 명확한 바운딩 박스 라벨 필수 |
| **신규 클래스 (New Vocabulary)** | 사전에 없는 가구/가전 | 500장 이상 | 기존 인지 클래스와 관계 문맥 제공 |

### 5-2. 실내 도메인 데이터 수집 시 필수 고려사항

**조명 다양성(Lighting Diversity):** 실내 이미지는 자연광, 형광등, LED 조명, 야간 조명 등 조명 조건이 극단적으로 다양합니다. PaliGemma의 사전학습 데이터(WebLI)는 대부분 낮 시간대 자연광 이미지로 구성되어 있으므로, **야간·저조도 환경 데이터**를 별도로 수집하면 파인튜닝 후 성능이 크게 향상됩니다.

**시점 다양성(Viewpoint Diversity):** 가구와 가전은 촬영 각도에 따라 형태가 크게 달라집니다. 특히 `Chair`, `Table`, `Couch` 등은 정면·측면·위에서 내려다보는 시점(Bird's eye view)을 모두 포함하는 것이 중요합니다. CCTV나 스마트홈 카메라는 대부분 **위에서 내려다보는 시점**으로 촬영되므로, 이 시점의 데이터를 특별히 강화해야 합니다.

**클러터(Clutter) 환경:** 실제 가정에서는 여러 사물이 겹치고 부분적으로 가려지는 경우(Occlusion)가 빈번합니다. 깔끔하게 배치된 단일 객체 이미지보다 **실제 생활 환경처럼 어수선하게 배치된 이미지**를 수집하면 실제 서비스 환경에서의 정확도가 높아집니다.

**해상도 전략:** PaliGemma는 224px, 448px, 896px 세 가지 해상도로 사전학습되었습니다. 실내 가전처럼 **작은 세부 특징(예: 리모컨 버튼, 전원 플러그 모양)** 이 중요한 클래스는 448px 이상의 고해상도 파인튜닝이 권장됩니다.

### 5-3. 파인튜닝 프롬프트 설계 예시

PaliGemma는 자연어 지시어(Prompt)를 통해 탐지 태스크를 수행합니다. 실내 도메인에서 효과적인 프롬프트 패턴은 다음과 같습니다.

```
# 단일 객체 탐지
detect chair
detect television
detect refrigerator

# 다중 객체 동시 탐지
detect chair ; table ; couch

# 공간 맥락 포함 (새로운 객체 학습 시 권장)
detect smart speaker on the table
detect robot vacuum on the floor
detect air purifier next to the couch

# 상태 포함 탐지 (고급)
detect open door
detect turned on television
```

---

## 6. 실내 도메인 사전학습의 한계 및 보완 전략

PaliGemma의 실내 도메인 사전학습이 커버하지 못하는 영역이 있으며, 이를 인지하고 데이터셋을 보완해야 합니다.

| 미커버 영역 | 예시 | 보완 전략 |
|---|---|---|
| **신규 스마트홈 기기** | 스마트 스피커, AI 로봇청소기, 스마트 잠금장치 | 500장 이상 집중 수집, 기존 클래스와 관계 문맥 제공 |
| **한국 특화 가전** | 김치냉장고, 전기밥솥, 비데 (Bidet은 있으나 한국형 특화) | 한국 가정 환경 이미지 별도 수집 |
| **가구 세부 상태** | 서랍 열림/닫힘, 의자 접힘/펼침 | 상태별 이미지 쌍(Pair) 수집 |
| **브랜드 특화 인지** | 삼성 vs LG 냉장고 구분 | 브랜드별 분류 레이블 추가 |
| **소형 생활용품** | 리모컨 배터리 칸, 콘센트 핀 구멍 | 고해상도(896px) 파인튜닝 필요 |

---

## References

[1] L. Beyer, et al., "PaliGemma: A versatile 3B VLM for transfer," *arXiv:2407.07726*, 2024. [Source](https://arxiv.org/abs/2407.07726)  
[2] A. Steiner, et al., "PaliGemma 2: A Family of Versatile VLMs for Transfer," *arXiv:2412.03555*, 2024. [Source](https://arxiv.org/abs/2412.03555)  
[3] "Open Images V7 Dataset," *Ultralytics Docs*. [Source](https://docs.ultralytics.com/datasets/detect/open-images-v7)  
[4] M. Oquab, et al., "DINOv2: Learning Robust Visual Features without Supervision," *arXiv:2304.07193*, 2023. [Source](https://arxiv.org/abs/2304.07193)  
[5] X. Zhai, et al., "Sigmoid Loss for Language Image Pre-Training (SigLIP)," *ICCV*, 2023. [Source](https://arxiv.org/abs/2303.15343)
