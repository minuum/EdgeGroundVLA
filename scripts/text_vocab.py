"""T4 어휘 확장 실험의 문구 목록(학습/보류 분리). plan: docs/plans/plan_20261108_text_vocab_expansion.md"""
POS_TRAIN = ["gray basket", "gray bin", "gray container", "the gray basket", "go to the gray bin", "basket",
    "grey basket", "the basket", "gray box", "gray tub", "gray crate", "go to the gray basket", "move to the gray basket", "find the gray basket",
    "head to the gray container", "approach the gray bin", "drive to the basket", "navigate to the gray bin", "reach the gray basket", "the gray container",
    "a gray basket", "gray storage bin", "gray plastic basket", "gray plastic bin", "the gray bin", "go to the basket", "go to the bin", "bin", "container",
    "gray tote", "go to the gray container", "move toward the gray bin", "the gray box", "gray laundry basket", "gray bucket basket", "find the bin",
    "approach the basket", "drive to the gray container", "reach the bin", "go to the gray box"]
NEG_TRAIN = ["red chair", "person", "blue bottle", "white cup", "yellow ball", "wooden table", "black backpack", "red apple", "orange cone", "human",
    "dog", "cat", "door", "window", "sofa", "bed", "tv", "keyboard", "computer mouse", "book", "pen", "shoe", "hat", "umbrella", "banana", "car", "bicycle",
    "flower pot", "go to the red chair", "go to the person", "go to the blue bottle", "find the white cup", "approach the yellow ball", "drive to the door",
    "reach the sofa", "go to the tv", "find the dog", "move to the window", "approach the wooden table", "go to the black backpack"]
POS_HELD = ["grey bucket", "the bin", "grey container", "gray carton", "go to the grey bin", "drive toward the gray basket", "walk to the gray container",
    "grey tote", "the gray tub", "gray pail", "the gray crate", "move toward the bin", "go to the grey basket", "approach the container", "gray receptacle", "gray holder"]
NEG_HELD = ["green plant", "black laptop", "purple vase", "pink teddy bear", "silver spoon", "white refrigerator", "blue towel", "red fire extinguisher",
    "wooden stool", "yellow duck", "go to the green plant", "find the black laptop", "approach the purple vase", "go to the pink toy"]
HARD_NEG = ["gray chair", "gray cat", "gray laptop", "gray sofa"]        # 색은 같고 물체가 다름 → 정지가 정답
AMBIG = ["red basket", "blue bin"]                                         # 속성만 다름 — 정답 불명확, 참고용
HELD = POS_HELD + NEG_HELD + HARD_NEG + AMBIG
ALL = POS_TRAIN + NEG_TRAIN + HELD
NP, NN = len(POS_TRAIN), len(NEG_TRAIN)
GROUPS = {"pos_held": (NP + NN, NP + NN + len(POS_HELD)),
          "neg_held": (NP + NN + len(POS_HELD), NP + NN + len(POS_HELD) + len(NEG_HELD))}
