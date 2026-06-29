import copy
import transformers
import torch

from robovlms.utils.model_utils import build_tokenizer


def build_vlm(vlm_config, tokenizer_config, precision="bf16", quantization_config=None):
    """
    Build Vision-Language Model with optional BitsAndBytes quantization
    
    Args:
        vlm_config: VLM configuration dict
        tokenizer_config: Tokenizer configuration dict
        precision: Model precision (bf16, fp16, fp32)
        quantization_config: Optional BitsAndBytesConfig for INT8/INT4 quantization
    """
    vlm_config = copy.deepcopy(vlm_config)
    model_path = vlm_config.get("pretrained_model_name_or_path")
    model_name = vlm_config.get("name")
    model_type = vlm_config.get("type", "AutoModel")
    
    if model_name == "paligemma":
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        tokenizer = AutoProcessor.from_pretrained(model_path)
        
    elif model_name == "llava":
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_base = None  # default is None
        model_path = vlm_config.get("pretrained_model_name_or_path")
        model_family_name = get_model_name_from_path(model_path)
        tokenizer, model, _, __ = load_pretrained_model(
            model_path,
            model_base,
            model_family_name,
            use_flash_attn=False,
            device_map="cpu",
        )
        
    elif model_name == "kosmos":
        # Kosmos-2: load from transformers.models.kosmos2.modeling_kosmos2 directly
        from transformers.models.kosmos2.modeling_kosmos2 import Kosmos2ForConditionalGeneration
        
        # BitsAndBytes INT8/INT4 quantization (VLA standard)
        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "trust_remote_code": True
        }
        
        if quantization_config is not None:
            # OpenVLA/BitVLA style quantization
            load_kwargs.update({
                "quantization_config": quantization_config,
                "device_map": "auto",  # Auto GPU allocation
                "torch_dtype": torch.float16  # BitsAndBytes requires FP16
            })
            print(f"🔧 Loading Kosmos-2 with BitsAndBytes INT8/INT4")

        # V4 모델의 확장된 Vocab(65037) 대응: config를 먼저 불러와서 vocab_size를 강제 설정
        from transformers import Kosmos2Config
        model_id = tokenizer_config["pretrained_model_name_or_path"]
        print(f"🔧 Forcing vocab_size to 65037 for model: {model_id}")
        
        # V4 모델의 확장된 Vocab(65037) 대응: 모델 로드 후 임베딩 레이어 강제 리사이즈
        load_kwargs.pop("pretrained_model_name_or_path", None)
        # 인터넷 연결을 시도하지 않고 로컬 가중치만 즉시 읽도록 local_files_only=True 강제
        model = Kosmos2ForConditionalGeneration.from_pretrained(
            model_id, **load_kwargs, ignore_mismatched_sizes=True, local_files_only=True
        )
        print(f"🔧 Resizing token embeddings to 65037 for checkpoint compatibility...")
        model.resize_token_embeddings(65037)
        
        # 토크나이저/프로세서 로드 (shortest_edge 에러 방지를 위한 폴백 적용)
        try:
            tokenizer = build_tokenizer(tokenizer_config)
        except Exception as e:
            print(f"⚠️ Standard build_tokenizer failed: {e}. Trying raw AutoProcessor fallback...")
            from transformers import AutoProcessor
            # 인터넷 연결을 시도하지 않고 로컬 가중치만 즉시 읽도록 local_files_only=True 강제
            tokenizer = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        
    else:
        # Handle deprecated AutoModelForVision2Seq -> AutoModelForImageTextToText
        if model_type == "AutoModelForVision2Seq":
            model_type = "AutoModelForImageTextToText"
        model = getattr(transformers, model_type).from_pretrained(
            model_path, trust_remote_code=True
        )
        tokenizer = build_tokenizer(tokenizer_config)

    return tokenizer, model
