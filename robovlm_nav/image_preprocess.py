"""
plan_20260622_train_inference_image_pipeline_unify.md §2 — 학습/추론이 암묵적으로
(HuggingFace AutoProcessor 내부 동작에 의존해서) 같은 224x224로 줄어들던 걸 명시적으로
강제한다. 동작 자체는 안 바뀐다(이미 둘 다 224x224였음) — processor 설정이 바뀌어도
학습/추론이 같은 크기를 보장받기 위한 안전망.
"""
from PIL import Image

VLM_INPUT_SIZE = 224


def resize_for_vlm(pil_img: Image.Image, size: int = VLM_INPUT_SIZE) -> Image.Image:
    """학습(h5 원본 프레임)과 추론(카메라 원본 프레임) 양쪽에서 VLM(Kosmos-2/PG2)에
    넘기기 전 동일하게 호출 — AutoProcessor의 암묵적 리사이즈에만 의존하지 않도록 명시.
    """
    if pil_img.size == (size, size):
        return pil_img
    return pil_img.resize((size, size), Image.BILINEAR)
