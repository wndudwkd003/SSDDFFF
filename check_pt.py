# check_pt.py
import torch

PT_PATH = "runs/20260110_121613_XCEPTION_224/KoDF/best_stage1.pth"

if __name__ == "__main__":
    ckpt = torch.load(PT_PATH, map_location="cpu")

    # state_dict 추출(체크포인트 구조가 다를 수 있어서 몇 가지 케이스만 커버)
    sd = None
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "backbone", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                break
    if sd is None:
        sd = ckpt if isinstance(ckpt, dict) else ckpt.state_dict()

    # Conv2d weight(4D) 중 "가장 이른 레이어"로 보이는 것(키 정렬상 첫 번째) 사용
    conv_keys = sorted(
        [
            k
            for k, v in sd.items()
            if isinstance(v, torch.Tensor) and v.ndim == 4 and k.endswith("weight")
        ]
    )
    if not conv_keys:
        raise RuntimeError(
            "4D conv weight를 찾지 못했습니다. (모델이 Conv2d 기반이 아닐 수 있습니다)"
        )

    k0 = conv_keys[0]
    w = sd[k0]  # (out_ch, in_ch, kH, kW)
    print(f"first_conv_key = {k0}")
    print(f"weight_shape   = {tuple(w.shape)}")
    print(f"in_channels    = {w.shape[1]}")
