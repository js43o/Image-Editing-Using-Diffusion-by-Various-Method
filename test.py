from huggingface_hub import snapshot_download

# 특정 모델의 로컬 저장 경로 확인
model_path = snapshot_download("runwayml/stable-diffusion-v1-5")
print(model_path)
