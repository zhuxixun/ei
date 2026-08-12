"""手动下载 lerobot/pusht 数据集（逐个文件 + 重试，比 snapshot_download 多线程更稳）"""
import time
from huggingface_hub import hf_hub_download

REPO = "lerobot/pusht"
FILES = [
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    "data/chunk-000/file-000.parquet",
    "videos/observation.image/chunk-000/file-000.mp4",
]

for i, path in enumerate(FILES, 1):
    for attempt in range(1, 6):
        try:
            local = hf_hub_download(REPO, path)
            print(f"[{i}/{len(FILES)}] ✅ {path} -> {local}")
            break
        except Exception as e:
            print(f"[{i}/{len(FILES)}] ⚠️  第 {attempt} 次失败: {type(e).__name__}，重试...")
            time.sleep(2 * attempt)
    else:
        print(f"[{i}/{len(FILES)}] ❌ {path} 下载失败，请检查网络")
        raise SystemExit(1)

print("\n🎉 数据集下载完成！")
