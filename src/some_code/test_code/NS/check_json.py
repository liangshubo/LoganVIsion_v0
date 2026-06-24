path = "/home/ubuntu4090/4T_disk/liangshubo/MSKAI/NerveSegment/dataset/rawdata/ubpb/all_json/1_111.json"

# 先读取原始字节
with open(path, 'rb') as f:
    raw = f.read()

# 尝试 UTF-8 解码
try:
    text = raw.decode('utf-8')
    print("✅ 文件是合法 UTF-8。")
except UnicodeDecodeError as e:
    print(f"❌ UTF-8 解码失败: {e}")
    start = max(0, e.start - 20)
    end = min(len(raw), e.start + 20)
    snippet = raw[start:end]
    print(f"出错附近的原始字节段 ({start}-{end}):")
    print(snippet)
    print("\n按 GBK 解码看看是什么字符：")
    print(snippet.decode('gbk', errors='replace'))

# 如果确认可以用 utf-8-sig
import json
data = json.loads(raw.decode('utf-8-sig'))
