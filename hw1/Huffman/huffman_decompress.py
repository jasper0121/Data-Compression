import os
from PIL import Image

IMAGE_SIZE = (512, 512)

def parse_encoded_file(filepath):
    """ 解析壓縮後的 Huffman 文字檔，回傳反向 code table 與去除填充位後的 bitstream。
    檔案格式：
        Header:
        Number of entries: X
        Symbol:   32  Code Length:  3  Code: 010
        Symbol:  -12  Code Length:  4  Code: 1100
        ...
        Extra bits (padding): 3

        Encoded Bitstream:
        010101001010...
    """
    rev_table, extra_bits, bitstream = {}, 0, ""
    with open(filepath, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    for i, line in enumerate(lines):
        if line.startswith("Symbol:"):
            # 格式範例: "Symbol:   32  Code Length:  3  Code: 010"
            left, code = line.split("Code:")
            try:
                symbol = int(left.split("Symbol:")[1].split("Code Length:")[0].strip())
            except ValueError:
                symbol = left.split("Symbol:")[1].split("Code Length:")[0].strip()
            rev_table[code.strip()] = symbol
        elif line.startswith("Extra bits"):
            extra_bits = int(line.split(":")[1].strip())
        elif line.startswith("Encoded Bitstream:"):
            # 將之後所有行合併為一個 bitstream
            bitstream = "".join(lines[i + 1:])
            break
    if extra_bits:
        bitstream = bitstream[:-extra_bits]
    return rev_table, bitstream

def decode_bitstream(rev_table, bitstream):
    """利用反向 code table 從 bitstream 解碼原始資料"""
    decoded, current = [], ""
    for bit in bitstream:
        current += bit
        if current in rev_table:
            decoded.append(rev_table[current])
            current = ""
    return decoded

def inverse_dpcm(values, width=IMAGE_SIZE[0]):
    """依據 DPCM 還原，每一列第一個像素加上 128，其餘像素累加還原"""
    restored = []
    for row_start in range(0, len(values), width):
        row = []
        # 還原該列的第一個像素
        first = values[row_start] + 128
        row.append(first)
        for delta in values[row_start + 1: row_start + width]:
            row.append(row[-1] + delta)
        restored.extend(row)
    return restored

def save_png(pixels, filepath):
    """將 pixel 資料轉成 512x512 灰階圖並存成 PNG"""
    if len(pixels) != IMAGE_SIZE[0] * IMAGE_SIZE[1]:
        raise ValueError("資料長度與影像尺寸不符")
    # 限制在 0~255 區間
    clipped = [max(0, min(255, v)) for v in pixels]
    img = Image.new('L', IMAGE_SIZE)
    img.putdata(clipped)
    img.save(filepath)

def process_decompression(in_path, out_path, is_dpcm=False):
    rev_table, bitstream = parse_encoded_file(in_path)
    decoded = decode_bitstream(rev_table, bitstream)
    if is_dpcm:
        decoded = inverse_dpcm(decoded, width=IMAGE_SIZE[0])
    save_png(decoded, out_path)
    print(f"已解壓縮並儲存 {out_path}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Results", "Huffman"))
    for folder in ["Interleave", "Plane"]:
        comp_folder = os.path.join(base_dir, folder, "Compression")
        out_folder = os.path.join(base_dir, folder, "Decompression")
        os.makedirs(out_folder, exist_ok=True)

        if not os.path.isdir(comp_folder):
            print(f"輸入目錄不存在: {comp_folder}")
            continue

        for filename in filter(lambda f: f.endswith(".txt"), os.listdir(comp_folder)):
            in_path = os.path.join(comp_folder, filename)
            base_name = os.path.splitext(filename)[0].replace("_huffman", "").replace("_dpcm_huffman", "")
            is_dpcm = "dpcm" in filename.lower()

            try:
                process_decompression(in_path, os.path.join(out_folder, f"{base_name}.png"), is_dpcm)
            except Exception as e:
                print(f"解壓縮 {in_path} 時發生錯誤: {e}")

if __name__ == "__main__":
    main()
