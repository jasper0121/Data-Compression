import os
import math
import heapq
from collections import Counter

# ---------------------------
# 熵計算相關函式
# ---------------------------
def compute_entropy(data):
    """計算資料序列的熵、元組資料的聯合熵（使用以 2 為底的對數）"""
    total = len(data)
    return -sum((c/total) * math.log2(c/total) for c in Counter(data).values())

def compute_conditional_entropy(pair_list):
    """計算條件熵 H(Y|X) = H(X,Y) - H(X)"""
    H_pair = compute_entropy(pair_list)
    H_x = compute_entropy([p[0] for p in pair_list])
    return H_pair - H_x

def rgb_to_yuv(pixels):
    """將 RGB 像素列表轉換為 YUV 色彩空間，並將結果限制在 0 到 255 的範圍內"""
    Y_list, U_list, V_list = [], [], []
    for (R, G, B) in pixels:
        Y_list.append(max(0, min(255, int(round( 0.299    * R + 0.587    * G + 0.114    * B)))))
        U_list.append(max(0, min(255, int(round(-0.168736 * R - 0.331264 * G + 0.5      * B + 128)))))
        V_list.append(max(0, min(255, int(round( 0.5      * R - 0.418688 * G - 0.081312 * B + 128)))))
    return Y_list, U_list, V_list

def neighbor_pairs(seq, width, height, pad):
    """針對任意序列產生左右 / 上方鄰居配對"""
    left_pairs, upper_pairs = [], []

    # 水平鄰居配對
    for row_idx in range(height):
        start = row_idx * width
        row = seq[start : start + width]
        # 行首的左鄰為 pad，其餘為該行的前一元素
        left_neighbors = [pad] + row[:-1]
        left_pairs.extend(zip(left_neighbors, row))

    # 垂直鄰居配對
    for col_idx in range(width):
        column = [seq[row_idx * width + col_idx] for row_idx in range(height)]
        # 列首的上鄰為 pad，其餘為該列的前一元素
        upper_neighbors = [pad] + column[:-1]
        upper_pairs.extend(zip(upper_neighbors, column))

    return left_pairs, upper_pairs

# ---------------------------
# Huffman 編碼相關函式
# ---------------------------
class Node:
    def __init__(self, freq, symbol=None, left=None, right=None):
        """初始化 Huffman Tree 節點，包含頻率、符號（若有）、左子節點和右子節點"""
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
    def __lt__(self, other):
        """定義節點之間的比較（以頻率為依據），以便在 heap 中正確排序"""
        return self.freq < other.freq

def build_frequency_table(data):
    """利用 Counter 建立資料序列的頻率表（字典）"""
    return dict(Counter(data))

def build_huffman_tree(freq):
    """從頻率表建立 Huffman Tree，並回傳樹的根節點（若資料為空則回傳 None）。"""
    heap = [Node(f, s) for s, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        heapq.heappush(heap, Node(n1.freq + n2.freq, None, n1, n2))
    return heap[0] if heap else None

def generate_codes(node, prefix="", table=None):
    """遞迴產生 Huffman 編碼
    從 Huffman Tree 中產生一個字典，該字典將每個符號映射到其對應的編碼（字串）
    """
    if table is None:
        table = {}
    if node is None:
        return table
    if node.symbol is not None:
        table[node.symbol] = prefix or "0"
    else:
        generate_codes(node.left, prefix + "0", table)
        generate_codes(node.right, prefix + "1", table)
    return table

def encode_data(data, table):
    """利用給定的 Huffman 編碼表，將資料序列編碼為一個位元字串"""
    return "".join(table[s] for s in data)

def bits_to_bytes(bitstring):
    """將位元字串補齊至 8 的倍數並回傳補齊後的字串及所添加的補齊位元數量"""
    extra = (-len(bitstring)) % 8
    return bitstring + "0" * extra, extra

def write_encoded_txt(table, extra, bitstring, filename):
    """將 Huffman 編碼表和編碼後的位元字串，以標頭資訊的格式寫入指定的文字檔中"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Header:\n")
        f.write("Number of entries: {}\n".format(len(table)))
        for s, code in sorted(table.items(), key=lambda x: x[0]):
            f.write("Symbol: {:>4}  Code Length: {:>2}  Code: {}\n".format(s, len(code), code))
        f.write("Extra bits (padding): {}\n\n".format(extra))
        f.write("Encoded Bitstream:\n" + bitstring + "\n")

def process_encoding(data, out_filename):
    """對輸入資料進行 Huffman 壓縮，將編碼結果寫入檔案，並回傳壓縮後的大小（位元組）"""
    freq = build_frequency_table(data)
    tree = build_huffman_tree(freq)
    table = generate_codes(tree)
    bits, extra = bits_to_bytes(encode_data(data, table))
    write_encoded_txt(table, extra, bits, out_filename)
    return len(bits) // 8  # 壓縮後大小（bytes）

# ---------------------------
# 影像讀取與轉換
# ---------------------------
def read_raw_image(filename, width=512, height=512, mode="RGB"):
    """從原始影像檔中讀取資料，並根據模式返回像素列表
    模式 "RGB" 會返回 (R, G, B) 元組的列表；模式 "Plane" 則將影像資料分為 R、G、B 三個平面
    """
    with open(filename, "rb") as f:
        raw = f.read()
    num_pixels = width * height
    exp = num_pixels * 3
    if len(raw) < exp:
        raise ValueError("檔案大小不符合預期")
    if mode == "RGB":
        return [(raw[i], raw[i+1], raw[i+2]) for i in range(0, exp, 3)]
    elif mode == "Plane":
        return list(zip(raw[:num_pixels], raw[num_pixels:2*num_pixels], raw[2*num_pixels:exp]))
    else:
        raise ValueError("不支援的格式")

def rgb_to_y_channel(pixels):
    """將 RGB 像素列表轉換為 Y（亮度）通道，並保證數值介於 0 與 255 之間"""
    return [max(0, min(255, int(round(0.299 * R + 0.587 * G + 0.114 * B)))) for (R, G, B) in pixels]

def dpcm_transform(y_vals, width=512):
    """對 Y 通道數值進行 DPCM 轉換
    每列第一個數值減去 128，其後每個數值與前一數值的差值被計算出來
    """
    dpcm = []
    for i in range(0, len(y_vals), width):
        row = y_vals[i:i + width]
        if row:
            dpcm.append(row[0] - 128)
            dpcm.extend(row[j] - row[j-1] for j in range(1, len(row)))
    return dpcm

# ---------------------------
# 統一處理單一影像
# ---------------------------
def process_image(filepath, mode):
    """處理單一影像檔案：
    - 讀取原始影像資料
    - 提取 RGB、YUV 和 Y 通道資料
    - 計算各種資訊熵，以及左右/上方相鄰條件的熵
    - 回傳像素資料、Y 通道資料與熵資訊字典
    """
    pixels = read_raw_image(filepath, mode=mode)
    R_list = [p[0] for p in pixels]
    G_list = [p[1] for p in pixels]
    B_list = [p[2] for p in pixels]
    Y_list, U_list, V_list = rgb_to_yuv(pixels)
    pixels_yuv = list(zip(Y_list, U_list, V_list))

    width, height = 512, 512

    # 計算各通道／向量的鄰居配對
    pair_left_y,   pair_upper_y   = neighbor_pairs(Y_list,   width, height,   128)
    pair_left_rgb, pair_upper_rgb = neighbor_pairs(pixels,   width, height, (128,128,128))
    pair_left_yuv, pair_upper_yuv = neighbor_pairs(pixels_yuv, width, height, (128,128,128))

    entropy_info = {
        # 一階熵
        **{f"H({ch})": compute_entropy(vals)
           for ch, vals in {"R": R_list, "G": G_list, "B": B_list, "Y": Y_list, "U": U_list, "V": V_list}.items()},
        # 聯合熵
        "Joint H(R,G,B)": compute_entropy(pixels),
        "Joint H(Y,U,V)": compute_entropy(list(zip(Y_list, U_list, V_list))),
        # 條件熵
        "Conditional H(RGB|Left)": compute_conditional_entropy(pair_left_rgb),
        "Conditional H(RGB|Upper)": compute_conditional_entropy(pair_upper_rgb),
        "Conditional H(YUV|Left)": compute_conditional_entropy(pair_left_yuv),
        "Conditional H(YUV|Upper)": compute_conditional_entropy(pair_upper_yuv),
        "Conditional H(Y|Left)": compute_conditional_entropy(pair_left_y),
        "Conditional H(Y|Upper)": compute_conditional_entropy(pair_upper_y),
    }

    return pixels, rgb_to_y_channel(pixels), entropy_info

# ---------------------------
# 主程式
# ---------------------------
def main():
    """主程式：
    - 處理 "Interleave" 與 "Plane" 兩個資料夾中的影像
    - 使用 Huffman 演算法分別對原始 Y 資料與 DPCM 後的 Y 資料進行編碼，並計算壓縮比率
    - 輸出各項熵資訊與壓縮結果
    """
    folders = ["Interleave", "Plane"]
    image_files = ["baboonRGB.raw", "lenaRGB.raw"]
    output_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Results", "Huffman")
    os.makedirs(output_root, exist_ok=True)
    mode_dict = {"Interleave": "RGB", "Plane": "Plane"}
    
    for folder in folders:
        print(f"處理資料夾 {folder} ...")
        base_out = os.path.join(output_root, folder)
        os.makedirs(base_out, exist_ok=True)
        compression_folder = os.path.join(base_out, "Compression")
        os.makedirs(compression_folder, exist_ok=True)
        
        for image_file in image_files:
            filepath = os.path.join("..", folder, image_file)
            print(f"讀取 {filepath} ...")
            try:
                pixels, y_values, ent_info = process_image(filepath, mode_dict[folder])
            except Exception as e:
                print(f"    失敗: {e}")
                continue

            print(f"Entropy for {image_file}:")
            print(f"H(R): {ent_info['H(R)']:.4f} | H(G): {ent_info['H(G)']:.4f} | H(B): {ent_info['H(B)']:.4f}, "
                  f"H(Y): {ent_info['H(Y)']:.4f} | H(U): {ent_info['H(U)']:.4f} | H(V): {ent_info['H(V)']:.4f}")
            print(f"Joint H(R,G,B): {ent_info['Joint H(R,G,B)']:.4f} | Joint H(Y,U,V): {ent_info['Joint H(Y,U,V)']:.4f}")
            print(f"Conditional H(RGB|Left): {ent_info['Conditional H(RGB|Left)']:.4f} | Conditional H(RGB|Upper): {ent_info['Conditional H(RGB|Upper)']:.4f}")
            print(f"Conditional H(YUV|Left): {ent_info['Conditional H(YUV|Left)']:.4f} | Conditional H(YUV|Upper): {ent_info['Conditional H(YUV|Upper)']:.4f}")
            print(f"Conditional H(Y|Left): {ent_info['Conditional H(Y|Left)']:.4f} | Conditional H(Y|Upper): {ent_info['Conditional H(Y|Upper)']:.4f}")
            
            original_size = len(y_values)
            # (1) 對原始 Y 資料進行 Huffman 壓縮 (原始)、(2) 對 DPCM 後的 Y 資料進行 Huffman 壓縮
            data_groups = [
                ("huffman", y_values, "原始"),
                ("dpcm_huffman", dpcm_transform(y_values), "DPCM")
            ]

            for suffix, data, label in data_groups:
                out_file = os.path.join(compression_folder, f"{image_file.split('.')[0]}_{suffix}.txt")
                comp_size = process_encoding(data, out_file)
                ratio = comp_size / original_size * 100
                print(f"    {label} Y Huffman: {out_file}")
                print(f"    ({label}) 原始 Y 大小: {original_size} bytes, 壓縮後: {comp_size} bytes, 壓縮率: {ratio:.2f}%\n")
            
if __name__ == "__main__":
    main()
