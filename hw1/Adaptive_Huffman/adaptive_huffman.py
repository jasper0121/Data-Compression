import os
from tqdm import tqdm

# Adaptive Huffman 節點
class AHNode:
    def __init__(self, weight=0, symbol=None, order=0, isNYT=False):
        self.weight = weight      # 節點權重
        self.symbol = symbol      # 葉節點儲存的符號
        self.order = order        # FGK 算法用來排序的編號
        self.isNYT = isNYT        # 是否是 NYT 節點
        self.parent = None        # 指向父節點
        self.left = None          # 指向左子節點
        self.right = None         # 指向右子節點

def get_code(node):
    '''取得節點編碼（由根至葉的路徑）'''
    code = ""
    while node.parent:
        code = ("0" if node.parent.left == node else "1") + code
        node = node.parent
    return code

def iter_nodes(node):
    '''使用生成器遍歷所有節點'''
    if node:
        yield node
        yield from iter_nodes(node.left)
        yield from iter_nodes(node.right)

def is_ancestor(a, b):
    '''判斷 a 是否為 b 的祖先'''
    current = b.parent
    while current:
        if current == a:
            return True
        current = current.parent
    return False

def swap_nodes(node1, node2):
    '''交換兩節點在樹中的位置（包括 order 與 parent 指標）'''
    if not node1.parent or not node2.parent:
        return
    p1, p2 = node1.parent, node2.parent
    if p1.left == node1:
        p1.left = node2
    else:
        p1.right = node2
    if p2.left == node2:
        p2.left = node1
    else:
        p2.right = node1
    node1.parent, node2.parent = p2, p1
    node1.order, node2.order = node2.order, node1.order

def find_block_leader(root, weight, current_order, current_node):
    '''尋找與 current_node 同權重且 order 較大且非祖先的節點（block leader）'''
    leader = None
    for node in iter_nodes(root):
        if (node.weight == weight and node.order > current_order and 
            node != current_node and not is_ancestor(node, current_node)):
            if leader is None or node.order > leader.order:
                leader = node
    return leader

def update_tree(current_node, root):
    '''自下而上更新樹（FGK 演算法）'''
    while current_node:
        leader = find_block_leader(root, current_node.weight, current_node.order, current_node)
        if leader and leader != current_node.parent:
            swap_nodes(current_node, leader)
        current_node.weight += 1
        current_node = current_node.parent

def adaptive_huffman_encode(data, fixed_bit_width=8):
    '''自適應 Huffman 編碼（data 為整數序列, fixed_bit_width 為新符號固定位數）'''
    MAX_ORDER = 512 # MAX_ORDER: 為樹中節點編號的起始最大值，用於維護 FGK 演算法的兄弟性質順序
    root = AHNode(order=MAX_ORDER, isNYT=True) # 初始化根節點為 NYT（Not Yet Transmitted）特殊節點
    NYT = root
    symbol_to_node = {}
    bitstream = ""
    
    for symbol in tqdm(data, desc="Encoding", unit="symbol", leave=False):
        if symbol in symbol_to_node:

            # 已見過的符號：直接從對應葉節點取得動態 Huffman 編碼
            node = symbol_to_node[symbol]
            bitstream += get_code(node)
        else:
            # 第一次出現的新符號：
            # (1) 先輸出目前 NYT 節點的編碼，通知解碼端接下來是新符號
            bitstream += get_code(NYT)

            # (2) 再輸出符號本身的固定長度二進位
            #     若 symbol 為負值，先加上 1<<fixed_bit_width 以處理二補數表示
            symbol_val = (1 << fixed_bit_width) + symbol if symbol < 0 else symbol
            bitstream += format(symbol_val, f'0{fixed_bit_width}b')

            # (3) 在 NYT 節點處分裂出兩個子節點：
            #     左子為新的 NYT，右子為此新符號的葉節點
            NYT.isNYT = False
            new_NYT = AHNode(order=NYT.order - 2, isNYT=True)
            new_leaf = AHNode(weight=1, symbol=symbol, order=NYT.order - 1, isNYT=False)
            NYT.left, NYT.right = new_NYT, new_leaf
            new_NYT.parent = new_leaf.parent = NYT

            # (4) 更新映射，方便下次遇到相同符號可直接查找葉節點
            symbol_to_node[symbol] = new_leaf

            # (5) 更新 NYT 指向到新分裂出的 NYT 節點
            NYT = new_NYT
            node = new_leaf # 從剛新增的葉節點開始更新樹權重

        # (6) 無論新舊符號，皆需從 node 自底向上呼叫更新樹的函式
        update_tree(node, root)
    return bitstream

def bits_to_bytes(bitstring):
    '''將 bit 字串轉為 byte 字串，並回傳填充位元數'''
    extra = (-len(bitstring)) % 8
    return bitstring + "0" * extra, extra

def write_encoded_txt(bitstring, extra, filename, fixed_bit_width):
    '''寫入編碼結果到檔案'''
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Header:\nAdaptive Huffman Encoding\n")
        f.write(f"Fixed bit width for new symbols: {fixed_bit_width}\n")
        f.write(f"Extra bits (padding): {extra}\n\nEncoded Bitstream:\n{bitstring}\n")

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

def compute_fixed_bit_width(dpcm_vals):
    '''計算 DPCM 差分所需固定 bit width（以兩補數表示）'''
    max_abs = max(abs(min(dpcm_vals)), abs(max(dpcm_vals)))
    n = 1
    while (1 << (n - 1)) <= max_abs:
        n += 1
    return n

def read_raw_image(filename, width=512, height=512, mode="RGB"):
    '''讀取 raw 影像資料與轉換為像素列表'''
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
    '''將 RGB 轉換為 Y 通道 (亮度)'''
    return [
        max(0, min(255, int(round(0.299 * R + 0.587 * G + 0.114 * B))))
        for (R, G, B) in pixels
    ]

def process_adaptive_encoding(data, out_filename, fixed_bit_width):
    '''呼叫 adaptive_huffman_encode 並寫入檔案'''
    bitstream = adaptive_huffman_encode(data, fixed_bit_width=fixed_bit_width)
    bitstring, extra = bits_to_bytes(bitstream)
    write_encoded_txt(bitstring, extra, out_filename, fixed_bit_width)
    return len(bitstring) // 8

def main():
    '''主程式：處理 "Interleave" 與 "Plane" 資料夾下的影像'''
    folders = ["Interleave", "Plane"]
    image_files = ["lenaRGB.raw", "baboonRGB.raw"]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    output_folder = os.path.join(parent_dir, "Results", "Adaptive_Huffman")
    os.makedirs(output_folder, exist_ok=True)
    mode_dict = {"Interleave": "RGB", "Plane": "Plane"}

    for folder in folders:
        print(f"處理資料夾 {folder} ...")
        base_out = os.path.join(output_folder, folder)
        os.makedirs(base_out, exist_ok=True)
        adaptive_out = os.path.join(base_out, "Compression")
        os.makedirs(adaptive_out, exist_ok=True)
        for image_file in image_files:
            filepath = os.path.join("..", folder, image_file)
            print(f"讀取 {filepath} ...")
            try:
                pixels = read_raw_image(filepath, mode=mode_dict[folder])
            except Exception as e:
                print(f"    失敗: {e}")
                continue

            y_values = rgb_to_y_channel(pixels)
            original_size = len(y_values)

            # 先將 DPCM 資料處理相關計算完成
            dpcm_vals = dpcm_transform(y_values)
            dpcm_fixed_bit = compute_fixed_bit_width(dpcm_vals)

            # (1) 處理原始 Y 資料、(2) 處理 DPCM 資料：先計算完整差分，再依據數據決定固定 bit width
            data_groups = [
                ("adaptive", y_values, "原始", 8),
                ("dpcm_adaptive", dpcm_vals, "DPCM", dpcm_fixed_bit)
            ]

            for suffix, data, label, fixed_bit in data_groups:  # 建立資料組合，每個 tuple 包含 (suffix, data, label, 固定位元數)
                out_file = os.path.join(adaptive_out, f"{image_file.split('.')[0]}_{suffix}.txt")
                comp_size = process_adaptive_encoding(data, out_file, fixed_bit)
                print(f"    Adaptive Huffman ({label}): {out_file}")
                print(f"    ({label}) 原始 Y: {original_size} bytes, 壓縮後: {comp_size} bytes, 壓縮率: {comp_size/original_size*100:.2f}%\n")

if __name__ == "__main__":
    main()
