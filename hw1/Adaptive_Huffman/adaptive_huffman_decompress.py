import os
from PIL import Image
from tqdm import tqdm

IMAGE_WIDTH ,IMAGE_HEIGHT= 512, 512

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
    '''取得從根到該節點的編碼路徑'''
    code = ""
    while node.parent:
        code = ("0" if node.parent.left == node else "1") + code
        node = node.parent
    return code

def get_all_nodes(node):
    '''取得整棵樹的所有節點（先序遍歷）'''
    return [node] + get_all_nodes(node.left) + get_all_nodes(node.right) if node else []

def is_ancestor(a, b):
    '''判斷 a 是否為 b 的祖先'''
    while b.parent:
        if b.parent == a:
            return True
        b = b.parent
    return False

def swap_nodes(node1, node2):
    '''交換兩個節點的位置（包括 parent 連結與 order）'''
    if not node1.parent or not node2.parent:
        return
    parent1, parent2 = node1.parent, node2.parent
    if parent1.left == node1:
        parent1.left = node2
    else:
        parent1.right = node2
    if parent2.left == node2:
        parent2.left = node1
    else:
        parent2.right = node1
    node1.parent, node2.parent = parent2, parent1
    node1.order, node2.order = node2.order, node1.order

def find_block_leader(root, weight, current_order, current_node):
    '''尋找同權重且 order 較大、又非祖先的區塊領導者'''
    return max(
        (node for node in get_all_nodes(root) 
         if node.weight == weight and node.order > current_order 
         and node != current_node and not is_ancestor(node, current_node)),
        key=lambda x: x.order, 
        default=None
    )

def update_tree(current_node, root):
    '''自下而上更新樹（FGK 演算法）'''
    while current_node:
        # 找出同權重區塊的領導者
        leader = find_block_leader(root, current_node.weight, current_node.order, current_node)
        if leader and leader != current_node.parent:
            swap_nodes(current_node, leader)
        current_node.weight += 1
        current_node = current_node.parent # 往上處理下一層

def parse_encoded_file(filepath):
    '''讀取 .txt 編碼結果：回傳 (fixed_bit_width, 去除填充後的 bitstream)'''
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    extra_bits, fixed_bit_width, bitstream = 0, 8, ""
    for idx, line in enumerate(lines):
        if line.startswith("Fixed bit width for new symbols:"):
            fixed_bit_width = int(line.split(":")[1].strip())
        if line.startswith("Encoded Bitstream:"):
            bitstream = "".join(l.strip() for l in lines[idx + 1:])
            break
        if line.startswith("Extra bits"):
            extra_bits = int(line.split(":")[1].strip())
    return fixed_bit_width, bitstream[:-extra_bits] if extra_bits else bitstream

def adaptive_huffman_decode(bitstream, fixed_bit_width, is_dpcm=False):
    '''解碼函式：同樣動態重建 Huffman 樹並讀出符號'''
    MAX_ORDER = 512
    root = AHNode(weight=0, order=MAX_ORDER, isNYT=True)
    decoded, index = [], 0

    with tqdm(total=len(bitstream), desc="Decoding bits", unit="bit", leave=False) as pbar:
        while index < len(bitstream):
            node = root

            # 路徑遍歷：沿樹向下，直到遇到葉節點
            while node.left and node.right:
                bit = bitstream[index]
                node = node.left if bit == "0" else node.right
                index += 1
                pbar.update(1)

            # 若為 NYT，表示接下來讀固定長度新符號
            if node.isNYT:
                # 讀取 fixed_bit_width 位元作為符號
                fixed_code = bitstream[index:index + fixed_bit_width]
                index += fixed_bit_width
                pbar.update(fixed_bit_width)

                # 將二進位轉整數
                symbol = int(fixed_code, 2)

                # 若是 DPCM，要做二補數還原
                symbol = symbol - (1 << fixed_bit_width) if is_dpcm and symbol >= (1 << (fixed_bit_width - 1)) else symbol
                decoded.append(symbol)

                # 把 NYT 分裂，新增新 NYT 與葉節點
                node.isNYT = False
                new_NYT = AHNode(order=node.order - 2, isNYT=True)
                new_leaf = AHNode(weight=1, symbol=symbol, order=node.order - 1)
                node.left, node.right = new_NYT, new_leaf
                new_NYT.parent, new_leaf.parent = node, node
                update_tree(new_leaf, root) # 更新樹結構
            else:
                # 已見符號，直接加到 output
                decoded.append(node.symbol)
                update_tree(node, root)

    return decoded

def inverse_dpcm(dpcm_vals, width=IMAGE_WIDTH):
    '''將 DPCM 差分逆運算還原到原始亮度值'''
    restored, idx = [], 0
    for _ in range(len(dpcm_vals) // width):
        row = [dpcm_vals[idx] + 128]
        idx += 1
        for i in range(1, width):
            row.append(row[-1] + dpcm_vals[idx])
            idx += 1
        restored.extend(row)
    return restored

def save_png(y_values, out_filepath):
    '''儲存灰階影像到 PNG'''
    if len(y_values) != IMAGE_WIDTH * IMAGE_HEIGHT:
        raise ValueError("資料長度與影像尺寸不符")
    y_values = [max(0, min(255, v)) for v in y_values]
    img = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT))
    img.putdata(y_values)
    img.save(out_filepath)
    print(f"已解壓縮並儲存 {out_filepath}")

def main():
    '''主流程：遍歷 Interleave/Plane 資料夾進行解壓'''
    input_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "Adaptive_Huffman")

    for folder in ["Interleave", "Plane"]:
        comp_folder = os.path.join(input_root, folder, "Compression")
        out_folder = os.path.join(input_root, folder, "Decompression")
        os.makedirs(out_folder, exist_ok=True)

        if not os.path.isdir(comp_folder):
            print(f"輸入目錄不存在: {comp_folder}")
            continue

        txt_files = [f for f in os.listdir(comp_folder) if f.endswith(".txt")]
        for filename in tqdm(txt_files, desc=f"Decompressing in {folder}", unit="file", leave=False):
            in_txt_path = os.path.join(comp_folder, filename)
            base_name = os.path.splitext(filename)[0].replace("_adaptive", "").replace("_dpcm_adaptive", "")
            is_dpcm = "dpcm" in filename.lower()
            try:
                fixed_bit_width, bitstream = parse_encoded_file(in_txt_path)
                decoded = adaptive_huffman_decode(bitstream, fixed_bit_width, is_dpcm=is_dpcm)
                if is_dpcm:
                    decoded = inverse_dpcm(decoded, width=IMAGE_WIDTH)
                save_png(decoded, os.path.join(out_folder, base_name + ".png"))
            except Exception as e:
                print(f"解壓縮 {in_txt_path} 時發生錯誤: {e}")

if __name__ == "__main__":
    main()
