import os
import math
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct

# --- 定義標準 JPEG 量化表 ---
# QY_std: 亮度量化表
QY_std = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
])
# QC_std: 色差量化表
QC_std = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

# --- 工具函式區 ---
def scale_quant_table(Q_ㄈstd, QF):
    """
    根據 Quality Factor (QF) 縮放標準量化表，
    QF<50 時 scale=5000/QF，否則 scale=200-2*QF；
    四捨五入後 clip 至 [1,255]。
    """
    scale = 5000 / QF if QF < 50 else 200 - 2 * QF
    Q = np.floor((Q_std * scale + 50) / 100)
    return np.clip(Q, 1, 255).astype(np.int32)

def dct2(block):
    """對 8×8 block 執行 2D 正交 DCT。"""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """對 8×8 block 執行 2D 正交反 DCT。"""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def psnr(orig, reco):
    """計算原圖與重建圖的 PSNR，MSE=0 回傳 inf。"""
    mse = np.mean((orig.astype(np.float32) - reco.astype(np.float32)) ** 2)
    return float('inf') if mse == 0 else 10 * math.log10(255 ** 2 / mse)

# --- I/O functions (合併灰階/RGB) ---
def read_raw(path, channels=1):
    data = np.fromfile(path, dtype=np.uint8)
    if channels > 1 and data.size % channels != 0:
        raise ValueError(f"Size {data.size} not divisible by {channels}")
    pixels = data.size // channels
    side = int(math.isqrt(pixels))
    if side * side != pixels:
        raise ValueError(f"Cannot reshape array of {pixels} pixels into square")
    return data.reshape((side, side, channels)) if channels > 1 else data.reshape((side, side))

def write_raw(path, img):
    img.astype(np.uint8).tofile(path)

# --- Zigzag scan ---
zigzag_index = [
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
]

def zigzag(block):
    """將 8×8 block 按 ZigZag 攤平成長度 64 向量。"""
    return block.flatten()[zigzag_index]

# --- Huffman builder & tables ---
def build_huffman(bits, vals):
    """
    根據 JPEG bits/vals 生成 (value -> (code, length)) 的字典。
    bits: 每個 code 長度出現次數，vals: 對應的值。
    """
    sizes = []
    for i, count in enumerate(bits, start=1):
        sizes += [i] * count
    codes = []
    code = 0
    si = sizes[0] if sizes else 0
    for s in sizes:
        while s > si:
            code <<= 1
            si += 1
        codes.append(code)
        code += 1
    return {v: (c, size) for v, c, size in zip(vals, codes, sizes)}

# 定義四種 Huffman bits/vals
DCY_bits = [0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0]
DCY_vals = list(range(12))
DCC_bits = [0,3,1,1,1,1,1,1,1,0,0,0,0,0,0,0]
DCC_vals = list(range(12))
ACY_bits = [0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,125]
ACY_vals = [
    0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,
    0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,
    0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
    0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,
    0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,
    0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
    0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,
    0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,
    0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,
    0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa
]
ACC_bits = [0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,119]
ACC_vals = [
    0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,
    0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xa1,0xb1,0xc1,0x09,0x23,0x33,0x52,0xf0,
    0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,
    0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,
    0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,
    0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,
    0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,
    0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,
    0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,
    0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa
]
# 建立最終的 Huffman 編碼字典
huff_DCY = build_huffman(DCY_bits, DCY_vals)
huff_DCC = build_huffman(DCC_bits, DCC_vals)
huff_ACY = build_huffman(ACY_bits, ACY_vals)
huff_ACC = build_huffman(ACC_bits, ACC_vals)

# --- 位元寫入器 (BitWriter) ---
class BitWriter:
    """將串接的 code bits 寫入 byte 並自動處理 0xFF 後插入 0x00 的填充需求。"""
    def __init__(self):
        self.buffer = 0
        self.nbits = 0
        self.data = bytearray()
    def write_bits(self, code, length):
        for i in range(length - 1, -1, -1):
            bit = (code >> i) & 1
            self.buffer = (self.buffer << 1) | bit
            self.nbits += 1
            if self.nbits == 8:
                self.data.append(self.buffer)
                # 若剛好等於 0xFF，JPEG 規範需插入 0x00
                if self.buffer == 0xFF:
                    self.data.append(0x00)
                self.buffer = 0
                self.nbits = 0
    def flush(self):
        """將剩餘 bits 左移補齊一個 byte 並加入 data。"""
        if self.nbits > 0:
            self.data.append(self.buffer << (8 - self.nbits))
            self.nbits = 0
    def get_bytes(self):
        """回傳累積的 byte array。"""
        return bytes(self.data)

# --- 單一 8×8 區塊編碼 (包含 DC 差分與 AC-RLE/Huffman) ---
def encode_block(z, prev_dc, huff_dc, huff_ac):
    """
    輸入 zigzag 後的 64 維向量 z，
    prev_dc: 前一塊的 DC 值；huff_dc/huff_ac: Huffman 表
    回傳 (encoded_bytes, current_dc) 供後續串接。
    """
    bw = BitWriter()
    dc = int(z[0])
    diff = dc - prev_dc

    # --- DC 編碼: size + 符號值 ---
    size = 0 if diff == 0 else int(math.floor(math.log2(abs(diff)))) + 1
    code, length = huff_dc[size]
    bw.write_bits(code, length)
    if size > 0:
        # 若為負數，依 JPEG 規範做補碼調整
        if diff < 0:
            diff = diff - 1 + (1 << size)
        bw.write_bits(diff & ((1 << size) - 1), size)

    # --- AC 編碼: run-length + Huffman ---
    run = 0
    for k in range(1, 64):
        v = int(z[k])
        if v == 0:
            run += 1
        else:
            # 超過 15 個零要先插入 ZRL (0xF0)
            while run > 15:
                c, l = huff_ac[0xF0]
                bw.write_bits(c, l)
                run -= 16
            # 計算 v 的 bit-length
            s = int(math.floor(math.log2(abs(v)))) + 1
            rs = (run << 4) | s
            c, l = huff_ac[rs]
            bw.write_bits(c, l)
            # 負數補碼處理
            if v < 0:
                v = v - 1 + (1 << s)
            bw.write_bits(v & ((1 << s) - 1), s)
            run = 0

    # 結尾 EOB
    if run > 0:
        c, l = huff_ac[0x00]
        bw.write_bits(c, l)

    bw.flush()
    return bw.get_bytes(), dc

# --- 壓縮 & 重建函式 ---
def compress_entropy(path, channels, q_tables, h_tables):
    img = read_raw(path, channels)
    arr = img.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    H, W, C = arr.shape
    prev = [0] * C
    stream = bytearray()
    for c in range(C):
        Qt = q_tables[c]
        hDC, hAC = h_tables[c]
        for i in range(0, H, 8):
            for j in range(0, W, 8):
                blk = arr[i:i+8, j:j+8, c] - 128
                q = np.round(dct2(blk) / Qt).astype(int)
                z = zigzag(q)
                bs, prev[c] = encode_block(z, prev[c], hDC, hAC)
                stream.extend(bs)
    return bytes(stream)

def reconstruct(img, q_tables):
    """
    重建影像 (逆量化 + 2D 反 DCT):
    1. 轉成 float，若為灰階則加上 channel 維度
    2. 對每個 channel, 每個 8×8 區塊：
       a. 對應量化表逆量化
       b. 執行 2D 反 DCT
       c. 加上 128 並 clip 至 [0,255]
    3. 若只有一個 channel，回傳 2D 重建結果，否則回傳 uint8 RGB array
    """
    arr = img.astype(np.float32)
    # 若灰階，擴維度至 H×W×1
    if arr.ndim == 2:
        arr = arr[:, :, None]
    H, W, C = arr.shape
    reco = np.zeros_like(arr)

    for c in range(C):
        Qt = q_tables[c] # 取得該通道的量化表
        # 以 8×8 區塊為單位進行遍歷
        for i in range(0, H, 8):
            for j in range(0, W, 8):
                block = arr[i:i+8, j:j+8, c] - 128  # 像素減 128，中心化到 [-128,127]
                coef = np.round(dct2(block) / Qt) * Qt # 先進行 DCT，再量化和逆量化（還原量化誤差）
                recon = idct2(coef) + 128 # 將量化後的係數做反 DCT，並加回 128
                reco[i:i+8, j:j+8, c] = np.clip(recon, 0, 255) # 將像素值限制在 [0,255] 範圍
    # 如果只有單一通道，回傳二維影像，否則回傳三維(彩色)影像
    return reco[:, :, 0] if C == 1 else reco.astype(np.uint8)

# --- 建立輸出資料夾 ---
def prepare_dirs():
    rd, pd = "result/raw", "result/jpeg"
    for d in (rd, pd):
        os.makedirs(d, exist_ok=True)
    return rd, pd

# --- Main entry point ---
def main():
    idir = "image"  # 輸入資料夾
    rd, pd = prepare_dirs()
    QFs = [90, 80, 50, 20, 10, 5]

    files = [f for f in os.listdir(idir) if f.lower().endswith('.raw')]
    for fn in files:
        ip = os.path.join(idir, fn)
        base = os.path.splitext(fn)[0]
        is_color = 'rgb' in fn.lower()
        channels = 3 if is_color else 1
        img = read_raw(ip, channels)

        for QF in QFs:
            QY = scale_quant_table(QY_std, QF)
            QC = scale_quant_table(QC_std, QF) if is_color else None
            q_tables = [QY] if channels == 1 else [QY, QC, QC]
            h_tables = [(huff_DCY, huff_ACY)] if channels == 1 else [(huff_DCY, huff_ACY), (huff_DCC, huff_ACC), (huff_DCC, huff_ACC)]

            # 重建並輸出
            reco = reconstruct(img, q_tables)
            write_raw(os.path.join(rd, f"{base}_QF{QF}.raw"), reco)
            mode = 'L' if channels == 1 else 'RGB'
            Image.fromarray(reco.astype(np.uint8), mode=mode).save(os.path.join(pd, f"{base}_QF{QF}.jpeg"))

            # 壓縮並計算統計
            p = psnr(img, reco) # 計算 PSNR
            bs = compress_entropy(ip, channels, q_tables, h_tables)
            comp_bits = len(bs) * 8
            orig_bits = img.size * channels * 8
            ratio = orig_bits / comp_bits
            bpp = comp_bits / img.size

            tag = 'COLOR' if channels == 3 else 'GRAY'
            print(f"[{tag}] {fn} QF={QF}: PSNR={p:.2f} dB, Bytes={len(bs)}, Ratio={ratio:.2f}:1, bpp={bpp:.3f}")

if __name__ == "__main__":
    main()
