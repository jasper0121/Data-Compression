#!/bin/bash
# 進入 Huffman 資料夾並執行程式
echo "進入 Huffman 資料夾，執行 huffman.py 與 huffman_decompress.py"
cd Huffman || { echo "找不到 Huffman 資料夾"; exit 1; }
python huffman.py
python huffman_decompress.py
cd ..  # 返回上一層

# 進入 Adaptive_Huffman 資料夾並執行程式
echo "進入 Adaptive_Huffman 資料夾，執行 adaptive_huffman.py 與 adaptive_huffman_decompress.py"
cd Adaptive_Huffman || { echo "找不到 Adaptive_Huffman 資料夾"; exit 1; }
python adaptive_huffman.py
python adaptive_huffman_decompress.py
cd ..  # 返回上一層

echo "所有程式執行完畢"
