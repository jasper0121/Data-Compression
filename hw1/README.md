# Huffman 與 Adaptive Huffman 壓縮

使用 Huffman 和 Adaptive Huffman 對Y通道做壓縮與解壓縮。

## 專案結構
```
├── Huffman
│   ├── huffman.py                     # Huffman 壓縮
│   ├── huffman_decompress.py          # Huffman 解壓縮
├── Adaptive_Huffman
│   ├── adaptive_huffman.py            # Adaptive Huffman 壓縮
│   ├── adaptive_huffman_decompress.py # Adaptive Huffman 解壓縮
├── Results                            # 產生的壓縮檔與解壓縮的灰階圖
└── run.sh                             # 一鍵執行腳本
```

## 環境需求
- pip install -r requirements.txt

## 使用說明

### 指令
1. 執行 Huffman 壓縮與解壓縮：
    ```bash
    cd Huffman
    python huffman.py
    python huffman_decompress.py
    cd ..
    ```
2. 執行 Adaptive Huffman 壓縮與解壓縮：
    ```bash
    cd Adaptive_Huffman
    python adaptive_huffman.py
    python adaptive_huffman_decompress.py
    cd ..
    ```

### 或是一鍵執行
```bash
bash run.sh
```

## 輸出結果
產生的壓縮檔案以及解壓縮後的灰階影像皆儲存在 `Results` 資料夾中。
