# JPEG-like 壓縮與重建

使用 Python 實作簡易 JPEG-like 壓縮流程

## 專案結構
```
├── main.py     # 主程式：壓縮與重建
├── image       # 輸入 .raw 圖像資料夾
├── result
│   ├── raw     # 輸出重建後的 .raw 檔
│   └── png     # 輸出重建後的 .png 圖檔
```

## 使用說明

### 指令
執行 Huffman 壓縮與解壓縮：
```bash
python main.py
```

## 輸出結果
產生的 .raw 和 .jpeg 檔將分別存放在 result 資料夾底下的 raw 和 jpeg 子資料夾底下
