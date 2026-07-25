import json

def filter_jsonl_by_job_id(original_path, filter_path, output_path):
    print("開始執行：收集 Filter 檔案中的 job_id...")
    
    # 使用 set 來儲存 job_id，以獲得最快的搜尋速度
    valid_job_ids = set()
    
    # 步驟 1：讀取過濾名單
    with open(filter_path, 'r', encoding='utf-8') as f_filter:
        for line in f_filter:
            # 去除頭尾空白後解析 JSON
            data = json.loads(line.strip())
            # 確保該行確實包含 job_id 欄位
            if 'job_id' in data:
                valid_job_ids.add(data['job_id'])
                
    print(f"收集完成！共找到 {len(valid_job_ids)} 個不重複的 job_id。")
    print("開始執行：過濾 Original 檔案並寫入新檔案...")
    
    # 步驟 2：讀取原始檔案並進行比對過濾
    matched_count = 0
    with open(original_path, 'r', encoding='utf-8') as f_original, \
         open(output_path, 'w', encoding='utf-8') as f_output:
        
        for line in f_original:
            data = json.loads(line.strip())
            # 檢查原始檔案的 job_id 是否在我們的過濾名單中
            if data.get('job_id') in valid_job_ids:
                # 為了保留原始格式並提升效能，我們直接寫入原始的字串(line)，而不是重新打包 JSON
                f_output.write(line)
                matched_count += 1
                
    print(f"處理完成！已將 {matched_count} 筆符合的資料寫入至：{output_path}")

# --- 參數設定區 ---
# 這裡放置你提供的檔案路徑，並自訂一個新的輸出路徑
FILTER_PATH = 'processed_iterable_dataset/test_50k_clean_90.jsonl'
ORIGINAL_PATH = 'processed_iterable_dataset_bge/test_50k.jsonl'
# 這是過濾後的新檔案名稱，你可以依據需求調整
OUTPUT_PATH = 'processed_iterable_dataset_bge/test_50k_filtered.jsonl'

# 執行主程式
if __name__ == '__main__':
    filter_jsonl_by_job_id(ORIGINAL_PATH, FILTER_PATH, OUTPUT_PATH)