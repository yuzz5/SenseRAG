import os
import sqlite3
import time
import torchaudio

# --- 配置 ---
AUDIO_DIR = "/root/autodl-tmp/Multimodal_AIGC/audio/clips_wav"  # 修改为指定目录
DB_PATH = "/root/autodl-tmp/Multimodal_AIGC/audio_index.db"
BATCH_SIZE = 1000
# --- 配置结束 ---


def setup_database(db_path):
    """创建数据库表和索引"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            duration_seconds REAL,
            sample_rate INTEGER
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON audio_files(filename)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_duration ON audio_files(duration_seconds)')
    conn.commit()
    conn.close()

def get_audio_files(audio_dir):
    """收集音频文件路径"""
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if os.path.splitext(file.lower())[1] in audio_extensions:
                audio_files.append(os.path.join(root, file))
    return audio_files

def index_files(audio_files, db_path, batch_size):
    """处理文件元数据并存入数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    batch_data = []
    start_time = time.time()

    for audio_path in audio_files:
        try:
            stat_info = os.stat(audio_path)
            size_bytes = stat_info.st_size
            filename = os.path.basename(audio_path)

            duration_seconds, sample_rate = None, None
            try:
                metadata = torchaudio.info(audio_path)
                sample_rate = metadata.sample_rate
                if sample_rate > 0:
                    duration_seconds = metadata.num_frames / sample_rate
            except Exception:
                pass # 元数据获取失败，保持 None

            batch_data.append((audio_path, filename, size_bytes, duration_seconds, sample_rate))

            if len(batch_data) >= batch_size:
                cursor.executemany('''
                    INSERT OR IGNORE INTO audio_files 
                    (filepath, filename, size_bytes, duration_seconds, sample_rate)
                    VALUES (?, ?, ?, ?, ?)
                ''', batch_data)
                conn.commit()
                batch_data = []

        except Exception as e:
            print(f"处理文件时出错 ({audio_path}): {e}")

    # 插入最后一批数据
    if batch_data:
        cursor.executemany('''
            INSERT OR IGNORE INTO audio_files 
            (filepath, filename, size_bytes, duration_seconds, sample_rate)
            VALUES (?, ?, ?, ?, ?)
        ''', batch_data)
        conn.commit()

    conn.close()
    end_time = time.time()
    print(f"索引完成，耗时: {end_time - start_time:.2f} 秒")


def main():
    """主函数"""
    print(f"开始索引音频文件: {AUDIO_DIR}")
    print(f"数据库路径: {DB_PATH}")

    setup_database(DB_PATH)
    audio_files = get_audio_files(AUDIO_DIR)
    
    if not audio_files:
        print("未找到音频文件。")
        return

    print(f"找到 {len(audio_files)} 个文件，开始处理...")
    index_files(audio_files, DB_PATH, BATCH_SIZE)


if __name__ == "__main__":
    main()



