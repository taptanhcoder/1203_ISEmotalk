import os
import time
import gc
from pydub import AudioSegment

def convert_audio_to_wav(input_path: str, output_dir: str = "data/predict_data") -> str:
    """
    Chuyển đổi file MP3, WEBM hoặc MP4 sang WAV và lưu vào thư mục output chỉ định.

    Args:
        input_path (str): Đường dẫn file âm thanh hoặc video đầu vào.
        output_dir (str): Thư mục lưu file WAV đã chuyển đổi.

    Returns:
        str: Đường dẫn file WAV đã được lưu.
    """
    total_start_time = time.time()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File {input_path} không tồn tại!")

    file_extension = os.path.splitext(input_path)[1].lower()
    supported_formats = [".mp3", ".webm", ".mp4"]

    if file_extension not in supported_formats:
        raise ValueError(f"Định dạng không hợp lệ! Hỗ trợ: {', '.join(supported_formats).upper()}.")

    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(input_path))[0] + ".wav"
    output_path = os.path.join(output_dir, file_name)

    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Đã xóa file WAV cũ: {output_path}")

    try:
        print(f"Đang chuyển đổi: {input_path} → {output_path}")

        load_start = time.time()
        audio = AudioSegment.from_file(input_path, format=file_extension[1:])
        load_time = time.time() - load_start
        print(f"Thời gian đọc file {file_extension.upper()}: {load_time:.6f} giây")

        export_start = time.time()
        audio.export(output_path, format="wav")
        export_time = time.time() - export_start
        print(f"Thời gian ghi file WAV: {export_time:.6f} giây")

        total_time = time.time() - total_start_time
        print(f"Tổng thời gian chuyển đổi: {total_time:.6f} giây")


        del audio
        gc.collect()

        if not os.path.exists(output_path):
            raise RuntimeError(f"Lỗi: File WAV không được tạo: {output_path}")

        print(f"Chuyển đổi thành công: {output_path}")

        return output_path

    except Exception as e:
        raise RuntimeError(f"Lỗi khi xử lý file {file_extension.upper()}: {e}")

if __name__ == "__main__":
    input_file = "data/raw_data/Sad (2).MP3"  
    output_directory = "data/predict_data"

    try:
        output_path = convert_audio_to_wav(input_file, output_directory)
        print(f"File đã được chuyển đổi và lưu tại: {output_path}")
    except Exception as e:
        print(f"Lỗi trong quá trình chuyển đổi: {e}")