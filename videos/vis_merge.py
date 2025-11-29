import subprocess
import os

# --- 설정 상수 ---
CAPTION_HEIGHT = 50
FONT_SIZE = 45                
TEXT_OFFSET_Y = 5             
FONT_NAME = "Varela Round Bold"    
PAD_COLOR = "white"           # 상단 칸 배경색 (#ffffff)
FONT_COLOR = "0x000000"       # 글꼴 색상 (#000000)

VIDEO_PADDING = 20            # 영상 주위에 추가할 여백 (픽셀)

def merge_videos_with_padding_and_caption(input_video_paths, output_video_path, crf=18, target_fps=10):
    """
    영상 사이에 흰색 여백을 추가하고 상단에 자막 칸을 만들어 1x4 그리드를 병합합니다.
    (hstack color 옵션 대신 pad color 옵션 사용)
    """
    if len(input_video_paths) != 4:
        print("오류: 정확히 4개의 입력 영상 경로가 필요합니다.")
        return

    output_codec = "h264"
    # ***STACK_COLOR가 오류를 일으키므로, 이 상수는 더 이상 사용하지 않습니다.***

    filter_graph = ""
    stream_labels = [] 

    for i in range(len(input_video_paths)):
        # 스트림 라벨 정의
        v_synced = f"[v_synced_{i}]"
        v_padded = f"[v_padded_{i}]"
        v_final_padded = f"[v_final_padded_{i}]" 
        v_final_caption = f"[v_final_caption_{i}]" 
        stream_labels.append(v_final_caption)
        
        view_number = i + 1
        caption_text = f"view {view_number}"
        
        # 1. FPS 통일 (fps 필터)
        filter_graph += f"[{i}:v]fps=fps={target_fps}{v_synced}; "
        
        # 2. 영상 주위에 흰색 여백 추가 (pad 필터)
        # ***수정: color=white를 명시적으로 지정하여 여백을 흰색으로 채웁니다.***
        filter_graph += f"{v_synced}pad=width=iw+{VIDEO_PADDING*2}:height=ih+{VIDEO_PADDING*2}:x={VIDEO_PADDING}:y={VIDEO_PADDING}:color=white{v_final_padded}; "
        
        # 3. 상단에 지정된 색상(white)의 캡션 칸 추가 (pad 필터)
        filter_graph += f"{v_final_padded}pad=width=iw:height=ih+{CAPTION_HEIGHT}:x=0:y={CAPTION_HEIGHT}:color={PAD_COLOR}{v_padded}; "
        
        # 4. 텍스트 추가 (drawtext 필터)
        drawtext_options = f"text='{caption_text}':x=(w-text_w)/2:y={TEXT_OFFSET_Y}:fontcolor={FONT_COLOR}:fontsize={FONT_SIZE}:font='{FONT_NAME}'"
        filter_graph += f"{v_padded}drawtext={drawtext_options}{v_final_caption}; "

    # 5. 최종적으로 모든 스트림을 수평 병합 (hstack 필터)
    # ***수정: color=white 옵션을 제거합니다.***
    hstack_inputs = "".join(stream_labels)
    filter_graph += f"{hstack_inputs}hstack=inputs={len(input_video_paths)}[v]"

    # FFmpeg 명령어 구성
    ffmpeg_command = [
        "ffmpeg",
        *sum([["-i", path] for path in input_video_paths], []), 
        
        "-filter_complex", filter_graph,
        
        "-map", "[v]",
        "-map", "0:a?", 
        
        "-c:v", output_codec,
        "-crf", str(crf), 
        
        output_video_path.replace(".webm", ".mp4") 
    ]

    print(f"생성된 FFmpeg 명령어: {' '.join(ffmpeg_command)}")

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"'{output_video_path.replace('.webm', '.mp4')}' 파일이 성공적으로 생성되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"오류: FFmpeg 명령어 실행 중 문제가 발생했습니다: {e}")
    except FileNotFoundError:
        print("오류: FFmpeg를 찾을 수 없습니다. FFmpeg가 시스템 PATH에 설치되어 있는지 확인하세요.")


if __name__ == "__main__":
    # --- 실행 설정 ---
    
    input_videos_list = [
        ["dexycb-437.mp4", "dexycb-430.mp4", "dexycb-433.mp4", "dexycb-436.mp4"],
        ["dexycb-017.mp4", "dexycb-014.mp4", "dexycb-015.mp4", "dexycb-010.mp4"],
        ["dexycb-100.mp4", "dexycb-107.mp4", "dexycb-103.mp4", "dexycb-104.mp4"],
        ["dexycb-300.mp4", "dexycb-307.mp4", "dexycb-303.mp4", "dexycb-305.mp4"],
        ["dexycb-560.mp4", "dexycb-564.mp4", "dexycb-563.mp4", "dexycb-566.mp4"],
        ["dexycb-741.mp4", "dexycb-744.mp4", "dexycb-746.mp4", "dexycb-743.mp4"],
        ["dexycb-804.mp4", "dexycb-807.mp4", "dexycb-806.mp4", "dexycb-801.mp4"],
    ]
    for input_videos in input_videos_list:
        output_file = input_videos[0][:9] + ".mp4"
        target_crf = 18 
        target_fps = 10  

        # --- 함수 호출 ---
        merge_videos_with_padding_and_caption(input_videos, output_file, target_crf, target_fps)

