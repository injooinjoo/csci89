import os
import sys
import subprocess

def convert_tex(tex_file):
    if not os.path.exists(tex_file):
        print(f"파일을 찾을 수 없습니다: {tex_file}")
        return

    # 파일 디렉토리로 이동
    tex_dir = os.path.dirname(os.path.abspath(tex_file))
    tex_name = os.path.basename(tex_file)

    print(f"변환 중: {tex_name}")

    # xelatex 사용 (한글 지원)
    try:
        result = os.system(f'cd "{tex_dir}" && xelatex "{tex_name}"')
        if result == 0:
            print("✅ 변환 성공!")
        else:
            print("❌ 변환 실패")
    except:
        print("❌ xelatex 실행 실패")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python simple_tex_converter.py <파일.tex>")
    else:
        convert_tex(sys.argv[1])