# File Converters

이 디렉토리에는 파일 변환을 위한 Python 스크립트들이 있습니다.

## 1. notebook_to_pdf.py - Jupyter Notebook을 HTML/PDF로 변환

### 사용법:
```bash
python notebook_to_pdf.py <notebook_path> [output_path]
```

### 예시:
```bash
python notebook_to_pdf.py assignment2/assignment2_InJooKim.ipynb
```

- Jupyter 노트북을 HTML로 변환
- 브라우저에서 자동으로 열림
- Ctrl+P로 PDF 저장 가능
- pandoc 설치 불필요

## 2. tex_to_pdf_simple.py - LaTeX 파일을 PDF로 변환

### 사용법:
```bash
python tex_to_pdf_simple.py <tex_path> [output_path]
```

### 예시:
```bash
python tex_to_pdf_simple.py lecture/lecture1.tex
python tex_to_pdf_simple.py test.tex output.pdf
```

### 요구사항:
- MiKTeX 설치 필요: `winget install MiKTeX.MiKTeX`
- 한글 문서의 경우 kotex 패키지 필요

### 한글 LaTeX 문서 작성 팁:
```latex
\documentclass{article}
\usepackage{kotex}  % 한글 지원
\title{제목}
\author{작성자}
\begin{document}
\maketitle
내용...
\end{document}
```

## 3. tex_to_pdf.py - 고급 LaTeX 변환기

- 여러 LaTeX 엔진 지원 (pdflatex, xelatex, lualatex)
- 자동 MiKTeX 업데이트 시도
- 더 많은 옵션 제공

### 사용법:
```bash
python tex_to_pdf.py <tex_path> [output_path] [engine]
```

## 문제 해결

### MiKTeX 관련 오류:
1. `winget install MiKTeX.MiKTeX`로 설치
2. `miktex-console`을 실행하여 업데이트
3. 필요한 패키지 자동 설치 허용

### 한글 LaTeX 오류:
- XeLaTeX 사용: `python tex_to_pdf.py file.tex "" xelatex`
- kotex 패키지 사용하여 문서 작성