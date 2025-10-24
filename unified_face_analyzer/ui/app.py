import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import traceback
from tkinterdnd2 import DND_FILES, TkinterDnD
import threading
from PIL import Image, ImageTk, ImageGrab
import cv2
from datetime import datetime
import queue # ✨ 추가: 분석 대기열(Queue)을 위한 모듈

# ✨ AI 분석 로직을 담당하는 클래스를 import 합니다.
# 이 코드를 실행하기 전에 hairstyle_analyzer.py 파일이 같은 폴더에 있어야 합니다.
from core import HairstyleAnalyzer
from utils import get_tcp_server


class StdoutRedirector:
    """표준 출력을 UI 로그 창으로 리다렉트하는 클래스"""
    def __init__(self, log_callback):
        self.log_callback = log_callback
        self.buffer = ""
        
    def write(self, message):
        # 원래 stdout에도 출력 (콘솔에도 보이도록)
        sys.__stdout__.write(message)
        
        # UI 로그에도 출력
        if message.strip():  # 빈 문자열이 아닐 때만
            self.log_callback(message.strip(), "INFO")
    
    def flush(self):
        sys.__stdout__.flush()

class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI 헤어스타일 분석기 (v2.0)")
        self.geometry("1600x1000")
        self.minsize(1200, 800)

        self.analyzer = None  # HairstyleAnalyzer 인스턴스를 저장할 변수
        self.history = []
        self.history_thumbnails = [] # PhotoImage 객체를 저장하여 가비지 컬렉션을 방지
        self.selected_history_path = None  # 선택된 히스토리 아이템의 경로

        # --- ✨ 수정된 부분: 대기열 및 워커 스레드 ---
        self.analysis_queue = queue.Queue()
        self.worker_thread = None
        # ---

        # TCP Server 인스턴스
        self.tcp_server = get_tcp_server()
        self.tcp_streaming = False

        self.create_widgets()
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.handle_drop)

        # 히스토리 캔버스에서만 마우스 휠이 작동하도록 수정
        self.history_canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

        # Ctrl+V 키 바인딩 (클립보드에서 이미지 붙여넣기)
        self.bind_all("<Control-v>", self.paste_from_clipboard)

        # 표준 출력을 UI 로그로 리다이렉트
        sys.stdout = StdoutRedirector(self.log)

        # 별도 스레드에서 분석기(모델) 초기화 시작
        threading.Thread(target=self.initialize_analyzer, daemon=True).start()

    def log(self, message, level="INFO"):
        """로그 창에 메시지를 출력합니다."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if level == "INFO":
            prefix = "ℹ️"
        elif level == "SUCCESS":
            prefix = "✅"
        elif level == "ERROR":
            prefix = "❌"
        elif level == "WARNING":
            prefix = "⚠️"
        else:
            prefix = "📝"
        
        log_message = f"[{timestamp}] {prefix} {message}\n"
        
        # UI 업데이트는 메인 스레드에서만 가능
        def update_log():
            self.log_text.config(state="normal")
            self.log_text.insert(tk.END, log_message)
            self.log_text.see(tk.END)  # 자동 스크롤
            self.log_text.config(state="disabled")
        
        # after를 사용하여 메인 스레드에서 UI 업데이트를 예약
        self.after(0, update_log)

    def initialize_analyzer(self):
        """별도 스레드에서 HairstyleAnalyzer를 초기화합니다."""
        try:
            self.after(0, self.status_label.config, {'text': "모델 로딩 중... (최초 실행 시 시간이 걸릴 수 있습니다)"})
            
            # HairstyleAnalyzer 초기화 - print 출력이 자동으로 UI 로그에 표시됨
            self.analyzer = HairstyleAnalyzer()
            
            self.after(0, self.status_label.config, {'text': "분석기 로딩 완료. 이미지를 드롭하여 분석을 시작하세요."})
            
        except Exception as e:
            error_msg = f"모델 로딩 실패: {str(e)}"
            self.after(0, self.log, error_msg, "ERROR")
            self.after(0, self.log, traceback.format_exc(), "ERROR")
            
            error_message = f"프로그램 초기화에 실패했습니다.\n\n오류: {e}\n\n로그 창의 상세 오류 내용을 확인해주세요."
            self.after(0, lambda: messagebox.showerror("치명적 오류", error_message))
            self.after(0, self.status_label.config, {'text': f"오류: 모델 로딩 실패. 프로그램을 재시작하세요."})

    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # PanedWindow로 상하 분할 (크기 조절 가능)
        paned_window = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned_window.pack(fill="both", expand=True)

        # --- 상단 패널 (분석 결과 + 히스토리) ---
        top_frame = ttk.Frame(paned_window)
        paned_window.add(top_frame, weight=3)

        # --- 왼쪽 패널 (분석 결과 표시) ---
        left_panel = ttk.Frame(top_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # 드롭 프레임 (시각화 이미지 표시)
        self.drop_frame = ttk.Frame(left_panel, relief="sunken", padding=10)
        self.drop_frame.pack(fill="both", expand=True)
        self.drop_frame.grid_rowconfigure(0, weight=1)
        self.drop_frame.grid_columnconfigure(0, weight=1)
        
        self.drop_info_label = ttk.Label(self.drop_frame, text="이곳에 분석할 이미지나 폴더를 드롭하세요", font=("Malgun Gothic", 16))
        self.drop_info_label.grid(row=0, column=0, sticky="nsew")
        
        self.image_label = ttk.Label(self.drop_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # 텍스트 결과창
        self.result_text = tk.Text(left_panel, height=8, wrap="word", state="disabled", font=("Consolas", 9), relief="flat", bg="#f0f0f0", padx=10, pady=5)
        self.result_text.pack(fill="x", pady=(10, 0))
        
        # --- 오른쪽 패널 (분석 기록) ---
        right_panel = ttk.Frame(top_frame, width=300)
        right_panel.pack_propagate(False)
        right_panel.pack(side="right", fill="y")

        # 분석 기록 헤더 + 저장 버튼
        history_header = ttk.Frame(right_panel)
        history_header.pack(fill="x", pady=5)

        ttk.Label(history_header, text="분석 기록", font=("Malgun Gothic", 14, "bold")).pack(side="left")

        self.save_all_btn = ttk.Button(
            history_header,
            text="💾 전체 저장",
            command=self.export_all_to_markdown,
            width=12
        )
        self.save_all_btn.pack(side="right", padx=(5, 0))

        canvas_frame = ttk.Frame(right_panel, relief="solid", borderwidth=1)
        canvas_frame.pack(fill="both", expand=True)
        
        self.history_canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.history_canvas.yview)
        self.history_frame = ttk.Frame(self.history_canvas)

        self.history_frame.bind("<Configure>", lambda e: self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all")))
        self.history_canvas.create_window((0, 0), window=self.history_frame, anchor="nw")
        self.history_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.history_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- 하단 로그 패널 (크기 조절 가능) ---
        log_container = ttk.LabelFrame(paned_window, text="📋 실행 로그", padding=5)
        paned_window.add(log_container, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_container, 
            height=8, 
            wrap="word", 
            state="disabled",
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#ffffff"
        )
        self.log_text.pack(fill="both", expand=True)

        # --- 하단 상태바 + TCP 토글 ---
        status_frame = ttk.Frame(self)
        status_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 5))

        self.status_label = ttk.Label(status_frame, text="프로그램 초기화 중...", anchor="w", relief="sunken")
        self.status_label.pack(side="left", fill="x", expand=True)

        # TCP 스트리밍 토글 버튼
        tcp_control_frame = ttk.Frame(status_frame)
        tcp_control_frame.pack(side="right", padx=(10, 0))

        ttk.Label(tcp_control_frame, text="Unreal 스트리밍:", font=("Malgun Gothic", 9)).pack(side="left", padx=(0, 5))

        self.tcp_toggle_btn = ttk.Button(
            tcp_control_frame,
            text="▶ 시작 (25Hz)",
            command=self.toggle_tcp_streaming,
            width=15
        )
        self.tcp_toggle_btn.pack(side="left")

        self.tcp_status_label = ttk.Label(tcp_control_frame, text="● OFF", foreground="gray", font=("Malgun Gothic", 9, "bold"))
        self.tcp_status_label.pack(side="left", padx=(5, 0))

    def _on_mousewheel(self, event):
        widget = self.winfo_containing(event.x_root, event.y_root)
        # Canvas 객체를 문자열로 변환하여 비교하도록 수정
        if widget is not None and str(widget).startswith(str(self.history_canvas)):
            self.history_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def handle_drop(self, event):
        """
        [✨ 수정] 드롭된 파일/폴더를 처리하여 분석 대기열에 추가합니다.
        """
        if not self.analyzer:
            self.log("모델이 아직 로딩되지 않았습니다.", "WARNING")
            messagebox.showwarning("준비 중", "아직 모델 로딩이 완료되지 않았습니다.")
            return

        dropped_items = self.tk.splitlist(event.data)
        image_files_to_process = []
        supported_extensions = ('.png', '.jpg', '.jpeg')

        for item in dropped_items:
            path = item.strip('{}')
            if os.path.isdir(path):
                self.log(f"폴더 스캔 중: {os.path.basename(path)}", "INFO")
                for filename in os.listdir(path):
                    if filename.lower().endswith(supported_extensions):
                        full_path = os.path.join(path, filename)
                        if os.path.isfile(full_path):
                            image_files_to_process.append(full_path)
            elif os.path.isfile(path):
                if path.lower().endswith(supported_extensions):
                    image_files_to_process.append(path)

        if not image_files_to_process:
            self.log("분석할 유효한 이미지 파일을 찾지 못했습니다.", "WARNING")
            messagebox.showwarning("파일 없음", "드롭한 대상 중에 지원하는 이미지(.png, .jpg, .jpeg)가 없습니다.")
            return

        # 찾은 파일들을 대기열에 추가
        for file_path in image_files_to_process:
            self.analysis_queue.put(file_path)

        q_size = self.analysis_queue.qsize()
        self.log(f"총 {len(image_files_to_process)}개의 이미지를 분석 대기열에 추가했습니다. (현재 {q_size}개 대기 중)", "SUCCESS")
        self.status_label.config(text=f"분석 대기 중... ({q_size}개 이미지)")
        
        # 워커 스레드가 없거나 죽었으면 새로 시작
        self.start_worker_if_needed()

    def start_worker_if_needed(self):
        """[✨ 추가] 워커 스레드가 실행 중이 아니면 시작합니다."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.process_queue, daemon=True)
            self.worker_thread.start()
            self.log("분석 워커 스레드를 시작합니다.", "INFO")

    def process_queue(self):
        """[✨ 추가] 백그라운드에서 대기열의 이미지를 하나씩 처리하는 함수."""
        while True:
            try:
                filepath = self.analysis_queue.get()
                
                # UI 업데이트는 메인 스레드로 전달
                self.after(0, lambda p=filepath: self.status_label.config(text=f"분석 중 ({self.analysis_queue.qsize()}개 남음): {os.path.basename(p)}"))

                # 분석 실행
                results, viz_image_np = self.analyzer.analyze_image(filepath)

                # TCP로 결과 업데이트 (스트리밍 중일 때 전송됨)
                if results and 'error' not in results:
                    self.tcp_server.update_result(results, filepath)

                # 결과 표시도 메인 스레드로 전달
                self.after(0, self.display_results, filepath, results, viz_image_np)

                self.analysis_queue.task_done()

            except Exception as e:
                # 워커 스레드 자체에서 예외 발생 시 로그 남기기
                self.log(f"워커 스레드 오류: {e}", "ERROR")
                traceback.print_exc()

    def display_results(self, original_filepath, results, viz_image_np):
        """분석 결과와 시각화 이미지를 화면에 표시합니다."""
        if 'error' in results or viz_image_np is None:
            error_msg = results.get('error', '알 수 없는 오류')
            self.log(f"이미지 처리 실패 '{os.path.basename(original_filepath)}': {error_msg}", "ERROR")
            # 여러 파일 처리 시 오류 팝업은 방해가 될 수 있으므로 로그로만 남깁니다.
            self.status_label.config(text=f"분석 실패: {os.path.basename(original_filepath)}")
            return

        classification = results.get('classification', 'Unknown')
        self.log(f"'{os.path.basename(original_filepath)}' 분석 결과: {classification}", "SUCCESS")
        
        self.show_main_content(original_filepath, results, viz_image_np)
        
        # 히스토리에 중복이 없으면 추가 (경로 기준)
        if not any(item['path'] == original_filepath for item in self.history):
            self.history.insert(0, {
                'path': original_filepath,
                'results': results,
                'viz_image': viz_image_np,
                'unreal_screenshot': None  # 언리얼 스크린샷 (PIL.Image)
            })
            self.update_history_view()

    def show_main_content(self, filepath, results, viz_image_np):
        """메인 패널(왼쪽)의 이미지와 텍스트를 업데이트합니다."""
        # 1. 시각화 이미지 표시
        try:
            viz_image_rgb = cv2.cvtColor(viz_image_np, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(viz_image_rgb)

            # 고정된 최대 크기 사용 (프레임 크기 변화 방지)
            max_w = 1200  # 고정 크기
            max_h = 800
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

            self.tk_image = ImageTk.PhotoImage(img) # 클래스 변수로 참조 유지
            self.drop_info_label.grid_remove()
            self.image_label.config(image=self.tk_image)
        except Exception as e:
            self.log(f"시각화 이미지 표시 오류: {str(e)}", "ERROR")
            self.image_label.config(image='')
            self.drop_info_label.grid()

        # 2. 텍스트 결과 표시
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        
        final_class = results.get('classification', 'Unknown')
        header = f"📁 {os.path.basename(filepath)}\n✨ 최종 분석 결과: {final_class} ✨\n\n"
        self.result_text.insert(tk.END, header)
        
        # CLIP 결과 추가 표시 - 수정된 부분
        if 'clip_results' in results:
            clip_data = results['clip_results']
            self.result_text.insert(tk.END, "📊 분석 상세 결과:\n")

            # 성별 결과 표시 (CLIP)
            gender = clip_data.get('gender', 'N/A')
            gender_conf = clip_data.get('gender_confidence', 0)
            self.result_text.insert(tk.END, f"  • 성별: {gender} ({gender_conf:.2%})\n")

            # 앞머리 결과 표시 (BiSeNet 기반)
            bangs_status = clip_data.get('bangs', 'N/A')
            bangs_method = "BiSeNet" if bangs_status in ['Bangs', 'No Bangs'] else "N/A"
            self.result_text.insert(tk.END, f"  • 앞머리: {bangs_status} (방법: {bangs_method})\n")

            # 안경과 수염 결과 (CLIP)
            self.result_text.insert(tk.END, f"  • 안경: {clip_data.get('glasses', 'N/A')} ({clip_data.get('glasses_confidence', 0):.2%})\n")
            self.result_text.insert(tk.END, f"  • 수염: {clip_data.get('beard', 'N/A')} ({clip_data.get('beard_confidence', 0):.2%})\n")
        
        self.result_text.config(state="disabled")
        
        q_size = self.analysis_queue.qsize()
        if q_size > 0:
            self.status_label.config(text=f"{q_size}개 이미지 분석 대기 중...")
        else:
            self.status_label.config(text=f"분석 완료: {os.path.basename(filepath)}")
        
    def update_history_view(self):
        """오른쪽 히스토리 패널을 업데이트합니다."""
        for widget in self.history_frame.winfo_children():
            widget.destroy()
        self.history_thumbnails.clear()

        for item in self.history:
            # 선택된 아이템인지 확인
            is_selected = (item['path'] == self.selected_history_path)

            # 전체 아이템 컨테이너
            frame_style = 'Selected.TFrame' if is_selected else 'Card.TFrame'
            item_frame = ttk.Frame(self.history_frame, relief="solid", borderwidth=1, style=frame_style)
            item_frame.pack(fill="x", padx=5, pady=(5, 0))

            # 좌측: 원본 분석 정보
            left_panel = ttk.Frame(item_frame)
            left_panel.pack(side="left", fill="both", expand=True)

            # 원본 이미지 썸네일
            try:
                img = Image.open(item['path'])
                img.thumbnail((60, 60))
                thumb = ImageTk.PhotoImage(img)
                self.history_thumbnails.append(thumb)

                thumb_label = ttk.Label(left_panel, image=thumb)
                thumb_label.pack(side="left", padx=5, pady=5)
            except Exception as e:
                self.log(f"썸네일 생성 오류: {str(e)}", "WARNING")

            final_class = item['results'].get('classification', 'N/A')
            summary_text = f"{os.path.basename(item['path'])[:20]}\n▶ {final_class}"
            summary_label = ttk.Label(left_panel, text=summary_text, justify="left", font=("Malgun Gothic", 8))
            summary_label.pack(side="left", padx=5, pady=5, anchor="w")

            # 우측: 언리얼 스크린샷 영역
            right_panel = ttk.Frame(item_frame, relief="sunken", borderwidth=1)
            right_panel.pack(side="right", padx=5, pady=5)

            if item['unreal_screenshot'] is not None:
                # 언리얼 스크린샷이 있으면 표시
                unreal_img = item['unreal_screenshot'].copy()
                unreal_img.thumbnail((60, 60))
                unreal_thumb = ImageTk.PhotoImage(unreal_img)
                self.history_thumbnails.append(unreal_thumb)

                unreal_label = ttk.Label(right_panel, image=unreal_thumb)
                unreal_label.pack(padx=5, pady=5)
            else:
                # 언리얼 스크린샷이 없으면 안내 텍스트
                placeholder_label = ttk.Label(
                    right_panel,
                    text="언리얼\n스크린샷",
                    justify="center",
                    font=("Malgun Gothic", 8),
                    foreground="gray"
                )
                placeholder_label.pack(padx=15, pady=15)

            # 클릭 이벤트 바인딩
            widgets_to_bind = [item_frame, left_panel, right_panel, thumb_label, summary_label]
            for widget in widgets_to_bind:
                widget.bind("<Button-1>", lambda e, i=item: self.on_history_click(i['path'], i['results'], i['viz_image']))
                widget.config(cursor="hand2")

    def on_history_click(self, filepath, results, viz_image_np):
        """분석 기록 클릭 시: 화면 표시 + TCP JSON 전송 + 선택 표시"""
        # 선택된 경로 저장
        self.selected_history_path = filepath

        # 히스토리 뷰 업데이트 (선택 표시)
        self.update_history_view()

        # 화면에 표시
        self.show_main_content(filepath, results, viz_image_np)

        # TCP 서버가 켜져있으면 JSON 전송
        if self.tcp_streaming:
            client_count = self.tcp_server.get_client_count()
            if client_count > 0:
                self.tcp_server.update_result(results, filepath)
                self.log(f"TCP 전송: {os.path.basename(filepath)} → {results.get('classification', 'Unknown')}", "SUCCESS")
            else:
                self.log(f"TCP 서버 ON이지만 연결된 클라이언트 없음 (전송 안 됨)", "WARNING")

    def paste_from_clipboard(self, event=None):
        """Ctrl+V: 클립보드에서 언리얼 스크린샷을 붙여넣기"""
        if self.selected_history_path is None:
            self.log("먼저 히스토리 아이템을 클릭하여 선택하세요.", "WARNING")
            return

        try:
            clipboard_image = ImageGrab.grabclipboard()

            if clipboard_image is None:
                self.log("클립보드에 이미지가 없습니다. Win+Shift+S로 스크린샷을 찍어주세요.", "WARNING")
                return

            if not isinstance(clipboard_image, Image.Image):
                self.log("클립보드에 이미지 형식이 아닌 데이터가 있습니다.", "WARNING")
                return

            # 선택된 히스토리 아이템에 언리얼 스크린샷 추가
            for item in self.history:
                if item['path'] == self.selected_history_path:
                    item['unreal_screenshot'] = clipboard_image.copy()
                    self.update_history_view()

                    filename = os.path.basename(self.selected_history_path)
                    self.log(f"'{filename}'에 언리얼 스크린샷을 추가했습니다.", "SUCCESS")
                    break

        except Exception as e:
            self.log(f"클립보드 붙여넣기 오류: {str(e)}", "ERROR")

    def export_all_to_markdown(self):
        """전체 히스토리를 Markdown 문서로 내보내기 (Notion 친화적 구조)"""
        if not self.history:
            self.log("저장할 분석 기록이 없습니다.", "WARNING")
            messagebox.showwarning("저장 불가", "분석 기록이 비어있습니다.")
            return

        try:
            # 타임스탬프로 폴더명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = os.path.join("documentation", f"export_{timestamp}")
            os.makedirs(export_dir, exist_ok=True)

            self.log(f"문서화 폴더 생성: {export_dir}", "INFO")

            # 헤어스타일별로 그룹핑
            hairstyle_groups = {}
            for idx, item in enumerate(self.history, start=1):
                results = item['results']
                classification = results.get('classification', 'Unknown')

                if classification not in hairstyle_groups:
                    hairstyle_groups[classification] = []

                hairstyle_groups[classification].append((idx, item))

            # README.md 생성 시작
            readme_lines = []
            readme_lines.append(f"# 헤어스타일 분석 결과 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n")

            # 개요 섹션
            readme_lines.append("## 개요\n\n")
            readme_lines.append(f"- **총 케이스 수**: {len(self.history)}\n")
            readme_lines.append("- **헤어스타일 분포**:\n")
            for hairstyle, cases in sorted(hairstyle_groups.items()):
                readme_lines.append(f"  - {hairstyle}: {len(cases)}개\n")
            readme_lines.append("\n---\n\n")

            # 각 히스토리 아이템 처리 (케이스별 폴더에 이미지 저장)
            for idx, item in enumerate(self.history, start=1):
                filename = os.path.basename(item['path'])

                # 케이스별 폴더 생성
                case_folder = f"case_{idx:03d}"
                case_dir = os.path.join(export_dir, case_folder)
                os.makedirs(case_dir, exist_ok=True)

                # 1. 분석 시각화 이미지 저장
                viz_image_rgb = cv2.cvtColor(item['viz_image'], cv2.COLOR_BGR2RGB)
                viz_img = Image.fromarray(viz_image_rgb)
                viz_img.save(os.path.join(case_dir, "analysis.png"))

                # 2. 언리얼 스크린샷 저장 (있으면)
                has_unreal = item['unreal_screenshot'] is not None
                if has_unreal:
                    item['unreal_screenshot'].save(os.path.join(case_dir, "unreal.png"))

                self.log(f"[{idx}/{len(self.history)}] {filename} 이미지 저장 완료", "INFO")

            # 헤어스타일별로 README.md 작성 (토글 형식)
            for hairstyle in sorted(hairstyle_groups.keys()):
                cases = hairstyle_groups[hairstyle]
                readme_lines.append(f"▶# {hairstyle} ({len(cases)}개)\n\n")

                for idx, item in cases:
                    filename = os.path.basename(item['path'])
                    results = item['results']
                    result_text = self._generate_result_text(results)
                    has_unreal = item['unreal_screenshot'] is not None
                    case_folder = f"case_{idx:03d}"

                    # 토글 내부 콘텐츠는 들여쓰기(tab) 필요
                    readme_lines.append(f"\t## Case {idx:03d} - {filename}\n\n")
                    readme_lines.append("\t**분석 결과:**\n")
                    readme_lines.append(f"\t```\n{result_text}\t```\n\n")
                    readme_lines.append("\t**이미지:**\n\n")

                    readme_lines.append("\t---\n\n")

            # README.md 파일 저장
            readme_path = os.path.join(export_dir, "README.md")
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.writelines(readme_lines)

            self.log(f"✅ 전체 {len(self.history)}개 케이스를 Markdown으로 저장했습니다: {export_dir}", "SUCCESS")
            messagebox.showinfo("저장 완료", f"문서가 저장되었습니다:\n{os.path.abspath(export_dir)}\n\nREADME.md를 Notion에 복사하세요.")

            # 폴더 열기 (Windows)
            if sys.platform == "win32":
                os.startfile(os.path.abspath(export_dir))

        except Exception as e:
            self.log(f"Markdown 저장 중 오류: {str(e)}", "ERROR")
            messagebox.showerror("저장 실패", f"문서 저장 중 오류가 발생했습니다:\n{str(e)}")

    def _generate_result_text(self, results):
        """분석 결과를 텍스트로 변환"""
        lines = []
        lines.append("=== 헤어스타일 분석 결과 ===\n\n")
        lines.append(f"최종 분류: {results.get('classification', 'Unknown')}\n\n")

        if 'clip_results' in results:
            clip_data = results['clip_results']
            lines.append("[상세 정보]\n")
            lines.append(f"- 성별: {clip_data.get('gender', 'N/A')} ({clip_data.get('gender_confidence', 0):.2%})\n")
            lines.append(f"- 앞머리: {clip_data.get('bangs', 'N/A')}\n")
            lines.append(f"- 안경: {clip_data.get('glasses', 'N/A')} ({clip_data.get('glasses_confidence', 0):.2%})\n")
            lines.append(f"- 수염: {clip_data.get('beard', 'N/A')} ({clip_data.get('beard_confidence', 0):.2%})\n")

        if 'geometric_analysis' in results:
            geo = results['geometric_analysis']
            lines.append("\n[기하학적 분석]\n")
            for key, value in geo.items():
                if isinstance(value, float):
                    lines.append(f"- {key}: {value:.2%}\n")
                else:
                    lines.append(f"- {key}: {value}\n")

        return ''.join(lines)

    def toggle_tcp_streaming(self):
        """TCP 스트리밍 토글 버튼 핸들러"""
        if self.tcp_streaming:
            # 서버 중지
            self.tcp_server.stop_server()
            self.tcp_streaming = False
            self.tcp_toggle_btn.config(text="▶ 시작 (25Hz)")
            self.tcp_status_label.config(text="● OFF", foreground="gray")
            self.log("TCP 서버를 중지했습니다.", "INFO")
        else:
            # 서버 시작
            self.tcp_server.start_server()
            self.tcp_streaming = True
            self.tcp_toggle_btn.config(text="■ 중지")
            self.tcp_status_label.config(text="● ON (25Hz)", foreground="green")
            self.log("TCP 서버를 시작했습니다 (0.0.0.0:5000, 25Hz). Unreal에서 연결하세요.", "SUCCESS")

            # 클라이언트 연결 모니터링 (UI 업데이트)
            self._monitor_connections()

    def _monitor_connections(self):
        """TCP 클라이언트 연결 수를 모니터링하여 UI에 표시"""
        if self.tcp_streaming:
            client_count = self.tcp_server.get_client_count()
            if client_count > 0:
                self.tcp_status_label.config(
                    text=f"● ON (25Hz) - {client_count} client(s)",
                    foreground="green"
                )
            else:
                self.tcp_status_label.config(
                    text="● ON (25Hz) - Waiting...",
                    foreground="orange"
                )
            # 1초마다 재확인
            self.after(1000, self._monitor_connections)

if __name__ == "__main__":
    app = App()

    # 간단한 스타일 추가
    style = ttk.Style(app)
    style.configure('Card.TFrame', background='white')
    style.configure('Selected.TFrame', background='lightblue', relief='solid', borderwidth=2)

    app.mainloop()