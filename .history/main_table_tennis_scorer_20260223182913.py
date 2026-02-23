"""
Ping Pong Smart Scoring System
Combining pose recognition and voice recognition technologies to achieve automated scoring statistics, 
real-time score display, and serving side indication
"""
# encoding: utf-8
import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time
import logging
from datetime import datetime
import sys
import os

# 设置系统编码为UTF-8
if sys.platform.startswith('win'):
    # Windows平台设置控制台编码
    os.system('chcp 65001')

# 添加语音播报功能
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("Warning: pyttsx3 module not found, text-to-speech feature unavailable")
    TTS_AVAILABLE = False

# 添加语音识别功能
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    print("Warning: speech_recognition module not found, voice scoring unavailable")
    SR_AVAILABLE = False

from config import *

class PlayerGestureState:
    def __init__(self):
        self.is_holding_high = False
        self.start_hold_time = 0
        self.cooldown_until = 0


class TableTennisScorer:
    def __init__(self):
        # 初始化变量
        self.score_a = 0
        self.score_b = 0
        self.game_active = True
        self.last_score_time = 0
        self.cooldown_period = COOLDOWN_PERIOD
        self.serve_side = 'A'  # 当前发球方
        self.score_source = ''  # 得分来源：'pose' 或 'voice'
        self.pose_detected_recently = False
        self.pose_detection_start_time = 0
        self.pose_detection_threshold = 1.0  # 需要持续检测1秒才确认得分
        
        # 添加总局比分追踪
        self.total_games_a = 0
        self.total_games_b = 0
        
        # 得分闪烁动画
        self.last_score_change_time = 0
        self.last_scoring_player = None
        self.score_flash_duration = 1.5  # 闪烁持续秒数
        
        # 初始化语音播报引擎
        self.tts_available = TTS_AVAILABLE
        self.tts_engine = None
        self.tts_lock = threading.Lock()
        self.tts_queue = queue.Queue(maxsize=50)
        self.tts_thread = None
        self.tts_stop_event = threading.Event()
        if self.tts_available:
            try:
                # 在独立线程中初始化并驱动TTS，避免跨线程调用导致后续无声
                self._start_tts_worker()
            except Exception:
                self.tts_available = False
                print("Failed to initialize text-to-speech engine")
        
        # 设置日志 - 使用FileHandler显式设置编码
        if LOG_TO_FILE:
            import logging.handlers
            handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
        else:
            # 如果不需要记录到文件，则只使用控制台输出
            logging.basicConfig(level=logging.INFO)
        
        # 初始化MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE
        )
        
        # 语音识别
        self.voice_recognition_available = SR_AVAILABLE
        self.sr_stop_event = threading.Event()
        self.sr_thread = None
        # 语音识别状态追踪（用于UI进度条）
        self.voice_state = 'idle'  # 'idle' / 'listening' / 'processing' / 'scored'
        self.voice_state_time = 0   # 状态开始时间
        self.voice_scored_player = None  # 最近识别到的得分方
        if self.voice_recognition_available:
            try:
                self.recognizer = sr.Recognizer()
                self.recognizer.energy_threshold = VOICE_ENERGY_THRESHOLD
                self.recognizer.dynamic_energy_threshold = False  # 固定阈值，防止自动调高
                self.recognizer.pause_threshold = 0.15  # 极短停顿即视为说完（加速）
                self.recognizer.phrase_threshold = 0.05  # 更敏感（加速）
                self.recognizer.non_speaking_duration = 0.1  # 缩短非语音段（加速）
                self.microphone = sr.Microphone()
                # 环境噪声校准
                with self.microphone as source:
                    print("Calibrating microphone for ambient noise (1s)...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print(f"Mic energy threshold after calibration: {self.recognizer.energy_threshold}")
                # 确保阈值不会被校准得太高
                if self.recognizer.energy_threshold > 2000:
                    self.recognizer.energy_threshold = 2000
                    print(f"Energy threshold capped to 2000")
                print(f"Voice recognition ready. Say 'A' or 'B' to score!")
            except Exception as e:
                self.voice_recognition_available = False
                print(f"Voice recognition init failed: {e}")
        else:
            print("Voice recognition unavailable")

        # 初始化选手举手状态机
        self.left_gesture = PlayerGestureState()
        self.right_gesture = PlayerGestureState()

    def start_listening(self):
        """启动语音识别后台线程"""
        if not self.voice_recognition_available:
            return
        if self.sr_thread and self.sr_thread.is_alive():
            return
        self.sr_stop_event.clear()
        self.sr_thread = threading.Thread(target=self._voice_listen_loop, daemon=True)
        self.sr_thread.start()
        print("Voice recognition thread started")

    def _voice_listen_loop(self):
        """持续监听麦克风，识别到 A/B 立即加分"""
        print("[Voice] Listen loop started")
        while not self.sr_stop_event.is_set():
            self.voice_state = 'listening'
            self.voice_state_time = time.time()
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(
                        source,
                        timeout=VOICE_LISTEN_TIMEOUT,
                        phrase_time_limit=VOICE_PHRASE_TIME_LIMIT
                    )
                self.voice_state = 'processing'
                self.voice_state_time = time.time()
                print("[Voice] Audio captured, recognizing...")
            except sr.WaitTimeoutError:
                self.voice_state = 'idle'
                continue
            except Exception as e:
                print(f"[Voice] Listen error: {e}")
                self.voice_state = 'idle'
                time.sleep(0.3)
                continue

            # 异步识别，避免阻塞监听循环
            recognize_thread = threading.Thread(
                target=self._recognize_and_score, args=(audio,), daemon=True)
            recognize_thread.start()

    def _recognize_and_score(self, audio):
        """识别音频并处理得分 —— 中英文并行请求，谁先匹配谁生效"""
        result_lock = threading.Lock()
        result = {'scored': None, 'matched_text': ''}
        done_event = threading.Event()

        def _try_recognize(language, label):
            """单语言识别子任务"""
            try:
                raw = self.recognizer.recognize_google(audio, language=language, show_all=True)
                if raw and isinstance(raw, dict):
                    for alt in raw.get('alternative', []):
                        t = alt.get('transcript', '').strip()
                        if t:
                            print(f"[Voice] {label} candidate: '{t}' (conf: {alt.get('confidence', '?')})")
                            s = self._match_voice_command(t)
                            if s:
                                with result_lock:
                                    if result['scored'] is None:
                                        result['scored'] = s
                                        result['matched_text'] = t
                                        done_event.set()
                                return
                elif raw and isinstance(raw, str):
                    print(f"[Voice] {label} heard: '{raw}'")
                    s = self._match_voice_command(raw)
                    if s:
                        with result_lock:
                            if result['scored'] is None:
                                result['scored'] = s
                                result['matched_text'] = raw
                                done_event.set()
                        return
                else:
                    print(f"[Voice] {label}: no result")
            except sr.RequestError as e:
                print(f"[Voice] {label} request error: {e}")
            except Exception as e:
                print(f"[Voice] {label} error: {e}")

        # ── 中英文并行识别 ──
        th_zh = threading.Thread(target=_try_recognize, args=("zh-CN", "zh"), daemon=True)
        th_en = threading.Thread(target=_try_recognize, args=("en-US", "en"), daemon=True)
        th_zh.start()
        th_en.start()

        # 等待任一线程匹配成功，或全部完成（最多等3秒防卡死）
        done_event.wait(timeout=3.0)
        th_zh.join(timeout=0.5)
        th_en.join(timeout=0.5)

        if result['scored']:
            print(f">>> Voice command: Player {result['scored']} scores! (heard: '{result['matched_text']}')")
            self.voice_state = 'scored'
            self.voice_scored_player = result['scored']
            self.voice_state_time = time.time()
            self.process_score('voice', result['scored'])
        else:
            print("[Voice] No match found")
            self.voice_state = 'idle'

    def _match_voice_command(self, text):
        """从识别文本中匹配得分指令，返回 'A'/'B' 或 None"""
        t = text.strip()
        tu = t.upper()

        # ── A 的各种可能识别结果 ──
        # 英文：A, a, Ay, Hey, Eh, Ace 等
        # 中文：诶, 哎, 唉, 啊, 嗯A, A分, 加A, A得分 等
        A_EXACT = {
            'A', 'a', 'AY', 'HEY', 'EI', 'AE', 'ACE', 'AH', 'HA', 'EH', 'YAY', 'YEAH', 'YA',
            'PLAYER A', 'SCORE A', 'POINT A', 'ADD A',
        }
        A_CN = {'诶', '哎', '唉', '啊', '嗯', '呃', '额', '欸',
                'a', 'A', '加a', '加A', 'a分', 'A分', 'a得分', 'A得分',
                '加诶', '加哎', '诶得分', '哎得分'}

        # ── B 的各种可能识别结果 ──
        B_EXACT = {
            'B', 'b', 'BE', 'BEE', 'BI', 'V', 'VE', 'BEA', 'P', 'PEE', 'VEE',
            'PLAYER B', 'SCORE B', 'POINT B', 'ADD B',
        }
        B_CN = {'币', '比', '必', '逼', '毕', '笔', '碧', '壁', '闭',
                'b', 'B', '加b', '加B', 'b分', 'B分', 'b得分', 'B得分',
                '加比', '加币', '比得分', '币得分'}

        # 精确匹配（原文）
        if t in A_CN:
            return 'A'
        if t in B_CN:
            return 'B'
        # 精确匹配（大写）
        if tu in A_EXACT:
            return 'A'
        if tu in B_EXACT:
            return 'B'

        # 逐词匹配
        words = tu.replace(',', ' ').replace('，', ' ').replace('。', ' ').split()
        if len(words) <= 5:
            for w in words:
                if w in A_EXACT or w in {x.upper() for x in A_CN}:
                    return 'A'
                if w in B_EXACT or w in {x.upper() for x in B_CN}:
                    return 'B'

        # 逐字符匹配中文关键字
        for ch in t:
            if ch in {'诶', '哎', '唉', '嗯', '呃', '欸'}:
                return 'A'
            if ch in {'币', '比', '必', '逼', '毕', '笔', '碧', '壁', '闭'}:
                return 'B'

        # 兜底：短文本包含A或B
        if len(tu) <= 8:
            has_a = 'A' in tu
            has_b = 'B' in tu
            if has_a and not has_b:
                return 'A'
            if has_b and not has_a:
                return 'B'

        return None

    def stop_listening(self):
        """停止语音识别"""
        self.sr_stop_event.set()
        if self.sr_thread and self.sr_thread.is_alive():
            self.sr_thread.join(timeout=3)
    
    def speak_score(self, player, current_score_a, current_score_b):
        """播报得分（中文简短版）"""
        if not self.tts_available:
            print("Text-to-speech not available")
            return
        # 中文播报：比分和发球方
        serve_text = f"{self.serve_side}方发球。"
        text = f"{current_score_a}比{current_score_b}，{serve_text}"
        print(f"TTS: {text}")
        self._speak_async(text)

    def _speak_async(self, text):
        """异步播报文本，避免阻塞主线程"""
        if not self.tts_available:
            return
        try:
            self.tts_queue.put_nowait(text)
        except queue.Full:
            # 队列满时丢弃最旧一条，优先播报最新比分
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.put_nowait(text)
            except Exception:
                pass

    def _start_tts_worker(self):
        """启动TTS工作线程（线程内初始化引擎）"""
        if self.tts_thread and self.tts_thread.is_alive():
            return
        self.tts_stop_event.clear()
        self.tts_thread = threading.Thread(target=self._tts_worker_loop, daemon=True)
        self.tts_thread.start()

    def _tts_worker_loop(self):
        """TTS线程主循环：每次播报都新建引擎实例，规避 pyttsx3 runAndWait 内部状态bug"""
        while not self.tts_stop_event.is_set():
            try:
                text = self.tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                break

            engine = None
            try:
                # 每次播报都新建引擎，彻底避免 _inLoop / endLoop 状态残留
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Text-to-speech playback failed: {e}")
            finally:
                # 确保引擎被彻底释放
                if engine is not None:
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    del engine

    def _stop_tts_worker(self):
        """停止TTS工作线程"""
        if not self.tts_available:
            return
        self.tts_stop_event.set()
        try:
            self.tts_queue.put_nowait(None)
        except Exception:
            pass
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=1.0)

    
    
    def detect_pose(self, image):
        """检测举手悬停手势，返回(side, is_high)"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                h, w = image.shape[:2]
                wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
                middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                wrist_x, wrist_y = wrist.x, wrist.y
                midtip_x, midtip_y = middle_tip.x, middle_tip.y
                high_threshold = 0.33
                is_high = (wrist_y < high_threshold) or (midtip_y < high_threshold)
                side = None
                if is_high:
                    if wrist_x < 0.5:
                        side = 'A'
                    else:
                        side = 'B'
                if side:
                    return side, True
        return None, False
    
    def process_score(self, source, target_side=None):
        """处理得分事件"""
        current_time = time.time()
        
        # 检查是否在冷却期内
        if current_time - self.last_score_time < self.cooldown_period:
            if DEBUG_MODE:
                print(f"Still in cooldown period, {self.cooldown_period - (current_time - self.last_score_time):.1f}s remaining")
            return False
        
        # 如果指定了目标方，则给指定方加分，否则给当前发球方加分
        if target_side == 'A':
            self.score_a += 1
        elif target_side == 'B':
            self.score_b += 1
        else:
            # 默认给当前发球方加分
            if self.serve_side == 'A':
                self.score_a += 1
            else:
                self.score_b += 1
                
        self.score_source = source
        self.last_score_time = current_time
        
        # ======== 发球方切换逻辑 ========
        total_score = self.score_a + self.score_b
        # 10平前，每SERVE_CHANGE_INTERVAL分换发；10平后，每1分换发
        if self.score_a >= 10 and self.score_b >= 10:
            serve_interval = 1
        else:
            serve_interval = SERVE_CHANGE_INTERVAL
        if total_score % serve_interval == 0:
            self.serve_side = 'B' if self.serve_side == 'A' else 'A'
        # ======== 胜负判定逻辑 ========
        scoring_player = 'A' if target_side == 'A' or (target_side is None and self.serve_side == 'A') else 'B'
        self.last_scoring_player = scoring_player
        self.last_score_change_time = current_time
        log_msg = f"Score! Player {scoring_player} scored, Source: {source}, Current Score A:{self.score_a} - B:{self.score_b}"
        print(log_msg)
        if LOG_TO_FILE:
            logging.info(log_msg)
        # ── 播报当前比分 ──
        self.speak_score(scoring_player, self.score_a, self.score_b)
        # ── 胜负判定 ──
        self._check_game_over()
        return True
    
    def _check_game_over(self):
        """统一胜负判定：必须先获得WINNING_SCORE分且领先2分才获胜，而不是10分"""
        score_diff = abs(self.score_a - self.score_b)
        max_score = max(self.score_a, self.score_b)

        # 只有当一方分数达到WINNING_SCORE及以上，且领先2分，才判定胜利
        game_won = (max_score >= WINNING_SCORE) and (score_diff >= MINIMUM_WINNING_DIFFERENCE)

        if not game_won:
            return

        winner = 'A' if self.score_a > self.score_b else 'B'
        if winner == 'A':
            self.total_games_a += 1
        else:
            self.total_games_b += 1

        log_msg = f"Game Over! Player {winner} wins! Score A:{self.score_a} - B:{self.score_b}, Total Games A:{self.total_games_a} - B:{self.total_games_b}"
        print(log_msg)
        if LOG_TO_FILE:
            logging.info(log_msg)

        match_winner = None
        if self.total_games_a >= 4:
            match_winner = 'A'
            self.game_active = False
        elif self.total_games_b >= 4:
            match_winner = 'B'
            self.game_active = False

        if match_winner:
            print(f"Match Over! Player {match_winner} wins! Final Games A:{self.total_games_a} - B:{self.total_games_b}")
            if LOG_TO_FILE:
                logging.info(f"Match Over! Player {match_winner} wins! Final Games A:{self.total_games_a} - B:{self.total_games_b}")
            if self.tts_available:
                self._speak_async(f"{match_winner}方赢得整场比赛，最终比分A{self.total_games_a}比B{self.total_games_b}")
        else:
            # 本局结束，重置比分继续
            self.score_a = 0
            self.score_b = 0
            self.serve_side = 'A'
            if self.tts_available:
                self._speak_async(f"{winner}方获得本局胜利，新局开始，比分清零")

    def _draw_translucent_rect(self, frame, x1, y1, x2, y2, color, alpha=0.6):
        """在 frame 上绘制半透明矩形"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _put_text_centered(self, frame, text, cx, cy, scale, color, thickness):
        """以 (cx, cy) 为中心绘制文字"""
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        tx = cx - text_size[0] // 2
        ty = cy + text_size[1] // 2
        # 文字描边（黑色轮廓），提升对比度
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 4)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def draw_ui(self, frame):
        """绘制美化后的用户界面 —— 超大比分、高对比度、一眼可见"""
        h, w, c = frame.shape
        now = time.time()

        # ── 颜色定义 ──
        COLOR_A = (200, 120, 50)     # 选手A 蓝橙色
        COLOR_B = (50, 50, 200)      # 选手B 红色
        COLOR_A_LIGHT = (230, 170, 80)
        COLOR_B_LIGHT = (80, 80, 240)
        COLOR_WHITE = (255, 255, 255)
        COLOR_YELLOW = (0, 255, 255)
        COLOR_GREEN = (0, 230, 118)
        COLOR_DARK = (30, 30, 30)

        # ── 计分面板高度（占画面上方 ~35%） ──
        panel_h = int(h * 0.35)
        mid_x = w // 2

        # ── 得分闪烁效果 ──
        flash_a = False
        flash_b = False
        if self.last_scoring_player and (now - self.last_score_change_time) < self.score_flash_duration:
            # 0.25秒闪一次
            blink = int((now - self.last_score_change_time) / 0.25) % 2 == 0
            if self.last_scoring_player == 'A':
                flash_a = blink
            else:
                flash_b = blink

        # ── 绘制左右半透明背景面板 ──
        alpha_a = 0.75 if flash_a else 0.55
        alpha_b = 0.75 if flash_b else 0.55
        bg_a = COLOR_A_LIGHT if flash_a else COLOR_A
        bg_b = COLOR_B_LIGHT if flash_b else COLOR_B
        self._draw_translucent_rect(frame, 0, 0, mid_x - 1, panel_h, bg_a, alpha_a)
        self._draw_translucent_rect(frame, mid_x + 1, 0, w, panel_h, bg_b, alpha_b)

        # ── 中央分割线 ──
        cv2.line(frame, (mid_x, 0), (mid_x, panel_h), COLOR_WHITE, 3)

        # ── 选手名称 ──
        name_y = 40
        self._put_text_centered(frame, 'A', mid_x // 2, name_y, 1.8, COLOR_WHITE, 3)
        self._put_text_centered(frame, 'B', mid_x + mid_x // 2, name_y, 1.8, COLOR_WHITE, 3)

        # ── 超大比分数字（核心：不戴眼镜也能看清） ──
        score_y = panel_h // 2 + 20
        score_scale = min(w, h) / 160.0  # 根据分辨率自适应，1280x720时约8.0
        score_scale = max(score_scale, 4.0)
        score_thick = max(int(score_scale * 1.8), 6)

        self._put_text_centered(frame, str(self.score_a), mid_x // 2, score_y,
                                score_scale, COLOR_WHITE, score_thick)
        self._put_text_centered(frame, str(self.score_b), mid_x + mid_x // 2, score_y,
                                score_scale, COLOR_WHITE, score_thick)

        # ── 中间 VS / 冒号 ──
        self._put_text_centered(frame, ':', mid_x, score_y, score_scale * 0.6, COLOR_YELLOW, score_thick - 2)

        # ── 总局比分（面板底部） ──
        game_y = panel_h - 15
        game_text = f'Games  {self.total_games_a} : {self.total_games_b}'
        self._put_text_centered(frame, game_text, mid_x, game_y, 1.2, COLOR_YELLOW, 3)

        # ── 发球方指示（小球图标 + 文字） ──
        serve_y = panel_h + 40
        serve_label = f'Serve >>  {self.serve_side}'
        # 在发球方一侧画一个小圆球
        if self.serve_side == 'A':
            ball_cx = mid_x // 2
        else:
            ball_cx = mid_x + mid_x // 2
        cv2.circle(frame, (ball_cx, serve_y - 5), 14, COLOR_YELLOW, -1)
        cv2.circle(frame, (ball_cx, serve_y - 5), 14, COLOR_DARK, 2)
        self._put_text_centered(frame, serve_label, mid_x, serve_y, 1.0, COLOR_GREEN, 2)

        # ── 冷却倒计时（面板下方居中，醒目红色） ──
        remaining = self.cooldown_period - (now - self.last_score_time)
        if 0 < remaining < self.cooldown_period:
            cd_text = f'Cooldown {remaining:.1f}s'
            cd_y = serve_y + 40
            self._put_text_centered(frame, cd_text, mid_x, cd_y, 0.9, (0, 0, 255), 2)

        # ── 底部状态栏（半透明黑条） ──
        bar_h = 36
        self._draw_translucent_rect(frame, 0, h - bar_h, w, h, COLOR_DARK, 0.65)
        # 左侧：按键提示
        cv2.putText(frame, "Q-Quit R-Reset F-FullReset | A/Z:A+/- B/X:B+/-", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        # 右侧：语音识别 + TTS状态
        vr_label = "MIC ON" if self.voice_recognition_available else "MIC OFF"
        vr_color = COLOR_GREEN if self.voice_recognition_available else (0, 0, 200)
        cv2.putText(frame, vr_label, (w - 260, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, vr_color, 2)
        tts_label = "TTS ON" if self.tts_available else "TTS OFF"
        tts_color = COLOR_GREEN if self.tts_available else (0, 0, 200)
        cv2.putText(frame, tts_label, (w - 130, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tts_color, 2)

        # ── 比赛结束大字幕 ──
        if not self.game_active:
            # 全屏半透明遮罩
            self._draw_translucent_rect(frame, 0, 0, w, h, COLOR_DARK, 0.6)
            winner = 'A' if self.total_games_a > self.total_games_b else 'B'
            win_color = COLOR_A_LIGHT if winner == 'A' else COLOR_B_LIGHT
            self._put_text_centered(frame, f'Player {winner} Wins!', mid_x, h // 2 - 30, 3.0, win_color, 6)
            final_score = f'Match  {self.total_games_a} : {self.total_games_b}'
            self._put_text_centered(frame, final_score, mid_x, h // 2 + 60, 2.0, COLOR_YELLOW, 4)

        # ── 举手手势进度/冷却反馈 ──
        HOLD_DURATION = 1.0   # 需要举手多久
        COOLDOWN_TOTAL = 3.0  # 冷却总时长
        ring_r = 28
        for player, gesture, cx in [('A', self.left_gesture, mid_x // 2),
                                      ('B', self.right_gesture, mid_x + mid_x // 2)]:
            ring_y = panel_h + 90
            if now < gesture.cooldown_until:
                # 冷却中：画红色倒计时弧
                cd_remain = gesture.cooldown_until - now
                ratio = cd_remain / COOLDOWN_TOTAL
                angle = int(360 * ratio)
                cv2.circle(frame, (cx, ring_y), ring_r, (60, 60, 60), 4)
                cv2.ellipse(frame, (cx, ring_y), (ring_r, ring_r), -90, 0, angle, (0, 60, 220), 4)
                self._put_text_centered(frame, f'CD {cd_remain:.1f}s', cx, ring_y + ring_r + 18, 0.55, (0, 80, 255), 2)
            elif gesture.is_holding_high:
                # 举手计时中：画绿色进度弧
                held = now - gesture.start_hold_time
                ratio = min(held / HOLD_DURATION, 1.0)
                angle = int(360 * ratio)
                cv2.circle(frame, (cx, ring_y), ring_r, (50, 50, 50), 4)
                cv2.ellipse(frame, (cx, ring_y), (ring_r, ring_r), -90, 0, angle, (0, 220, 80), 5)
                label = '+1' if ratio >= 1.0 else 'Hold'
                color = (0, 255, 100) if ratio >= 1.0 else COLOR_WHITE
                self._put_text_centered(frame, label, cx, ring_y, 0.7, color, 2)

        return frame
    
    def reset_game(self):
        """重置游戏（保留总局比分）"""
        self.score_a = 0
        self.score_b = 0
        self.game_active = True
        self.last_score_time = 0
        self.serve_side = 'A'
        self.score_source = ''
        self.last_scoring_player = None
        self.last_score_change_time = 0
        print("Current game reset, totals kept")
        
        if LOG_TO_FILE:
            logging.info("Current game reset, totals kept")
    
    def manual_adjust_score(self, player, delta):
        """手动调整比分
        player: 'A' 或 'B'
        delta: +1 加分, -1 减分
        """
        if player == 'A':
            self.score_a = max(0, self.score_a + delta)
        else:
            self.score_b = max(0, self.score_b + delta)
        
        # 重新计算发球方
        total_score = self.score_a + self.score_b
        # 根据总分重新推算发球方：初始A发，每SERVE_CHANGE_INTERVAL分轮换
        switches = total_score // SERVE_CHANGE_INTERVAL
        self.serve_side = 'A' if switches % 2 == 0 else 'B'
        
        # 更新闪烁动画
        self.last_scoring_player = player if delta > 0 else None
        self.last_score_change_time = time.time()
        self.score_source = 'manual'
        
        action = '+1' if delta > 0 else '-1'
        log_msg = f"Manual adjust: Player {player} {action}, Score A:{self.score_a} - B:{self.score_b}"
        print(log_msg)
        
        # 播报当前比分
        if self.tts_available:
            self._speak_async(f"{self.score_a}比{self.score_b}，{self.serve_side}方发球")
        
        if LOG_TO_FILE:
            logging.info(log_msg)

        # 胜负判定（键盘加分也需要检查）
        if delta > 0:
            self._check_game_over()
    
    def full_reset(self):
        """全局重置（包括总局比分）"""
        self.score_a = 0
        self.score_b = 0
        self.total_games_a = 0
        self.total_games_b = 0
        self.game_active = True
        self.last_score_time = 0
        self.serve_side = 'A'
        self.score_source = ''
        self.last_scoring_player = None
        self.last_score_change_time = 0
        print("Full reset: all scores cleared")
        if self.tts_available:
            self._speak_async("比分已全部清零")
        if LOG_TO_FILE:
            logging.info("Full reset: all scores cleared")

    def run(self):
        """运行主循环"""
        print("Starting Ping Pong Smart Scoring System...")
        print(f"Text-to-Speech: {'Enabled' if self.tts_available else 'Unavailable'}")
        print(f"Voice Recognition: {'Enabled' if self.voice_recognition_available else 'Unavailable'}")
        print(f"Use touch gestures or say 'A'/'B' to score")
        print("Keys: Q-Quit  R-Reset  F-Full Reset")
        print("      A/S - Player A +1/-1")
        print("      B/N - Player B +1/-1")
        
         # 启动语音识别
        self.start_listening()
        
        # 打开摄像头
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        # 创建窗口（支持全屏）
        if WINDOW_FULLSCREEN:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        
        while cap.isOpened() and self.game_active:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # 检测举手悬停手势
            side, pose_detected = self.detect_pose(frame)
            current_time = time.time()
            # 左右两侧状态机
            for player, gesture in [('A', self.left_gesture), ('B', self.right_gesture)]:
                if current_time < gesture.cooldown_until:
                    gesture.is_holding_high = False
                    continue
                if pose_detected and side == player:
                    if not gesture.is_holding_high:
                        gesture.is_holding_high = True
                        gesture.start_hold_time = current_time
                    else:
                        held_duration = current_time - gesture.start_hold_time
                        if held_duration > 1.0:
                            self.process_score('pose', player)
                            gesture.cooldown_until = current_time + 3.0
                            gesture.is_holding_high = False
                else:
                    gesture.is_holding_high = False

            # 绘制UI
            frame = self.draw_ui(frame)

            # 显示帧
            cv2.imshow(WINDOW_NAME, frame)

            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_game()
            elif key == ord('a'):
                self.manual_adjust_score('A', +1)  # A方 +1
            elif key == ord('s'):
                self.manual_adjust_score('A', -1)  # A方 -1
            elif key == ord('b'):
                self.manual_adjust_score('B', +1)  # B方 +1
            elif key == ord('n'):
                self.manual_adjust_score('B', -1)  # B方 -1
            elif key == ord('f'):
                self.full_reset()                  # 全局重置
        
        # 释放资源
        self.stop_listening()
        cap.release()
        cv2.destroyAllWindows()
        self._stop_tts_worker()


if __name__ == "__main__":
    scorer = TableTennisScorer()
    scorer.run()