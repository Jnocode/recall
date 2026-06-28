#!/usr/bin/env python3
"""Generate recall-sqlite explainer video — for non-developer audience.

v2 — Fixed: card cut-off bugs, non-dev language, added setup tutorial scene.
Zero terminal commands, zero code, pure visual storytelling.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, subprocess, tempfile, shutil

# ─── Config ─────────────────────────────────────────────────────
W, H = 1080, 720           # 16:9
FPS = 30
BG = "#0d1117"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
YELLOW = "#d29922"
GRAY = "#8b949e"
WHITE = "#e6edf3"
DARK_CARD = "#161b22"
BORDER = "#30363d"
OUTPUT = "D:/Workspace/03_Dev_Projects/recall/demo/recall-demo-video.mp4"
TMP = tempfile.mkdtemp(prefix="recall_vid_")

# ─── Font helpers ───────────────────────────────────────────────
def get_font(size):
    candidates = [
        "C:/Windows/Fonts/msjh.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msjhbd.ttc",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/msgothic.ttc",
        None,
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except (OSError, AttributeError):
            continue
    return ImageFont.load_default()

FONT_LG = get_font(52)
FONT_MD = get_font(36)
FONT_SM = get_font(26)
FONT_XS = get_font(18)

def make_frame(bg=BG):
    img = Image.new("RGB", (W, H), bg)
    return img, ImageDraw.Draw(img)

def tw(draw, text, font):
    """Text width helper."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]

def center_text(draw, text, y, font=FONT_LG, fill=WHITE):
    w = tw(draw, text, font)
    x = (W - w) // 2
    draw.text((x, y), text, fill=fill, font=font)

def draw_card(draw, x, y, w, h, bg=DARK_CARD):
    draw.rounded_rectangle([x, y, x+w, y+h], radius=12, fill=bg, outline=BORDER)

# ─── Scene generators ───────────────────────────────────────────

def scene_title(n_frames):
    """Hook: 你的 AI 助理有失憶症嗎？"""
    frames = []
    for i in range(n_frames):
        img, draw = make_frame()
        draw.rectangle([200, 80, W-200, 86], fill=ACCENT)
        center_text(draw, "你的 AI 助理", H//2 - 80, FONT_LG, GRAY)
        center_text(draw, "有失憶症嗎？", H//2 - 20, get_font(64), WHITE)
        center_text(draw, "你跟它說過的，它真的記得嗎？", H//2 + 50, FONT_MD, GRAY)
        draw.rectangle([200, H-100, W-200, H-94], fill=ACCENT)
        p = min(1.0, i / max(n_frames*0.6, 1))
        draw.rectangle([200, H-80, 200+int(p*(W-400)), H-74], fill=GREEN)
        frames.append(np.array(img))
    return frames

def scene_problem(n_frames):
    """痛點視覺故事：你告訴 AI → AI 忘記 → 你要重講"""
    frames = []
    items = [
        ("💬", "你告訴 AI 你的偏好", "「用 Docker 部署」", 100),
        ("❌", "AI 忘記了", "「我創建一個 Dockerfile...」", 280),
        ("😤", "你要再說一次", "「我說過了！docker-compose！」", 460),
    ]
    for i in range(n_frames):
        img, draw = make_frame()
        p = min(1.0, (i+1) / (n_frames*0.7))
        draw.text((50, 30), "你每天在重複的事", fill=GRAY, font=FONT_SM)
        draw.rectangle([50, 70, int(50+(W-100)*p), 76], fill=ACCENT)
        for j, (icon, title, desc, y) in enumerate(items):
            fp = max(0, min(1, p*3 - j*0.5))
            if fp <= 0:
                continue
            draw.ellipse([120, y, 200, y+80], fill=DARK_CARD, outline=BORDER)
            draw.text((130, y+10), icon, fill=WHITE, font=get_font(40))
            draw.text((230, y+5), title, fill=WHITE, font=get_font(32))
            draw.text((230, y+45), desc, fill=GRAY, font=get_font(22))
            if j < len(items)-1 and fp > 0.5:
                draw.text((W//2-15, y+95), "↓", fill=GREEN, font=get_font(30))
        frames.append(np.array(img))
    return frames

def scene_pain_points(n_frames):
    """三大痛點 — 改用一般人看得懂的語言，修正被切掉的 bug"""
    frames = []
    cards = [
        ("🔄", "換不同 AI", "都要重新教一次", GRAY),
        ("🗑️", "開新對話", "一切從頭來過", GRAY),
        ("💰", "長期記憶很貴", "每個月燒掉一堆費用", GRAY),
    ]
    # Fix: card width 290, gap=30, 3 cards fit: 60+290+30+290+30+290=990 < 1080
    CARD_W, GAP = 290, 30
    starts = [60, 60+CARD_W+GAP, 60+2*(CARD_W+GAP)]
    for i in range(n_frames):
        img, draw = make_frame()
        p = min(1.0, (i+1) / (n_frames*0.7))
        center_text(draw, "所有 AI 記憶方案的問題", 30, FONT_SM, GRAY)
        draw.rectangle([W//2-150, 65, int(W//2-150+400*p), 71], fill=ACCENT)

        # Non-dev explanation at top
        if p > 0.15:
            center_text(draw, "對話塞不下記憶", 120, FONT_XS, GRAY)
        if p > 0.3:
            center_text(draw, "租線上資料庫要花錢維護", 150, FONT_XS, GRAY)
        if p > 0.45:
            center_text(draw, "叫 AI 自己記更是燒錢", 180, FONT_XS, GRAY)

        for j, (icon, title, desc, color) in enumerate(cards):
            fp = max(0, min(1, p*2.5 - j*0.5))
            if fp <= 0:
                continue
            cx = starts[j]
            cy = 280
            draw_card(draw, cx, cy, CARD_W, 220)
            # Icon centered in card
            draw.text((cx+CARD_W//2, cy+35), icon, fill=WHITE, font=get_font(48), anchor="mt")
            draw.text((cx+CARD_W//2, cy+100), title, fill=WHITE, font=get_font(30), anchor="mt")
            draw.text((cx+CARD_W//2, cy+155), desc, fill=GRAY, font=get_font(22), anchor="mt")

        # Bottom summary
        if p > 0.75:
            draw_card(draw, 60, 540, W-120, 80)
            center_text(draw, "解決方案：要嘛燒錢、要嘛重教、要嘛換工具一切歸零", 560, FONT_XS, GRAY)

        frames.append(np.array(img))
    return frames

def scene_solution(n_frames):
    """recall-sqlite 登場"""
    frames = []
    for i in range(n_frames):
        img, draw = make_frame()
        p = min(1.0, (i+1) / (n_frames*0.6))
        glow = int(p * 300)
        draw.ellipse([W//2-glow, H//2-glow, W//2+glow, H//2+glow], fill="#1a2332")
        if p > 0.1:
            center_text(draw, "🧠 解決方案", H//2-100, FONT_SM, GRAY)
        if p > 0.25:
            center_text(draw, "recall-sqlite", H//2-30, get_font(64), ACCENT)
        if p > 0.45:
            center_text(draw, "給你的 AI 一個不會失憶的大腦", H//2+50, FONT_MD, WHITE)
            center_text(draw, "一條指令安裝 · 單一檔案 · 開源免費", H//2+110, FONT_SM, GRAY)
        frames.append(np.array(img))
    return frames

def scene_setup(n_frames):
    """新增：安裝教學 — 三步驟，一般人也看得懂"""
    frames = []
    steps = [
        ("①", "下載 LM Studio", "免費，載入 nomic-embed 模型", "約 150MB"),
        ("②", "安裝 recall", "pip install recall-sqlite", "一條指令"),
        ("③", "開始使用", "AI 自動存取記憶", "零設定"),
    ]
    for i in range(n_frames):
        img, draw = make_frame()
        p = min(1.0, (i+1) / (n_frames*0.65))
        center_text(draw, "開始使用，只要三步", 30, FONT_SM, GRAY)
        draw.rectangle([W//2-100, 65, int(W//2-100+300*p), 71], fill=ACCENT)

        for j, (num, title, desc, tag) in enumerate(steps):
            fp = max(0, min(1, p*2.5 - j*0.3))
            if fp <= 0:
                continue
            cx = 60 + j * 340
            cy = 150
            draw_card(draw, cx, cy, 300, 380)
            # Number circle
            draw.ellipse([cx+110, cy+30, cx+190, cy+110], fill="#1a2332", outline=ACCENT)
            draw.text((cx+150, cy+70), num, fill=ACCENT, font=get_font(44), anchor="mt")
            # Title
            draw.text((cx+150, cy+150), title, fill=WHITE, font=get_font(30), anchor="mt")
            # Description
            draw.text((cx+150, cy+210), desc, fill=GRAY, font=get_font(22), anchor="mt")
            # Tag
            draw_card(draw, cx+50, cy+280, 200, 45, bg="#1c2128")
            draw.text((cx+150, cy+302), tag, fill=GREEN, font=FONT_XS, anchor="mt")

            # Arrows between steps
            if j < len(steps)-1 and fp > 0.5:
                ax = cx + 300 + 10
                draw.text((ax, cy+160), "→", fill=ACCENT, font=get_font(40))

        if p > 0.75:
            center_text(draw, "不用 GPU、不用 API Key、不用上雲端", H-80, FONT_XS, GRAY)

        frames.append(np.array(img))
    return frames

def scene_benefits(n_frames):
    """三大優勢 — 一般人也懂的語言"""
    frames = []
    benefits = [
        ("⚡", "超快回應", "80 毫秒，比眨眼還快 10 倍", GREEN),
        ("🔒", "完全離線", "資料不離開你的電腦，不需網路", ACCENT),
        ("💰", "完全免費", "不用月費、沒有隱藏費用", YELLOW),
    ]
    for i in range(n_frames):
        img, draw = make_frame()
        p = min(1.0, (i+1) / (n_frames*0.65))
        center_text(draw, "為什麼不一樣", 30, FONT_SM, GRAY)
        draw.rectangle([W//2-80, 65, int(W//2-80+280*p), 71], fill=ACCENT)

        for j, (icon, title, desc, color) in enumerate(benefits):
            fp = max(0, min(1, p*2.5 - j*0.3))
            if fp <= 0:
                continue
            cx = 60 + j * 340
            cy = 150
            draw_card(draw, cx, cy, 300, 380)
            draw.text((cx+150, cy+40), icon, fill=color, font=get_font(64), anchor="mt")
            draw.text((cx+150, cy+130), title, fill=WHITE, font=get_font(34), anchor="mt")
            # Word-wrap desc
            dl = [desc[i:i+16] for i in range(0, len(desc), 16)]
            for li, line in enumerate(dl):
                draw.text((cx+150, cy+180+li*35), line, fill=GRAY, font=get_font(22), anchor="mt")

        if p > 0.7:
            draw_card(draw, 60, 560, W-120, 100)
            center_text(draw, "對比：其他方案需要雲端 AI、需要資料庫、需要網路連線", 580, FONT_XS, GRAY)
            center_text(draw, "recall-sqlite：只需要一個檔案", 610, FONT_SM, GREEN)

        frames.append(np.array(img))
    return frames

def scene_how_it_works(n_frames):
    """運作流程 — 4 步驟，修正被切掉 bug"""
    frames = []
    steps = [
        ("🗣️", "你告訴 AI", "「用 Docker 部署」"),
        ("🧠", "AI 記住", "存入 SQLite"),
        ("🔄", "跨工具記憶", "換 AI 也記得"),
        ("⚡", "瞬間找到", "80ms 就找到"),
    ]
    # Fix: 4 cards, width 230, gap=20 → 30+230+20+230+20+230+20+230=1010 < 1080
    CARD_W, GAP = 230, 20
    starts = [30 + j*(CARD_W+GAP) for j in range(4)]
    for i in range(n_frames):
        img, draw = make_frame()
        p = min(1.0, (i+1) / (n_frames*0.6))
        center_text(draw, "運作方式", 30, FONT_SM, GRAY)
        draw.rectangle([W//2-80, 65, int(W//2-80+200*p), 71], fill=ACCENT)

        for j, (icon, title, desc) in enumerate(steps):
            fp = max(0, min(1, p*3 - j*0.3))
            if fp <= 0:
                continue
            cx = starts[j]
            cy = 170
            draw_card(draw, cx, cy, CARD_W, 320)
            draw.text((cx+CARD_W//2, cy+35), icon, fill=WHITE, font=get_font(56), anchor="mt")
            draw.text((cx+CARD_W//2, cy+130), title, fill=WHITE, font=get_font(28), anchor="mt")
            draw.text((cx+CARD_W//2, cy+190), desc, fill=GRAY, font=get_font(22), anchor="mt")

            # Arrow between cards
            if j < len(steps)-1 and fp > 0.5:
                ax = cx + CARD_W + 5
                draw.text((ax, cy+140), "→", fill=ACCENT, font=get_font(30))

        if p > 0.8:
            center_text(draw, "三種方法同時找：文字相似度 + 關鍵字 + 全文搜尋", H-120, FONT_XS, GRAY)
            center_text(draw, "不需要叫 AI 來記，不需要上雲端", H-85, FONT_XS, GRAY)

        frames.append(np.array(img))
    return frames

def scene_cta(n_frames):
    """Call to action"""
    frames = []
    for i in range(n_frames):
        img, draw = make_frame()
        p = min(1.0, (i+1) / (n_frames*0.5))
        if p > 0.1:
            center_text(draw, "開始使用", H//2-100, FONT_SM, GRAY)
        if p > 0.2:
            center_text(draw, "pip install recall-sqlite", H//2-30, get_font(40), GREEN)
        if p > 0.4:
            center_text(draw, "免費安裝 · 不用伺服器 · 零花費", H//2+30, FONT_SM, WHITE)
        if p > 0.55:
            center_text(draw, "🧠 github.com/Jnocode/recall-memory", H//2+90, FONT_MD, ACCENT)
        if p > 0.65:
            center_text(draw, "🏆 已納入 Hermes Agent 官方記憶供應商", H//2+140, FONT_XS, GRAY)
        if p > 0.75:
            draw_card(draw, W//2-150, H-130, 300, 70)
            center_text(draw, "開源 · Apache 2.0 · 免費", H-110, FONT_XS, GRAY)
        frames.append(np.array(img))
    return frames

# ─── Video assembly ──────────────────────────────────────────────

def render_scene(scene_fn, duration_sec, name):
    n = int(duration_sec * FPS)
    frames = scene_fn(n)
    paths = []
    for idx, frame in enumerate(frames):
        fname = os.path.join(TMP, f"{name}_{idx:04d}.png")
        Image.fromarray(frame).save(fname)
        paths.append(fname)
    return paths

def assemble_video(scene_specs, output_path):
    all_paths = []
    for name, duration in scene_specs:
        paths = render_scene(globals()[f"scene_{name}"], duration, name)
        all_paths.extend(paths)
    print(f"  Generating {len(all_paths)} frames...")

    seq_dir = os.path.join(TMP, "seq")
    os.makedirs(seq_dir, exist_ok=True)
    gidx = 0
    for name, duration in scene_specs:
        n = int(duration * FPS)
        for j in range(n):
            src = os.path.join(TMP, f"{name}_{j:04d}.png")
            dst = os.path.join(seq_dir, f"frame_{gidx:06d}.png")
            if os.path.exists(src):
                os.rename(src, dst)
            else:
                img, _ = make_frame()
                img.save(dst)
            gidx += 1
    print(f"  Total frames: {gidx}")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-pattern_type", "sequence",
        "-start_number", "0",
        "-i", os.path.join(seq_dir, "frame_%06d.png"),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-vf", "fps=30",
        output_path
    ]
    print("  Encoding video...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FFmpeg error: {result.stderr[:500]}")
        return False
    size_mb = os.path.getsize(output_path) / (1024*1024)
    print(f"  ✅ Video: {output_path} ({size_mb:.1f} MB)")
    return True

# ─── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    scenes = [
        ("title", 7.0),
        ("problem", 7.0),
        ("pain_points", 9.0),
        ("solution", 5.0),
        ("setup", 8.0),          # NEW: installation tutorial
        ("benefits", 11.0),
        ("how_it_works", 9.0),
        ("cta", 7.0),
    ]
    total = sum(d for _, d in scenes)
    print(f"🎬 recall-sqlite explainer video v2")
    print(f"   Duration: {total:.1f}s @ {FPS}fps = {int(total*FPS)} frames")
    print(f"   Resolution: {W}x{H}")
    print(f"   Scenes: {len(scenes)}")
    print()
    success = assemble_video(scenes, OUTPUT)
    if success:
        print(f"\n✅ Complete!")
    else:
        print(f"\n❌ Failed")
    shutil.rmtree(TMP, ignore_errors=True)
