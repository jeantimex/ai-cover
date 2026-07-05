# AI Cover Mac

[English](README.md)

一个 Mac 原生的 AI 翻唱工具，可以将歌曲中的人声转换成另一位歌手的声音。专为 Apple Silicon 优化，支持 MPS 加速。

## 功能特点

- **YouTube 支持** — 直接从 YouTube 下载歌曲和参考人声
- **三阶段人声分离** — 使用 UVR MDX-Net 模型从任何歌曲中提取干净人声
- **零样本声音转换** — 只需 10-30 秒的参考音频即可转换人声（无需训练）
- **F0 音高保持** — 保留原始旋律和音高
- **专业音效处理** — 自动添加混响、压缩和混音

## 工作原理

```
输入歌曲 → 分离人声 → 转换声音 → 添加效果 → 混音 → 输出
                ↓
         伴奏（保留）
         和声（保留）
         主人声 → Seed-VC → 转换后的人声
```

## 安装

### 前置要求

- macOS，Apple Silicon（M1/M2/M3）或 Intel 芯片
- Python 3.10+
- ffmpeg（`brew install ffmpeg`）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/my-singer.git
cd my-singer

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

首次运行会下载所需模型（约 2GB）。

## 快速开始

### 基本用法

```bash
# 使用 YouTube 歌曲和 YouTube 参考人声
python pipeline.py "https://youtube.com/watch?v=歌曲ID" \
    --reference "https://youtube.com/watch?v=人声ID"
```

### 使用本地文件

```bash
# 本地歌曲 + 本地参考音频
python pipeline.py /path/to/song.mp3 --reference /path/to/voice.wav

# YouTube 歌曲 + 本地参考音频
python pipeline.py "https://youtube.com/watch?v=..." --reference /path/to/voice.wav
```

### 输出文件

结果保存在 `song_output/<歌曲ID>/` 目录：
- `cover.wav` — 最终混音成品
- `converted_vocal.wav` — 仅转换后的人声
- `instrumental.wav` — 分离的伴奏
- `dry_main_vocal.wav` — 原始分离的人声

## 音高调整指南

当源声音和目标声音音域不同时，使用 `--pitch` 参数进行半音调整。

### 跨性别转换

| 转换类型 | 命令 |
|---------|------|
| 女声 → 男声 | `--pitch -12`（降低一个八度） |
| 男声 → 女声 | `--pitch 12`（升高一个八度） |

**示例：女声歌曲转男声**
```bash
python pipeline.py "https://youtube.com/watch?v=女声歌曲" \
    --reference "https://youtube.com/watch?v=男声参考" \
    --pitch -12
```

**示例：男声歌曲转女声**
```bash
python pipeline.py "https://youtube.com/watch?v=男声歌曲" \
    --reference "https://youtube.com/watch?v=女声参考" \
    --pitch 12
```

### 同性别转换

同性别但音域不同的情况：

| 情况 | 命令 |
|-----|------|
| 源声音略高于参考 | `--pitch -2` 到 `--pitch -3` |
| 源声音明显高于参考 | `--pitch -4` 到 `--pitch -5` |
| 源声音略低于参考 | `--pitch 2` 到 `--pitch 3` |
| 源声音明显低于参考 | `--pitch 4` 到 `--pitch 5` |

**示例：高音女声转低音女声**
```bash
python pipeline.py "高音歌曲.mp3" \
    --reference "低音参考.wav" \
    --pitch -4
```

### 女声音域参考

| 声部类型 | 音域 | 典型调整 |
|---------|------|---------|
| 女高音 (Soprano) | 高 (C4-C6) | 参考基准 |
| 女中音 (Mezzo) | 中 (A3-A5) | 与女高音相差 ±2-3 半音 |
| 女低音 (Alto) | 低 (F3-F5) | 与女高音相差 ±4-5 半音 |

### 自动 F0 vs 手动控制

| 设置 | 使用场景 |
|-----|---------|
| `--auto-f0`（默认） | 跨性别转换或音域差异很大时 |
| `--no-auto-f0` | 同性别、音域相近、或需要保持原始旋律时 |

**注意：** `--auto-f0` 会将源音高移动到参考声音的音域范围。这对跨性别转换很有帮助，但可能会改变旋律。如果音质不错但旋律听起来不同，请尝试 `--no-auto-f0`。

### 调整技巧

- **旋律听起来不同？** — 尝试 `--no-auto-f0` 保持原始旋律
- **需要时添加 `--pitch`** — 当自动调整不够时手动补充
- **小步调整** — 每次尝试 ±2 半音
- **注意失真** — 调整过大可能导致音质下降

## 参数说明

### 必需参数

| 参数 | 说明 |
|-----|------|
| `source` | YouTube 链接或本地歌曲路径 |
| `--reference` | YouTube 链接或参考人声路径（10-30秒） |

### 声音转换参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--pitch` | 0 | 音高偏移，单位半音（-12 到 +12） |
| `--steps` | 50 | 扩散步数（30-100，越高质量越好） |
| `--cfg` | 0.7 | 无分类器引导率（0.5-0.9） |
| `--auto-f0` | 开启 | 自动调整音高以匹配参考音域 |
| `--no-auto-f0` | — | 禁用自动音高调整 |

### 输出参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--output-dir` | `song_output` | 输出目录 |
| `--format` | `wav` | 输出格式（`wav` 或 `mp3`） |
| `--no-keep-files` | — | 删除中间文件 |

### 高级参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--engine` | `seedvc` | 转换引擎（`seedvc` 或 `rvc`） |
| `--no-separate-ref` | — | 不分离参考音频中的人声 |

## 使用示例

### 高质量转换

增加扩散步数以获得更好的效果：

```bash
python pipeline.py "song.mp3" \
    --reference "voice.wav" \
    --steps 100
```

### 增强声音相似度

提高 CFG 率以更接近参考声音：

```bash
python pipeline.py "song.mp3" \
    --reference "voice.wav" \
    --cfg 0.85
```

### 女声流行歌曲 → 男声摇滚

```bash
python pipeline.py "https://youtube.com/watch?v=女声流行" \
    --reference "https://youtube.com/watch?v=男声摇滚" \
    --pitch -12 \
    --steps 75
```

### 男声 R&B → 女声

```bash
python pipeline.py "https://youtube.com/watch?v=男声RNB" \
    --reference "https://youtube.com/watch?v=女声参考" \
    --pitch 12 \
    --cfg 0.8
```

### 同一歌手，不同歌曲

当参考和源声音相似时，几乎不需要调整：

```bash
python pipeline.py "新歌曲.mp3" \
    --reference "同一歌手片段.wav"
```

## 参考音频建议

参考音频的质量直接影响转换效果：

1. **长度**：15-25 秒的演唱效果最佳
2. **质量**：干净的人声（无背景音乐）
3. **风格**：与源歌曲风格相似效果更好
4. **音域**：尽量包含高音和低音部分

**推荐的参考音频来源：**
- 清唱（A cappella）表演
- 分离的人声音轨
- 清晰的现场演出
- YouTube 视频中清晰的人声（会自动分离）

## 常见问题

### 声音听起来机械/有杂音
- 增加 `--steps` 到 75 或 100
- 尝试更干净的参考音频

### 音高听起来不对
- 添加 `--pitch` 调整（参考上方指南）
- 尝试 `--no-auto-f0` 并手动设置音高

### 声音不像参考
- 增加 `--cfg` 到 0.8 或 0.85
- 使用更长/更干净的参考片段
- 确保参考音频风格相似

### 内存不足
- 关闭其他应用程序
- 处理更短的歌曲
- 减少 `--steps`

### 模型下载失败
- 检查网络连接
- 设置 `HF_TOKEN` 环境变量加速下载
- 重试 — 下载会被缓存

## 项目结构

```
my-singer/
├── pipeline.py        # 主入口
├── download.py        # YouTube/本地音频获取
├── separate.py        # 三阶段人声分离
├── convert_seedvc.py  # Seed-VC 声音转换
├── convert_rvc.py     # RVC 声音转换（开发中）
├── effects.py         # 音频效果和混音
├── qa.py              # 质量检测工具
├── models/            # 下载的模型文件
└── song_output/       # 输出目录
```

## 许可说明

本项目仅供个人学习和研究使用。使用时请尊重歌曲版权。

## 致谢

- [Seed-VC](https://github.com/BytedanceSpeech/seed-vc) — 零样本声音转换
- [audio-separator](https://github.com/karaokenerds/python-audio-separator) — 人声分离
- [UVR MDX-Net](https://github.com/kuielab/mdx-net) — 分离模型
