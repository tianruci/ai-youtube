import os
import yt_dlp
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class VideoProcessor:
    """视频处理器，使用yt-dlp下载和转换视频"""
    
    def __init__(self):
        self.ydl_opts = {
            'format': 'bestaudio/best',  # 优先下载最佳音频源
            'outtmpl': '%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                # 直接在提取阶段转换为单声道 16k（空间小且稳定）
                'preferredcodec': 'm4a',
                'preferredquality': '192'
            }],
            # 全局FFmpeg参数：单声道 + 16k 采样率 + faststart
            'postprocessor_args': ['-ac', '1', '-ar', '16000', '-movflags', '+faststart'],
            'prefer_ffmpeg': True,
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,  # 强制只下载单个视频，不下载播放列表
        }
    
    async def download_and_convert(self, url: str, output_dir: Path) -> tuple[str, str]:
        """
        下载视频并转换为m4a格式
        
        Args:
            url: 视频链接
            output_dir: 输出目录
            
        Returns:
            转换后的音频文件路径
        """
        try:
            # 创建输出目录
            output_dir.mkdir(exist_ok=True)
            
            # 生成唯一的文件名
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            output_template = str(output_dir / f"audio_{unique_id}.%(ext)s")
            
            # 更新yt-dlp选项
            ydl_opts = self.ydl_opts.copy()
            ydl_opts['outtmpl'] = output_template

            # 如果设置了 YTDLP_COOKIEFILE 环境变量，则将其传递给 yt-dlp
            # 期望为一个本地文件路径，格式为 Netscape cookies（浏览器导出格式）
            cookiefile = os.getenv("YTDLP_COOKIEFILE")
            if cookiefile:
                ydl_opts['cookiefile'] = cookiefile
            
            logger.info(f"开始下载视频: {url}")
            
            # 直接同步执行，不使用线程池
            # 在FastAPI中，IO密集型操作可以直接await
            import asyncio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # 获取视频信息（放到线程池避免阻塞事件循环）
                info = await asyncio.to_thread(ydl.extract_info, url, False)
                video_title = info.get('title', 'unknown')
                expected_duration = info.get('duration') or 0
                logger.info(f"视频标题: {video_title}")
                
                # 下载视频（放到线程池避免阻塞事件循环）
                await asyncio.to_thread(ydl.download, [url])
            
            # 查找生成的m4a文件
            audio_file = str(output_dir / f"audio_{unique_id}.m4a")
            
            if not os.path.exists(audio_file):
                # 如果m4a文件不存在，查找其他音频格式
                for ext in ['webm', 'mp4', 'mp3', 'wav']:
                    potential_file = str(output_dir / f"audio_{unique_id}.{ext}")
                    if os.path.exists(potential_file):
                        audio_file = potential_file
                        break
                else:
                    raise Exception("未找到下载的音频文件")
            
            # 校验时长，如果和源视频差异较大，尝试一次ffmpeg规范化重封装
            try:
                import subprocess
                # 使用列表参数避免在 Windows 上因引号/转义导致的 "Invalid argument"
                out = subprocess.check_output([
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_file
                ]).decode().strip()
                actual_duration = float(out) if out else 0.0
            except Exception:
                actual_duration = 0.0
            
            if expected_duration and actual_duration and abs(actual_duration - expected_duration) / expected_duration > 0.1:
                logger.warning(
                    f"音频时长异常，期望{expected_duration}s，实际{actual_duration}s，尝试重封装修复…"
                )
                try:
                    fixed_path = str(output_dir / f"audio_{unique_id}_fixed.m4a")
                    # 使用列表参数调用 ffmpeg，避免 shell 字符串引号问题
                    subprocess.check_call([
                        "ffmpeg", "-y", "-i", audio_file,
                        "-vn", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", fixed_path
                    ])
                    # 用修复后的文件替换
                    audio_file = fixed_path
                    # 重新探测
                    out2 = subprocess.check_output([
                        "ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        audio_file
                    ]).decode().strip()
                    actual_duration2 = float(out2) if out2 else 0.0
                    logger.info(f"重封装完成，新时长≈{actual_duration2:.2f}s")
                except Exception as e:
                    logger.error(f"重封装失败：{e}")
            
            logger.info(f"音频文件已保存: {audio_file}")
            return audio_file, video_title
            
        except Exception as e:
            logger.error(f"下载视频失败: {str(e)}")
            raise Exception(f"下载视频失败: {str(e)}")
    
    def get_video_info(self, url: str) -> dict:
        """
        获取视频信息
        
        Args:
            url: 视频链接
            
        Returns:
            视频信息字典
        """
        try:
            # 支持可选的 cookies 文件以处理需要登录/会话的站点（如抖音）
            info_opts = {'quiet': True}
            cookiefile = os.getenv("YTDLP_COOKIEFILE")
            if cookiefile:
                info_opts['cookiefile'] = cookiefile

            with yt_dlp.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', ''),
                    'upload_date': info.get('upload_date', ''),
                    'description': info.get('description', ''),
                    'view_count': info.get('view_count', 0),
                }
        except Exception as e:
            logger.error(f"获取视频信息失败: {str(e)}")
            raise Exception(f"获取视频信息失败: {str(e)}")

    async def convert_local_video(self, local_video_path: Path, output_dir: Path) -> tuple[str, str]:
        """
        将本地视频文件提取音频并转换为m4a，返回音频路径和视频标题（文件名）。
        仅允许项目下的文件路径。
        """
        try:
            # 确保输入存在
            if not local_video_path.exists():
                raise Exception(f"本地视频文件不存在: {local_video_path}")

            # 创建输出目录
            output_dir.mkdir(exist_ok=True)

            import uuid, subprocess
            unique_id = str(uuid.uuid4())[:8]
            out_audio = output_dir / f"audio_{unique_id}.m4a"

            # 使用 ffmpeg 提取音频并转换为设定的参数
            cmd = [
                "ffmpeg", "-y", "-i", str(local_video_path),
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "160k",
                str(out_audio)
            ]
            subprocess.check_call(cmd)

            video_title = local_video_path.stem
            return str(out_audio), video_title
        except Exception as e:
            logger.error(f"处理本地视频失败: {e}")
            raise