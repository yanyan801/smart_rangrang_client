"""
Pi 端音频播放模块
- 中断安全：收到 stop 信号立即清空缓冲并停止播放
- 低延迟写入：预缓冲后立即写出所有可用数据，避免 PyAudio 缓冲区排空
"""

import asyncio
import logging

logger = logging.getLogger("pi.player")

_FRAMES_PER_BUFFER = 4096  # PyAudio 内部缓冲 (256ms @ 16kHz)，防网络抖动


class AudioPlayer:
    """可立即中断的音频播放器"""

    def __init__(self, sample_rate: int = 16000, chunk_samples: int = 640,
                 buffer_chunks: int = 4, device_index: int | None = None):
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.buffer_chunks = buffer_chunks
        self.device_index = device_index
        self._p = None
        self._stream = None
        self._running = False
        self._started = False

    # ---------- 生命周期 ----------

    def start(self) -> asyncio.Queue:
        """启动播放器，返回音频数据消费队列"""
        import pyaudio

        self._p = pyaudio.PyAudio()
        self._stream = self._p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            output_device_index=self.device_index,
            frames_per_buffer=_FRAMES_PER_BUFFER,
        )
        self._running = True
        self._started = False

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Player started: rate={self.sample_rate}, "
                     f"buffer={self.buffer_chunks} chunks, frames_per_buffer={_FRAMES_PER_BUFFER}")
        return self._queue

    async def _run_loop(self):
        """预缓冲 → 连续写出，保持 PyAudio 缓冲区不排空"""
        loop = asyncio.get_event_loop()
        buf: list[bytes] = []

        while self._running or buf:
            if not self._running and not buf:
                break

            try:
                if not self._started:
                    # 预缓冲：攒够 buffer_chunks 再开始播放
                    while len(buf) < self.buffer_chunks:
                        try:
                            chunk = await asyncio.wait_for(
                                self._queue.get(), timeout=0.5
                            )
                            buf.append(chunk)
                        except asyncio.TimeoutError:
                            if not self._running:
                                break
                    if not buf:
                        break
                    self._started = True
                else:
                    # 非阻塞地排空队列中所有可用 chunk
                    try:
                        while True:
                            buf.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        pass

                    if not buf:
                        # 短暂等待下一个 chunk
                        try:
                            chunk = await asyncio.wait_for(
                                self._queue.get(), timeout=0.03
                            )
                            buf.append(chunk)
                        except asyncio.TimeoutError:
                            continue

                if buf:
                    data = b''.join(buf)
                    buf = []
                    await loop.run_in_executor(
                        None, _write, self._stream, data
                    )
            except Exception as e:
                if self._running:
                    logger.error(f"Playback error: {e}")
                break

        logger.debug("Player loop ended")

    # ---------- 中断控制 ----------

    def stop(self):
        """立即停止播放"""
        logger.info("Player stop requested")
        self._running = False
        self._started = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def flush_and_restart(self):
        """清空播放队列并重置预缓冲（不重启 stream，避免 ALSA 线程竞争崩溃）"""
        self._started = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info("Player flushed after interrupt")

    # ---------- 清理 ----------

    def _close_stream(self):
        if self._stream:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    async def close(self):
        """完全关闭播放器"""
        self.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._close_stream()
        if self._p:
            self._p.terminate()
            self._p = None
        logger.info("Player closed")

    # ---------- 设备信息 ----------

    @staticmethod
    def list_devices():
        """列出所有输出设备"""
        import pyaudio
        p = pyaudio.PyAudio()
        count = p.get_device_count()
        for i in range(count):
            info = p.get_device_info_by_index(i)
            if info["maxOutputChannels"] > 0:
                print(f"  [{i}] {info['name']} "
                      f"(out={info['maxOutputChannels']}, "
                      f"rate={int(info['defaultSampleRate'])})")
        p.terminate()


def _write(stream, data: bytes):
    """在线程池中执行的同步写入"""
    stream.write(data, exception_on_underflow=False)
