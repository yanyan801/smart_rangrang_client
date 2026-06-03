"""
Pi 端音频播放模块
- 中断安全：收到 stop 信号立即清空缓冲并停止播放
- 批量写入：攒多个 chunk 一次写入，减少 executor 开销和 ALSA underrun
- 预缓冲：积累足够 chunk 再开始播放
"""

import asyncio
import logging

logger = logging.getLogger("pi.player")

_BATCH_CHUNKS = 4  # 一次写入 4 个 chunk (160ms)，减少 executor 调用


class AudioPlayer:
    """可立即中断的音频播放器"""

    def __init__(self, sample_rate: int = 16000, chunk_samples: int = 640,
                 buffer_chunks: int = 16, device_index: int | None = None):
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
            frames_per_buffer=self.chunk_samples * _BATCH_CHUNKS,
        )
        self._running = True
        self._started = False

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Player started: rate={self.sample_rate}, "
                     f"buffer={self.buffer_chunks} chunks, batch={_BATCH_CHUNKS}")
        return self._queue

    async def _run_loop(self):
        """批量消费 Queue → executor 一次写入"""
        loop = asyncio.get_event_loop()
        buf: list[bytes] = []

        while self._running or buf:
            if not self._running and not buf:
                break

            # 攒到 buffer_chunks 才开始播
            need = self.buffer_chunks if not self._started else _BATCH_CHUNKS
            timeout = 0.5 if not self._started else 0.3

            try:
                while len(buf) < need:
                    try:
                        chunk = await asyncio.wait_for(
                            self._queue.get(), timeout=timeout
                        )
                        buf.append(chunk)
                    except asyncio.TimeoutError:
                        if self._started and buf:
                            break
                        if not self._running:
                            break

                if not buf:
                    break

                # 攒够初始缓冲后标记为已启动
                if not self._started and len(buf) >= self.buffer_chunks:
                    self._started = True

                # 批量写入：取最多 _BATCH_CHUNKS 个拼接
                batch = buf[:_BATCH_CHUNKS]
                buf = buf[_BATCH_CHUNKS:]
                data = b''.join(batch)

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
        """排空后重新开始（打断恢复）"""
        self.stop()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        self._close_stream()
        self.start()
        logger.info("Player restarted after flush")

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
