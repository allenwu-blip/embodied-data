"""Shared video helpers for AgiBot → LeRobot v3 conversion.

The re-encode obeys the LeRobot v3 video contract: h264 codec, no B-frames
(``bf=0``), short GOP (``g=2``), monotonic PTS aligned to ``frame_index``.
This keeps DTS == PTS so the mp4 muxer never sees out-of-order packets and
``decode_video_frames``-style readers can seek by frame index.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av


@dataclass(frozen=True)
class VideoMetadata:
    """Probe results for an upstream video file."""

    n_frames: int
    fps: float
    duration: float
    width: int
    height: int
    codec_name: str


def probe_video(src: Path) -> VideoMetadata:
    """Read header-only metadata. No frame decode; cheap."""
    src = Path(src)
    container = av.open(str(src))
    try:
        stream = container.streams.video[0]
        ctx = stream.codec_context
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        fps = float(rate) if rate is not None else 0.0
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        n_frames = int(stream.frames) if stream.frames else 0
        return VideoMetadata(
            n_frames=n_frames,
            fps=fps,
            duration=duration,
            width=int(ctx.width),
            height=int(ctx.height),
            codec_name=str(ctx.codec.name) if ctx.codec else "",
        )
    finally:
        container.close()


def reencode_video(src: Path, dst: Path, *, fps: int) -> None:
    """Re-encode ``src`` to ``dst`` per the LeRobot v3 video contract.

    Uses h264 with ``bf=0`` (no B-frames), ``g=2`` (short GOP), ``crf=30``,
    ``yuv420p``. Frame PTS is assigned monotonically as ``frame_index``, so
    timestamps == frame_index/fps.
    """
    time_base = Fraction(1, fps)
    in_container = av.open(str(src))
    in_stream = in_container.streams.video[0]
    width = in_stream.codec_context.width
    height = in_stream.codec_context.height

    out_container = av.open(str(dst), mode="w")
    out_stream = out_container.add_stream("h264", rate=fps)
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": "30", "g": "2", "bf": "0"}
    out_stream.codec_context.time_base = time_base

    try:
        for i, frame in enumerate(in_container.decode(in_stream)):
            new_frame = frame.reformat(format="yuv420p")
            new_frame.pts = i
            new_frame.time_base = time_base
            for packet in out_stream.encode(new_frame):
                out_container.mux(packet)
        for packet in out_stream.encode():
            out_container.mux(packet)
    finally:
        out_container.close()
        in_container.close()


def count_reencoded_frames(path: Path) -> int:
    """Return the number of frames in ``path`` (post re-encode)."""
    with av.open(str(path)) as c:
        return int(c.streams.video[0].frames)
