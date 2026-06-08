from live_recognition.piece_graph_cli import build_parser


def test_graph_command_exposes_min_sticker_segment_ratio():
    args = build_parser().parse_args(
        [
            "--source",
            "frame.jpg",
            "--min-sticker-segment-ratio",
            "0.4",
        ]
    )

    assert args.min_sticker_segment_ratio == 0.4
