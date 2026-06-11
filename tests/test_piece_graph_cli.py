from live_recognition.piece_graph_cli import build_parser


def test_graph_command_exposes_min_sticker_area():
    args = build_parser().parse_args(
        [
            "--source",
            "frame.jpg",
            "--min-sticker-area",
            "40",
        ]
    )

    assert args.min_sticker_area == 40
