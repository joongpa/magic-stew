from live_recognition.net_cli import build_parser


def test_net_command_exposes_raw_net_outputs():
    args = build_parser().parse_args(
        [
            "--source",
            "frame.jpg",
            "--save-net",
            "runs/net.png",
            "--save-graph",
            "runs/graph.jpg",
        ]
    )

    assert args.source == "frame.jpg"
    assert str(args.save_net) == "runs/net.png"
    assert str(args.save_graph) == "runs/graph.jpg"
