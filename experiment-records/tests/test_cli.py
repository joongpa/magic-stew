from live_recognition.cli import _parse_source, build_parser


def test_parse_source_accepts_pasted_quoted_image_paths():
    source = '"/Users/jeffpark/Downloads/cubingVideos/frames/cubingMethoComparison/frame_000177.jpg\n"'

    assert (
        _parse_source(source)
        == "/Users/jeffpark/Downloads/cubingVideos/frames/cubingMethoComparison/frame_000177.jpg"
    )


def test_parse_source_preserves_camera_index_support():
    assert _parse_source("0") == 0


def test_display_is_opt_in():
    args = build_parser().parse_args([])

    assert args.display is False


def test_display_flag_enables_preview_window():
    args = build_parser().parse_args(["--display"])

    assert args.display is True


def test_yolo_detector_parser_options():
    args = build_parser().parse_args(
        ["--detector", "yolo", "--model", "runs/500ExamplesWeights.pt", "--imgsz", "640"]
    )

    assert args.detector == "yolo"
    assert args.model == "runs/500ExamplesWeights.pt"
    assert args.imgsz == 640


def test_color_plot_video_options():
    args = build_parser().parse_args(
        ["--color-plot", "runs/colors.png", "--color-plot-frames", "12"]
    )

    assert str(args.color_plot) == "runs/colors.png"
    assert args.color_plot_frames == 12
    assert args.color_plot_samples_per_frame == 800


def test_no_detect_parser_option():
    args = build_parser().parse_args(["--no-detect", "--color-debug"])

    assert args.no_detect is True
    assert args.color_debug is True
