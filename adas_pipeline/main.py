"""
main.py — CLI entry point for the ADAS pipeline.

Usage:
    # Process a single video file:
    python main.py --input input/video.mp4

    # Process JAAD dataset:
    python main.py --mode jaad

    # Resume from last successful checkpoint:
    python main.py --mode jaad --resume

    # Clear all checkpoints and restart:
    python main.py --mode jaad --fresh

    # Use a specific JAAD split/subset:
    python main.py --mode jaad --jaad-split default --jaad-subset train
"""

import argparse
import logging
import sys
import os

# Make sure local imports work regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from pipeline_runner import PipelineRunner
import modules.input_handler as input_handler
import modules.frame_extractor as frame_extractor
import modules.frame_cleaner as frame_cleaner
import modules.detector as detector
import modules.tracker as tracker
import modules.pose_estimator as pose_estimator
import modules.behavior_analyzer as behavior_analyzer
import modules.intent_predictor as intent_predictor
import modules.tagger as tagger
import exporter


def build_stages():
    return [
        ("input_validation",  input_handler.run),
        ("frame_extraction",  frame_extractor.run),
        ("frame_cleaning",    frame_cleaner.run),
        ("detection",         detector.run),
        ("tracking",          tracker.run),
        ("pose_estimation",   pose_estimator.run),
        ("behavior_analysis", behavior_analyzer.run),
        ("intent_prediction", intent_predictor.run),
        ("tagging",           tagger.run),
        ("export",            exporter.run),
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="ADAS Sensor Data Cleaning & Behavior Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to input video file (required for --mode video)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["video", "jaad", "pie"],
        default="jaad",
        help="Input mode: 'video' for a single file, 'jaad' for JAAD dataset (default: jaad)",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Skip already-completed stages using checkpoints",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear all checkpoints and run from scratch",
    )
    parser.add_argument(
        "--jaad-split",
        default=config.JAAD_SPLIT,
        choices=["default", "all_videos", "high_visibility"],
        help="Which JAAD split to use (default: %(default)s)",
    )
    parser.add_argument(
        "--jaad-subset",
        default=config.JAAD_SUBSET,
        choices=["train", "val", "test", "all"],
        help="Which subset of the split to process (default: %(default)s)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of JAAD videos to process (for quick testing)",
    )
    parser.add_argument(
        "--no-bytetrack",
        action="store_true",
        help="Use simple IoU tracker instead of ByteTrack",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging to console",
    )
    return parser.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    # Use a UTF-8 stream so Unicode chars don't crash on Windows cp1252 consoles
    utf8_stdout = open(
        sys.stdout.fileno(),
        mode="w",
        encoding="utf-8",
        errors="replace",
        closefd=False,
        buffering=1,
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(utf8_stdout)],
    )


def main():
    args = parse_args()
    setup_logging(args.verbose)

    log = logging.getLogger("main")
    log.info("=" * 60)
    log.info("ADAS Pipeline starting")
    log.info(f"  Mode:   {args.mode}")
    log.info(f"  Resume: {args.resume}")
    log.info("=" * 60)

    # Override config from CLI flags
    if args.no_bytetrack:
        config.USE_BYTETRACK = False
    if args.max_videos:
        config.JAAD_MAX_VIDEOS = args.max_videos

    # Build initial context
    context = {"mode": args.mode}

    if args.mode == "video":
        if not args.input:
            log.error("--input is required when --mode video")
            sys.exit(1)
        context["video_path"] = args.input

    elif args.mode == "jaad":
        context["jaad_split"] = args.jaad_split
        context["jaad_subset"] = args.jaad_subset

    elif args.mode == "pie":
        log.error(
            "PIE mode is not yet implemented in this version.\n"
            "To use PIE data, run extract_pie.py first to generate crops,\n"
            "then use --mode video with individual PIE video files."
        )
        sys.exit(1)

    # Build and run pipeline
    stages = build_stages()
    runner = PipelineRunner(stages, resume=args.resume)

    if args.fresh:
        log.info("--fresh: clearing all checkpoints")
        runner.clear_checkpoints()

    try:
        result = runner.run(context)
        log.info("=" * 60)
        log.info("Pipeline complete!")
        log.info(f"  JSON output: {result.get('json_path', 'N/A')}")
        log.info(f"  CSV output:  {result.get('csv_path', 'N/A')}")
        log.info("=" * 60)
    except RuntimeError as e:
        log.error(str(e))
        log.info("Run with --resume to continue from the last successful stage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
