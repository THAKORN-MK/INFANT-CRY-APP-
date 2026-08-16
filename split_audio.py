"""สุ่มแบ่งชุดข้อมูลเสียงแบบ stratified เป็น train 80% และ test 20%.

สคริปต์นี้คัดลอกไฟล์ .wav โดยไม่แก้ไขหรือลบข้อมูลต้นฉบับ
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT / "data_set_dbl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_set_dbl_split"


@dataclass(frozen=True)
class ClassSplit:
    label: str
    train_count: int
    test_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "สุ่มแบ่งไฟล์ .wav รายคลาสจาก data_set_dbl เป็น "
            "data_set_dbl_split/train และ data_set_dbl_split/test"
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"โฟลเดอร์ข้อมูลต้นฉบับ (ค่าเริ่มต้น: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"โฟลเดอร์ผลลัพธ์ (ค่าเริ่มต้น: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="สัดส่วนข้อมูล train ต่อคลาส (ค่าเริ่มต้น: 0.80)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ค่า random seed เพื่อให้แบ่งซ้ำได้ผลเดิม (ค่าเริ่มต้น: 42)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="ลบโฟลเดอร์ผลลัพธ์เดิมก่อนสร้างใหม่",
    )
    return parser.parse_args()


def wav_files(class_dir: Path) -> list[Path]:
    """คืนรายการไฟล์ WAV ในคลาสตามลำดับคงที่ก่อนนำไปสุ่ม."""
    return sorted(
        (path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"),
        key=lambda path: path.relative_to(class_dir).as_posix().casefold(),
    )


def test_count_for(total: int, train_ratio: float) -> int:
    """คำนวณจำนวน test แบบปัดเศษครึ่งขึ้นและคงทั้งสองชุดเมื่อทำได้."""
    if total < 2:
        return 0

    test_count = int(total * (1.0 - train_ratio) + 0.5)
    return min(max(test_count, 1), total - 1)


def validate_paths(source: Path, output: Path, train_ratio: float) -> tuple[Path, Path]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ข้อมูลต้นฉบับ: {source}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio ต้องมากกว่า 0 และน้อยกว่า 1")
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("โฟลเดอร์ต้นฉบับและผลลัพธ์ต้องไม่ซ้อนกัน")

    return source, output


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"โฟลเดอร์ผลลัพธ์มีข้อมูลอยู่แล้ว: {output}\n"
                "หากต้องการสร้างใหม่ ให้ตรวจสอบปลายทางแล้วระบุ --overwrite"
            )
        shutil.rmtree(output)

    (output / "train").mkdir(parents=True, exist_ok=True)
    (output / "test").mkdir(parents=True, exist_ok=True)


def copy_to_partition(
    files: list[Path],
    class_dir: Path,
    destination: Path,
) -> None:
    for source_file in files:
        relative_path = source_file.relative_to(class_dir)
        destination_file = destination / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)


def split_dataset(
    source: Path,
    output: Path,
    train_ratio: float = 0.80,
    seed: int = 42,
    overwrite: bool = False,
) -> list[ClassSplit]:
    """สุ่มและคัดลอกไฟล์แต่ละคลาสไปยัง train/test."""
    source, output = validate_paths(source, output, train_ratio)
    class_dirs = sorted(
        (path for path in source.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    if not class_dirs:
        raise ValueError(f"ไม่พบโฟลเดอร์คลาสภายใน: {source}")

    files_by_class = [(class_dir, wav_files(class_dir)) for class_dir in class_dirs]
    empty_classes = [class_dir.name for class_dir, files in files_by_class if not files]
    if empty_classes:
        raise ValueError(f"ไม่พบไฟล์ .wav ในคลาส: {', '.join(empty_classes)}")

    prepare_output(output, overwrite)
    rng = random.Random(seed)
    results: list[ClassSplit] = []

    for class_dir, files in files_by_class:
        rng.shuffle(files)
        test_count = test_count_for(len(files), train_ratio)
        test_files = files[:test_count]
        train_files = files[test_count:]

        copy_to_partition(train_files, class_dir, output / "train" / class_dir.name)
        copy_to_partition(test_files, class_dir, output / "test" / class_dir.name)
        results.append(
            ClassSplit(
                label=class_dir.name,
                train_count=len(train_files),
                test_count=len(test_files),
            )
        )

    return results


def main() -> int:
    args = parse_args()
    results = split_dataset(
        source=args.source,
        output=args.output,
        train_ratio=args.train_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    print("แบ่งข้อมูลเรียบร้อย")
    for result in results:
        print(
            f"- {result.label}: train={result.train_count}, "
            f"test={result.test_count}"
        )
    print(
        f"รวม: train={sum(item.train_count for item in results)}, "
        f"test={sum(item.test_count for item in results)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
