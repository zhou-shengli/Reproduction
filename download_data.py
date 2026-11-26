"""下载并校验固定版本的 Petrobras 合成试井数据。"""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import ProxyHandler, build_opener

from src.source import DATA_PATH, URL, verify_csv


def download(output: Path, proxy: str | None = None) -> None:
    if output.exists():
        verify_csv(output.read_bytes())
        print(f"Verified existing dataset: {output}")
        return

    opener = build_opener(ProxyHandler({"https": proxy})) if proxy else build_opener()
    with opener.open(URL, timeout=120) as response:
        raw = response.read()
    verify_csv(raw)

    output.parent.mkdir(parents=True, exist_ok=True)
    # 同目录临时文件支持原子替换，写入中断时训练程序不会读到半份 CSV。
    with NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_bytes(raw)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified {len(raw):,} bytes: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", help="Optional HTTP(S) proxy URL")
    parser.add_argument("--output", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    download(args.output, args.proxy)


if __name__ == "__main__":
    main()
