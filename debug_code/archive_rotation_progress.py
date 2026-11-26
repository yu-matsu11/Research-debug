"""Create an independent dated TeX snapshot; compile its PDF with LuaLaTeX."""
import argparse
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--date', required=True, help='Progress date, YYYY-MM-DD')
    args = parser.parse_args()
    from datetime import date
    day = date.fromisoformat(args.date)
    source = args.source.resolve()
    text = source.read_text(encoding='utf-8')
    root = Path(__file__).resolve().parent.parent
    folder = root / 'seminar/research_progress' / day.isoformat()
    folder.mkdir(parents=True, exist_ok=False)
    text = re.sub(r'\\date\{[^\n]*\}',
                  lambda _: f'\\date{{{day.year}年{day.month}月{day.day}日}}', text)
    for relative in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text):
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.parent / relative, target)
    stem = f'rotation_report_{day:%Y%m%d}'
    (folder / f'{stem}.tex').write_text(text, encoding='utf-8')
    csv_source = source.parent / 'psnr_ssim_results.csv'
    if csv_source.exists():
        shutil.copy2(csv_source, folder / csv_source.name)
    (folder / 'BUILD.txt').write_text(
        f'研究進捗日: {day.isoformat()}\n'
        'このフォルダ内で次のコマンドを実行してください。\n'
        f'lualatex -interaction=nonstopmode -halt-on-error {stem}.tex\n'
        f'lualatex -interaction=nonstopmode -halt-on-error {stem}.tex\n'
        '図はfiguresフォルダに同梱しています。日付ごとの版を上書きせず保存します。\n',
        encoding='utf-8')
    print(folder)


if __name__ == '__main__':
    main()
