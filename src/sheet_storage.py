"""Google Sheets保存用の軽量アダプター。

``gspread`` を必須依存にはせず、Worksheet互換オブジェクトを受け取る設計にして
いる。Colabでもローカルでも認証処理だけ利用者が用意すれば使える。
"""

from __future__ import annotations

from typing import Any, Dict

from .story_generator import Story


def append_row_by_columns(sheet: Any, data: Dict[str, Any]) -> None:
    """1行目のヘッダー名に対応する列へ辞書の値を追加する。"""
    headers = sheet.row_values(1)
    new_row = [""] * len(headers)

    for column_name, value in data.items():
        if column_name not in headers:
            continue
        new_row[headers.index(column_name)] = value

    sheet.append_row(new_row)


def story_to_row(story: Story, output_path: str = "") -> Dict[str, Any]:
    """StoryをSheets保存向けの代表的な列へ変換する。"""
    if not isinstance(story, Story):
        raise ValueError("story must be Story instance")

    return {
        "title": story.title,
        "plot_type": story.plot_type,
        "protagonist": story.characters.protagonist.name,
        "messenger": story.characters.messenger.name,
        "supporter": story.characters.supporter.name,
        "adversary": story.characters.adversary.name,
        "chapter_count": len(story.chapters),
        "word_count": sum(len(chapter) for chapter in story.chapters),
        "output_path": output_path,
    }


def append_story(sheet: Any, story: Story, output_path: str = "") -> None:
    """StoryをWorksheetの末尾へ追加する。"""
    append_row_by_columns(sheet, story_to_row(story, output_path=output_path))
