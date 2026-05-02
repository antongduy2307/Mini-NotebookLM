from __future__ import annotations

from mini_notebooklm_rag.storage.sqlite import connect, initialize_database


def test_initialize_database_is_idempotent(tmp_path) -> None:
    db_path = tmp_path / "app.db"

    initialize_database(db_path)
    initialize_database(db_path)

    with connect(db_path) as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]

    assert {"workspaces", "documents", "chunks", "chat_sessions", "chat_messages"}.issubset(tables)
    assert foreign_keys == 1
