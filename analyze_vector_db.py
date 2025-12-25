import os
import sqlite3
import argparse
import json

def analyze_chroma_db(persist_dir, show_docs=False, limit=5):
    db_path = os.path.join(persist_dir, "chroma.sqlite3")
    if not os.path.exists(db_path):
        print(f"❌ 找不到数据库文件: {db_path}")
        return

    print(f"✅ 找到数据库文件: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # === 列出所有表 ===
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print("\n[DEBUG] 数据库中的表:", tables)

    # === 打印每个表的字段结构 ===
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = [col[1] for col in cursor.fetchall()]
        print(f"[DEBUG] {table} 表字段: {cols}")

    # === 检查 collections 表 ===
    if "collections" not in tables:
        print("❌ 没有找到 collections 表")
        return

    cursor.execute("SELECT id, name, dimension FROM collections")
    collections = cursor.fetchall()

    print("\n=== Collections 信息 ===\n")
    for col_id, col_name, col_dim in collections:
        print(f"📂 Collection: {col_name}")
        print(f"   ID: {col_id}")
        print(f"   维度: {col_dim if col_dim else '未知'}")

        # === 找 segments ===
        if "segments" in tables:
            cursor.execute("PRAGMA table_info(segments)")
            seg_cols = [c[1] for c in cursor.fetchall()]
            if "collection" in seg_cols:
                cursor.execute("SELECT id, type FROM segments WHERE collection=?", (col_id,))
                segments = cursor.fetchall()
                print(f"   Segments: {len(segments)} 个")
            else:
                print("❌ 无法识别 segments 表的关联字段")
        else:
            print("❌ 没有 segments 表")

        # === 尝试读取 embeddings ===
        embedding_table = None
        for candidate in ["embedding", "embeddings", "embedding_fulltext", "segment_items"]:
            if candidate in tables:
                embedding_table = candidate
                break

        if embedding_table:
            print(f"   使用数据表: {embedding_table}")
            cursor.execute(f"PRAGMA table_info({embedding_table})")
            embed_cols = [c[1] for c in cursor.fetchall()]
            print(f"   [DEBUG] {embedding_table} 表字段: {embed_cols}")

            # 尝试计数
            if "id" in embed_cols:
                cursor.execute(f"SELECT COUNT(*) FROM {embedding_table}")
                count = cursor.fetchone()[0]
                print(f"   共 {count} 条向量")

            # 如果 show_docs，尝试找文本字段
            if show_docs and limit > 0:
                text_field = None
                for candidate in ["document", "content", "text", "value"]:
                    if candidate in embed_cols:
                        text_field = candidate
                        break
                if text_field:
                    cursor.execute(f"SELECT {text_field} FROM {embedding_table} LIMIT ?", (limit,))
                    rows = cursor.fetchall()
                    print(f"\n   === 示例文档（前 {limit} 条）===")
                    for i, r in enumerate(rows, 1):
                        print(f"   {i}. {r[0][:80]}...")
                else:
                    print("   ❌ 没有找到文本字段")
        else:
            print("❌ 没有找到 embedding 表")

        print("-" * 60)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 Chroma SQLite 数据库结构")
    parser.add_argument("--dir", type=str, default="../vector_db", help="Chroma 持久化目录")
    parser.add_argument("--show-docs", action="store_true", help="是否显示示例文档内容")
    parser.add_argument("--limit", type=int, default=5, help="示例文档条数")
    args = parser.parse_args()

    persist_dir = os.path.abspath(args.dir)
    analyze_chroma_db(persist_dir, show_docs=args.show_docs, limit=args.limit)
