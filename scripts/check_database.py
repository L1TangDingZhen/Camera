#!/usr/bin/env python3
"""检查database.db中的数据"""

import sqlite3
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

db_path = 'data/database.db'

if not os.path.exists(db_path):
    print(f"❌ 数据库文件不存在: {db_path}")
    sys.exit(1)

print(f"✓ 找到数据库: {db_path}")
print(f"  文件大小: {os.path.getsize(db_path)} bytes\n")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查看所有表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

if not tables:
    print("❌ 数据库中没有表")
    conn.close()
    sys.exit(1)

print("="*60)
print("数据库表结构")
print("="*60)

for table in tables:
    table_name = table[0]
    print(f"\n📊 表名: {table_name}")

    # 表结构
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f"   列: {', '.join([col[1] for col in columns])}")

    # 记录数
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"   记录数: {count}")

    if count > 0:
        # 示例数据
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
        rows = cursor.fetchall()
        print(f"   示例数据 (前3条):")
        for i, row in enumerate(rows, 1):
            print(f"     {i}. {row}")

print("\n" + "="*60)
print("总结")
print("="*60)

total_records = 0
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
    count = cursor.fetchone()[0]
    total_records += count
    print(f"  {table[0]}: {count} 条记录")

print(f"\n总记录数: {total_records}")

conn.close()
