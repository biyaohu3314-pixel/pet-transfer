import os, csv
# TODO: ROOT + os.chdir，和前面一样（注意 dirname 要两层！）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

rows = []
for name in sorted(os.listdir("outputs")):          # 列出 outputs 下所有文件和文件夹的名字
    path = os.path.join("outputs", name, "result.txt")
    if not os.path.exists(path):                     # _smoke、features、.gitkeep 都没有 result.txt，跳过
        continue
    row = {}
    with open(path, encoding="utf-8") as f:
        for line in f:                               # 每行形如 "test_acc=0.8932\n"
            key, value = line.strip().split("=")                         # TODO: 怎么把一行拆成等号两边？提示：strip()、split("=")
            row[key] = value
    rows.append(row)

# TODO: 用 csv.DictWriter 写到 outputs/results.csv
#       fieldnames 可以用 rows[0].keys()
out_path = os.path.join("outputs","results.csv")
with open(out_path,"w",newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"汇总了 {len(rows)} 组")                       # 预测：应该是几？