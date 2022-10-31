import csv
file1 = 'data_A.csv'
with open(file1, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    data = [line['y'].rstrip().encode("utf-8") for line in reader]
print(data)

res_strs = sorted([val.decode("utf-8") for val in data])
with open('data_A_result.csv', mode="w", encoding="utf-8") as f:
    for line in res_strs:
        f.write(line)
        f.write(",")
