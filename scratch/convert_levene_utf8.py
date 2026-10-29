with open("reports/levene_test_output.txt", "r", encoding="utf-16") as f:
    content = f.read()

with open("reports/levene_test_output.txt", "w", encoding="utf-8") as f:
    f.write(content)

print("Converted reports/levene_test_output.txt to UTF-8 successfully.")
print("Preview of first 300 chars:")
print(content[:300])
