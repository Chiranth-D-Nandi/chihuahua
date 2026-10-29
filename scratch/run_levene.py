import subprocess

res = subprocess.run(["python", "scripts/levene_test.py"], capture_output=True, text=True)
print(res.stdout)

with open("reports/levene_test_output.txt", "w", encoding="utf-8") as f:
    f.write(res.stdout)

print("Saved reports/levene_test_output.txt cleanly in UTF-8.")
