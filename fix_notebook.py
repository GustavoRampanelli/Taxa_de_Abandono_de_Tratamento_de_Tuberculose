import json

path = "tuberculosis-ltfu-prediction-main/feature_engineering.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)

OLD = "X_enc[col] = pd.Categorical(X_enc[col].astype(str)).codes.replace(-1, np.nan)\n"
NEW = ("codes = pd.Categorical(X_enc[col].astype(str)).codes.astype(float)\n"
       "    codes[codes == -1] = np.nan\n"
       "    X_enc[col] = codes\n")

fixed = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            if OLD in line:
                new_source.append(line.replace(OLD, NEW))
                fixed = True
            else:
                new_source.append(line)
        cell["source"] = new_source

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Fixed:", fixed)
