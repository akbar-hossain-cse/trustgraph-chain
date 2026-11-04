from src.features.load_data import load_raw

df = load_raw()
atk = df[df.Label == 1]
print("Attack SOURCE IPs:\n", atk.srcip.value_counts().head(15))
print("\nAttack DEST IPs:\n", atk.dstip.value_counts().head(15))
print("\nNormal SOURCE IPs:\n", df[df.Label == 0].srcip.value_counts().head(15))
print("\nSrc IPs that appear in BOTH normal and attack:",
      len(set(atk.srcip) & set(df[df.Label == 0].srcip)))