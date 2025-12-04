import shutil
import os

# 1️⃣ Créer le dossier
share_folder = "ML_Dashboard"
os.makedirs(share_folder, exist_ok=True)

# 2️⃣ Créer requirements.txt
with open(os.path.join(share_folder, "requirements.txt"), "w") as f:
    f.write("""streamlit==1.28.0
pandas==2.1.0
numpy==1.24.3
plotly==5.17.0
scikit-learn
""")

# 3️⃣ Créer README.md
with open(os.path.join(share_folder, "README.md"), "w") as f:
    f.write("""# FX VOLATILITY DASHBOARD

## INSTALLATION
1. Install Python 3.8+
2. Open terminal/command prompt
3. Run: pip install -r requirements.txt

## USAGE
```bash
streamlit run app.py
