# Dans Jupyter, exécutez ceci :
import shutil
import os

# Créer un dossier de partage
share_folder = "C:/ML/FX_Dashboard_Package"
os.makedirs(share_folder, exist_ok=True)

# Copier les fichiers nécessaires
shutil.copy("app.py", share_folder)
shutil.copytree("exchange_rate_results", 
                os.path.join(share_folder, "exchange_rate_results"))

# Créer un fichier requirements.txt
with open(os.path.join(share_folder, "requirements.txt"), "w") as f:
    f.write("""streamlit==1.28.0
pandas==2.1.0
numpy==1.24.3
plotly==5.17.0
""")

# Créer un README
with open(os.path.join(share_folder, "README.txt"), "w") as f:
    f.write("""FX VOLATILITY DASHBOARD
=======================

INSTALLATION:
1. Install Python 3.8+
2. Open terminal/command prompt
3. Run: pip install -r requirements.txt

USAGE:
1. Open terminal in this folder
2. Run: streamlit run app.py
3. Open browser at: http://localhost:8501

REQUIREMENTS:
- Python 3.8 or higher
- Internet connection (first run only)
""")

print(f"✅ Package créé dans : {share_folder}")