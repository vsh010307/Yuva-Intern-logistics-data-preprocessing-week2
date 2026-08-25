# Logistics Data Collection, Cleaning and Preprocessing

Week 2 – Logistics Data Analyst Internship.

## Repository contents
- `shipment_data.csv` – synthetic raw logistics dataset with intentional data-quality issues.
- `preprocess_logistics.py` – preprocessing pipeline.
- `cleaned_shipment_data.csv` – cleaned output.
- `requirements.txt` – dependencies.

## Steps implemented
Missing-value handling, date/category standardization, invalid-value removal, duplicate removal, IQR outlier detection, feature engineering, one-hot encoding, and Min-Max scaling.

## Run
```bash
pip install -r requirements.txt
python preprocess_logistics.py
```

Dataset is synthetic and for educational use.
