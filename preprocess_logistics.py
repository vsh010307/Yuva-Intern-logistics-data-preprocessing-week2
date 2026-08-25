import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_logistics_data(filepath):
    df = pd.read_csv(filepath)

    # Missing values
    df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())
    df["carrier"] = df["carrier"].fillna(df["carrier"].mode()[0])
    df = df.dropna(subset=["delivery_date"])

    # Standardize formats
    df["pickup_date"] = pd.to_datetime(df["pickup_date"], format="%d-%m-%Y")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], format="%d-%m-%Y")
    df["carrier"] = df["carrier"].str.title()
    df["status"] = df["status"].replace({"Pendingg": "Pending"})

    # Invalid values and duplicates
    df = df[(df["distance_km"] >= 0) & (df["weight_kg"] >= 0)]
    df = df.drop_duplicates(subset=["shipment_id"], keep="first")

    # Feature engineering
    df["transit_hours"] = (df["delivery_date"] - df["pickup_date"]).dt.total_seconds() / 3600
    df["delay_hours"] = df["transit_hours"] - df["expected_transit_hours"]
    df["delivery_performance"] = df["delay_hours"].apply(
        lambda x: "Early" if x <= 0 else ("On-time" if x <= 2 else "Delayed")
    )
    df["pickup_day_of_week"] = df["pickup_date"].dt.day_name()
    df["transit_efficiency"] = df["distance_km"] / df["transit_hours"]

    # IQR outlier treatment
    q1, q3 = df["transit_hours"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["transit_outlier"] = (df["transit_hours"] < lower) | (df["transit_hours"] > upper)
    df = df[~df["transit_outlier"]].copy()

    # Encoding and scaling
    df = pd.get_dummies(df, columns=["carrier"], prefix="carrier")
    scaler = MinMaxScaler()
    df["distance_scaled"] = scaler.fit_transform(df[["distance_km"]])
    return df

if __name__ == "__main__":
    clean_df = preprocess_logistics_data("shipment_data.csv")
    clean_df.to_csv("cleaned_shipment_data.csv", index=False)
    print(f"Preprocessing completed. Cleaned records: {len(clean_df)}")
