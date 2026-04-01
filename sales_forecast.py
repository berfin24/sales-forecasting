# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:35:10 2026

@author: berfi
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")


np.random.seed(42)  #veri dönüyor
dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq="W")  #veri oluşturduk
trend = np.linspace(1000, 3000, len(dates))  #zamanla artan satış 
seasonality = 500 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)  #mevsimsel dalgalanma
noise = np.random.normal(0, 150, len(dates))  #gerçek hayattaki rasgele gürültü
sales = trend + seasonality + noise

df = pd.DataFrame({"ds": dates, "y": sales})
df["y"] = df["y"].clip(lower=100)
df["x"] = np.arange(len(df))  # sayısal index

print("=" * 50)
print("VERİ ÖZETİ")
print("=" * 50)
print(df[["ds","y"]].describe())
print(f"\nToplam Hafta: {len(df)}")
print(f"Tarih Aralığı: {df['ds'].min().date()} → {df['ds'].max().date()}")



def set_date_ticks(ax, dates_series, n_ticks=7):
    """Sayısal x eksenine tarih etiketi yazar — dateutil kullanmaz"""
    idx = np.linspace(0, len(dates_series)-1, n_ticks, dtype=int)
    ax.set_xticks(idx)
    ax.set_xticklabels(
        [dates_series.iloc[i].strftime("%Y-%m") for i in idx],
        rotation=30, ha="right"
    )


fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["x"], df["y"], color="steelblue", linewidth=1.5, label="Gerçek Satışlar")
ax.set_title("Haftalık Satış Verisi (2021-2023)", fontsize=14, fontweight="bold")
ax.set_xlabel("Tarih")
ax.set_ylabel("Satış Miktarı")
set_date_ticks(ax, df["ds"])
ax.legend()
ax.grid(alpha=0.3)
plt.subplots_adjust(bottom=0.2)
plt.savefig("01_gercek_veri.png", dpi=150)
plt.show()
print("✅ Grafik 1 kaydedildi.")



split_date = "2023-07-01"
train = df[df["ds"] < split_date].copy()
test  = df[df["ds"] >= split_date].copy()

print(f"\nEğitim seti: {len(train)} hafta")
print(f"Test seti  : {len(test)} hafta")


lr = LinearRegression()
lr.fit(train[["x"]], train["y"])
test = test.copy()
test["lr_pred"] = lr.predict(test[["x"]])

lr_mae  = mean_absolute_error(test["y"], test["lr_pred"])
lr_rmse = np.sqrt(mean_squared_error(test["y"], test["lr_pred"]))

print("\n" + "=" * 50)
print("MODEL 1: LINEAR REGRESSION")
print("=" * 50)
print(f"MAE : {lr_mae:.2f}")
print(f"RMSE: {lr_rmse:.2f}")



model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode="additive"
)
model.fit(train[["ds", "y"]])

future   = model.make_future_dataframe(periods=len(test), freq="W")
forecast = model.predict(future)

prophet_test_pred = forecast.tail(len(test))["yhat"].values
prophet_mae  = mean_absolute_error(test["y"], prophet_test_pred)
prophet_rmse = np.sqrt(mean_squared_error(test["y"], prophet_test_pred))

print("\n" + "=" * 50)
print("MODEL 2: PROPHET")
print("=" * 50)
print(f"MAE : {prophet_mae:.2f}")
print(f"RMSE: {prophet_rmse:.2f}")



all_dates = df["ds"].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(train["x"], train["y"], label="Eğitim Verisi", color="steelblue")
ax.plot(test["x"],  test["y"],  label="Gerçek (Test)", color="green", linewidth=2)
ax.plot(test["x"],  test["lr_pred"],
        label=f"Linear Reg (RMSE={lr_rmse:.0f})", color="orange", linestyle="--", linewidth=2)
ax.plot(test["x"],  prophet_test_pred,
        label=f"Prophet (RMSE={prophet_rmse:.0f})", color="red", linestyle="--", linewidth=2)
ax.axvline(train["x"].iloc[-1], color="gray", linestyle=":", label="Train/Test Sınırı")
ax.set_title("Satış Tahmini Karşılaştırması", fontsize=14, fontweight="bold")
ax.set_xlabel("Tarih")
ax.set_ylabel("Satış")
set_date_ticks(ax, all_dates)
ax.legend()
ax.grid(alpha=0.3)
plt.subplots_adjust(bottom=0.2)
plt.savefig("02_tahmin_karsilastirma.png", dpi=150)
plt.show()
print("✅ Grafik 2 kaydedildi.")



future_12   = model.make_future_dataframe(periods=len(test)+12, freq="W")
forecast_12 = model.predict(future_12)
gelecek     = forecast_12.tail(12)[["ds","yhat","yhat_lower","yhat_upper"]].copy()
gelecek["x"] = np.arange(len(df), len(df)+12)

print("\n" + "=" * 50)
print("SONRAKİ 12 HAFTA TAHMİNİ")
print("=" * 50)
print(gelecek[["ds","yhat","yhat_lower","yhat_upper"]].to_string(index=False))

combined_dates = pd.concat([df["ds"], gelecek["ds"]], ignore_index=True)
combined_x     = np.arange(len(combined_dates))

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df["x"], df["y"], label="Geçmiş Satışlar", color="steelblue")
ax.plot(gelecek["x"], gelecek["yhat"],
        label="12 Haftalık Tahmin", color="red", linewidth=2.5)
ax.fill_between(gelecek["x"], gelecek["yhat_lower"], gelecek["yhat_upper"],
                alpha=0.2, color="red", label="Güven Aralığı")
ax.set_title("Gelecek 12 Hafta Satış Tahmini", fontsize=14, fontweight="bold")
ax.set_xlabel("Tarih")
ax.set_ylabel("Satış")
set_date_ticks(ax, combined_dates)
ax.legend()
ax.grid(alpha=0.3)
plt.subplots_adjust(bottom=0.2)
plt.savefig("03_gelecek_tahmin.png", dpi=150)
plt.show()
print("✅ Grafik 3 kaydedildi.")


with pd.ExcelWriter("sales_forecast_results.xlsx", engine="openpyxl") as writer:
    df[["ds","y"]].to_excel(writer, sheet_name="Ham_Veri", index=False)
    test[["ds","y","lr_pred"]].to_excel(writer, sheet_name="LR_Tahmin", index=False)
    gelecek[["ds","yhat","yhat_lower","yhat_upper"]].to_excel(
        writer, sheet_name="Prophet_Gelecek", index=False)

print("\n✅ Excel kaydedildi: sales_forecast_results.xlsx")
