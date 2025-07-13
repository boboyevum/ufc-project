import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Loading UFC Dataset...")
ufc = pd.read_csv('data/ufc-master.csv')
print(f"Dataset loaded: {ufc.shape[0]} fights, {ufc.shape[1]} features")

print("\n DATASET OVERVIEW:")
print("=" * 50)
print(f"Total Fights: {len(ufc):,}")
print(f"Date Range: {ufc['Date'].min()} to {ufc['Date'].max()}")
print(f"Unique Fighters: {len(set(ufc['RedFighter'].tolist() + ufc['BlueFighter'].tolist())):,}")

ufc['Date'] = pd.to_datetime(ufc['Date'])

print("\n FIGHT OUTCOMES ANALYSIS:")
print("=" * 50)
winner_counts = ufc['Winner'].value_counts()
print(f"Red Corner Wins: {winner_counts.get('Red', 0):,} ({winner_counts.get('Red', 0)/len(ufc)*100:.1f}%)")
print(f"Blue Corner Wins: {winner_counts.get('Blue', 0):,} ({winner_counts.get('Blue', 0)/len(ufc)*100:.1f}%)")

finish_counts = ufc['Finish'].value_counts()
print(f"\nMost Common Finish Types:")
for finish, count in finish_counts.head(5).items():
    print(f"  {finish}: {count:,} ({count/len(ufc)*100:.1f}%)")

print("\n WEIGHT CLASS ANALYSIS:")
print("=" * 50)
weight_class_counts = ufc['WeightClass'].value_counts()
print("Fights by Weight Class:")
for wc, count in weight_class_counts.head(10).items():
    print(f"  {wc}: {count:,} fights")

print("\n BETTING ODDS ANALYSIS:")
print("=" * 50)
odds_data = ufc[['RedOdds', 'BlueOdds', 'Winner']].dropna()
print(f"Fights with betting odds: {len(odds_data):,}")

odds_data['RedImpliedProb'] = 1 / (1 + odds_data['RedOdds']/100)
odds_data['BlueImpliedProb'] = 1 / (1 + odds_data['BlueOdds']/100)

odds_data['RedFavorite'] = odds_data['RedOdds'] < odds_data['BlueOdds']
odds_data['Upset'] = ((odds_data['RedFavorite'] & (odds_data['Winner'] == 'Blue')) | 
                      (~odds_data['RedFavorite'] & (odds_data['Winner'] == 'Red')))

upset_rate = odds_data['Upset'].mean()
print(f"Upset Rate: {upset_rate:.1%}")

print("\n FIGHTER STATISTICS:")
print("=" * 50)

# Most active fighters
red_fighters = ufc['RedFighter'].value_counts()
blue_fighters = ufc['BlueFighter'].value_counts()
all_fighters = pd.concat([red_fighters, blue_fighters]).groupby(level=0).sum().sort_values(ascending=False)

print("Most Active Fighters:")
for fighter, fights in all_fighters.head(10).items():
    print(f"  {fighter}: {fights} fights")

print("\n TIME TRENDS:")
print("=" * 50)
ufc['Year'] = ufc['Date'].dt.year
yearly_fights = ufc.groupby('Year').size()
print("Fights per Year:")
for year, count in yearly_fights.tail(5).items():
    print(f"  {year}: {count:,} fights")

print("\n LOCATION ANALYSIS:")
print("=" * 50)
location_counts = ufc['Location'].value_counts()
print("Top Fight Locations:")
for location, count in location_counts.head(5).items():
    print(f"  {location}: {count:,} fights")

print("\n TITLE FIGHTS:")
print("=" * 50)
title_fights = ufc[ufc['TitleBout'] == True]
print(f"Total Title Fights: {len(title_fights):,}")
print(f"Title Fight Rate: {len(title_fights)/len(ufc)*100:.1f}%")

print("\n FIGHT DURATION:")
print("=" * 50)
duration_data = ufc[ufc['TotalFightTimeSecs'].notna()]
print(f"Average Fight Duration: {duration_data['TotalFightTimeSecs'].mean()/60:.1f} minutes")
print(f"Shortest Fight: {duration_data['TotalFightTimeSecs'].min()/60:.1f} minutes")
print(f"Longest Fight: {duration_data['TotalFightTimeSecs'].max()/60:.1f} minutes")

print("\n PHYSICAL ATTRIBUTES:")
print("=" * 50)
height_data = ufc[['RedHeightCms', 'BlueHeightCms']].dropna()
reach_data = ufc[['RedReachCms', 'BlueReachCms']].dropna()
weight_data = ufc[['RedWeightLbs', 'BlueWeightLbs']].dropna()

print(f"Average Height: {height_data.values.flatten().mean():.1f} cm")
print(f"Average Reach: {reach_data.values.flatten().mean():.1f} cm")
print(f"Average Weight: {weight_data.values.flatten().mean():.1f} lbs")

print("\n PERFORMANCE METRICS:")
print("=" * 50)
sig_str_cols = [col for col in ufc.columns if 'AvgSigStr' in col]
td_cols = [col for col in ufc.columns if 'AvgTD' in col]

sig_str_data = ufc[sig_str_cols].dropna()
td_data = ufc[td_cols].dropna()

print(f"Average Significant Strikes Landed: {sig_str_data.values.flatten().mean():.2f}")
print(f"Average Takedowns Landed: {td_data.values.flatten().mean():.2f}")

print("\n Data Analysis Complete!") 