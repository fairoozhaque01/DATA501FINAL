import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon
from shapely.wkt import loads  


df = pd.read_csv("calgary_community_boundaries.tsv", sep="\t")

print(df.columns)


df["geometry"] = df["MULTIPOLYGON"].apply(lambda x: loads(x) if pd.notna(x) else None)

gdf = gpd.GeoDataFrame(df, geometry="geometry")

gdf.to_file("calgary_community_boundaries.geojson", driver="GeoJSON")

print("GeoJSON file saved as calgary_community_boundaries.geojson")


calgary_map = gpd.read_file("calgary_community_boundaries.geojson")  # Replace with actual file path

crime_df = pd.read_csv("calgary_crime_cleaned.csv")

crime_by_community = crime_df.groupby("Community")["Crime Count"].sum().reset_index()


calgary_map["NAME"] = calgary_map["NAME"].str.lower().str.strip()
crime_by_community["Community"] = crime_by_community["Community"].str.lower().str.strip()

crime_map_data = calgary_map.merge(crime_by_community, left_on="NAME", right_on="Community", how="left")

crime_map_data["Crime Count"] = crime_map_data["Crime Count"].fillna(0)

fig, ax = plt.subplots(figsize=(12, 8))

crime_map_data.plot(column="Crime Count", 
                    cmap="Reds",  # Color scale: Light red (low crime) → Dark red (high crime)
                    linewidth=0.8, 
                    edgecolor="black", 
                    legend=True, 
                    legend_kwds={"label": "Crime Count by Community", "orientation": "vertical"},
                    ax=ax)

plt.title("Crime Density Across Calgary's Communities (2021-2025)", fontsize=14)

ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

plt.show()
