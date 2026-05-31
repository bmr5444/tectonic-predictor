import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
from pathlib import Path
import pygplates

BASE = Path("data/Young_etal_2018_GeosciFrontiers")
COASTLINES = BASE / "Coastlines/Global_coastlines_Young_et_al_low_res.shp"
CONT_POLY = BASE / "ContinentalPolygons/PresentDay_ContPolygons_Young_etal.shp"
PALEO_PLATE_POLYS = BASE / "Topologies/Global_Paleozoic_plate_bounds_Young_etal.gpml"
MESO_PLATE_POLYS = BASE / "Topologies/Global_Mesozoic-Cenozoic_plate_bounds_Young_etal.gpml"

coastlines = gpd.read_file(COASTLINES)
cont_polys = gpd.read_file(CONT_POLY)
paleo_features = pygplates.FeatureCollection(str(PALEO_PLATE_POLYS))
meso_features = pygplates.FeatureCollection(str(MESO_PLATE_POLYS))
rotation_model = pygplates.RotationModel([
        "data/Young_etal_2018_GeosciFrontiers/Rotations/Global_250-0Ma_Young_et_al.rot",
        "data/Young_etal_2018_GeosciFrontiers/Rotations/Global_410-250Ma_Young_et_al.rot"
    ])

resolved_sections = []
pygplates.resolve_topologies(
    [paleo_features, meso_features],
    rotation_model,
    [],
    0,
    resolved_sections
)

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()}, figsize=(15,8))
ax.set_global()
ax.set_facecolor("#0a1628")
fig.patch.set_facecolor("white")

for section in resolved_sections:
    feature_type = section.get_feature().get_feature_type()
    if "MidOceanRidge" in str(feature_type):
        color = "red"
    elif "SubductionZone" in str(feature_type):
        color = "blue"
    elif "Transform" in str(feature_type):
        color = "yellow"
    else:
        color = "white"

    geometry = section.get_topological_section_geometry()
    lats, lons = geometry.to_lat_lon_array().T
    ax.plot(
        lons, lats, 
        color=color, 
        transform=ccrs.PlateCarree(), 
        linewidth=0.85
    )

cont_polys.plot(
    ax=ax, 
    transform=ccrs.PlateCarree(), 
    edgecolor="white",
    linewidth=0.85,
)
coastlines.plot(
    ax=ax, 
    transform=ccrs.PlateCarree(), 
    facecolor="green", 
    edgecolor="black",
    linewidth=0.7,
)

plt.show()