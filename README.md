# Global Habitat Suitability Prediction for the Fall Armyworm Based on Deep Learning Models

This repository accompanies a study that uses a U-Net to map monthly habitat suitability for the fall armyworm (*Spodoptera frugiperda*, FAW) across global croplands. The model combines FAW occurrence records with 17 environmental predictors on a common 0.5° × 0.5° grid (360 latitude cells × 720 longitude cells).

## Study workflow

1. Clean and harmonize FAW occurrence records from the contributing monitoring sources.
2. Align the environmental predictors to the common grid and stack them as 17-channel monthly inputs.
3. Apply the cropland mask. For each year and month with occurrence records, sample an equal number of pseudo-absences in cropland outside the 1° buffers around recorded occurrences. Match both sets of samples to the environmental rasters for the same year and month.
4. Train the U-Net in PyTorch and use the validation set for model selection. Evaluate the selected model on the independent test set.

| Split | Years | Role |
| --- | --- | --- |
| Training | 1981–2018 | Fit the model |
| Validation | 2019–2020 | Monitor performance and select the model |
| Independent test | 2021–2025 | Evaluate the selected model |

## Data availability

The full datasets are omitted from this repository because of their size. The links below point to upstream data products and monitoring platforms. The monthly 17-channel rasters, cropland mask, and presence and pseudo-absence sample sets are prepared for this study from the source inputs.

| Input | Source or product | Access |
| --- | --- | --- |
| FAW occurrence records | Global Biodiversity Information Facility (GBIF) | [GBIF](https://www.gbif.org/) |
| FAW field monitoring records | FAO Fall Armyworm Monitoring and Early Warning System (FAMEWS) | [FAO FAMEWS overview](https://www.fao.org/e-agriculture/news/focus-fall-armyworm-monitoring-and-early-warning-system-famews) |
| Atmospheric and land-surface meteorological variables | ECMWF ERA5-Land reanalysis, Copernicus Climate Data Store | [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) |
| Normalized Difference Vegetation Index (NDVI) | MODIS/Terra monthly vegetation indices, 1 km, Version 6.1 (MOD13A3) | [MOD13A3 V061](https://doi.org/10.5067/MODIS/MOD13A3.061) |
| Land use and land cover | MODIS/Terra+Aqua annual land cover, 500 m, Version 6.1 (MCD12Q1) | [MCD12Q1 V061](https://doi.org/10.5067/MODIS/MCD12Q1.061) |
| Elevation and slope | CGIAR-CSI SRTM 90 m Digital Elevation Database v4.1. Slope was derived from the elevation data. | [CGIAR SRTM data page](https://bigdata.cgiar.org/srtm-90m-digital-elevation-database/) |
| Sand, silt, and clay contents | ISRIC SoilGrids250m | [SoilGrids](https://soilgrids.org/) |

The manuscript also identifies the National FAW Monitoring and Management Information Platform as a source of occurrence records in China. Access details for the exact extract used in the study are pending. The 2026 projection additionally uses CMIP5-derived climate inputs. The specific product and download details for those inputs are pending.

## Citation

If you use this code, please cite the accompanying manuscript, *Global Habitat Suitability Prediction for the Fall Armyworm Based on Deep Learning Models*, and acknowledge the original data providers.
