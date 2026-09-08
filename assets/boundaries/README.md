# Cartographic boundary layers

Reference layers for cartography only, not inputs to indicator extraction or validation.

| File | Layers | Source and licence | Native CRS |
|---|---|---|---|
| `shenzhen_districts.gpkg` | 9 district-level polygons (Luohu, Futian, Nanshan, Baoan, Longgang, Yantian, Longhua, Pingshan, Guangming) | Alibaba Cloud DataV administrative-boundary service (GB/T 2260 adcode 4403xx), attributes only, free for cartographic reuse | WGS 84 (EPSG:4326) |

Prepared on 2026-09-08 from `shenzhen.json`. Coordinates were checked against the 465-location
Shenzhen manifest used by the restorative-quality and six-dimension tasks: 463/465 points (99.6%)
fall inside the district union, with the two remaining points within 180 m of a district edge, so
no coordinate-system correction (e.g. GCJ-02 offset) is applied. The previous OSM municipality
outline (`shenzhen_boundaries.gpkg`, relation 3464353) is no longer used because its coastline
generalisation excluded a visible fraction of the same manifest points.
