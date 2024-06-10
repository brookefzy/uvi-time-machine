# Time-machine
## Summarize the urban changes in each city
* This folder summarizes the changes happened in cities from 2015 to today, focusing on parameters of walking environment.
* The elements (SVF) to extract and compare are:
1. existence of sidewalk
2. existence of greenery
3. people
4. street furnitures
5. buildings
6. car
7. bike

* The changes will be aggregated to different h3 index level, and city level: 6, 9, 12.
* parameters/features to calculate includes:
1. percentage of h3 grids with significant changes of SVF_i above from 2015 to 2022
2. road accident for each city from 2015 to 2022
3. obesity rate from 2015 to 2022 for each city (adjusted from country level to city level)
4. type II diabetes from 2015 to 2022 for each city (adjusted from country level to city leve)
5. Depression (or Stress)
6. vehicle miles traveled from 2015 to 2022 for each city
7. CO2 emission from 2015 to 2022
8. Child/Infant motality rate


* Control variables to collect:
1. population at 2015 and 2022
2. economic status 2015 and 2022 (GDP) or 
3. smart-mobile phone penetration rate

## Code Structure:
1. run oneformer to extract pixel level and object level detection results per images
2. run `04_pano_post.py` and `04_seg_post-full.py` for data summary
3. run `d-experiment/01_combine_seg_cat.py` to merge all variables.

## Data
### Source of Fatality
#### DALYs (Disability Adjusted Life Years)
```DALYs are a combination of the sum of the years of potential life lost due to premature mortality and years of productive life lost due to a disability per 100 000 population. (https://www.thelancet.com/journals/lanplh/article/PIIS2542-5196(19)30263-3/fulltext#supplementaryMaterial)
```

* https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates/global-health-estimates-leading-causes-of-dalys
#### Fatality
* US: https://www.cdc.gov/nchs/pressroom/sosmap/accident_mortality/accident.htm

### Source of Emissions (FFDAS)
`Z:\_world_data\01_carbon_emission_ffdas\ffdas_flux_2013_2015.nc.gz`
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009JD013439

### Source of Obesity or Type II Diabetes
* US: https://www.cdc.gov/places/measure-definitions/health-outcomes/index.html#obesity
* 
### Vehicle Miles Traveled
* 

## Reference
* https://www.thelancet.com/journals/lanplh/article/PIIS2542-5196(19)30263-3/fulltext#supplementaryMaterial
* https://www.thelancet.com/cms/10.1016/S2542-5196(19)30263-3/attachment/4ab590e0-e73f-406f-8432-3e01929010c2/mmc1.pdf
* https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(17)32130-X/fulltext#tbl3
* https://www.thelancet.com/cms/10.1016/S0140-6736(17)32130-X/attachment/a8bdc974-3ee6-4a6a-8980-ba0b755c24ec/mmc1.pdf
