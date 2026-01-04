# this script reads the raw parquet files downloaded from overtunemaps
# filter to banks (financial districts), tourism districts (museums, tourist_info, etc)
# then create h3 index at resolution 7,8,9 and classify the h3 indexes into landuse types (cbd (mixed uses), toursim, others)

import h3
import os
import pandas as pd
import geopandas as gpd
import duckdb as dk

CBD_KEYWORDS = ["bank", "mall", "finance", "business", "office", "commercial"]
TOURISM_KEYWORDS = [
    "museum",
    "tour",
    "tourist_info",
    "attraction",
    "gallery",
    "amusement_park",
    "theme_park",
    "souvenir",
    "cultural",
    "historical",
    "temple",
    "overlook",
    "viewpoint",
]
RESIDENTIAL_KEYWORDS = [
    "school",
    "library",
    "kindergarten",
    "college",
    "hospital",
    "daycare",
    "clinic",
]
INDUSTRIAL_KEYWORDS = ["industrial", "factory", "warehouse"]


def classify_poi(places_df: pd.DataFrame) -> pd.DataFrame:
    places_df["top_category"] = places_df["categories"].apply(
        lambda x: x["primary"] if isinstance(x, dict) and "primary" in x else "unknown"
    )

    def classify_row(cat: str) -> str:
        cat_lower = cat.lower()
        if any(keyword in cat_lower for keyword in CBD_KEYWORDS):
            return "cbd"
        elif any(keyword in cat_lower for keyword in TOURISM_KEYWORDS):
            return "tourism"
        elif any(keyword in cat_lower for keyword in RESIDENTIAL_KEYWORDS):
            return "residential"
        elif any(keyword in cat_lower for keyword in INDUSTRIAL_KEYWORDS):
            return "industrial"
        else:
            return "other"

    places_df["landuse_type"] = places_df["top_category"].apply(classify_row)
    return places_df


def classify_h3_index(places_df: pd.DataFrame) -> pd.DataFrame:
    # create h3 index at resolution 7,8,9
    for res in [7, 8, 9]:
        places_df[f"h3_res{res}"] = places_df.apply(
            lambda row: h3.latlng_to_cell(row["geometry"].y, row["geometry"].x, res),
            axis=1,
        )

    # explode the dataframe to have one row per h3 index
    records = []
    for _, row in places_df.iterrows():
        for res in [7, 8, 9]:
            records.append(
                {
                    "h3_index": row[f"h3_res{res}"],
                    "landuse_type": row["landuse_type"],
                    "resolution": res,
                }
            )
    h3_df = pd.DataFrame(records)

    # aggregate to get the dominant landuse type per h3 index
    # for each h3_index, get its rank based on count of each type of landuse_type
    # for example, if an h3_index ranked 1 for 'cbd', 2 for 'tourism', then we classify it as 'cbd'
    resultdf = dk.query(
        """
            WITH aggregated AS (
        SELECT 
            h3_index, 
            landuse_type, 
            resolution,
            COUNT(*) as count
        FROM h3_df
        WHERE landuse_type <> 'other'
        GROUP BY resolution, h3_index, landuse_type
        ),
        ranked_by_type AS (
        SELECT 
            resolution, 
            h3_index, 
            RANK() OVER (PARTITION BY resolution ORDER BY count DESC) as type_rank,
            'cbd' as landuse_type
        FROM aggregated
        WHERE landuse_type = 'cbd'
        UNION ALL
        SELECT 
            resolution, 
            h3_index, 
            RANK() OVER (PARTITION BY resolution ORDER BY count DESC) as type_rank,
            'tourism' as landuse_type
        FROM aggregated
        WHERE landuse_type = 'tourism'
        UNION ALL
        SELECT 
            resolution, 
            h3_index, 
            RANK() OVER (PARTITION BY resolution ORDER BY count DESC) as type_rank,
            'residential' as landuse_type
        FROM aggregated
        WHERE landuse_type = 'residential'
        UNION ALL
        SELECT 
            resolution, 
            h3_index, 
            RANK() OVER (PARTITION BY resolution ORDER BY count DESC) as type_rank,
            'industrial' as landuse_type
        FROM aggregated
        WHERE landuse_type = 'industrial'
        ),
        -- select the landuse_type with the highest rank (lowest type_rank) for each h3_index
        final_classification AS (
        SELECT 
            resolution, 
            h3_index, 
            MIN(type_rank) as rank
        FROM ranked_by_type
        GROUP BY resolution, h3_index
        ),
        selected AS (
        SELECT 
            f.resolution,
            f.h3_index,
            r.landuse_type
        FROM final_classification f
        JOIN ranked_by_type r
            ON f.h3_index = r.h3_index 
            AND f.resolution = r.resolution
            AND f.rank = r.type_rank
            )
        SELECT
            s.resolution,
            s.h3_index,
            CASE 
                WHEN COUNT(*) > 1 THEN array_to_string(list_sort(ARRAY_AGG(s.landuse_type)), ', ')
                ELSE MAX(s.landuse_type)
            END as landuse_type
        FROM selected s
        GROUP BY s.resolution, s.h3_index
        """
    ).to_df()
    print(resultdf["landuse_type"].value_counts())
    return resultdf


def main():
    input_folder = (
        "/Users/yuan/Dropbox (Personal)/Personal Work/_commondata/POI/overtunemaps/raw"
    )
    output_folder = "/Users/yuan/Dropbox (Personal)/Personal Work/_Projects2025/urban-sim-flow/_data/_transformed/landuse"

    cityls = os.listdir(input_folder)
    valid_cityls = [x for x in cityls if x.endswith(".parquet")]
    cityls = [x.replace(".parquet", "") for x in valid_cityls]

    for city_lower in cityls:
        print("=" * 50)
        print(f"Processing city: {city_lower}")
        print("=" * 50)
        # check finished or not

        file_path = os.path.join(input_folder, f"{city_lower}.parquet")
        output_path = os.path.join(output_folder, f"{city_lower}_h3_landuse.parquet")
        if os.path.exists(output_path):
            print(f"{output_path} already exists. Skipping...")
            continue
        places_df = gpd.read_parquet(file_path)

        classified_places_df = classify_poi(places_df)
        h3_classified_df = classify_h3_index(classified_places_df)

        h3_classified_df["resolution"] = h3_classified_df["resolution"].astype(str)
        h3_classified_df.to_parquet(output_path, index=False)
        print(f"Saved classified h3 landuse to {output_path}")


if __name__ == "__main__":
    main()
