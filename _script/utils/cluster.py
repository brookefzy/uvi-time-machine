class ClusterAttr:
    variables_sel_top1_order = [
    'skyscraper',
 'light',
 "road",
  'sidewalk', 
    'traffic light',
    'window',
     "building",
    'signage',
    'pole',
      'trashcan',
      'installation', 
      "railing",
    'shrub', 
    'grass',
    'tree',
    'lake+waterboday',
     'sky',
     'sportsfield',
     'mountain+hill',
]
    variables_sel_top1_order_all = [
        
        'skyscraper',   
    "car", 
    "van", 
    'light',"road",
    'sidewalk', 
        'traffic light',
        "bus",
            'window',
            "building",
                'signage',
        'pole',
        'truck',
            'bike',
        'person',
        'trashcan',
        'installation', 
        "railing",
        'shrub', 
        'grass',
        'tree',
        'lake+waterboday',
        'sky',
        'sportsfield',
        'mountain+hill',
    ]
    len(variables_sel_top1_order)
    variables_sel_order = {
    "_built_environment": variables_sel_top1_order,
    "": variables_sel_top1_order_all
    }