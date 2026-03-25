# Disease class mapping (index to disease name)
DISEASE_MAPPING = {
    0: 'Pepper__bell___Bacterial_spot',
    1: 'Pepper__bell___healthy',
    2: 'Potato___Early_blight',
    3: 'Potato___Late_blight',
    4: 'Potato___healthy',
    5: 'Tomato_Bacterial_spot',
    6: 'Tomato_Early_blight',
    7: 'Tomato_Late_blight',
    8: 'Tomato_Leaf_Mold',
    9: 'Tomato_Septoria_leaf_spot',
    10: 'Tomato_Spider_mites_Two_spotted_spider_mite',
    11: 'Tomato__Target_Spot',
    12: 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    13: 'Tomato__Tomato_mosaic_virus',
    14: 'Tomato_healthy'
}

# Reverse mapping (disease name to index)
REVERSE_MAPPING = {v: k for k, v in DISEASE_MAPPING.items()}

def get_disease_name(class_index):
    """Convert class index to disease name"""
    return DISEASE_MAPPING.get(class_index, f'Unknown Class {class_index}')

def get_class_names():
    """Get all disease names in order"""
    return [DISEASE_MAPPING[i] for i in range(len(DISEASE_MAPPING))]
