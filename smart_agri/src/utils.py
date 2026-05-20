"""
utils.py — Shared constants, feature metadata, and helper functions.
"""

FEATURE_RANGES = {
    'N':           {'min': 0,   'max': 200, 'default': 50,  'unit': 'mg/kg', 'label': 'Nitrogen (N)'},
    'P':           {'min': 0,   'max': 200, 'default': 50,  'unit': 'mg/kg', 'label': 'Phosphorous (P)'},
    'K':           {'min': 0,   'max': 250, 'default': 40,  'unit': 'mg/kg', 'label': 'Potassium (K)'},
    'temperature': {'min': 0,   'max': 50,  'default': 25,  'unit': '°C',    'label': 'Temperature'},
    'humidity':    {'min': 1,   'max': 100, 'default': 65,  'unit': '%',     'label': 'Humidity'},
    'ph':          {'min': 0,   'max': 14,  'default': 6.5, 'unit': '',      'label': 'Soil pH'},
    'rainfall':    {'min': 0,   'max': 400, 'default': 100, 'unit': 'mm',    'label': 'Rainfall'},
}

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Cluster agronomic profiles — dynamically assigned after training via cluster center analysis
CLUSTER_PROFILES = {
    0: {
        'name': 'High Fertility Zone',
        'description': 'Nutrient-rich, balanced NPK. Ideal for high-demand crops.',
        'recommendations': 'Rice, Maize, Banana, Sugarcane',
        'action': 'Maintain current nutrient levels; monitor micronutrients regularly.',
    },
    1: {
        'name': 'Low Nutrient Zone',
        'description': 'Depleted soil requiring fertilization intervention.',
        'recommendations': 'Legumes (nitrogen-fixing), Cover crops',
        'action': 'Apply balanced NPK fertilizer; prioritize nitrogen-fixing rotation.',
    },
    2: {
        'name': 'Moderate Soil Zone',
        'description': 'Average conditions suitable for a broad range of crops.',
        'recommendations': 'Wheat, Cotton, Chickpea, Lentil',
        'action': 'Optimize pH to 6.0–7.0; supplement phosphorus for better yields.',
    },
    3: {
        'name': 'High Potassium Zone',
        'description': 'Potassium-rich with moderate nitrogen content.',
        'recommendations': 'Grapes, Apple, Root vegetables, Fruits',
        'action': 'Focus on nitrogen supplementation; avoid excess potassium application.',
    },
}


def get_cluster_info(cluster_id):
    return CLUSTER_PROFILES.get(int(cluster_id), {
        'name': f'Zone {cluster_id}',
        'description': 'Soil zone identified by KMeans nutrient profiling.',
        'recommendations': 'General crops',
        'action': 'Conduct detailed soil testing for site-specific recommendations.',
    })


def compute_yield_bounds(y_pred, margin_pct=0.15):
    """Return (lower, mid, upper) yield confidence bounds."""
    return max(0.0, y_pred * (1 - margin_pct)), float(y_pred), y_pred * (1 + margin_pct)


def get_data_dictionary():
    return {
        'N':           'Nitrogen content in soil (mg/kg)',
        'P':           'Phosphorous content in soil (mg/kg)',
        'K':           'Potassium content in soil (mg/kg)',
        'temperature': 'Average ambient temperature (°C)',
        'humidity':    'Relative humidity (%)',
        'ph':          'Soil pH level (0–14)',
        'rainfall':    'Annual rainfall (mm)',
        'label':       'Crop type — classification target (22 classes)',
        'yield':       'Crop yield in kg/hectare — regression target',
    }
