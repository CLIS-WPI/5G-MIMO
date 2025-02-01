# config.py
import tensorflow as tf

# General Configuration
# config.py
CONFIG = {
    "num_samples": 50000,         # Increased total samples
    "batch_size": 32,           
    "random_seed": 42,            
    "splits": {
        "train": 0.7,            # 70% for training
        "validation": 0.15,      # 15% for validation
        "test": 0.15            # 15% for test
    },
    "output_files": {
        "train": "./data/training/train_channel_dataset.h5",
        "validation": "./data/validation/val_channel_dataset.h5",
        "test": "./data/test/test_channel_dataset.h5"
    }
}

# MIMO and Antenna Configuration
ANTENNA_CONFIG = {
    "num_rows": 4,                    # Number of rows of elements
    "num_cols": 4,                    # Number of columns of elements
    "polarization": "dual",           # Must be either "single" or "dual"
    "polarization_type": "VH",        # For dual polarization, must be "VH" or "cross"
    "pattern": "38.901",             # Element radiation pattern, either "omni" or "38.901"
    "carrier_frequency": 3.5e9,      # Carrier frequency in Hz
    "spacing": 0.5,                  # Element spacing in multiples of wavelength
    "vertical_spacing": 0.5,         # Optional: Element vertical spacing (defaults to 0.5 if not specified)
    "horizontal_spacing": 0.5        # Optional: Element horizontal spacing (defaults to 0.5 if not specified)
}

# OFDM Configuration
OFDM_CONFIG = {
    "num_subcarriers": 64,       
    "subcarrier_spacing": 30e3,  
    "num_ofdm_symbols": 14,      
    "cyclic_prefix_length": 6    
}

# Channel Configuration
CHANNEL_CONFIG = {
    "model": "A",  # Change from "A" to "TDL-A",# - TDL models: "A", "B", "C", "D", "E", "A30", "B100", "C300"  ||TDL-A channel model which is suitable for static urban environments           
    "delay_spread": 300e-9,     
    "min_speed": 0.0, # Set to 0.0 for no mobility         
    "max_speed": 0.0, # Set to 0.0 for no mobility  
    "num_ant": 4,  # Changed to 4 for 4x4 MIMO            
    "snr_range": (10, 30),
    "los_probability": 1.0,  # Line of sight probability
    "bandwidth": 100e6  # 100 MHz bandwidth      
}

# Path Loss Configuration
PATHLOSS_CONFIG = {
    "model": "3gpp_38901",     # 3GPP path loss model
    "scenario": "umi",         # Urban microcell
    "min_distance": 10,        # Minimum distance in meters
    "max_distance": 200        # Maximum distance in meters
}

# Debugging Configuration
DEBUG_CONFIG = {
    "print_channel_stats": True,  # Print channel statistics
    "save_intermediate": False,   # Save intermediate results
    "log_level": "INFO"          # Logging level
}

# Channel Characteristics Configuration
CHANNEL_CHARACTERISTICS_CONFIG = {
    "compute_condition_number": True,
    "compute_path_loss": True,
    "compute_delay_spread": True,
    "compute_doppler": False, #set to False since there's no mobility
    "compute_angular_spread": True
}

# Beamforming Configuration
BEAMFORMING_CONFIG = {
    "compute_optimal_weights": True,
    "compute_beam_patterns": False,
    "store_beam_directions": True,
    "num_beams": 4
}

# Performance Metrics Configuration
METRICS_CONFIG = {
    "compute_capacity": True,
    "compute_sinr": True,
    "compute_spectral_efficiency": True,
    "compute_ber": True
}


# Channel Characteristics Configuration
CHANNEL_CHARACTERISTICS_CONFIG = {
    "compute_condition_number": True,  # Compute condition number of channel matrix
    "compute_path_loss": True,        # Compute path loss
    "compute_delay_spread": True,     # Compute delay spread
    "compute_doppler": False,         # Compute Doppler shift (set to False for static scenario)
    "compute_angular_spread": True    # Compute angular spread
}

# Beamforming Configuration
BEAMFORMING_CONFIG = {
    "compute_optimal_weights": True,   # Compute optimal beamforming weights
    "compute_beam_patterns": True,     # Compute beam patterns
    "store_beam_directions": True,     # Store beam directions
    "num_beams": 4                    # Number of beams (matching num_streams)
}

# Performance Metrics Configuration
METRICS_CONFIG = {
    "compute_capacity": True,         # Compute channel capacity
    "compute_sinr": True,            # Compute SINR
    "compute_ber": False,            # Compute Bit Error Rate (if applicable)
    "compute_ser": False             # Compute Symbol Error Rate (if applicable)
}