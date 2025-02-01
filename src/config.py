# config.py
import tensorflow as tf

# General Configuration
# config.py
CONFIG = {
    "num_samples": 100,         # Increased total samples
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
    "num_rows": 2,                    # Number of rows of elements
    "num_cols": 2,                    # Number of columns of elements
    "polarization": "single",           # Must be either "single" or "dual"
    "polarization_type": "V",        # For dual polarization, must be "VH" or "cross"
    "pattern": "38.901",             # Element radiation pattern, either "omni" or "38.901"
    "carrier_frequency": 3.5e9,      # Carrier frequency in Hz
    "spacing": 0.5,                  # Element spacing in multiples of wavelength????????
    "vertical_spacing": 0.5,         # Optional: Element vertical spacing (defaults to 0.5 if not specified)
    "horizontal_spacing": 0.5        # Optional: Element horizontal spacing (defaults to 0.5 if not specified)
}

MIMO_CONFIG = {
    "num_tx": 4,                  # Number of transmitters
    "num_rx": 4,                  # Number of receivers
    "num_streams_per_tx": 1       # Streams per transmitter (for spatial multiplexing)
}

# OFDM Configuration
OFDM_CONFIG = {
    "num_subcarriers": 64,       
    "subcarrier_spacing": 30e3,  #30 kHz subcarrier spacing
    "num_ofdm_symbols": 14,      
    "cyclic_prefix_length": 9  # CP length in samples for 30 kHz spacing
}

# Channel Configuration
CHANNEL_CONFIG = {
    "model": "A",  #"TDL-A",# - TDL models: "A", "B", "C", "D", "E", "A30", "B100", "C300"  ||TDL-A channel model which is suitable for static urban environments           
    "delay_spread": 30e-9,   # Reduced to 30ns for TDL-A  
    "min_speed": 0.0, # Set to 0.0 for no mobility         
    "max_speed": 0.0, # Set to 0.0 for no mobility  
    "num_ant": 4,  # Changed to 4 for 4x4 MIMO            
    "snr_range": (10, 30),
    "los_probability": 0.5,     # Changed to 0.5 for realistic urban scenario
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

