# channel_generator.py
import os
import tensorflow as tf
import sionna
from sionna.channel.tr38901 import AntennaArray  # Changed from PanelArray
from sionna.channel.tr38901 import TDL
from sionna.ofdm import ResourceGrid
from sionna.channel import OFDMChannel
import h5py
from config import CONFIG, ANTENNA_CONFIG, OFDM_CONFIG, CHANNEL_CONFIG
from sionna.nr import PUSCHConfig, PUSCHDMRSConfig
import numpy as np
from config import (
    CONFIG, 
    ANTENNA_CONFIG, 
    OFDM_CONFIG, 
    CHANNEL_CONFIG,
    CHANNEL_CHARACTERISTICS_CONFIG,
    BEAMFORMING_CONFIG,
    METRICS_CONFIG
)

def setup_arrays():
    """Create BS and UE antenna arrays"""
    array_params = {
        "num_rows": ANTENNA_CONFIG["num_rows"],
        "num_cols": ANTENNA_CONFIG["num_cols"],
        "polarization": ANTENNA_CONFIG["polarization"],
        "polarization_type": ANTENNA_CONFIG["polarization_type"],
        "antenna_pattern": ANTENNA_CONFIG["pattern"],
        "carrier_frequency": ANTENNA_CONFIG["carrier_frequency"],
        "horizontal_spacing": ANTENNA_CONFIG["spacing"],
        "vertical_spacing": ANTENNA_CONFIG["spacing"]
    }
    return AntennaArray(**array_params), AntennaArray(**array_params)

def setup_resource_grid():
    """Create OFDM resource grid"""
    return ResourceGrid(
        num_ofdm_symbols=OFDM_CONFIG["num_ofdm_symbols"],
        fft_size=OFDM_CONFIG["num_subcarriers"],
        subcarrier_spacing=OFDM_CONFIG["subcarrier_spacing"],
        num_tx=1,
        num_streams_per_tx=CHANNEL_CONFIG["num_ant"],
        cyclic_prefix_length=OFDM_CONFIG["cyclic_prefix_length"]
    )

def setup_channel_model(rg, bs_array, ue_array):
    """Create TDL channel model for 4x4 MIMO configuration"""
    tdl = TDL(
        model=CHANNEL_CONFIG["model"],           # e.g., "A", "B", "C", etc.
        delay_spread=CHANNEL_CONFIG["delay_spread"],
        carrier_frequency=ANTENNA_CONFIG["carrier_frequency"],
        min_speed=CHANNEL_CONFIG["min_speed"],
        max_speed=CHANNEL_CONFIG["max_speed"],
        num_rx_ant=4,  # Set to 4 for 4x4 MIMO
        num_tx_ant=4,  # Set to 4 for 4x4 MIMO
        # Optional parameters for spatial correlation
        spatial_corr_mat=None,  # Add if you want to model spatial correlation
        rx_corr_mat=None,       # Add if you want to model RX correlation
        tx_corr_mat=None,       # Add if you want to model TX correlation
        dtype=tf.complex64
    )
    
    # Create OFDM channel with the TDL model
    return OFDMChannel(
        channel_model=tdl,
        resource_grid=rg,
        add_awgn=True
    )

def setup_resource_grid():
    """Create OFDM resource grid"""
    return ResourceGrid(
        num_ofdm_symbols=OFDM_CONFIG["num_ofdm_symbols"],
        fft_size=OFDM_CONFIG["num_subcarriers"],
        subcarrier_spacing=OFDM_CONFIG["subcarrier_spacing"],
        num_tx=4,  # Changed from 1 to 4 for 4x4 MIMO
        num_streams_per_tx=1,  # Changed to 1 since we have 4 transmitters
        cyclic_prefix_length=OFDM_CONFIG["cyclic_prefix_length"]
    )

# Add these new functions
def compute_channel_characteristics(h, snr_db):
    """Compute various channel characteristics"""
    characteristics = {}
    
    if CHANNEL_CHARACTERISTICS_CONFIG["compute_condition_number"]:
        # Compute condition number for each subcarrier using SVD
        h_matrix = tf.reshape(h, [-1, CHANNEL_CONFIG["num_ant"], CHANNEL_CONFIG["num_ant"]])
        # Compute SVD
        s = tf.linalg.svd(h_matrix, compute_uv=False)
        # Condition number is ratio of largest to smallest singular value
        characteristics["condition_number"] = tf.math.divide_no_nan(
            tf.math.reduce_max(s, axis=-1),
            tf.math.reduce_min(s, axis=-1)
        )
    
    if CHANNEL_CHARACTERISTICS_CONFIG["compute_path_loss"]:
        # Compute path loss
        characteristics["path_loss"] = tf.reduce_mean(tf.abs(h)**2, axis=[-1, -2])
    
    if CHANNEL_CHARACTERISTICS_CONFIG["compute_delay_spread"]:
        # Compute delay spread from frequency domain channel
        characteristics["delay_spread"] = compute_delay_spread(h)
    
    if CHANNEL_CHARACTERISTICS_CONFIG["compute_doppler"]:
        # Compute Doppler shift
        characteristics["doppler_shift"] = compute_doppler_shift(CHANNEL_CONFIG["max_speed"], 
                                                            ANTENNA_CONFIG["carrier_frequency"])
    return characteristics

def compute_beamforming_data(h):
    """Compute beamforming-related data"""
    beamforming_data = {}
    
    if BEAMFORMING_CONFIG["compute_optimal_weights"]:
        # Compute optimal beamforming weights using SVD
        _, _, v = tf.linalg.svd(h)
        beamforming_data["optimal_weights"] = v[..., 0]
    
    return beamforming_data

def compute_performance_metrics(h, snr_db):
    """Compute performance metrics with reduced memory usage"""
    metrics = {}  # Initialize empty dictionary
    
    if METRICS_CONFIG["compute_capacity"]:
        # Process in smaller batches to reduce memory usage
        batch_size = 32  # Reduce this if still running into memory issues
        num_batches = tf.shape(h)[0] // batch_size + (1 if tf.shape(h)[0] % batch_size != 0 else 0)
        capacities = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = tf.minimum(start_idx + batch_size, tf.shape(h)[0])
            
            # Get batch of channel matrices
            h_batch = h[start_idx:end_idx]
            snr_batch = snr_db[start_idx:end_idx]
            
            # Compute SNR in linear scale
            snr_linear = tf.pow(10.0, snr_batch/10.0)
            
            # Reshape h_matrix for batch
            h_matrix = tf.reshape(h_batch, [-1, CHANNEL_CONFIG["num_ant"], CHANNEL_CONFIG["num_ant"]])
            
            # Create identity matrix
            identity = tf.eye(CHANNEL_CONFIG["num_ant"], dtype=h.dtype)
            identity = tf.expand_dims(identity, 0)
            
            # Compute H * H^H
            h_conj_t = tf.transpose(h_matrix, perm=[0, 2, 1], conjugate=True)
            h_product = tf.matmul(h_matrix, h_conj_t)
            
            # Prepare SNR for broadcasting
            snr_linear = tf.cast(snr_linear, h.dtype)
            snr_linear = tf.expand_dims(tf.expand_dims(snr_linear, -1), -1)
            
            # Compute capacity for this batch
            capacity_batch = tf.math.real(
                tf.math.log(
                    tf.linalg.det(identity + snr_linear * h_product)
                )
            ) / tf.math.log(2.0)
            
            capacities.append(capacity_batch)
        
        # Combine results from all batches
        metrics["capacity"] = tf.concat(capacities, axis=0)
    
    if METRICS_CONFIG["compute_sinr"]:
        print("h shape:", h.shape)
        # Compute signal power by averaging over OFDM symbols and subcarriers
        signal_power = tf.reduce_mean(tf.abs(h)**2, axis=[-1, -2])  # Average over last two dims
        print("signal_power shape:", signal_power.shape)
        
        # Compute noise power
        noise_power = tf.pow(10.0, -snr_db/10.0)
        
        # Add dimensions to match signal_power shape for broadcasting
        noise_power = tf.expand_dims(noise_power, axis=-1)  # For num_rx dimension
        noise_power = tf.expand_dims(noise_power, axis=-1)  # For num_rx_ant dimension
        
        # Based on the shapes printed in the error message:
        # h shape: (32, 1, 4, 14, 64)
        # signal_power shape: (32, 1, 4)
        # We need noise_power shape to be (32, 1, 1) for proper broadcasting
        print("noise_power shape:", noise_power.shape)
        
        # Compute SINR
        metrics["sinr"] = signal_power / noise_power

    return metrics  # Make sure to return the metrics dictionary


# Modify your generate_dataset function
def generate_dataset():
    """Generate channel dataset for 4x4 MIMO configuration"""
    # Setup
    tf.random.set_seed(CONFIG["random_seed"])
    bs_array, ue_array = setup_arrays()
    rg = setup_resource_grid()
    channel = setup_channel_model(rg, bs_array, ue_array)
    
    with h5py.File(CONFIG["output_file"], 'w') as f:
        for i in range(0, CONFIG["num_samples"], CONFIG["batch_size"]):
            batch_size = min(CONFIG["batch_size"], CONFIG["num_samples"] - i)
            
            # Generate SNR values
            snr_db = tf.random.uniform([batch_size, 1], 
                                    minval=CHANNEL_CONFIG["snr_range"][0],
                                    maxval=CHANNEL_CONFIG["snr_range"][1])
            
            # Create input tensor with correct shape for 4x4 MIMO
            # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers]
            input_shape = [batch_size, 4, 1,  # 4 transmitters with 1 stream each
                        OFDM_CONFIG["num_ofdm_symbols"], 
                        OFDM_CONFIG["num_subcarriers"]]
            input_bits = tf.zeros(input_shape, dtype=tf.complex64)
            
            # Generate channel realizations
            h = channel([input_bits, snr_db])

            # Handle frequency domain channel response if returned
            if isinstance(h, tuple):
                h, _ = h  # Ignore frequency domain response since it's not needed
            
            # Handle frequency domain channel response if returned
            if isinstance(h, tuple):
                h, freq_h = h
            
            # Compute additional characteristics
            characteristics = compute_channel_characteristics(h, snr_db)
            beamforming_data = compute_beamforming_data(h)
            performance_metrics = compute_performance_metrics(h, snr_db)
            
            # Save batch data
            grp = f.create_group(f'batch_{i//CONFIG["batch_size"]}')
            
            # Save channel response
            # h shape: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
            grp.create_dataset('channel', data=h.numpy())
            grp.create_dataset('snr', data=snr_db.numpy())
            
            # Save channel characteristics
            char_grp = grp.create_group('characteristics')
            for key, value in characteristics.items():
                char_grp.create_dataset(key, data=value.numpy())
            
            # Save beamforming data
            beam_grp = grp.create_group('beamforming')
            for key, value in beamforming_data.items():
                beam_grp.create_dataset(key, data=value.numpy())
            
            # Save performance metrics
            metrics_grp = grp.create_group('metrics')
            for key, value in performance_metrics.items():
                metrics_grp.create_dataset(key, data=value.numpy())
            
            # Save additional MIMO-specific information
            mimo_grp = grp.create_group('mimo_info')
            mimo_grp.create_dataset('num_tx_ant', data=4)
            mimo_grp.create_dataset('num_rx_ant', data=4)
            
            print(f"Generated batch {i//CONFIG['batch_size'] + 1}")
            
            # Optional: Add progress information
            progress = (i + batch_size) / CONFIG["num_samples"] * 100
            print(f"Progress: {progress:.1f}%")


def compute_delay_spread(h):
    """Compute RMS delay spread from frequency domain channel"""
    # Convert to time domain
    h_time = tf.signal.fft(h)
    
    # Compute power delay profile
    power_profile = tf.abs(h_time)**2
    
    # Compute mean delay
    delays = tf.range(tf.shape(h_time)[-1], dtype=tf.float32)
    mean_delay = tf.reduce_sum(delays * power_profile, axis=-1) / tf.reduce_sum(power_profile, axis=-1)
    
    # Compute RMS delay spread
    mean_delay = tf.expand_dims(mean_delay, -1)
    rms_ds = tf.sqrt(
        tf.reduce_sum(((delays - mean_delay)**2) * power_profile, axis=-1) / 
        tf.reduce_sum(power_profile, axis=-1)
    )
    
    return rms_ds

def compute_doppler_shift(speed, carrier_freq):
    """Compute maximum Doppler shift"""
    c = 299792458.0  # Speed of light
    return 2 * np.pi * speed * carrier_freq / c

if __name__ == "__main__":
    # Calculate samples for each split
    total_samples = CONFIG["num_samples"]
    split_samples = {
        "train": int(total_samples * CONFIG["splits"]["train"]),
        "validation": int(total_samples * CONFIG["splits"]["validation"]),
        "test": int(total_samples * CONFIG["splits"]["test"])
    }
    
    # Generate each dataset
    for split_name, num_samples in split_samples.items():
        print(f"\nGenerating {split_name} dataset...")
        CONFIG["num_samples"] = num_samples  # Temporarily update num_samples
        CONFIG["output_file"] = CONFIG["output_files"][split_name]
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(CONFIG["output_file"]), exist_ok=True)
        
        # Generate dataset
        generate_dataset()
        print(f"{split_name} dataset saved to {CONFIG['output_file']}")
        print(f"Number of samples: {num_samples} ({CONFIG['splits'][split_name]*100:.1f}%)")