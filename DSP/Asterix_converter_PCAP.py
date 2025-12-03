import pandas as pd
import struct
import math
import os
import time

# --- ASTERIX ENCODING FUNCTIONS ---

def float_to_asterix_azimuth(deg):
    """Encodes degrees (0-360) to ASTERIX 16-bit azimuth."""
    val = int((deg % 360) / 360.0 * 65536)
    return max(0, min(65535, val))

def float_to_asterix_range(nmi):
    """Encodes nautical miles to ASTERIX 16-bit range."""
    val = int(nmi * 256)
    return max(0, min(65535, val))

def float_to_flight_level(altitude_ft):
    """Encodes altitude in feet to ASTERIX Flight Level."""
    val = int(altitude_ft / 25.0)
    return max(-32768, min(32767, val))

def create_asterix_message(row, sac=0, sic=1):
    """Constructs the raw ASTERIX payload."""
    
    # I048/010 Data Source Identifier
    b_010 = struct.pack('>BB', sac, sic)
    
    # I048/040 Measured Position
    rng_val = float_to_asterix_range(row['Range_nmi'])
    az_val = float_to_asterix_azimuth(row['Azimuth_deg'])
    b_040 = struct.pack('>HH', rng_val, az_val)
    
    # I048/090 Flight Level
    fl_val = float_to_flight_level(row['Altitude_ft'])
    b_090 = struct.pack('>h', fl_val)
    
    # I048/140 Time of Day
    time_val = int(row['Time_Sec'] * 128) % (24 * 3600 * 128)
    b_140 = struct.pack('>I', time_val)[1:] 
    
    # FSPEC: 1101 0100 = 0xD4 (DataSource, Time, Position, FlightLevel)
    fspec_byte = 0xD4
    
    # Data Block
    data_block = struct.pack('B', fspec_byte) + b_010 + b_140 + b_040 + b_090
    
    # Header: CAT (048) + Length
    total_len = 1 + 2 + len(data_block)
    header = struct.pack('>BH', 48, total_len)
    
    return header + data_block

# --- PCAP & NETWORK HEADERS ---

def get_pcap_global_header():
    """Returns the 24-byte PCAP global header."""
    # Magic(4), Major(2), Minor(2), Zone(4), SigFigs(4), SnapLen(4), Network(4)
    # Network 1 = Ethernet
    return struct.pack('<IHHiIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1)

def get_pcap_packet_header(ts_sec, ts_usec, length):
    """Returns the 16-byte PCAP packet header."""
    return struct.pack('<IIII', int(ts_sec), int(ts_usec), length, length)

def create_udp_packet(payload):
    """Wraps payload in Ethernet/IP/UDP headers."""
    
    # 1. Ethernet Header (14 bytes)
    # Dst MAC (dummy), Src MAC (dummy), EtherType (0x0800 for IPv4)
    eth = b'\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00' + b'\x08\x00'
    
    # 2. IP Header (20 bytes)
    # Ver/IHL, TOS, TotLen, ID, Frag, TTL, Proto (17=UDP), Cksum, SrcIP, DstIP
    ip_len = 20 + 8 + len(payload)
    # Src: 127.0.0.1 (0x7f000001), Dst: 127.0.0.1
    ip = struct.pack('>BBHHHBBHII', 
                     0x45, 0, ip_len, 0, 0, 64, 17, 0, 
                     0x7f000001, 0x7f000001)
    
    # 3. UDP Header (8 bytes)
    # SrcPort, DstPort (8600 for ASTERIX), Len, Cksum
    udp = struct.pack('>HHHH', 5000, 8600, 8 + len(payload), 0)
    
    return eth + ip + udp + payload

# --- MAIN CONVERSION ---

def convert_csv_to_pcap(csv_path, output_path):
    print(f"Reading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return

    print(f"Converting {len(df)} records to ASTERIX PCAP...")
    
    with open(output_path, 'wb') as f:
        # Write Global Header
        f.write(get_pcap_global_header())
        
        start_time = time.time()
        
        for index, row in df.iterrows():
            # Create Raw ASTERIX Message
            asterix_msg = create_asterix_message(row, sac=10, sic=43)
            
            # Wrap in UDP/IP/Ethernet
            full_packet = create_udp_packet(asterix_msg)
            
            # Create PCAP Packet Header (using simulated time for timestamp)
            # Sim time is row['Time_Sec'], we add it to a base epoch
            pkt_time = start_time + row['Time_Sec']
            ts_sec = int(pkt_time)
            ts_usec = int((pkt_time - ts_sec) * 1_000_000)
            
            pcap_header = get_pcap_packet_header(ts_sec, ts_usec, len(full_packet))
            
            # Write to file
            f.write(pcap_header)
            f.write(full_packet)
            
    print(f"Success! PCAP file written to {output_path}")
    print("Open this file in Wireshark. Ensure 'ASTERIX' protocol is enabled.")

if __name__ == "__main__":
    csv_file = 'radar_track_data.csv'
    pcap_file = 'radar_tracks.pcap'
    convert_csv_to_pcap(csv_file, pcap_file)