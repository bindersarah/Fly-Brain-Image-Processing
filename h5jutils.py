
import numpy as np
import h5py
import tempfile
import subprocess
import sys
import imageio_ffmpeg as ff
import os
import uuid
import warnings

def get_hevc_codecs():
    process = subprocess.Popen(
                    ['ffmpeg', '-codecs'],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.PIPE
              )
    stdout_data, stderr_data = process.communicate()
    
    hevc_codecs = []
    for line in stdout_data.decode().split('\n'):
        if 'hevc' in line:
            hevc_codecs.extend(line.split('encoders: ')[1].split(')')[0].split(' '))

    return hevc_codecs

def extract_channel(raw_data):
    """
    Extract the raw x264 data into an numpy array

    Parameters
    ----------
    raw_data: byte
        byte string of the original data

    Returns
    -------
    np.ndarray: Nframes by height by width
    """
    with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as temp_file:
        temp_file.write(raw_data)
        temp_file_name = temp_file.name

        reader = ff.read_frames(temp_file_name)
        meta = next(reader)

        width, height = meta['size'] # or ["source_size"] ?

        if meta['pix_fmt'] == 'gray12le(tv)':
            fmt = 'gray12le'
            dtype = np.uint16
        else:
            fmt = 'yuv444p'
            dtype = np.uint8

        process = subprocess.Popen(
                        ['ffmpeg', '-i', temp_file_name, '-c:v', 'rawvideo',
                         '-pix_fmt', fmt, '-f', 'rawvideo', 
                         'pipe:1'],
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE
                  )
        stdout_data, stderr_data = process.communicate()
        
        frames = np.frombuffer(stdout_data, dtype = dtype).reshape(-1, height, width)

    return frames

def read_h5j(filename, channels = None, reference = False):
    """
    Read an h5j file according to the [format specification](https://github.com/JaneliaSciComp/workstation/blob/master/docs/H5JFileFormat.md)

    Parameters
    ----------
    filename: str
        Path to the .h5j file
    channels: str
        default: None, to get all signal channels
        Specify any combinations of 'RGB' in a single string in any order, for
        .h5j file with 3 color channels, otherwise supply 'Y' for .h5j with only
        single channel.
    reference: bool
        default: False
        Whether to read the reference background

    Returns
    -------
    np.ndarray: The signal channels, either Nz by height by width by N,
                where N is the number of channel requested,
                if `channels` is an empty string, then will return None
    np.ndarray: The reference channel, Nz by height by width.
                If `reference` is False, will return None
    
        
    
    """
    with h5py.File(filename, 'r') as h5file:
        # Read global attributes
        attributes = {tmp: h5file.attrs[tmp] for tmp in h5file.attrs}
        
        # Read channel attributes
        channel_attributes = {tmp: h5file['Channels'].attrs[tmp] for tmp in h5file['Channels'].attrs}
        nframes = channel_attributes['frames'][0]
        height = channel_attributes['height'][0] + channel_attributes['pad_bottom'][0]
        width = channel_attributes['width'][0] + channel_attributes['pad_right'][0]
        
        # Detect available channels
        channel_keys = list(h5file['Channels'].keys())
        num_channels = len(channel_keys) - 1  # Assuming last channel is 'reference'

        if num_channels == 3:
            signals = {'R': 0, 'G': 1, 'B': 2}
            if channels is None:
                channels = 'RGB'
        elif num_channels == 1:
            signals = {'Y': 0}
            if channels is None:
                channels = 'Y'
        
        # Extract image channels
        signal_channels = []
        for channel in channels:
            channel_data = extract_channel(h5file["Channels"][f"Channel_{signals[channel]}"][()].tobytes())
            signal_channels.append(channel_data)
        
        # Concatenate RGB or return single channel
        signal = None
        if num_channels == 3:
            signal = np.concatenate([channel[:, :, :, None] for channel in signal_channels], axis=3)  # Stack R, G, B
        elif num_channels == 1:
            signal = signal_channels[0]  # Single channel

        # Extract reference channel
        if reference:
            reference = extract_channel(h5file["Channels"][f'Channel_{num_channels}'][()].tobytes())
        else:
            reference = None
    
    return signal, reference, attributes


_hevc_codecs = get_hevc_codecs()
if not len(_hevc_codecs):
    warnings.warn("No HEVC codec found on system - can only read h5j")
    _hevc_codecs = None
else:
    def encode_channel(arr, codec = _hevc_codecs[0]):
        nframes, height, width = arr.shape
        temp_dir = tempfile.gettempdir()
        temp_file_name = os.path.join(temp_dir, f"tempfile_{uuid.uuid4().hex}.mp4")
    
        process = subprocess.Popen(
                ['ffmpeg', '-f', 'rawvideo', '-pix_fmt', 'gray', 
                 '-s', f'{width}x{height}', 
                 '-r', '25',
                 '-i', 'pipe:0', 
                 '-c:v', codec,
                 '-pix_fmt', 'gray',
                 temp_file_name],
                stdin = subprocess.PIPE,
                stderr = subprocess.PIPE
        )
        print(temp_file_name)
        stdout_data, stderr_data = process.communicate(input = arr.tobytes())    
        with open(temp_file_name, 'rb') as f:
            cc = np.frombuffer(f.read(), dtype = np.int8)
        os.remove(temp_file_name)
        return cc

    
    def write_binary_h5j(filename, arr, codec = _hevc_codecs[0]):
        """
        Parameters
        ----------
        filename : str
          Path of the file to be created
        arr : np.ndarray
          Must be np.uint8
          An array should be of Nframes by height by width for single channel,
          or of Nframes by height by width by Nchannels for multiple channels.
        codec : str
          The codec to use for encoding. ffmpeg must have this codec supported.
          By default, an hevc codec supported by ffmpeg will be read
        """
        if len(arr.shape) == 3:
            nframes, height, width = arr.shape
            nchannels = 1
            arr = arr[:,:,:,None]
        elif len(arr.shape) == 4:
            nframes, height, width, nchannels = arr.shape
        else:
            raise ValueError("arr must be 4d or 3d.")
        attrs = {'frames': np.array([nframes]),
                 'height': np.array([height]),
                 'pad_bottom': np.array([0]),
                 'pad_right': np.array([0]),
                 'width': np.array([width])}
        
        with h5py.File(filename, 'w') as f:
            grp = f.create_group('Channels')
            grp.attrs.update(attrs)
            for n in range(nchannels):
                data = encode_channel(arr[:,:,:,n], codec = codec)
                ds = f['Channels'].create_dataset(f'Channel_{n}', data = data)
                ds.attrs['content_type'] = b'signal'

        
def mip(signal, axis = 0):
    """
    Perform a maximum intensity projection of the 3D image, along the axis
    """
    return signal.max(axis = axis)
