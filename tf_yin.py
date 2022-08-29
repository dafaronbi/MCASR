import tensorflow as tf
import librosa

def tf_yin(
    y,
    *,
    fmin,
    fmax,
    sr=22050,
    frame_length=2048,
    win_length=None,
    hop_length=None,
    trough_threshold=0.1,
    center=True,
    pad_mode="constant",
):
    """Fundamental frequency (F0) estimation using the YIN algorithm.
    YIN is an autocorrelation based method for fundamental frequency estimation [#]_.
    First, a normalized difference function is computed over short (overlapping) frames of audio.
    Next, the first minimum in the difference function below ``trough_threshold`` is selected as
    an estimate of the signal's period.
    Finally, the estimated period is refined using parabolic interpolation before converting
    into the corresponding frequency.
    .. [#] De Cheveigné, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.
    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported..
    fmin : number > 0 [scalar]
        minimum frequency in Hertz.
        The recommended minimum is ``librosa.note_to_hz('C2')`` (~65 Hz)
        though lower values may be feasible.
    fmax : number > 0 [scalar]
        maximum frequency in Hertz.
        The recommended maximum is ``librosa.note_to_hz('C7')`` (~2093 Hz)
        though higher values may be feasible.
    sr : number > 0 [scalar]
        sampling rate of ``y`` in Hertz.
    frame_length : int > 0 [scalar]
        length of the frames in samples.
        By default, ``frame_length=2048`` corresponds to a time scale of about 93 ms at
        a sampling rate of 22050 Hz.
    win_length : None or int > 0 [scalar]
        length of the window for calculating autocorrelation in samples.
        If ``None``, defaults to ``frame_length // 2``
    hop_length : None or int > 0 [scalar]
        number of audio samples between adjacent YIN predictions.
        If ``None``, defaults to ``frame_length // 4``.
    trough_threshold : number > 0 [scalar]
        absolute threshold for peak estimation.
    center : boolean
        If ``True``, the signal `y` is padded so that frame
        ``D[:, t]`` is centered at `y[t * hop_length]`.
        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.
        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
        time grid by means of ``librosa.core.frames_to_samples``.
    pad_mode : string or function
        If ``center=True``, this argument is passed to ``np.pad`` for padding
        the edges of the signal ``y``. By default (``pad_mode="constant"``),
        ``y`` is padded on both sides with zeros.
        If ``center=False``,  this argument is ignored.
        .. see also:: `np.pad`
    Returns
    -------
    f0: np.ndarray [shape=(..., n_frames)]
        time series of fundamental frequencies in Hertz.
        If multi-channel input is provided, f0 curves are estimated separately for each channel.
    See Also
    --------
    librosa.pyin :
        Fundamental frequency (F0) estimation using probabilistic YIN (pYIN).
    Examples
    --------
    Computing a fundamental frequency (F0) curve from an audio input
    >>> y = librosa.chirp(fmin=440, fmax=880, duration=5.0)
    >>> librosa.yin(y, fmin=440, fmax=880)
    array([442.66354675, 441.95299983, 441.58010963, ...,
        871.161732  , 873.99001454, 877.04297681])
    """

    if fmin is None or fmax is None:
        raise ParameterError('both "fmin" and "fmax" must be provided')

    # Set the default window length if it is not already specified.
    if win_length is None:
        win_length = frame_length // 2

    if win_length >= frame_length:
        raise ParameterError(
            "win_length={} cannot exceed given frame_length={}".format(
                win_length, frame_length
            )
        )

    # Set the default hop if it is not already specified.
    if hop_length is None:
        hop_length = frame_length // 4

    # Pad the time series so that frames are centered
    if center:
        padding = [(0, 0) for _ in y.shape]
        padding[-1] = (frame_length // 2, frame_length // 2)
        y = np.pad(y, padding, mode=pad_mode)

    # Frame audio.
    y_frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

    # Calculate minimum and maximum periods
    min_period = max(int(np.floor(sr / fmax)), 1)
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)

    # Calculate cumulative mean normalized difference function.
    yin_frames = librosa._cumulative_mean_normalized_difference(
        y_frames, frame_length, win_length, min_period, max_period
    )

    # Parabolic interpolation.
    parabolic_shifts = _parabolic_interpolation(yin_frames)

    # Find local minima.
    is_trough = util.localmin(yin_frames, axis=-2)
    is_trough[..., 0, :] = yin_frames[..., 0, :] < yin_frames[..., 1, :]

    # Find minima below peak threshold.
    is_threshold_trough = np.logical_and(is_trough, yin_frames < trough_threshold)

    # Absolute threshold.
    # "The solution we propose is to set an absolute threshold and choose the
    # smallest value of tau that gives a minimum of d' deeper than
    # this threshold. If none is found, the global minimum is chosen instead."
    target_shape = list(yin_frames.shape)
    target_shape[-2] = 1

    global_min = np.argmin(yin_frames, axis=-2)
    yin_period = np.argmax(is_threshold_trough, axis=-2)

    global_min = global_min.reshape(target_shape)
    yin_period = yin_period.reshape(target_shape)

    no_trough_below_threshold = np.all(~is_threshold_trough, axis=-2, keepdims=True)
    yin_period[no_trough_below_threshold] = global_min[no_trough_below_threshold]

    # Refine peak by parabolic interpolation.

    yin_period = (
        min_period
        + yin_period
        + np.take_along_axis(parabolic_shifts, yin_period, axis=-2)
    )[..., 0, :]

    # Convert period to fundamental frequency.
    f0 = sr / yin_period
    return f0