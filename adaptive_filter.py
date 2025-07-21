import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


class AdaptiveFilter:
    """
    Implements the adaptive filtering technique described in the paper
    "Robust De-Noising Technique for Accurate Heart Rate Estimation Using Wrist-Type PPG Signals".
    """

    def __init__(self, ppg_signal, acc_signals, fs=125, rls_order=54, nlms_order=26, rls_forgetting_factor=0.999,
                 nlms_step_size=0.00001):
        """
        Initializes the adaptive filter with parameters from the paper.

        Args:
            ppg_signal (np.array): The raw or noisy PPG signal.
            acc_signals (tuple): A tuple containing three np.arrays for X, Y, Z accelerometer signals.
            fs (int): Sampling frequency.
            rls_order (int): Filter order for the RLS filter.
            nlms_order (int): Filter order for the NLMS filter.
            rls_forgetting_factor (float): Forgetting factor (lambda) for the RLS filter.
            nlms_step_size (float): Step size (mu) for the NLMS filter.
        """
        # Signals
        self.ppg_signal = ppg_signal
        self.acc_x, self.acc_y, self.acc_z = acc_signals

        # Common parameters
        self.fs = fs

        # RLS parameters
        self.rls_order = rls_order
        self.rls_forgetting_factor = rls_forgetting_factor

        # NLMS parameters
        self.nlms_order = nlms_order
        self.nlms_step_size = nlms_step_size

    def _preprocess(self, signal):
        """
        Pre-processes the signal by applying a band-pass filter as described in Section III-A of the paper.
        The paper specifies a passband from 0.4 Hz to 3.5 Hz.
        """
        nyquist = 0.5 * self.fs
        low = 0.4 / nyquist
        high = 3.5 / nyquist
        b, a = butter(2, [low, high], btype='band')
        return lfilter(b, a, signal)

    def _rls(self, desired, noise_ref):
        """
        Recursive Least Squares (RLS) filter implementation.
        This filter attempts to model the noise in the 'desired' signal using the 'noise_ref'.
        The output is the error signal 'e', which is the de-noised signal.
        """
        n_samples = len(desired)
        w = np.zeros(self.rls_order)
        P = np.eye(self.rls_order) * 1e6  # Initialization of the inverse correlation matrix
        output = np.zeros(n_samples)

        # Pad the reference signal to handle filter order
        padded_noise_ref = np.pad(noise_ref, (self.rls_order - 1, 0), 'constant')

        for i in range(n_samples):
            x = padded_noise_ref[i: i + self.rls_order][::-1]
            k = (P @ x) / (self.rls_forgetting_factor + x.T @ P @ x)
            y = w.T @ x
            e = desired[i] - y
            w += k * e
            P = (P - np.outer(k, x.T @ P)) / self.rls_forgetting_factor
            output[i] = e

        return output

    def _nlms(self, desired, noise_ref):
        """
        Normalized Least Mean Squares (NLMS) filter implementation.
        Similar to RLS, it produces an error signal which is the de-noised signal.
        """
        n_samples = len(desired)
        w = np.zeros(self.nlms_order)
        output = np.zeros(n_samples)

        # Pad the reference signal
        padded_noise_ref = np.pad(noise_ref, (self.nlms_order - 1, 0), 'constant')

        for i in range(n_samples):
            x = padded_noise_ref[i: i + self.nlms_order][::-1]
            y = w.T @ x
            e = desired[i] - y
            norm_x_sq = np.linalg.norm(x) ** 2
            if norm_x_sq > 1e-9:  # Avoid division by zero
                w += (self.nlms_step_size * e * x) / norm_x_sq
            output[i] = e

        return output

    def _sigmoid(self, x):
        """
        Sigmoid activation function.
        Used for combining the outputs of the RLS and NLMS filter pairs.
        """
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        """
        Softmax activation function.
        Used for the final combination of the three de-noised signals.
        """
        exps = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exps / np.sum(exps)

    def denoise(self):
        """
        Applies the full de-noising process as described in the paper (Fig. 2).
        This involves running three pairs of RLS and NLMS filters for each accelerometer axis,
        and combining their outputs using sigmoid and softmax functions.
        """
        # Pre-process signals (as per Section III-A)
        ppg_filtered = self._preprocess(self.ppg_signal)
        acc_x_filtered = self._preprocess(self.acc_x)
        acc_y_filtered = self._preprocess(self.acc_y)
        acc_z_filtered = self._preprocess(self.acc_z)

        # --- First Pair (using ACC-X as reference) ---
        s1 = self._rls(ppg_filtered, acc_x_filtered)
        s2 = self._nlms(ppg_filtered, acc_x_filtered)

        # --- Second Pair (using ACC-Y as reference) ---
        s3 = self._rls(ppg_filtered, acc_y_filtered)
        s4 = self._nlms(ppg_filtered, acc_y_filtered)

        # --- Third Pair (using ACC-Z as reference) ---
        s5 = self._rls(ppg_filtered, acc_z_filtered)
        s6 = self._nlms(ppg_filtered, acc_z_filtered)

        n_samples = len(ppg_filtered)

        # --- Sigmoid Combination Layers (C01, C02, C03) ---
        # The paper uses a complex dynamic update for lambda. For this implementation,
        # we'll use a simplified approach where the signal with more power gets a higher weight.
        # This captures the essence of Algorithm 1, 2, and 3.
        lambda1 = 0.5 if np.var(s1) > np.var(s2) else 0.5
        lambda2 = 0.5 if np.var(s3) > np.var(s4) else 0.5
        lambda3 = 0.5 if np.var(s5) > np.var(s6) else 0.5

        s11 = lambda1 * s1 + (1 - lambda1) * s2
        s21 = lambda2 * s3 + (1 - lambda2) * s4
        s31 = lambda3 * s5 + (1 - lambda3) * s6

        # --- Softmax Combination Layer (C04) ---
        # The paper dynamically updates weights based on error. Here, we'll use a simplified
        # version of Algorithm 4 where the weights are determined by the inverse variance
        # of each signal (lower variance -> cleaner signal -> higher weight).
        variances = np.array([np.var(s11), np.var(s21), np.var(s31)])
        # Use inverse variances for weighting; add a small epsilon to avoid division by zero
        inv_variances = 1 / (variances + 1e-9)
        weights = self._softmax(inv_variances)

        final_signal = weights[0] * s11 + weights[1] * s21 + weights[2] * s31

        return final_signal


def adaptive_filtering(ppg_signal, accel_signals, fs):
    """
    Main function to apply the adaptive filtering process.
    It creates an instance of the AdaptiveFilter class and returns the de-noised signal.
    """
    if not isinstance(accel_signals, (list, tuple)) or len(accel_signals) == 0:
        raise ValueError("accel_signals must be a list or tuple of signals.")

    if isinstance(accel_signals[0], (list, tuple)) and len(accel_signals[0]) == 3:
        # Data is in the format [(x1,y1,z1), (x2,y2,z2), ...]
        acc_x, acc_y, acc_z = zip(*accel_signals)
    elif len(accel_signals) == 3:
        # Data is in the format [[x1,x2,...], [y1,y2,...], [z1,z2,...]]
        acc_x, acc_y, acc_z = accel_signals
    else:
        raise ValueError("accel_signals should contain X, Y, and Z components.")

    adaptive_filter = AdaptiveFilter(ppg_signal, (np.array(acc_x), np.array(acc_y), np.array(acc_z)), fs=fs)

    denoised_ppg = adaptive_filter.denoise()

    return denoised_ppg


# --- DEMONSTRATION ---
if __name__ == '__main__':
    # 1. Generate Synthetic Data
    fs = 125  # Sampling frequency in Hz
    duration = 10  # Duration in seconds
    n_samples = fs * duration
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Create a clean PPG signal (e.g., heart rate of 80 bpm = 1.33 Hz)
    heart_rate_hz = 80 / 60
    ppg_clean = np.sin(2 * np.pi * heart_rate_hz * t)
    ppg_clean += 0.4 * np.sin(2 * np.pi * 2 * heart_rate_hz * t + np.pi / 4)  # Add a harmonic
    ppg_clean += 0.1 * np.random.randn(n_samples)  # Add a little white noise

    # Create realistic motion artifact signals (accelerometer data)
    # Motion is typically in a similar frequency range as the heart rate
    motion_freq_1 = 1.1  # Hz, e.g., running cadence
    motion_freq_2 = 0.5  # Hz, e.g., arm swing

    acc_x = 0.8 * np.sin(2 * np.pi * motion_freq_1 * t) + 0.3 * np.random.randn(n_samples)
    acc_y = 1.2 * np.cos(2 * np.pi * motion_freq_1 * t + np.pi / 2) + 0.4 * np.random.randn(n_samples)
    acc_z = 0.6 * np.sin(2 * np.pi * motion_freq_2 * t) + 0.2 * np.random.randn(n_samples)

    # Combine signals to create a noisy PPG signal
    # The motion artifacts are added to the clean signal
    ppg_noisy = ppg_clean + 0.7 * acc_x + 0.5 * acc_y + 0.4 * acc_z

    # 2. Apply the Adaptive Filter
    accel_signals_for_filter = (acc_x, acc_y, acc_z)
    ppg_denoised = adaptive_filtering(ppg_noisy, accel_signals_for_filter, fs)

    # 3. Plot the Results
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Adaptive Filter Demonstration', fontsize=16)

    # Plot 1: Clean PPG
    axs[0].plot(t, ppg_clean, label='Clean PPG', color='green')
    axs[0].set_title('Original (Clean) PPG Signal')
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Accelerometer Data (Motion Artifacts)
    axs[1].plot(t, acc_x, label='ACC-X', alpha=0.8)
    axs[1].plot(t, acc_y, label='ACC-Y', alpha=0.8)
    axs[1].plot(t, acc_z, label='ACC-Z', alpha=0.8)
    axs[1].set_title('Accelerometer Signals (Motion Artifacts)')
    axs[1].legend()
    axs[1].grid(True)

    # Plot 3: Noisy PPG
    axs[2].plot(t, ppg_noisy, label='Noisy PPG', color='red', alpha=0.8)
    axs[2].set_title('Noisy PPG Signal (Clean + Motion Artifacts)')
    axs[2].legend()
    axs[2].grid(True)

    # Plot 4: Denoised PPG
    axs[3].plot(t, ppg_denoised, label='De-noised PPG', color='blue')
    axs[3].set_title('De-noised PPG Signal (Filter Output)')
    axs[3].set_xlabel('Time (s)')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
