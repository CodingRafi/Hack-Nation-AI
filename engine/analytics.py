"""Analitik time-series (rolling window) dan kalkulasi FCR / penghematan pakan."""

import collections

HARGA_PAKAN_IDR_PER_KG = 12000
EFISIENSI_PAKAN = 0.15  # estimasi 15% pakan terbuang yang bisa dihemat


class TelemetryAnalytics:
    def __init__(self, window_size=10):
        self.ph_history = collections.deque(maxlen=window_size)
        self.temp_history = collections.deque(maxlen=window_size)
        self.turbidity_history = collections.deque(maxlen=window_size)

    def update(self, ph, temp, turbidity):
        self.ph_history.append(ph)
        self.temp_history.append(temp)
        self.turbidity_history.append(turbidity)

    def reset(self):
        self.ph_history.clear()
        self.temp_history.clear()
        self.turbidity_history.clear()

    def rolling_avg(self):
        """Rata-rata (pH, suhu, turbidity) pada window; None jika belum ada data."""
        if not self.ph_history:
            return None
        n = len(self.ph_history)
        return (
            round(sum(self.ph_history) / n, 2),
            round(sum(self.temp_history) / n, 2),
            round(sum(self.turbidity_history) / n, 1),
        )

    def get_trends(self):
        """Return (label, ph_delta) dari perubahan awal -> akhir window."""
        if len(self.ph_history) < 2:
            return "STABLE", 0.0

        ph_delta = round(self.ph_history[-1] - self.ph_history[0], 2)
        temp_delta = round(self.temp_history[-1] - self.temp_history[0], 2)

        if ph_delta > 0.5 or temp_delta > 1.0:
            return "LONJAKAN_AMONIA", ph_delta
        return "STABLE", ph_delta


def calculate_fcr(feed_kg, biomass_gain_kg):
    """FCR = total pakan / pertambahan biomassa (kg)."""
    if biomass_gain_kg <= 0:
        return 0.0
    return round(feed_kg / biomass_gain_kg, 2)


def calculate_savings(feed_kg):
    """Estimasi pakan (kg) dan biaya (Rp) yang dihemat berkat monitoring."""
    saved_feed_kg = feed_kg * EFISIENSI_PAKAN
    cost_saved_idr = int(saved_feed_kg * HARGA_PAKAN_IDR_PER_KG)
    return round(saved_feed_kg, 2), cost_saved_idr
