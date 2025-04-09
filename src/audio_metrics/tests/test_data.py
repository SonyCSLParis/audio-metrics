import torch

from audio_metrics.data import AudioMetricsData


def test_incremental_stats():
    d = 8
    x1 = torch.randn((1, d))
    x2 = torch.randn((10, d))
    x12 = torch.cat((x1, x2))
    amd = AudioMetricsData()
    amd.add(x1)
    amd.add(x2)
    m_a = amd.mean
    c_a = amd.cov
    amd = AudioMetricsData()
    amd.add(x12)
    m_b = amd.mean
    c_b = amd.cov
    torch.testing.assert_close(m_a, m_b)
    torch.testing.assert_close(c_a, c_b)
