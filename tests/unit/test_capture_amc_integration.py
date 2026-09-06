"""Controlled new-domain observation: channel tuning is not AMC-transparent."""

from iqwav.amr import DatasetSeeds, PRIMARY_AMC_LABELS, SyntheticDatasetConfig, classify_modulation, generate_synthetic_dataset
from iqwav.dsp import apply_frequency_offset, extract_band


def test_channelized_records_remain_deterministic_primary_amc_inputs():
    records = generate_synthetic_dataset(SyntheticDatasetConfig(n_parents_per_class=1, n_variants_per_parent=3, seeds=DatasetSeeds(801, 802, 803, 804))).records
    labels = []
    for record in records:
        shifted = apply_frequency_offset(record.samples, record.fs, 6000.0)
        tuned = extract_band(shifted, record.fs, 0.0, 12000.0, transition_hz=1000.0, numtaps=51)
        labels.append(classify_modulation(tuned.samples).label)
    assert len(labels) == len(records)
    assert set(labels) <= set(PRIMARY_AMC_LABELS)
