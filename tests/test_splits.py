import pandas as pd

from boneage.build_splits import find_duplicate_groups, hamming_distance


def duplicate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [10, 11, 12, 13],
            "sha256": ["a" * 64, "a" * 64, "b" * 64, "c" * 64],
            "phash": [
                "0000000000000000",
                "ffffffffffffffff",
                "0000000000000001",
                "f0f0f0f0f0f0f0f0",
            ],
        }
    )


def test_hamming_distance() -> None:
    assert hamming_distance("0" * 16, "0" * 15 + "1") == 1
    assert hamming_distance("invalid", "0" * 16) == 64


def test_exact_sha_duplicates_are_grouped() -> None:
    groups, audit = find_duplicate_groups(duplicate_frame(), confirm_pair=lambda *_: False)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
    assert audit["exact_sha256_groups"] == 1


def test_near_phash_requires_content_confirmation() -> None:
    rejected, rejected_audit = find_duplicate_groups(
        duplicate_frame(), confirm_pair=lambda *_: False
    )
    accepted, accepted_audit = find_duplicate_groups(
        duplicate_frame(), confirm_pair=lambda first, second: {first, second} == {0, 2}
    )
    assert rejected[0] != rejected[2]
    assert accepted[0] == accepted[2]
    assert rejected_audit["confirmed_near_duplicate_pairs"] == 0
    assert accepted_audit["confirmed_near_duplicate_pairs"] == 1
