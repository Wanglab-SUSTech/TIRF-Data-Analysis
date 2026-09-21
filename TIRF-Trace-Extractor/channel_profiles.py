"""
channel_profiles.py - shared channel naming helpers.

The ALEX workflow stores excitation and emission as distinct logical channels.
Registration still operates on physical emission channels (488/532/638).
"""

ALEX_LOGICAL_CHANNELS = (
    "532ex_532",
    "532ex_638",
    "488ex_532",
    "488ex_488",
)

ALEX_6CH_INDEX_MAP = (0, 1, 3, 5)
ALEX_4CH_INDEX_MAP = (0, 1, 2, 3)

ALEX_PHYSICAL_CHANNELS = {
    "532ex_532": "532",
    "532ex_638": "638",
    "488ex_532": "532",
    "488ex_488": "488",
}


def resolve_channel_profile(raw_num_channels: int) -> dict:
    raw_num_channels = int(raw_num_channels)
    if raw_num_channels == 6:
        return {
            "profile": "alex_4ch",
            "channel_names": list(ALEX_LOGICAL_CHANNELS),
            "channel_index_map": list(ALEX_6CH_INDEX_MAP),
        }
    if raw_num_channels == 4:
        return {
            "profile": "alex_4ch",
            "channel_names": list(ALEX_LOGICAL_CHANNELS),
            "channel_index_map": list(ALEX_4CH_INDEX_MAP),
        }
    return {
        "profile": "standard",
        "channel_names": None,
        "channel_index_map": list(range(max(0, raw_num_channels))),
    }


def physical_channel_name(channel_name) -> str:
    name = str(channel_name)
    return ALEX_PHYSICAL_CHANNELS.get(name, name)


def is_reference_logical_channel(channel_name, reference_channel: str = "532") -> bool:
    return physical_channel_name(channel_name) == str(reference_channel)
