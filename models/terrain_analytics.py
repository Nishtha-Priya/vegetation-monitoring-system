def dominant_terrain(stats):

    if not stats:
        return "Unknown"

    return max(
        stats,
        key=stats.get
    )