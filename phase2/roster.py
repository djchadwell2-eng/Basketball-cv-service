"""Per-game roster -- the CLOSED SET jersey OCR matches against.

This is a PARTIAL CALIBRATION roster for TEST1, built from jersey numbers that are
actually legible in the footage (OCR discovery pass + visual verification). It is
NOT full ground truth -- a real roster would be coach-entered (both full squads).
Its job here is (a) turn open OCR into closed-set matching (a read only counts if
it is a roster number), and (b) provide seed identities for the 3-outcome test.

SEED_LABELS are a stand-in for the coach seeding a player BY EYE (a first signal
INDEPENDENT of the automated OCR reader): I visually confirmed track 17 wears #13
and track 6 wears #5 from the crops. Those seeds carry a jersey number that
position-continuity then carries through breaks -- so a later OCR read can AGREE
(confirm) or DISAGREE (swap flag) with what position says.
"""

# Numbers discovered legibly by OCR on this footage (multi-sighting). #1/#23 were
# seen once each and excluded as likely misreads -> kept STRICT.
ROSTER_NUMBERS = {5, 13, 24}

# Loose team/color note (matching is by NUMBER; colour is not relied on).
TEAMS = {
    "Milford (white)": {13},
    "Little Miami (green)": {5, 24},
}

# track_id -> jersey number, hand-verified by eye (coach-seed stand-in, per clip).
SEED_LABELS = {
    "TEST1": {17: 13, 6: 5},
}


def seed_number_for(clip: str, track_id: int):
    return SEED_LABELS.get(clip, {}).get(track_id)
