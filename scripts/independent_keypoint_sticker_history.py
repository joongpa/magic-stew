"""Four-observation histories keyed by the 54 persistent physical stickers."""
STICKER_COUNT = 54
EMPTY = bytes(STICKER_COUNT)
MASK = (1 << STICKER_COUNT)-1
# High nibble: observation count, saturated at four. Low nibble: newest four
# outcomes, newest in bit zero; one means incorrect.
_MISMATCH = bytes(int(value >> 4 == 4 and (value & 15).bit_count() >= 3)
                  for value in range(256))
_QUALIFIED = bytes(int(value >> 4 == 4) for value in range(256))
# Fixed four-observation denominator: a first error already costs 1/16,
# without treating one early error as four errors. Missing observations do
# not vote, and a sticker that is not seen retains its previous penalty.
_WRONG_SQUARED = bytes((value & 15).bit_count() ** 2 for value in range(256))


def observe(history, payload):
    observed = int(payload.get('sticker_observed_mask_hex', '0'), 16)
    correct = int(payload.get('sticker_correct_mask_hex', '0'), 16)
    if len(history) != STICKER_COUNT or observed & ~MASK or correct & ~observed:
        raise ValueError('Invalid physical sticker observation masks or history')
    if not observed:
        return history
    result = bytearray(history)
    remaining = observed
    while remaining:
        bit = remaining & -remaining
        index = bit.bit_length()-1
        previous = result[index]
        count = min(4, (previous >> 4)+1)
        wrong = int(not correct & bit)
        result[index] = (count << 4) | (((previous & 15) << 1 | wrong) & 15)
        remaining ^= bit
    return bytes(result)


def mismatch_count(history):
    return sum(history.translate(_MISMATCH))


def penalty(history):
    return sum(history.translate(_WRONG_SQUARED)) / 16.


def summary(history):
    mismatched = sum(1 << i for i,value in enumerate(history) if _MISMATCH[value])
    return dict(sticker_mismatch_count=mismatched.bit_count(),
                sticker_history_penalty=penalty(history),
                sticker_history_qualified_count=sum(history.translate(_QUALIFIED)),
                sticker_mismatched_mask_hex=f'{mismatched:014x}')
