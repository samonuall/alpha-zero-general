from collections import Counter
import numpy as np

# standardise rank mapping (accept both 'T' and '10')
_RANK_STR_TO_INT = {
    '2': 2,  '3': 3,  '4': 4,  '5': 5,  '6': 6,  '7': 7,
    '8': 8,  '9': 9,  'T': 10, '10': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

def classify_board_texture(street, hole_cards, community_cards):
    """
    Classify the board by (suit_type, paired, connected).

    Parameters:
      street:            str, e.g. 'flop', 'turn', 'river'
      hole_cards:        tuple[str,str], e.g. ('s8','dK')  # currently unused
      community_cards:   list[str], e.g. ['hA','dT','c9']

    Returns:
      {'suit_type': str,   # 'rainbow'|'two-tone'|'monotone'
       'paired':    bool,  # any rank appears more than once
       'connected': bool}  # run of ≥3 unique consecutive ranks
    """
    # 1) normalize and validate
    comm = [c.strip().upper() for c in community_cards]
    if not 3 <= len(comm) <= 5:
        raise ValueError(f"Expected 3–5 community cards, got {len(comm)}")

    # 2) extract suits & ranks
    suits = [c[0] for c in comm]
    ranks = []
    for c in comm:
        rank_str = c[1:]
        if rank_str not in _RANK_STR_TO_INT:
            raise KeyError(f"Unknown rank '{rank_str}' in card '{c}'")
        ranks.append(_RANK_STR_TO_INT[rank_str])

    # 3) suit_type
    cnt = Counter(suits)
    mx = max(cnt.values())
    if mx == 1:
        suit_type = 'rainbow'
    elif mx == len(comm):
        suit_type = 'monotone'
    else:
        suit_type = 'two-tone'

    # 4) paired?
    rank_cnt = Counter(ranks)
    paired = any(v > 1 for v in rank_cnt.values())

    # 5) connected? (longest run in unique ranks)
    uniq = sorted(set(ranks))
    longest = cur = 1
    for i in range(1, len(uniq)):
        if uniq[i] == uniq[i-1] + 1:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    connected = (longest >= 3)

    return {
        'suit_type': suit_type,
        'paired': paired,
        'connected': connected
    }

def board_texture_to_onehot(texture: dict) -> np.ndarray:
    """
    Convert a board‐texture dict into a 7‐dim one‐hot vector.

    texture keys:
      'suit_type': 'rainbow' | 'two-tone' | 'monotone'
      'paired':    bool
      'connected': bool

    One‐hot layout (total 7 dims):
      [rainbow, two-tone, monotone,
       paired=False, paired=True,
       connected=False, connected=True]
    """
    # suit_type one‐hot
    suit_map = ['rainbow', 'two-tone', 'monotone']
    suit_vec = np.zeros(len(suit_map), dtype=np.int32)
    try:
        suit_vec[suit_map.index(texture['suit_type'])] = 1
    except ValueError:
        raise ValueError(f"Unknown suit_type: {texture['suit_type']}")

    # paired one‐hot
    paired_vec = np.zeros(2, dtype=np.int32)
    paired_vec[1 if texture['paired'] else 0] = 1

    # connected one‐hot
    conn_vec = np.zeros(2, dtype=np.int32)
    conn_vec[1 if texture['connected'] else 0] = 1

    # concatenate
    return np.concatenate([suit_vec, paired_vec, conn_vec], axis=0)

# ——— Quick smoke tests ———
if __name__ == "__main__":
    # rainbow & connected
    r1 = classify_board_texture('flop', (), ['s8','d9','hT'])
    assert r1 == {'suit_type':'rainbow','paired':False,'connected':True}

    # two-tone & paired
    r2 = classify_board_texture('turn',(),['dA','hA','d5','dK'])
    assert r2 == {'suit_type':'two-tone','paired':True,'connected':False}

    # monotone & not connected
    r3 = classify_board_texture('river',(),['s2','s7','sJ','sQ','sA'])
    assert r3 == {'suit_type':'monotone','paired':False,'connected':False}

    print("All board texture tests passed!")

    example = {'suit_type':'two-tone','paired':True,'connected':False}
    oh = board_texture_to_onehot(example)
    # Expect: [0,1,0, 0,1, 1,0]
    print("one-hot:", oh)