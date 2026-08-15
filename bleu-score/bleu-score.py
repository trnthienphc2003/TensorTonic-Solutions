from math import exp, log

def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    # Write code here
    bleu = 0.
    c = len(candidate)
    if c == 0:
        return 0.
    r = len(reference)
    
    bp = 1. if c >= r else exp(1. - r / c)
    max_n = min(max_n, r)
    if max_n > c:
        return 0.
    
    sum_log_pn = 0.
    for n in range(1, max_n + 1):
        # compute ref value first
        ref_cnts = {}
        for i in range(r - n + 1):
            ref_ng = tuple(reference[i:i+n])
            ref_cnts[ref_ng] = ref_cnts.get(ref_ng, 0) + 1

        can_cnts = {}
        for i in range(c - n + 1):
            can_ng = tuple(candidate[i:i+n])
            can_cnts[can_ng] = can_cnts.get(can_ng, 0) + 1

        p_n = 0
        for can_ng, can_cnt in can_cnts.items():
            p_n += min(can_cnt, ref_cnts.get(can_ng, 0))
        p_n /= (c - n + 1)
        if p_n < 1e-8:
            return 0.
        sum_log_pn += log(p_n)

    bleu = bp * exp(sum_log_pn / max_n)
    return bleu