import numpy as np

def levenshtein(u, v):
    prev = None
    curr = [0] + list(range(1, len(v) + 1))
    # Operations: (SUB, DEL, INS)
    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
    for x in range(1, len(u) + 1):
        prev, curr = curr, [x] + ([None] * len(v))
        prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
        for y in range(1, len(v) + 1):
            delcost = prev[y] + 1
            addcost = curr[y - 1] + 1
            subcost = prev[y - 1] + 2*int(u[x - 1] != v[y - 1])
            curr[y] = min(subcost, delcost, addcost)
            if curr[y] == subcost:
                (n_s, n_d, n_i) = prev_ops[y - 1]
                curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
            elif curr[y] == delcost:
                (n_s, n_d, n_i) = prev_ops[y]
                curr_ops[y] = (n_s, n_d + 1, n_i)
            else:
                (n_s, n_d, n_i) = curr_ops[y - 1]
                curr_ops[y] = (n_s, n_d, n_i + 1)
    return curr[len(v)], curr_ops[len(v)]

def compute_error(ref, hyp):
    wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
    cer_s, cer_i, cer_d, cer_n = 0, 0, 0, 0
    sen_err = 0

    # update CER statistics
    _, (s, i, d) = levenshtein(ref, hyp)
    cer_s += s
    cer_i += i
    cer_d += d
    cer_n += len(ref)
    # update WER statistics
    _, (s, i, d) = levenshtein(ref.split(), hyp.split())
    wer_s += s
    wer_i += i
    wer_d += d
    wer_n += len(ref.split())
    
    # update SER statistics
    if s + i + d > 0:
        sen_err += 1

    if cer_n > 0 and wer_n > 0:
        return(
            (100.0 * (cer_s + cer_i + cer_d)) / cer_n,
            (100.0 * (wer_s + wer_i + wer_d)) / wer_n)
    
    return np.nan, np.nan


def compute_errors(ref, hyp):
    wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
    cer_s, cer_i, cer_d, cer_n = 0, 0, 0, 0
    sen_err = 0
    for n in range(len(ref)):
        # update CER statistics
        _, (s, i, d) = levenshtein(ref[n], hyp[n])
        cer_s += s
        cer_i += i
        cer_d += d
        cer_n += len(ref[n])
        # update WER statistics
        _, (s, i, d) = levenshtein(ref[n].split(), hyp[n].split())
        wer_s += s
        wer_i += i
        wer_d += d
        wer_n += len(ref[n].split())
        # update SER statistics
        if s + i + d > 0:
            sen_err += 1

    if cer_n > 0 and wer_n > 0:
        return(
            (100.0 * (cer_s + cer_i + cer_d)) / cer_n,
            (100.0 * (wer_s + wer_i + wer_d)) / wer_n,
            (100.0 * sen_err) / len(ref))

    return np.nan, np.nan