#!/usr/bin/env python

from tqdm import tqdm

def boundaries_to_intervals(boundaries):
    """
    Convert a list of boundaries to a list of intervals.
    Args:
        boundaries (list): A list of boundaries, where 1 indicates a boundary. The boundary mast start with 0 and end with 1.
    return:
        intervals (list): A list of intervals, where each interval is a tuple of (start, end).
    """
    assert boundaries[-1] == 1
    n = len(boundaries)
    intervals = []
    start = end = 0
    while end < n:
        if boundaries[end] != 0:
            intervals.append((start, end))
            start = end
        end += 1
    return intervals


def boundary_metric(ref, seg, tolerance=0):
    n_tokens_seg = 0
    n_tokens_correct = 0
    for i_boundary, boundary_ref in tqdm(enumerate(ref),total=len(ref),desc="evaluating"):
        boundary_seg = seg[i_boundary]
        try:
            assert len(boundary_ref) == len(boundary_seg)
            assert boundary_ref[-1] == boundary_seg[-1] == 1
        except:
            breakpoint()

        # Build list of ((word_start_lower, word_start_upper), (word_end_lower, word_end_upper))
        word_bound_intervals = []
        for word_start, word_end in boundaries_to_intervals(boundary_ref):
            word_bound_intervals.append((
                (word_start - tolerance, word_start + tolerance),
                (word_end - tolerance, word_end + tolerance)
                ))
        seg_intervals = boundaries_to_intervals(boundary_seg)
        try:
            assert len(seg_intervals) == len(word_bound_intervals)
        except:
            breakpoint()
        n_tokens_seg += len(seg_intervals) * 2

        for i in range(len(seg_intervals)):
            seg_start, seg_end = seg_intervals[i]
            (start_lower, start_upper), (end_lower, end_upper) = word_bound_intervals[i]
            if start_lower <= seg_start <= start_upper:
                n_tokens_correct += 1
            if end_lower <= seg_end <= end_upper:
                n_tokens_correct += 1
    
    precision = n_tokens_correct/n_tokens_seg
    return precision

if __name__ == '__main__':
    boundaries = [1,0,0,1,0,0,1,0,1]
    print(boundaries_to_intervals(boundaries))