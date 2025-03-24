import random
import tqdm


def shuffle_stream(iterator, buffer_size=100, seed=None, min_age=0, desc=None):
    """
    Shuffles an iterator using a fixed-size buffer with a minimum age constraint.

    Once the buffer is filled, each new item replaces an item from an "eligible"
    region of the buffer. The eligible region is defined such that an item that was
    just inserted is kept out of that region until at least 'min_age' subsequent
    replacements have occurred.

    To ensure at least one slot is always eligible, we clamp min_age to at most
    (buffer_size - 1). The eligible window is then of size:
          n_eligible = len(buffer) - effective_min_age
    where effective_min_age = min(min_age, len(buffer)-1).

    The algorithm uses an indices list and an offset pointer. The eligible region
    is the consecutive block starting at `offset` (modulo the buffer length) of
    size n_eligible.

    Parameters:
      iterator (iterable): The input iterable.
      buffer_size (int): The number of items to store in the buffer.
      seed (int, optional): A seed for random number generation.
      min_age (int): The minimum number of new insertions that must occur before
                     a slot can be replaced again.

    Yields:
      Items from the iterator in a shuffled order.
    """
    buffer = []
    indices = []
    offset = 0  # points to the beginning of the eligible region

    # Set up the random number generator.
    rng = random if seed is None else random.Random(seed)

    tqdm_kwargs = {"desc": desc, "leave": False} if desc else {"disable": True}
    progress = tqdm.tqdm(**tqdm_kwargs)
    progress.total = 0

    # Fill the buffer.
    for i in range(buffer_size):
        try:
            buffer.append(next(iterator))
            progress.total += 1
            progress.refresh()
            indices.append(i)
        except StopIteration:
            break

    total = len(buffer)
    if total == 0:
        return

    # Clamp min_age so that effective_min_age is at most total-1.
    effective_min_age = min(min_age, total - 1)
    # The eligible window size is then:
    n_eligible = total - effective_min_age  # always at least 1

    # Process new items from the iterator.
    for item in iterator:
        progress.total += 1
        progress.refresh()
        # Pick a random index from the eligible window.
        # Eligible window positions in 'indices' are offset, offset+1, ..., offset+n_eligible-1 (mod total).
        pos = rng.randrange(n_eligible)
        j = (offset + pos) % total
        idx = indices[j]
        yield buffer[idx]
        progress.update() and progress.refresh()
        # Replace the chosen slot with the new item.
        buffer[idx] = item
        # Swap the chosen index with the one at the current offset.
        indices[j], indices[offset] = indices[offset], indices[j]
        # Advance offset cyclically.
        offset = (offset + 1) % total

    # When the iterator is exhausted, yield the remaining items in random order.
    rng.shuffle(indices)
    for i in indices:
        yield buffer[i]
        progress.update() and progress.refresh()
    progress.close()
