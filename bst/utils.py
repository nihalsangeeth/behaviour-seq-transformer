def pad(seq, max_seq_len, pad_with=0):
    seq_len = len(seq)
    return [pad_with]*(max_seq_len - seq_len) + seq

def batch_fn(user_seq, context_features, batch_size, max_seq_len, shuffle=True):
    if shuffle:
        data = list(zip(user_seq, context_features))
        random.shuffle(data)
        user_seq, context_features = zip(*data)
    context_features = np.array(context_features).T
    for start_idx in range(0, len(user_seq) - batch_size + 1, batch_size):
        batch = user_seq[start_idx:start_idx + batch_size]
        context_batch = context_features[..., start_idx:start_idx + batch_size].tolist()
        batch = [seq[-max_seq_len:] for seq in batch]
        user_seq_batch = []
        for seq in batch:
            pseq = pad(seq, max_seq_len)
            user_seq_batch += [pseq]
        yield user_seq_batch, context_batch
